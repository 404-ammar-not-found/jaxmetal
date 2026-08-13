# Chunked training steps (many SGD steps per command buffer)

**Since v0.2.0** · `src/ops/mlp.mm` · `include/jaxmetal/ops/mlp.h` ·
`tests/python/test_mlp_auto.py`

## The problem

The resident MLP already did the thing everyone gets wrong — it encoded a whole
forward + backward + SGD step into a single Metal command buffer. It was still
slower than the CPU at every batch size tested at `hidden=128`:

| batch | GPU step | JAX-CPU step | ratio |
|------:|---------:|-------------:|------:|
| 1 | 0.446 ms | 0.019 ms | 0.04× |
| 128 | 0.619 ms | 0.169 ms | 0.27× |
| 2048 | 1.314 ms | 1.075 ms | 0.82× |

A linear fit separated it cleanly:

- **GPU: 0.510 ms fixed + 0.392 µs/sample**
- **CPU: 0.078 ms fixed + 0.487 µs/sample**

The GPU's *marginal* rate already won. It lost entirely on a **fixed ~0.5 ms per
step** — and at batch 1, where the step does essentially no arithmetic, it still
took 0.446 ms. So the problem was not compute.

## Where the half-millisecond went

Profiling a batch=1 step (`JAXMETAL_PROFILE=1`, still available) decomposed it:

```
~95 µs   command buffer round trip  (irreducible with a per-step sync)
~100 µs  CPU-side encoding
~124 µs  GPU execution
~140 µs  driver scheduling latency (wait − gpu)
```

Measuring a bare `1×1×1` resident matmul — one dispatch, one `commit`, one
`waitUntilCompleted` — costs **95–110 µs** on an M4 Pro. That is the floor for
*any* host-synchronised step, and it is why per-step submission could never win at
small batch.

## What changed

Five things, in order of measured payoff:

1. **`train_steps(n)` encodes `n` consecutive SGD steps into ONE command buffer.**
   The round trip is paid once per chunk instead of once per step, and encoding step
   *i+1* overlaps the GPU executing step *i*. Steps stay correctly ordered because
   Metal hazard-tracks within a command buffer, and each step's SGD update is a
   read-after-write on the same resident parameter buffers.

2. **`nn_reduce_sum_axis0` rewritten** as one threadgroup per output column with a
   tree reduction. The previous one-thread-per-column version dispatched **10
   threads** for `db2` (`C = 10` classes), each walking the entire batch serially, and
   scaled linearly in batch size. This single change took batch=2048 from 1.31 ms to
   0.79 ms.

3. **MPS objects cached per `(batch, chunk slot)`.** `MPSMatrixMultiplication` +
   `MPSMatrix` views were being rebuilt every step — about 28 Objective-C allocations
   per step, for objects whose shapes never change.

4. **MPS `transposeLeft` / `transposeRight`** for the three backward contractions, so
   `xᵀ` / `h1ᵀ` / `W2ᵀ` are never materialised. Removed 3 dispatches, 3 buffers, and
   their memory traffic. The `MPSMatrix` descriptors stay the *stored* (untransposed)
   shapes; `interiorColumns` is K *after* transposition.

5. **A `Coalescer`** holds one compute encoder open across consecutive kernel
   dispatches and closes it only when an MPS matmul needs the buffer — PyTorch MPS's
   `endKernelCoalescing()`. Serial dispatch means read-after-write chains inside a
   group need no explicit barriers.

Loss is also reduced on the GPU into `loss_sums[slot]`; the host no longer walks the
batch after each step.

## Load-bearing invariants

> **`train_steps(n)` must stay bit-identical to `n` separate `train_step()` calls.**
> The whole premise is that batching changes *when* work is submitted, never what is
> computed. `test_chunked_matches_per_step` in `tests/python/test_mlp_auto.py` asserts
> exact equality (`max|Δ| == 0`) on all four parameter arrays, not a tolerance.

- Chunk slot `i` reads rows `[i·batch, (i+1)·batch)`. The two GEMMs that read `x` use
  `MPSMatrix initWithBuffer:offset:descriptor:` at that row offset; the label buffer
  binding is offset to match.
- Input/label buffers are sized `max_batch × chunk_steps`; activations stay sized to
  one batch. Sizing activations by the chunk would multiply their memory by
  `chunk_steps` for no benefit.
- `upload_chunk` must supply `n_steps × batch` rows. The C side memcpys exactly that
  many, so a short upload reads past the end of the host array — this segfaulted
  during development and is now checked in `_capi.py`.

## Measurements

`.venv/bin/python examples/train_mnist.py --bench-only --batch <B> --hidden <H>`, or
the full sweep with `--calibrate`. Apple M4 Pro, `chunk_steps=128`:

| batch | `hidden=128` GPU | CPU | speedup | `hidden=1024` GPU | CPU | speedup |
|------:|-----------------:|----:|--------:|------------------:|----:|--------:|
| 32 | 0.124 ms | 0.097 ms | 0.79× | 0.200 ms | 0.319 ms | **1.60×** |
| 128 | 0.137 ms | 0.179 ms | **1.31×** | 0.274 ms | 0.664 ms | **2.42×** |
| 512 | 0.206 ms | 0.446 ms | **2.17×** | 0.605 ms | 1.827 ms | **3.02×** |
| 2048 | 0.496 ms | 1.382 ms | **2.79×** | 2.023 ms | 6.400 ms | **3.16×** |

**~4.4× faster per step** at `hidden=128` (batch 128: 0.619 → 0.137 ms). The
GPU-beats-CPU crossover moved from **batch >2048 to ~60**. MNIST still trains to
98.15%.

## Refuted: replacing MPS with our own kernel

The hypothesis was that MPS's one-encoder-per-matmul cost 9 encoders per step and
that using our own GEMM would collapse them into 1, winning at small batch. Our
kernel gained transpose support and the full step was rerouted through it.

**It was 2.7× slower at batch 1** (0.918 ms vs 0.336 ms), and slower at every batch
size tested. The 64×64 threadgroup tile dispatches only **2 threadgroups** for
`[1,784] @ [784,128]` — the skinny-M shapes that dominate small batches waste almost
the entire tile, and MPS has specialised kernels for exactly that case.

Reverted. The encoder-count theory was also wrong on its own terms: measuring
`forward` (4 encoders) against `train_step` (9 encoders) showed only **~4.6 µs** of
CPU cost per encoder. Encoders were never the expensive part.

## Limits and things left out

- **Below batch ~60 the CPU still wins, and that is a hard limit.** A Metal command
  buffer round trip cannot go below ~95 µs; the CPU's entire step at batch 1 is
  ~20 µs. No amount of kernel work closes that.
- **Encode and GPU execution are still sequential within a chunk** — roughly 74 µs of
  CPU encode against 67 µs of GPU time per step. A ring of command buffers overlapping
  them is worth up to ~1.6×. Not done: it needs double-buffered activations to avoid
  cross-command-buffer races, and the correctness risk was not worth taking
  unprompted.
- **`MTLIndirectCommandBuffer` (Metal's CUDA-Graphs analogue) is not used.** Compute
  ICBs exist, but MPS cannot be encoded into one, so it would force our own GEMM —
  which the refuted experiment above shows is a loss. Revisit only if profiling after
  chunking still shows encode cost.
