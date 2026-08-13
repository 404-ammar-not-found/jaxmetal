# Documentation

## Design

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layer diagram, runtime primitives, execution surfaces, verification strategy |
| [PJRT_PLUGIN.md](PJRT_PLUGIN.md) | Roadmap for a real `metal` PJRT device (not yet built) |

## Features

One document per feature, added when the feature lands. Each answers the same four
questions, because a feature that cannot answer them is not finished:

1. **What problem does it solve**, stated as something that was measurably wrong before.
2. **How it works**, at the level of detail needed to change it safely.
3. **What it measures**, with the command that reproduces the numbers.
4. **What it does not do** — the limits, and the things deliberately left out.

| Feature | Since | Summary |
|---|---|---|
| [features/GPU_CHOLESKY.md](features/GPU_CHOLESKY.md) | v0.4.0 | Blocked right-looking Cholesky, one command buffer, MPS GEMM for the trailing update. Parity with direct LAPACK at N=4096, 13× faster than Apple's own MPS Cholesky. Records two refuted optimisations. |
| [features/COMPENSATED_REDUCTIONS.md](features/COMPENSATED_REDUCTIONS.md) | v0.3.0 | Neumaier f32 summation. 127× more accurate than a tree sum on adversarial input, at no measurable cost. Works around Apple GPUs having no `float64`. |
| [features/CHUNKED_TRAINING.md](features/CHUNKED_TRAINING.md) | v0.2.0 | Many SGD steps per Metal command buffer. Cut MLP per-step cost ~4.4× and moved the GPU-beats-CPU crossover from batch >2048 to ~60. |
| [features/DEVICE_ROUTING.md](features/DEVICE_ROUTING.md) | v0.2.0 | `Mlp(device="auto")`. Picks GPU or CPU from a measured crossover model, because the GPU is not faster at every size. |

## Conventions for new feature docs

- **Lead with the measurement, not the design.** State the number that was wrong and
  the number it is now. A feature doc with no numbers in it is a design doc.
- **Record refuted hypotheses.** Approaches that were tried and measured as worse are
  the most expensive knowledge in the repo and the easiest to lose. Say what was
  tried, what it measured, and why it lost.
- **Name the load-bearing invariants.** Anything that silently breaks the feature if
  changed — a compiler flag, a constant that must match between host and kernel, an
  ordering assumption — belongs in the doc *and* in a test.
- **State the limits honestly.** Every feature here has a regime where it loses. Say
  where, and whether that is a hard limit or unfinished work.
