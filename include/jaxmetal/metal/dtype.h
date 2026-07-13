#pragma once
#include <cstddef>

namespace jaxmetal {

// Scalar element types. f32 is the only compute type for now (see CLAUDE.md);
// i32/bool exist for future convert/compare/select support.
enum class DType { F32, I32, Bool };

inline size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::F32: return 4;
    case DType::I32: return 4;
    case DType::Bool: return 1;
  }
  return 0;
}

inline const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::F32: return "f32";
    case DType::I32: return "i32";
    case DType::Bool: return "bool";
  }
  return "?";
}

}  // namespace jaxmetal
