# Run via: cmake -DIN=<src.metal> -DOUT=<hdr.h> -DVAR=<name> -P embed_one.cmake
# Emits a header exposing the MSL source as: jaxmetal::kernels::<VAR> (const char*).
file(READ "${IN}" _content)
set(_gen "// Auto-generated from ${IN} by cmake/embed_one.cmake. Do not edit.\n")
string(APPEND _gen "#pragma once\n")
string(APPEND _gen "namespace jaxmetal { namespace kernels {\n")
string(APPEND _gen "inline constexpr const char* ${VAR} = R\"MSL(\n")
string(APPEND _gen "${_content}")
string(APPEND _gen ")MSL\";\n")
string(APPEND _gen "}}  // namespace jaxmetal::kernels\n")
file(WRITE "${OUT}" "${_gen}")
