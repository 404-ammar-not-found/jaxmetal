# embed_metal(SRC HDR VAR)
#   Registers a *build-time* rule that regenerates HDR from the MSL file SRC
#   whenever SRC changes, exposing its contents as jaxmetal::kernels::VAR. Appends
#   HDR to EMBED_OUTPUTS in the caller's scope.
set(_EMBED_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/embed_one.cmake)

function(embed_metal SRC HDR VAR)
  add_custom_command(
    OUTPUT "${HDR}"
    COMMAND ${CMAKE_COMMAND} -DIN=${SRC} -DOUT=${HDR} -DVAR=${VAR} -P ${_EMBED_SCRIPT}
    DEPENDS "${SRC}" "${_EMBED_SCRIPT}"
    COMMENT "Embedding kernel ${SRC}"
    VERBATIM)
  set(EMBED_OUTPUTS ${EMBED_OUTPUTS} "${HDR}" PARENT_SCOPE)
endfunction()
