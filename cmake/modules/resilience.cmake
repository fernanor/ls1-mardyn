# mpi
option(ENABLE_RESILIENCE "Enables Resilience Plugin" OFF)
if(ENABLE_RESILIENCE)
    set(LZ4_ENABLED ON CACHE INTERNAL "Enables the module LZ4")
    message(STATUS "Resilience Enabled")
else()
    set(LZ4_ENABLED OFF CACHE INTERNAL "Enables the module LZ4")
    message(STATUS "Resilience Disabled")
endif()
