# mpi
option(ENABLE_RESILIENCE "Enables Resilience Plugin" OFF)
if(ENABLE_RESILIENCE)
    set(LZ4_ENABLED ON CACHE INTERNAL)
    message(STATUS "Resilience Enabled")
else()
    set(LZ4_ENABLED OFF CACHE INTERNAL)
    message(STATUS "Resilience Disabled")
endif()