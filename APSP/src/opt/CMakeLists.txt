add_library(apsp_opt SHARED)
target_sources(apsp_opt PRIVATE apsp.cc)

option(USE_OPENMP      "Set to ON to build use openmp"  ON) 
option(USE_SSE         "Set to ON to build use SSE"  ON)
option(USE_AVX         "Set to ON to build use AVX"  ON)

if (USE_OPENMP)
    find_package(OpenMP QUIET)   
    if (OPENMP_FOUND)
        message(STATUS "Use OpenMP")
        add_definitions(-DUSE_OPENMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()
# Use SSE                                                 
if (USE_SSE)
    add_definitions(-DUSE_SSE)
    message(STATUS "Use SSE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()
# Use AVX
if (USE_AVX)
    add_definitions(-DUSE_AVX)
    message(STATUS "Use AVX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()

find_package(Threads REQUIRED)
target_link_libraries(apsp_opt Threads::Threads)
find_package(OpenMP REQUIRED)
target_link_libraries(apsp_opt OpenMP::OpenMP_CXX)
add_library(sse4_1 INTERFACE IMPORTED)
target_compile_options(apsp_opt PRIVATE -msse4.2)
set_target_properties(${TARGET} PROPERTIES
COMPILE_FLAGS "-pthread"
LINK_FLAGS "-pthread")
add_library(avx INTERFACE IMPORTED)
target_compile_options(apsp_opt PRIVATE -mavx -mavx2 -mavx512f)


target_compile_options(apsp_opt PRIVATE -Ofast)