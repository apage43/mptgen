cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("minmpt.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(BUILD_SHARED_LIBS ON)


#
# Option list
#

# general
option(MINMPT_STATIC                 "minmpt: static link libraries"                          OFF)
option(MINMPT_NATIVE                 "minmpt: enable -march=native flag"                      OFF)
option(MINMPT_LTO                    "minmpt: enable link time optimization"                  OFF)

# debug
option(MINMPT_ALL_WARNINGS           "minmpt: enable all compiler warnings"                   ON)
option(MINMPT_ALL_WARNINGS_3RD_PARTY "minmpt: enable all compiler warnings in 3rd party libs" OFF)
option(MINMPT_GPROF                  "minmpt: enable gprof"                                   OFF)

# sanitizers
option(MINMPT_SANITIZE_THREAD        "minmpt: enable thread sanitizer"                        OFF)
option(MINMPT_SANITIZE_ADDRESS       "minmpt: enable address sanitizer"                       OFF)
option(MINMPT_SANITIZE_UNDEFINED     "minmpt: enable undefined sanitizer"                     OFF)

# instruction set specific
option(MINMPT_AVX                    "minmpt: enable AVX"                                     ON)
option(MINMPT_AVX2                   "minmpt: enable AVX2"                                    ON)
option(MINMPT_AVX512                 "minmpt: enable AVX512"                                  OFF)
option(MINMPT_AVX512_VBMI            "minmpt: enable AVX512-VBMI"                             OFF)
option(MINMPT_AVX512_VNNI            "minmpt: enable AVX512-VNNI"                             OFF)
option(MINMPT_FMA                    "minmpt: enable FMA"                                     ON)
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(MINMPT_F16C               "minmpt: enable F16C"                                    ON)
endif()

# 3rd party libs
option(MINMPT_ACCELERATE             "minmpt: enable Accelerate framework"                    ON)
option(MINMPT_OPENBLAS               "minmpt: use OpenBLAS"                                   OFF)
option(MINMPT_CUBLAS                 "minmpt: use cuBLAS"                                     OFF)
option(MINMPT_CLBLAST                "minmpt: use CLBlast"                                    OFF)

#
# Build info header
#

# Generate initial build-info.h
include(${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.cmake)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${REAL_GIT_DIR}")
    endif()

    # Add a custom target for build-info.h
    add_custom_target(BUILD_INFO ALL DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build-info.h")

    # Add a custom command to rebuild build-info.h when .git/index changes
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/build-info.h"
        COMMENT "Generating build details from Git"
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.cmake"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS "${GIT_DIR}/index"
        VERBATIM
    )
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
endif()

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT MSVC)
    if (MINMPT_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (MINMPT_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (MINMPT_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND MINMPT_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(MINMPT_EXTRA_LIBS ${MINMPT_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (MINMPT_OPENBLAS)
    if (MINMPT_STATIC)
        set(BLA_STATIC ON)
    endif()

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "OpenBLAS found")

        add_compile_definitions(GGML_USE_OPENBLAS)
        add_link_options(${BLAS_LIBRARIES})
        set(MINMPT_EXTRA_LIBS ${MINMPT_EXTRA_LIBS} openblas)

        # find header file
        set(OPENBLAS_INCLUDE_SEARCH_PATHS
            /usr/include
            /usr/include/openblas
            /usr/include/openblas-base
            /usr/local/include
            /usr/local/include/openblas
            /usr/local/include/openblas-base
            /opt/OpenBLAS/include
            $ENV{OpenBLAS_HOME}
            $ENV{OpenBLAS_HOME}/include
            )
        find_path(OPENBLAS_INC NAMES cblas.h PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS})
        add_compile_options(-I${OPENBLAS_INC})
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

if (MINMPT_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_CUDA_SOURCES ggml-cuda.cu ggml-cuda.h)

        add_compile_definitions(GGML_USE_CUBLAS)

        if (MINMPT_STATIC)
            set(MINMPT_EXTRA_LIBS ${MINMPT_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        else()
            set(MINMPT_EXTRA_LIBS ${MINMPT_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (MINMPT_CLBLAST)
    find_package(CLBlast)
    if (CLBlast_FOUND)
        message(STATUS "CLBlast found")

        set(GGML_OPENCL_SOURCES ggml-opencl.c ggml-opencl.h)

        add_compile_definitions(GGML_USE_CLBLAST)

        set(MINMPT_EXTRA_LIBS ${MINMPT_EXTRA_LIBS} clblast)
    else()
        message(WARNING "CLBlast not found")
    endif()
endif()

if (MINMPT_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wdouble-promotion
            -Wshadow
            -Wstrict-prototypes
            -Wpointer-arith
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
            -Wno-multichar
        )
    else()
        # todo : msvc
    endif()

    add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (MINMPT_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (MINMPT_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (MINMPT_GPROF)
        add_compile_options(-pg)
    endif()
    if (MINMPT_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            # Apple M1, M2, etc.
            # Raspberry Pi 3, 4, Zero 2 (64-bit)
            add_compile_options(-mcpu=native)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            # Raspberry Pi 2
            add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            add_compile_options(-mfp16-format=ieee -mno-unaligned-access)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (MINMPT_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (MINMPT_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (MINMPT_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
        elseif (MINMPT_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (MINMPT_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        if (MINMPT_F16C)
            add_compile_options(-mf16c)
        endif()
        if (MINMPT_FMA)
            add_compile_options(-mfma)
        endif()
        if (MINMPT_AVX)
            add_compile_options(-mavx)
        endif()
        if (MINMPT_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (MINMPT_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
        if (MINMPT_AVX512_VBMI)
            add_compile_options(-mavx512vbmi)
        endif()
        if (MINMPT_AVX512_VNNI)
            add_compile_options(-mavx512vnni)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    add_compile_options(-mcpu=native -mtune=native)
    #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
else()
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

add_library(ggml OBJECT
            ggml.c
            ggml.h
            ${GGML_CUDA_SOURCES}
            ${GGML_OPENCL_SOURCES})

target_include_directories(ggml PUBLIC .)
target_compile_features(ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PUBLIC Threads::Threads ${MINMPT_EXTRA_LIBS})

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(minmpt
            mpt.cpp
            mpt.h
            mpt-util.h
            minmpt.cpp
            minmpt.h)


target_include_directories(minmpt PUBLIC .)
target_compile_features(minmpt PUBLIC cxx_std_11) # don't bump
target_link_libraries(minmpt PRIVATE ggml ${MINMPT_EXTRA_LIBS})

add_executable(quantize
               quantize.cpp)

target_include_directories(quantize PUBLIC .)
target_compile_features(quantize PUBLIC cxx_std_11) # don't bump
target_link_libraries(quantize PRIVATE ggml minmpt ${MINMPT_EXTRA_LIBS})


if (BUILD_SHARED_LIBS)
    set_target_properties(minmpt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(minmpt PRIVATE MINMPT_SHARED MINMPT_BUILD)
endif()

if (GGML_CUDA_SOURCES)
    message(STATUS "GGML CUDA sources found, configuring CUDA architecture")
    set_property(TARGET ggml PROPERTY CUDA_ARCHITECTURES OFF)
    set_property(TARGET ggml PROPERTY CUDA_SELECT_NVCC_ARCH_FLAGS "Auto")
    set_property(TARGET minmpt PROPERTY CUDA_ARCHITECTURES OFF)
endif()