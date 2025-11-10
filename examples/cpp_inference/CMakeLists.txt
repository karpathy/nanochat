cmake_minimum_required(VERSION 3.14)
project(nanochat_cpp_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build LibTorch example
option(BUILD_LIBTORCH_EXAMPLE "Build LibTorch inference example" ON)

# Option to build ONNX Runtime example
option(BUILD_ONNX_EXAMPLE "Build ONNX Runtime inference example" ON)

# LibTorch example
if(BUILD_LIBTORCH_EXAMPLE)
    find_package(Torch REQUIRED)
    
    add_executable(libtorch_inference libtorch_inference.cpp)
    target_link_libraries(libtorch_inference "${TORCH_LIBRARIES}")
    
    # Set C++17 for LibTorch
    set_property(TARGET libtorch_inference PROPERTY CXX_STANDARD 17)
    
    # Copy DLLs on Windows
    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET libtorch_inference
                         POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E copy_if_different
                         ${TORCH_DLLS}
                         $<TARGET_FILE_DIR:libtorch_inference>)
    endif()
    
    message(STATUS "LibTorch inference example will be built")
endif()

# ONNX Runtime example
if(BUILD_ONNX_EXAMPLE)
    # Find ONNX Runtime
    # You can set ONNXRUNTIME_DIR to point to your ONNX Runtime installation
    if(DEFINED ENV{ONNXRUNTIME_DIR})
        set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
    endif()
    
    if(ONNXRUNTIME_DIR)
        message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
        
        add_executable(onnx_inference onnx_inference.cpp)
        
        target_include_directories(onnx_inference PRIVATE 
            ${ONNXRUNTIME_DIR}/include
        )
        
        if(WIN32)
            target_link_libraries(onnx_inference 
                ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
            )
        elseif(APPLE)
            target_link_libraries(onnx_inference 
                ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib
            )
        else()
            target_link_libraries(onnx_inference 
                ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
            )
        endif()
        
        message(STATUS "ONNX Runtime inference example will be built")
    else()
        message(WARNING "ONNXRUNTIME_DIR not set. ONNX example will not be built.")
        message(WARNING "Set ONNXRUNTIME_DIR environment variable or pass -DONNXRUNTIME_DIR=/path/to/onnxruntime")
    endif()
endif()

# Print build configuration
message(STATUS "")
message(STATUS "nanochat C++ Inference Examples")
message(STATUS "================================")
message(STATUS "Build LibTorch example: ${BUILD_LIBTORCH_EXAMPLE}")
message(STATUS "Build ONNX example: ${BUILD_ONNX_EXAMPLE}")
message(STATUS "")
