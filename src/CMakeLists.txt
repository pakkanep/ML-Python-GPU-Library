cmake_minimum_required(VERSION 3.18)
project(ml_gpu_library LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate nanobind's CMake config via its Python package
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
  
find_package(nanobind CONFIG REQUIRED)

# Build the module
nanobind_add_module(ml_gpu_library binding.cpp)

#CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  target_sources(ml_gpu_library PRIVATE cuda/matmul.cu) # this compiles if cuda found
  target_compile_definitions(ml_gpu_library PRIVATE HAVE_CUDA)
  set_target_properties(ml_gpu_library PROPERTIES CUDA_ARCHITECTURES native)
  message(STATUS "CUDA backend: ON")
endif()


find_package(OpenCL)
if(OpenCL_FOUND)
  target_sources(ml_gpu_library PRIVATE opencl/matmul.cpp) # this compiles if cuda not found
  target_link_libraries(ml_gpu_library PRIVATE OpenCL::OpenCL) 
  target_compile_definitions(ml_gpu_library PRIVATE HAVE_OPENCL CL_TARGET_OPENCL_VERSION=120)
  message(STATUS "OpenCL backend: ON")
endif()