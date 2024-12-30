cmake_minimum_required(VERSION 3.10)
project(IsingModelTraining LANGUAGES CXX)


# Set the C++ compiler
set(CMAKE_CXX_COMPILER clang++)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Collect all .cpp files in the src folder and its subdirectories
file(GLOB SOURCES "src/*.cpp")
file(GLOB_RECURSE ADDITIONAL_SOURCES "src/**/*.cpp")


# Combine them together
list(APPEND SOURCES ${ADDITIONAL_SOURCES})


# Include directories if needed
include_directories(${CMAKE_SOURCE_DIR}/include)

# Linker flags
set(LINK_LIBS yaml-cpp)

# Target: configuration_generator
add_executable(configuration_generator ${SOURCES} configuration_generator.cpp)
target_link_libraries(configuration_generator ${LINK_LIBS})

# Target: mem_ising_training_main
add_executable(mem_ising_training_main ${SOURCES} training-main.cpp)
target_link_libraries(mem_ising_training_main PRIVATE ${LINK_LIBS})

add_executable(dynamical_mem_ising_training_main ${SOURCES} dynamical-ising-training-main.cpp)
target_link_libraries(dynamical_mem_ising_training_main PRIVATE ${LINK_LIBS})

add_executable(partition_function_inference ${SOURCES} partition-function-inference.cpp)
target_link_libraries(partition_function_inference PRIVATE ${LINK_LIBS})

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mem_ising_training_main PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(mem_ising_training_main PRIVATE ${OpenMP_CXX_FLAGS})
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -latomic -g -O3")

# Custom clean command
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/configuration_generator
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mem_ising_training_main
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dynamical-ising-training-main
    COMMENT "Cleaning generated binaries"
)
