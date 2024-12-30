cmake_minimum_required(VERSION 3.10)

project(snn-deployment-simulation VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_compile_options(-Werror -Wextra -Wpedantic -O3)
add_compile_options(-Wno-unused-parameter)


file(GLOB_RECURSE SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/**/*.cpp
)
file(GLOB_RECURSE HEADER_FILES
    ${CMAKE_SOURCE_DIR}/include/*.hpp
    ${CMAKE_SOURCE_DIR}/include/**/*.cpp
)


add_subdirectory(./snn-lib-cpp)

# Add an executable target
add_executable(lif-simulation-main ${SOURCE_FILES} ${HEADER_FILES} lif-simulation-main.cpp)
add_executable(possion-lif-main ${SOURCE_FILES} ${HEADER_FILES} test_possion_lif_neurons.cpp)

# Specify the include directories for header files
target_include_directories(lif-simulation-main
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(lif-simulation-main
    PRIVATE
    ${CMAKE_SOURCE_DIR}/snn-lib-cpp/include
)

target_include_directories(possion-lif-main 
    PRIVATE
    ${CMAKE_SOURCE_DIR}/snn-lib-cpp/include
)

target_include_directories(possion-lif-main 
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)



file(GLOB_RECURSE SNN_LIB_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/snn-lib-cpp/src/*.cpp
    ${CMAKE_SOURCE_DIR}/snn-lib-cpp/src/**/*.cpp
)
file(GLOB_RECURSE SNN_LIB_INCLUDE_FILES
    ${CMAKE_SOURCE_DIR}/snn-lib-cpp/include/*.hpp
    ${CMAKE_SOURCE_DIR}/snn-lib-cpp/include/**/*.hpp
)
list(REMOVE_ITEM SNN_LIB_SOURCE_FILES ${CMAKE_SOURCE_DIR}/snn-lib-cpp/src/snn_main.cpp)


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()


add_library(snn-lib-cpp STATIC ${SNN_LIB_SOURCE_FILES})
target_include_directories(snn-lib-cpp
    PUBLIC
    ${CMAKE_SOURCE_DIR}/snn-lib-cpp/include
)
target_link_libraries(lif-simulation-main snn-lib-cpp)
target_link_libraries(possion-lif-main snn-lib-cpp)

target_compile_options(snn-lib-cpp PRIVATE -Wno-unused-parameter -g)

file(GLOB_RECURSE EXPERIMENT_LIB_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/experiments/src/*.cpp
    ${CMAKE_SOURCE_DIR}/experiments/src/**/*.cpp
)
file(GLOB_RECURSE EXPERIMENT_LIB_INCLUDE_FILES
    ${CMAKE_SOURCE_DIR}/experiments/include/*.hpp
    ${CMAKE_SOURCE_DIR}/experiments/include/**/*.hpp
)
