cmake_minimum_required(VERSION 3.10)

project(AttnPThread VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

include_directories(${INCLUDE_DIR})

file(GLOB SOURCES "${SRC_DIR}/*.c")

add_executable(AttnPThread ${SOURCES})

target_compile_options(AttnPThread PRIVATE -Wall -Wextra -pedantic)

target_link_libraries(AttnPThread -lm)