CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
project(CSP)

set(CMAKE_CXX_STANDARD 17)
include_directories(src)
add_executable(CSP src/csp.cpp)
