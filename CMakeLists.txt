cmake_minimum_required(VERSION 2.8)

project(MPISort)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(SOURCES
    task.cpp
)

add_executable(mpisort ${SOURCES})

target_link_libraries(mpisort PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(mpisort PRIVATE MPI::MPI_CXX)
