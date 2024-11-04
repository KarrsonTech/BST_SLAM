cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 20)
file(GLOB_RECURSE SRC _*.cpp *_*.cpp)
add_executable(${PROJECT_NAME} ${SRC}
 OAKCamera.h
)
source_group(TREE 
 ${CMAKE_CURRENT_SOURCE_DIR} 
 FILES ${SRC}
)

find_package(depthai CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} 
 PRIVATE depthai::opencv
)