# 设置cmake版本，如果cmake版本过高，可能会出现错误
cmake_minimum_required(VERSION 3.5.1)
project(NCNN_test)

# 设置C++编译版本
set(CMAKE_CXX_STANDARD 11)

# 设置程序编译出的可执行文件
set(MAIN_FILE main.cpp)
set(EXECUTABLE_OUTPUT_PATH ./)

# 分别设置ncnn的链接库和头文件
set(NCNN_LIBS /home/pc/bbuf_project/ncnn/build/install/lib/libncnn.a)
set(NCNN_INCLUDE_DIRS /home/pc/bbuf_project/ncnn/build/install/include/ncnn)

# 配置OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 配置OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${NCNN_INCLUDE_DIRS})

# 建立链接依赖
add_executable(NCNN_SEG main.cpp)
target_link_libraries(NCNN_SEG ${NCNN_LIBS})
target_link_libraries(NCNN_SEG ${OpenCV_LIBS})