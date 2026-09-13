# 1. 规定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 2. 定义项目名称，并声明这是一个 C++ 项目
project(cpu_gemm LANGUAGES CXX)

# 3. 设置 C++ 标准（推荐使用现代的 C++17）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. 告诉编译器去哪里找头文件 (对应你的 include/ 目录)
include_directories(${PROJECT_SOURCE_DIR}/include)

# 5. 定义要编译的源文件列表
# (根据手册，我们第一阶段要写 main.cpp 和 gemm_ijk.cpp)
# 1. 收集公共的源文件（所有算子实现和测速工具）
set(COMMON_SOURCES
    src/baseline/gemm_ijk.cpp
    src/baseline/gemm_ikj.cpp
    src/baseline/gemm_kij.cpp
    src/benchmark_utils.cpp
)

# 2. 跑分程序：main.cpp + 公共源文件
add_executable(benchmark_gemm
    src/main.cpp
    ${COMMON_SOURCES}
)

# 3. 单元测试程序：test_gemm.cpp + 公共源文件
add_executable(test_gemm
    tests/test_gemm.cpp
    ${COMMON_SOURCES}
)

enable_testing()
add_test(NAME GemmCorrectnessTest COMMAND test_gemm)

# 确保在 add_executable 之后加上这几行
if(MSVC)
    target_compile_options(benchmark_gemm PRIVATE /utf-8)
    target_compile_options(test_gemm PRIVATE /utf-8)
endif()

# 7. 基础性能优化选项配置
# (手册要求：MSVC 记录 /O2，GCC/Clang 从 -O2 或 -O3 开始)
if(MSVC)
    target_compile_options(benchmark_gemm PRIVATE /O2)
else()
    target_compile_options(benchmark_gemm PRIVATE -O3)
endif()
