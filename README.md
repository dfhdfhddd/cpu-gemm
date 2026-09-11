# CPU GEMM

这是一个通过逐步优化矩阵乘法来学习 CPU 缓存、SIMD、多线程和性能测量的实验项目。

## 当前状态

截至 2026-09-12，`gemm_ijk` baseline 已完成主要实现闭环：

- `src/gemm_ijk.cpp` 提供朴素 `ijk` 三重循环实现；
- `benchmark_gemm` 和 `test_gemm` 都通过 CMake 构建，并调用同一个 `gemm_ijk`；
- 正确性测试覆盖标量、全 1、单位矩阵、手算矩阵、固定随机矩阵和逐元素参考检查；
- benchmark 使用固定随机种子、3 次预热和 20 次正式计时，输出中位数、最小值、标准差、GFLOPS 和 checksum；
- 结果写入 `results/benchmark_results.csv`；当前代码中的测试规模为 `64、128、256、512、1024`。

当前还不能标记 `baseline-v1`：提交前应在本机 Release 构建后确认测试和 benchmark 成功，并确保 CSV 不含旧的 `N=4` 或 `N=16` 记录。README、手册和结果文件必须与最终一次运行保持一致。

## 实验环境

- CPU：Intel Core i7-13650HX
- 内存：2x16GB DDR5
- 编译器：Microsoft C/C++ x64 优化编译器 19.50.35721
- CMake：4.4.3
- 编译模式：Release，MSVC `/O2`
- 电源模式：增强
- 测试日期：2026.9.12
- 预热次数：3 次
- 正式计时：20 次

## 构建与测试

在 Windows PowerShell 中，从本目录执行：

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

运行正确性测试：

```powershell
.\build\Release\test_gemm.exe
```

运行 benchmark（性能数据只采用 Release）：

```powershell
.\build\Release\benchmark_gemm.exe
```

程序会覆盖写入 `results/benchmark_results.csv`。计时只包含一次 GEMM 调用，随机数生成、内存分配、预热和 checksum 不计入计时区间。

## 性能记录要求

报告中同时记录矩阵规模、数据类型、编译器和编译参数、CPU、电源模式、测试日期、迭代次数及 CSV 原始结果。比较不同实现时必须保持这些条件一致；中位数作为主要指标，平均值、最小值和标准差作为辅助指标。

当前已知环境事项：本机 Release 构建已由用户确认通过；Codex 环境复跑时因进程环境变量同时存在 `Path` 和 `PATH`，MSBuild 报环境错误，这不属于项目源码错误。

完整学习路线、重要程度标记、逐项进度和最终验收标准见 [CPU GEMM 项目手册](../CPU_GEMM_项目手册.md)。
