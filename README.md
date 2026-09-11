# CPU GEMM

这是一个通过逐步优化矩阵乘法来学习 CPU 缓存、SIMD、多线程和性能测量的实验项目。

## 当前状态

截至 2026-09-10，项目处于 baseline 阶段早期：

- 已有 `gemm_ijk` 接口、朴素三重循环实现和固定种子输入；
- 当前 CMake 只编译 `src/main.cpp`，尚未编译或调用 `src/gemm_ijk.cpp`；
- `tests/test_gemm.cpp` 尚未接入 CMake，而且测试的是文件内另一份实现；
- benchmark、GFLOPS、CSV 和多规模实验尚未完成。

因此，当前可执行程序显示的 `GEMM time: 0ms` 不是性能结果，只表示计时区间里还没有计算。

## 当前构建方式

在 Windows PowerShell 中：

```powershell
cmake -S . -B build
cmake --build build --config Debug
.\build\Debug\matrix_app.exe
```

性能数据只能使用 Release 构建；在 benchmark 接通后使用：

```powershell
cmake --build build --config Release
.\build\Release\matrix_app.exe
```

## 最近里程碑

完成可验证的 `gemm_ijk` baseline：项目实现同时被主程序和测试调用，测试检查确定性案例及浮点误差，Release benchmark 输出多次运行的中位数、GFLOPS、checksum 和 CSV。

完整学习路线、当前逐项进度和验收标准见 [CPU GEMM 项目手册](../CPU_GEMM_项目手册.md)。
