# 调试与 profiling

先 `./lab check ID --case 0`，再逐 case 扩大。CLI 报告本关 case、输入模式、seed、stream 类型、输出名和首个失败坐标。先保存 raw JSON/日志，再构造最小失败输入；修改支持域时同时更新契约，不在测试中偷偷换输入。`CUDA_LAUNCH_BLOCKING=1` 只用于定位 launch。

`./lab check`/`bench` 在独立子进程运行，默认 180 秒；首次编译可 `--timeout 300`，smoke 默认 600 秒。父进程只终止自己的进程组。对于异步错误，发生非法访问的子进程退出后重新启动，不能信任同一 CUDA context 的后续结果。不要以无限等待或系统级 GPU reset 解决学生死锁。

```bash
./lab sanitize s00_softmax --tool memcheck --case 0
./lab sanitize s33_tma --tool racecheck --case 0 --timeout 180
./lab sanitize s35_pipeline --tool synccheck --case 0 --timeout 180
./lab sanitize s38_async_gemm --tool initcheck --case 0 --timeout 180
```

sanitizer 不是形式证明；尤其跨引擎/异步协议仍需源码审查和生命周期证明。若工具不支持某指令或报限制，保留具体消息，不能将跳过算通过。

`./lab profile ID --case 0` 先执行正确性检查和 warmup，再只对 `lab_profile` NVTX 范围做 NCU 收集；参考计算、初始化、JIT 不在该范围。启动前查询所选 NCU 版本是否提供所需的 `LaunchStats`、`MemoryWorkloadAnalysis`、`SpeedOfLight` sections，原始 `.ncu-rep` 保存在 `artifacts/profiles/`，不覆盖旧报告。

```bash
source scripts/activate.sh
mkdir -p artifacts/environment
lab_ncu="$(python -c 'from flash_attention_lab.runtime import require_tool; print(require_tool("ncu", "LAB_NCU"))')"
"$lab_ncu" --list-sections
"$lab_ncu" --query-metrics --query-metrics-mode all > artifacts/environment/ncu-metrics.txt
```

可查询的候选指标包括 `dram__bytes_read.sum`、`dram__bytes_write.sum`、`gpu__time_duration.sum`、`sm__throughput.avg.pct_of_peak_sustained_elapsed`。具体名称和可用性以所选 NCU 版本的查询结果为准；原始查询与报告只保存在本地 `artifacts/`，不提交。

先在实际输出中核对 metric 名称，再增加 counters。初期问题：s0 合并访问与 launch 开销；s1 HBM/L2 流量和输出重写；s2 CTA 数、warp 通信、非矩阵操作与 Tensor Core 指令；s3 producer/consumer stall、SMEM/TMEM 生命周期；s4 指数/MMA overlap、correction、spills 和调度尾部。吞吐低不能独自区分“等待”与“计算少”，须结合源码、数据依赖和时间线。

NCU replay 会改变时间和缓存行为，正式 timing 用独立 benchmark；时间线可另用 `nsys profile -t cuda,nvtx` 包住同样的命令。计数器权限若返回 `ERR_NVGPUCTRPERM`，记录为环境限制，不修改系统权限。工具路径可通过 `LAB_NCU`、`LAB_SANITIZER` 配置。

## VS Code / clangd

clangd 需要项目头文件、Python/PyTorch 头文件和正确的 CUDA Toolkit 路径。
在仓库根目录激活独立环境，再生成本机的编辑器编译数据库：

```bash
source scripts/activate.sh
# 若 PATH 中的 nvcc 不是项目要求的 12.8，先设置 LAB_CUDA_HOME。
python scripts/configure_clangd.py
```

脚本遵循 `LAB_CUDA_HOME` / `CUDA_HOME` / PATH 和 `LAB_GPU` / `CUDA_VISIBLE_DEVICES`，
检查所选 GPU 与构建环境后，为 CUDA 源文件、公共头文件和 C++ 绑定生成
`compile_commands.json`。该文件含本机绝对路径，已被 Git 忽略；切换工具链、
Conda 环境或新增 CUDA 文件后重新生成。脚本只配置解析，不编译或运行学生 kernel。
`.vscode/` 中的个人编辑器配置也被 Git 忽略。

在 VS Code 命令面板执行 **clangd: Restart language server**。使用支持 SM100a 的
clangd；可通过 `clangd --check=students/cuda/s00_softmax/kernel.cu` 独立检查解析诊断。
编辑器使用 Clang CUDA 参数，实际构建仍使用 `build.py` 的 nvcc 参数；
无编辑器诊断不代表 nvcc 构建成功或学生正确性通过。
