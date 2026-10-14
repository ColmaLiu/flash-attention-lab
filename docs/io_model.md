# 从循环顺序推导 IO

以下按单个 head、元素数计，暂忽略缓存、尾块与初始化；乘 B×H、各 tensor 的真实 element bytes 后才是字节。算法读写次数不是 NCU 的 DRAM 事务数。

设 Q 行 tile 高度 B_r、K/V tile 高度 B_c，T_r=ceil(N/B_r)，T_c=ceil(N/B_c)。这些符号不是 CUDA blockDim。

朴素 attention 在 HBM/global memory 中物化 S 和 P：S 写 N²、softmax 读 S N²、写 P N²、PV 读 P N²，单中间量流量已达约 4N² 个元素，另有 Q/K/V/O 与 reduction 临时状态。若你的朴素实现反复重算/读取，应按源码加入额外项。

FA1 教学约束是 KV 外层与 Q 内层：当前 K/V tile 在遍历 Q tiles 时复用，因此完整 K/V 的理想读入为 2ND。每个 KV tile 都要再遍历全部 Q，共 ND×T_c。若按本关状态约定把更新后的 O/m/l 保存到 HBM，O 每轮读和写各 ND，共 2ND×T_c；m/l 各读写 N，共 4N×T_c。加总得到约 `2ND + 3ND*T_c + 4N*T_c`，再单列初始化及最终 LSE。第一轮若不读旧 O/state，需扣除相应项。这个式子解释重复访问从哪里发生，而不是要求你照此预设 CUDA layout。

FA2 的 query tile ownership 让一个 query tile 遍历其有效 KV 期间保持自己的输出状态：Q 理想总读 ND、O 最终总写 ND、LSE 总写 N；每个 query tile 读取完整 K/V，所以逻辑 K/V 读约 2ND×T_r。中间分数/概率不写 HBM。收益还涉及非矩阵操作和 warp 通信，不能仅从这一个 IO 式子断言速度。

因果模式必须按各 query tile 的有效 key 范围求和：整个对角 tile 可能仍执行物理 MMA，只把无效概率屏蔽。有效 pair 数 N(N+1)/2 不等于实际执行的乘加数。尾块按有效元素统计理论模型，按实际指令/传输粒度统计执行流量。

对每个版本填写：tensor 在哪个引擎/线程中产生 → 存在哪里 → 哪个循环复用 → 何时搬运/覆盖/释放。FA4 还要分别统计 SMEM 读写、TMEM ld/st 和寄存器压力；HBM 流量降低不保证指数单元或 SMEM 带宽不成为瓶颈。

NCU 中重复的逻辑 global load 可能命中 L2，实际 DRAM bytes 更少；事务合并/对齐、局部 spill、writeback、replay 和 profiler cache 控制也会影响计数。记录原始报告、metric、cache/replay 设置，不把 `dram__bytes_*` 直接当作论文理想 IO 的证明。
