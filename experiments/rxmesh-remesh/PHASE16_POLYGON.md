# 阶段 16：完成态精度来源与约束邻区重建

## 结论

本轮取得实质诊断及候选生成进展，但总目标未达到。完成态 3816/3817 可以继续重建，候选消除了约 100 单位的低质量面积；邻区和全局质量仍有退化，未采用。默认算法保持原样，GPU 算子和接缝调度未改写。新增工具未接入完整流程。

固定全局 h=7.623706340789795、几何误差预算 1.524741292、诊断阈值 0.026690566912293434。区域 .95 是明确的尺寸参数变化，不能计入固定尺寸算法收益。所有局部实验复用完成态，不是原始 STL 完整流程计时。

## 原因和证据

1. `3816_convexity.json`：修复参考 3816 有 3 个凸分量；完成态有 47 个近似平面分量，其中 2 个不满足凸检查。最严重凹陷距离 1.5131e-5、平面偏差 8.7803e-6。原参考最小边侧值约 -5.68e-12。该差异说明量化影响了分组；不是证明所有拒绝都是数值误差。
2. `tools/restore_reference_precision.py`：只恢复能精确对应不可变参考锚点的计算坐标。87470 个源锚点恢复，double 最大差 3.04356e-5；写出的 float32 坐标逐点逐位相同，面和约束身份不变。重合锚点和来源身份合并拒绝。原始 STL 仍是误差参照，计算快照不能重置误差预算。3816 重建分量降到 6，并通过既有检查；不是全部回到原参考的 3 分量。
3. 单独细化 3816 仍使 3817 的邻接面无法安全三角化；完整动态规划搜索也没有找到合法表示。联合选择 3816/3817 解决这项边界协调问题。不能把联合选区的收益归功于动态规划搜索本身。
4. 现有无约束渐变 Delaunay 在联合区域边界上产生 float32 退化或反向面。`tools/constrained_neighbor_geometry.py` 试验了固定所有边界段的动态规划三角化、合法内部点插入及有界内部对角线优化。只面向源三角形的有序细分边界；不是任意带孔、多分量或自交区域重建器。

新三角化通过已有拓扑、锚点和源面采样误差检查，但其最终质量需要区域/全局验收。工作量限制 128 边界点、4096 内部点、24 次内部优化扫描。无法表示的候选拒绝；输入和已完成输出不被覆盖。

## 候选对照

对照为 `phase14_selection/full/remeshed.ply`。

| 候选 | 低质量面积 | 全局质量退化 | 邻区问题 | 接缝拓扑 |
|---|---:|---|---|---|
| 完成态对照 | 4727.792922 | — | 既有未解决区域 | 原有 81 开边 |
| 两区域细化，完整普通三角化 | 4636.528616 | 均值、面积加权均值 | 489/982/3818 | 源快照检查通过 |
| 两区域细化，约束渐变三角化 | 4633.945909 | 均值、面积加权均值 | 489/982/3818 | 81 开边、Euler=-1 |
| 局部 GPU，显式区域目标 | 4625.585972 | 均值 | 982 面积加权均值 | 608 开边、Euler=-121，拒绝 |
| 局部 GPU，统一全局目标 | 4628.002509 | 均值、面积加权均值 | 489/982/3818 | 81 开边、Euler=-1 |

显式目标 GPU 候选的局部边界再次被尺寸场细分，直接拼接会裂缝；原顶点未动不等于接缝一致。该候选明确拒绝，不能把有效局部网格冒充有效全局网格。统一目标候选拓扑通过，但仍质量失败。

约束渐变候选中 3816 低质量面积 87.418056→0.292355，3817 为 12.067204→0.236590。邻区 982 面积加权质量 0.897787→0.804560，说明锁住每个原始邻接三角形的外围会限制尺寸过渡。之后的显式目标 GPU 能恢复到 0.896238，却没有完全达到原质量，并导致外部接缝再细分。需要保持外部边界契约的连通邻区处理，不能不断扩张特殊名单或直接接受退化。

## 有效拓扑候选的独立检查

统一全局目标的 GPU 候选 `halo_uniform_joined.ply`：

| 指标 | 对照 | 候选 |
|---|---:|---:|
| 平均质量 | 0.568358022 | 0.565395041 |
| P05 | 0.032024174 | 0.032972095 |
| 面积加权平均 | 0.752342857 | 0.751773987 |
| 低质量面积 | 4727.792922 | 4628.002509 |
| 最大低质量区域 | 55.522459 | 55.522459 |
| 低质量连通分量 | 7189 | 7211 |
| 全局 h 短边 | 678731 | 687237 |
| 全局 h 长边 | 0 | 0 |
| 面数 | 499935 | 505193 |

92885 条特征标记链、92105 条接口链、81 条开放边界链无缺失，源锚点误差为零。连通分量 1、Euler=-1；非流形、内部绕向冲突、重复面、float32 零面积均为零。继承的特征标记不是完整真实 CAD 分类认证。

原始 STL 双向采样最大误差 1.477164 / 1.304286，小于不变预算；前向值较对照的 1.267034 增加，说明连续局部投影可能累积误差，必须始终对原始输入评价。采样不是连续误差证明。最近源面方向异常 456→456，不是严格真实翻转/自交证明。原生局部尺寸未对拼接后的网格重新评价，不能宣称尺寸达标。最大坏区域仍是 3842/3843，未处理。

## 耗时和复现

计算精度来源恢复 1.106 秒，联合约束渐变重建 8.818 秒，统一目标五区域 GPU batch 1.096 秒，独立审查 89.362 秒。这些是复用完成态的阶段测量，且部分诊断、测试存在并发，不能用于完整流程性能结论。

```powershell
python experiments/rxmesh-remesh/tests/test_constrained_neighbor.py
python experiments/rxmesh-remesh/tests/test_reference_precision.py
python experiments/rxmesh-remesh/tools/restore_reference_precision.py --incumbent experiments/rxmesh-remesh/results/phase15_flat/incumbent.cadpart --reference experiments/rxmesh-remesh/results/phase14_selection/full/strip_repair/pass_01/input.cadpart --output experiments/rxmesh-remesh/results/phase16_polygon/canonical_reproduce.cadpart --report experiments/rxmesh-remesh/results/phase16_polygon/canonical_reproduce.json

# 本地原型：固定来源、固定两区域和参数，未集成默认流程。
python experiments/rxmesh-remesh/results/phase16_polygon/probe_constrained.py
python experiments/rxmesh-remesh/tools/extract_partition_blocks.py experiments/rxmesh-remesh/results/phase16_polygon/constrained_rebuilt.cadpart experiments/rxmesh-remesh/results/phase16_polygon/halo.cadpart --patches 489 982 3816 3817 3818 --block-size 1
experiments/rxmesh-remesh/build_rx/Release/cad_raw_partition_cli.exe experiments/rxmesh-remesh/results/phase16_polygon/halo.cadpart experiments/rxmesh-remesh/results/phase16_polygon/halo_uniform.ply --target 7.623706340789795 --max-error 1.524741292 --iters 12 --workers 4 --gpu-concurrency 4 --patches-per-task 1 --smooth-passes 3 --collapse-passes 8 --flip-passes 8 --low-quality-threshold 0.026690566912293434 --select-final-regions
# CLI 返回 1 表示候选保留、终点失败，不能解释为验收成功。
python experiments/rxmesh-remesh/results/phase16_polygon/evaluate_halo.py halo_uniform
python experiments/rxmesh-remesh/tools/audit_phase1.py --source examples/2.stl --snapshot experiments/rxmesh-remesh/results/phase14_selection/full/strip_repair/pass_01/input.cadpart --repair-report experiments/rxmesh-remesh/results/phase14_selection/full/strip_repair/pass_01/repair.json --baseline experiments/rxmesh-remesh/results/phase14_selection/full/remeshed.ply --candidate experiments/rxmesh-remesh/results/phase16_polygon/halo_uniform_joined.ply --output experiments/rxmesh-remesh/results/phase16_polygon/reproduce_audit.json --target 7.623706340789795 --max-error 1.524741292 --threshold 0.026690566912293434
```

## 独立的完整流程工作线程实验

完整运行完成，107.3933895 秒（上一阶段 4 线程为 109.4936743 秒），仍未低于 90 秒。保持原始 STL、ModelSeeds=250、所有尺寸/误差/算子参数及 SelectFinalRegions 一致，只将 Workers 从 4 改为 8，GpuConcurrency 仍为 4。此实验未启用本阶段重建工具。数据在 `results/phase16_polygon/full_workers8`。

分区及打包 44.438076 秒，GPU batch 48.002441 秒，其中 tasks 31.720640、assembly 12.878952、boundary 1.880602 秒；候选比较和另一种最终细分在 assembly 内，不能重复相加。第二轮窄带缓存/诊断约 2.316 秒。完整计时包含原始 STL 分区、打包、预处理、输入读取、所有轮次和输出，其他开销包含在总时间里。

原始分区快照、修复快照和最终 PLY 与 4 线程对照 **SHA256 逐字节一致**，因此可以复用该对照的网格质量和硬检查证据。`quality_accepted=false`，不能因输出相同或零长边宣称终点成功。这是单次跨阶段比较，不足以保证稳定的 2.10 秒收益。完整运行期间没有其他 remesh、测试、构建或独立网格审查；仅执行了少量只读日志、指标读取和文档写入。

```powershell
experiments/rxmesh-remesh/run_raw_partition.ps1 -InputMesh examples/2.stl -OutputDirectory experiments/rxmesh-remesh/results/phase16_polygon/full_workers8_reproduce -ModelSeeds 250 -Workers 8 -GpuConcurrency 4 -TargetLength 7.623706340789795 -MaxError 1.524741292 -Iterations 12 -SmoothPasses 3 -LowQualityThreshold 0.026690566912293434 -SelectFinalRegions
```

38 项针对性检查通过：34 项 Python 检查及 4 项 C++/GPU 安全检查。作用范围分别覆盖新增精度恢复、约束三角化、既有区域选择、修复缓存和边界/投影安全；它们不构成对所有几何案例的证明。九项已有无关 tracked 修改与接手时一致，remesh 范围 diff 检查通过。未提交或推送。

下一步应优先处理 3842/3843 最大坏区域，并设计带外部边界契约的连通邻区候选，而非采用已知退化的候选。性能方面，分区/打包和任务阶段仍是主要成本；增加工作线程没有解决瓶颈。
