# 第六轮：共享区域尺寸场与审查缓存

目标仍未完成。所有试验保持 h=7.623706340789795、原始几何预算=1.524741292、质量诊断阈值=0.026690566912293434。没有提交或推送。

## 有限改动

- BoundarySizingField 增加任务子集，GPU 与 CPU 使用同一空间评价函数；显式区域覆盖不再把原始边界细碎采样当成尺寸要求。
- 可选 `-UnifiedRegionalSizing` / CLI `--patch-target id:h` 默认关闭。不能与曲率特征细化同时使用。没有重写 GPU 编辑算子。
- 显式尺寸场不收紧或放宽几何预算。装配时保留已有更细的恢复目标，避免扩大其他困难区域的尺寸目标。
- 修正填入有效尺寸后意外激活整任务质量让步的逻辑；初次执行不启用这项让步，已有显式恢复分支单独保留。
- 独立距离审查缓存空间查询结构；区域局部网格去掉未引用顶点，保留面顺序、查询容差与距离并列时的选择规则。

主要文件：`BoundarySizingField.h/.cpp`、`ReferenceSurfaceGpu.cuh`、`RawCudaRemesher.cu`、`RawCudaBatch.cpp`、`raw_partition_cli.cpp`、`run_raw_partition.ps1`、`cached_proximity.py`、`audit_phase1.py`、`compare_regional_candidate.py`。相关回归断言在 `test_raw_projection.cu`、`test_raw_batch.cpp`、`test_phase1_audit.py`。

## 对照需要纠正的来源差异

第五轮保存的程序与当前程序的默认结果不同。第五轮程序在部分初次几何失败的任务中继续拆成单 patch，当前程序保留合法的温和候选。49 个任务存在这类差异；不能把第五轮到本轮的所有变化归因于尺寸场。第五轮报告中的结果仍是当时保存程序的实测数据，不能当作当前源代码的默认结果。

因此另用当前程序、同一已修复完整分区输入运行默认对照：499297 面、均值 0.568807、P05 0.031948、面积加权均值 0.752271，batch 阶段 49.7447 秒。这是阶段对照，不是完整流程耗时。合法温和候选优先保留会中断后续拆分候选的探索，是下一轮需要明确比较的接受机制问题。

## 结果

全流程区域场原型从原始 STL 开始，ModelSeeds=250、Iterations=12、SmoothPasses=3；五个区域目标 1.9，区域 CPU 邻域生成与 GPU 增长参数均 0.5。

| 指标 | 完整区域场原型 |
|---|---:|
| 完整流程，包含区域验收 | 380.2834 秒 |
| 分区打包 | 44.7598 秒 |
| 首次窄带预处理 | 12.038 秒 |
| batch | 43.6087 秒 |
| 区域对照审查 | 272.4042 秒 |
| 三角形 | 504753 |
| 均值 / P05 | 0.567938 / 0.033275 |
| 面积加权均值 | 0.752492 |
| 低质量面积 / 最大连通面积 | 4544.8767 / 40.1940 |
| 全局 h 的超长边 / 原生局部目标超长边 | 0 / 1 |
| 原始输入双向采样最大误差 | 1.267034 / 1.304286 |

特征、开放边界、接缝与拓扑检查通过，但区域和最终质量验收失败，输出主文件保留第五轮参考，候选另存 `regional_provisional.ply`。与当前程序默认对照相比，P05 和面积加权均值提高，平均质量降低，且仍有区域退化。不能采用为最终结果。

随后修正装配时不扩大已有更细目标，仅复用分区运行：43.8597 秒、505817 面、均值 0.567527、P05 0.033220、面积加权均值 0.752570；原生低质量面积 4529.6144，局部目标超长边 **414**。这一最新版本尚无从原始 STL 的完整实测，不能引用 380.28 秒作为最新代码耗时，也不能用全局目标边数掩盖局部目标缺陷。

小范围增长参数 0.25 / 0.5 / 1 的扫描在 `results/phase6_sizing/guarded_comparison.json`，同时保存无 GPU 尺寸场的相同 CPU 邻域输入对照。增长参数变化与算法变化分别记录；不能把 0.25 GPU 增长与 CPU 0.5 增长当成一致配置。

## 验收性能优化

同一候选、同一当前程序默认参考的 644 个实际变化区域，审查时间从 168.3151 秒降至缓存后的 91.7271 秒，再到区域顶点压缩后的 46.7434 秒（最后一次包含 profiler 开销）。三份 JSON 除 `seconds` 外逐项相同；随机、共面、等距离且相反方向的平行面及未引用顶点的最近点、距离和三角形索引均与已安装 trimesh 逐项相同。这是审查实现的收益，没有改变网格或验收口径。

## 复现

```powershell
.\experiments\rxmesh-remesh\run_raw_partition.ps1 `
  -InputMesh .\examples\2.stl `
  -OutputDirectory .\experiments\rxmesh-remesh\results\regional-field-reproduce `
  -ModelSeeds 250 -Iterations 12 -SmoothPasses 3 `
  -RegionalPatchTargets .\experiments\rxmesh-remesh\results\phase3_regional\targets_1.9.json `
  -RegionalReferenceOutput .\experiments\rxmesh-remesh\results\phase5_geometry\full_reverse_guard\remeshed.ply `
  -GradedRegionalNeighbors -UnifiedRegionalSizing

python experiments/rxmesh-remesh/tests/test_phase1_audit.py
ctest --test-dir experiments/rxmesh-remesh/build_rx -C Release -R '^(region_quality|raw_batch_safety|raw_projection_safety)$' --output-on-failure
```

上述完整命令运行当前最新版，预期仍是待改进结果，不能保证复现原型的精确指标。
