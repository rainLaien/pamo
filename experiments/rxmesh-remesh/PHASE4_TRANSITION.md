# 第四轮：邻面尺寸过渡的受控实验

## 结论

实现了默认关闭的 CPU 邻面渐变候选，并完成局部对照及原始 STL 完整运行。候选未通过最终验收，未采用。还有优化空间，但仅在预处理中增加渐变点不能保证后续 GPU 使用同一个尺寸场。不能据此宣称质量达标或达到 90 秒。

本轮没有修改分区、GPU 算子、接缝调度或最终接受规则。拒绝候选后保留第一阶段 remesh 参考结果，其 SHA256 与参考逐字节一致；困难区域候选另存，便于继续研究。参考结果本身也存在待解决缺陷，保留参考不表示参考已达标。

## 改动范围与回退

- `tools/graded_neighbor_geometry.py`：在受区域细化影响的平面源三角形内部生成渐变点，保持给定边界站点，检查边界、方向及 float32 退化。
- `tools/repair_narrow_strips.py`：可选接入候选，缓存参数包含该开关，记录内部点与误差；无法合法重建时沿用已有支持区域扩张路径。
- `tools/extract_partition_blocks.py`：提取完整计算块并补齐支持引用闭包，保留原 ID 映射，供受控诊断。
- `run_raw_partition.ps1`：增加 `-GradedRegionalNeighbors`，要求同时指定区域目标；默认关闭。
- 相关测试与比较元数据。省略该开关即恢复第三轮邻面生成行为。

实验尺寸场为 h(x)=min(全局 h, 局部 h+0.5×到细化边的距离)。增长率、点间距与点数上限是实验参数，尚未校准为通用质量阈值。新内部点不是几何特征，不能通过冻结这些点强迫算法保留实验影响。

## 控制变量与阶段实验

固定原输入、全局尺寸 7.623706340789795、误差预算 1.524741292、12 次迭代、3 次平滑、4 个 worker/GPU 并发。区域尺寸 1.9 沿用第三轮实验，属于参数变化；渐变邻面是算法变化，分别报告。

计算块诊断覆盖 848 个支持闭包 patch。抽取后外部上下文变化，因此这些结果只能相互比较，不能当作完整模型耗时或直接等同完整模型指标。

| 抽取数据 | 每任务 16 patch 的 batch 秒数 | 每任务 1 patch 的 batch 秒数 |
|---|---:|---:|
| 原参考输入 | 5.18 | 17.50 |
| 区域 1.9 输入 | 5.51 | 18.01 |

缩小计算任务仍有邻面质量损失，阶段耗时约增加到 3.4 倍，故没有采用任务大小 1。任务耦合存在，但不是邻面损失的唯一解释。

渐变候选触发现有路径扩张到 3817、3818。为避免把新增 3818 的影响算作渐变收益，另生成相同五区域、相同细化网格相位的无渐变对照。渐变实际只影响 20 个邻面源三角形，增加 110 个内部点：

| patch | 对照低质量面积 | 渐变低质量面积 | 对照面积加权质量 | 渐变面积加权质量 |
|---|---:|---:|---:|---:|
| 607 | 2.059776 | 0.123390 | 0.783421 | 0.777810 |
| 982 | 5.417501 | 0.818940 | 0.811561 | 0.831681 |

这是阶段对照：低质量面积改善，但 607 的面积加权质量仍下降。不能用单一指标接受候选。

## 从原始 examples/2.stl 开始的完整结果

完整耗时 **171.9907 秒**，包含分区打包 45.7019 秒、区域比较审查 63.0968 秒，以及预处理、batch 和输出。原生日志 batch 为 43.6539 秒。生成流程扣除比较审查仍约 108.8939 秒，未达到 90 秒。复用分区实验未计为完整运行。

| 指标 | 第一阶段参考 | 本轮未采用候选 |
|---|---:|---:|
| 三角形 | 488037 | 492497 |
| 平均质量 | 0.564172 | 0.561471 |
| P05 | 0.033410 | 0.034559 |
| 面积加权平均质量 | 0.751832 | 0.750771 |
| 低质量面积 | 4677.9655 | 4516.3405 |
| 最大低质量连通区域面积 | 55.5225 | 40.1940 |
| 低质量连通区域数 | 6996 | 6988 |
| 过短边计数 | 659414 | 667066 |
| 超长边计数 | 0 | 0 |
| 相邻面尺寸比 P95 | 6.1893 | 6.1291 |

低质量阈值固定为 0.026690566912293434，尺寸统计沿用前轮定义；这些是连续对照口径，不是新宣称的通用达标阈值。完整候选的区域扩张范围不同于第三轮纯 1.9 候选，不能把两次完整运行差值全部归因于算法；纯算法收益以五区域阶段对照为准。

实际变化 63 个 patch，其中 53 个在原声明几何邻域之外；40 个变化 patch 有质量退步。真实特征链、孔边界链、分区接缝和拓扑硬检查通过；没有新增统计中的退化面，但 3815 的参考方向异常从 1 增至 2，尚未证明为真实翻转，不能声称翻转检查全面通过。

相对原输入的双向采样：输出到原输入最大 1.3331073，原输入到输出最大 **1.5397929 > 1.524741292**。采样不构成连续几何误差证明。超预算缺陷此前已有，但仍阻止最终验收。原生退出 4，区域事务和最终结果均未接受；fallback=0，unchanged=49，重试 231。不能把 unchanged 数量当作失败区域数量。

## 证据、限制与下一步

代码事实：`RawCudaRemesher.cu` 明确拒绝 `LocalSizing`；`RawCudaBatch.cpp` 组装目标初始化为全局 `constantLength`。新增内部点在完整候选中没有一个按链容差原位保留，110 个均移动或消失。这不是失败指标，也不是某个 GPU 算子的隔离证明，但表明预处理加点不保证后续尺寸场一致。

下一项有依据的有限工作是统一局部尺寸场的数据传递和评价，先核对组装、拆分及投影是否保持同一目标，独立验证后再决定 GPU 支持方式。本轮没有直接修改 GPU。原输入反向误差、方向异常、远邻质量回退及审查开销仍需分别解决，不能靠放宽预算或跳过困难区域处理。

22 个针对性测试通过。计时后补充了渐变邻面相对源三角形的 float32 双向采样检查：前向最大 2.8974e-5，反向最大 1.4823e-5；重新生成快照与计时快照 SHA256 完全一致。上述完整耗时不包含这次补充检查的新增开销，不能当作最终版本精确性能承诺。

九项已有无关 tracked 修改与任务初始 diff 一致，未提交或推送。汇总、哈希和保护核对在 `results/phase4_isolation/comparison.json`；完整报告在 `full_graded_1_9/regional_comparison.json`，失败候选为 `regional_provisional.ply`，最终 `remeshed.ply` 为保留的参考。

## 复现

使用第三轮与原快照 SHA 绑定的 `targets_1.9.json`：

```powershell
.\experiments\rxmesh-remesh\run_raw_partition.ps1 `
  -InputMesh .\examples\2.stl `
  -OutputDirectory .\experiments\rxmesh-remesh\results\graded-reproduce `
  -ModelSeeds 250 -Iterations 12 -SmoothPasses 3 `
  -RegionalPatchTargets .\experiments\rxmesh-remesh\results\phase3_regional\targets_1.9.json `
  -RegionalReferenceOutput .\experiments\rxmesh-remesh\results\phase1_20260926\new\full_release\remeshed.ply `
  -GradedRegionalNeighbors

python experiments/rxmesh-remesh/tests/test_repair_narrow_strips.py
python experiments/rxmesh-remesh/tests/test_phase1_audit.py
python experiments/rxmesh-remesh/tests/test_phase1_cli.py
ctest --test-dir experiments/rxmesh-remesh/build_rx -C Release -R '^(region_quality|raw_batch_safety|raw_projection_safety)$' --output-on-failure
```

完整命令预计仍返回质量未接受，候选可审查，参考输出保留。阶段抽取数据、四组任务对照、五区域控制及渐变结果均保存于 `results/phase4_isolation/`，不应替代完整命令进行性能验收。
