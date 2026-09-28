# 阶段 15：严格平面证据与完成态局部重建试验

## 结论和范围

总目标仍未达到。本阶段保留诊断工具和有界重建入口，未接入默认完整流程；没有修改 GPU 算子、分区算法或接缝调度。所有实验采用 h=7.623706340789795、原始几何误差预算 1.524741292、质量诊断阈值 0.026690566912293434。局部尺寸 .95 是单独的尺寸参数试验，不是固定 h 算法优化的收益。

上一阶段原始 STL 完整流程 109.4936743 秒。本阶段复用了上一阶段完成态输出，没有新的原始 STL 完整流程计时，不能据此声称 90 秒达标。

## 代码和证据

- `tools/merge_certified_flat_regions.py`：只有逐顶点精确轴坐标相等、面方向一致、普通无支持角色、二面非硬内部接口，才允许合并。原始 2.stl 快照 6810 分区中 565 个被证明为轴对齐平面，但可合并接口数为 0；该路径没有收益。
- `tools/optimize_flat_diagonals.py`：完成态上有界两轮对角线候选。保持原坐标文本、标签、特征链及接缝；四点精确同轴平面、凸向一致、没有既存或新产生的重复对角线、两面质量最小值/均值/面积加权总量非退化。新对角线不长于旧边，且不短于全局 0.8h。工具只生成阶段候选，不进行原生局部尺寸或终点认证；仍需独立区域、原生目标和原始几何审查。
- `tools/export_incumbent_snapshot.py`：将已完成网格导出给局部重建。保留坐标、面和标签；源链映射必须沿现有边且匹配两侧分区身份，重合坐标锚点、身份冲突或未映射接口拒绝。坐标近邻查询只用于查找锚点，绝不焊接顶点。源快照和原始 STL 继续作为不可变误差参照。
- `tools/repair_narrow_strips.py --only-requested-regions`：必须显式选择区域；仅处理选中区域及必要接缝邻居，不重新处理所有自动窄带。默认关闭，缓存复用校验包含区域范围。该选项用于完成态阶段试验，未集成全流程自动接受。

初次完成态重建试验尚未限定自动选区，实际 196 个候选、188 重建、8 拒绝；该结果未作为局部收益、验收结果或默认输出。其报告保留在 `incumbent_repair.json`。后续有界试验才严格限定 3816。

## 相同完成态输入的平面对角线对照

原始完成态为 `results/phase14_selection/full/remeshed.ply`；候选为 `results/phase15_flat/guarded_diagonal_candidate.ply`。候选共 130 次交换、259 个不同面、16 个分区。早期候选和增加尺寸保护后的候选字节哈希一致，独立审查适用于二者。

| 指标 | 完成态对照 | 平面候选 |
|---|---:|---:|
| 平均质量 | 0.568358022 | 0.568412615 |
| P05 | 0.032024174 | 0.032024174 |
| 面积加权平均 | 0.752342857 | 0.753009144 |
| 低质量面积 | 4727.792922 | 4727.792922 |
| 最大坏区域面积 | 55.522459 | 55.522459 |
| 坏区域连通分量 | 7189 | 7189 |
| 全局 h 短边 | 678731 | 678731 |
| 全局 h 长边 | 0 | 0 |

16 个变化分区的五项质量指标无退化。92885 特征标记链、92105 分区接口链、81 开放边界链无缺失，锚点误差为零；连通分量 1、Euler=-1；非流形、绕向冲突、重复面、float32 零面积均为零。真实几何分类仍是继承声明，不能把全部标记链称为真实 CAD 特征认证。

原始 STL 双向采样最大误差维持 1.267034 / 1.304286，小于未变预算；采样不构成连续误差证明。最近源面方向异常 456→456，该指标也不是实际翻转或自交证明。相邻面尺寸比 P95 5.894672、最大 9085.940487 未改善。3816/3842/3843 和重建窄带的质量未改善。原生局部目标未重新评价，不作原生尺寸达标声明。

平面候选生成阶段 7.162 秒（含读写和保护链映射）；独立审查另耗 91.971 秒。本次独立审查与早期候选生成和只读指标检查有部分时间重叠，不能用来评价性能。该收益集中在普通平面，没有消除主要坏区域，因此暂不纳入默认流程。

## 完成态困难区域重建

完成态快照导出 13.488 秒，250007 顶点、499935 面、100586 条细分后约束段，源身份和拓扑校验通过。此开销是阶段工具开销，不能直接加入生产入口并期待加速。

仅请求 3816、局部 h=.95、保留全局 h 和原始误差预算，耗时 1.328 秒；该区域被现有凸分量重建器以 `nonconvex component` 拒绝。返回完成态输入副本，记录拒绝原因，而非回到原始劣质输入。这不是成功处理，该区域继续未解决。尚未证实非凸源于真实形状还是量化后的微小凹陷，需要面与边界证据后再决定一般多边形重建或曲线尺寸策略；不能仅放宽凸性容差。

## 验证和保护

24 项针对性检查通过：平面候选 9、完成态导出 3、窄带修复 12。覆盖硬边、开边界、接缝、真实折角、非平面、角色支持、尺寸拒绝、坐标身份、重合锚点拒绝、区域范围和缓存隔离。remesh 范围 diff 空白检查通过；九项已有无关 tracked 修改与接手时一致。未提交或推送。tracked_checkpoint.patch 不包含新增工具，不能作为完整备份。

## 复现

```powershell
python experiments/rxmesh-remesh/tests/test_flat_candidates.py
python experiments/rxmesh-remesh/tests/test_incumbent_snapshot.py
python experiments/rxmesh-remesh/tests/test_repair_narrow_strips.py

python experiments/rxmesh-remesh/tools/optimize_flat_diagonals.py --incumbent experiments/rxmesh-remesh/results/phase14_selection/full/remeshed.ply --snapshot experiments/rxmesh-remesh/results/phase14_selection/full/strip_repair/pass_01/input.cadpart --output experiments/rxmesh-remesh/results/phase15_flat/reproduce.ply --report experiments/rxmesh-remesh/results/phase15_flat/reproduce.json --target 7.623706340789795 --threshold 0.026690566912293434

python experiments/rxmesh-remesh/tools/repair_narrow_strips.py experiments/rxmesh-remesh/results/phase15_flat/incumbent.cadpart experiments/rxmesh-remesh/results/phase15_flat/rebuild_reproduce.cadpart --report experiments/rxmesh-remesh/results/phase15_flat/rebuild_reproduce.json --target 7.623706340789795 --max-error 1.524741292 --patch-targets experiments/rxmesh-remesh/results/phase15_flat/incumbent_targets.json --graded-neighbors --only-requested-regions
```

独立检查复现：

```powershell
python experiments/rxmesh-remesh/tools/audit_phase1.py --source examples/2.stl --snapshot experiments/rxmesh-remesh/results/phase14_selection/full/strip_repair/pass_01/input.cadpart --repair-report experiments/rxmesh-remesh/results/phase14_selection/full/strip_repair/pass_01/repair.json --baseline experiments/rxmesh-remesh/results/phase14_selection/full/remeshed.ply --candidate experiments/rxmesh-remesh/results/phase15_flat/guarded_diagonal_candidate.ply --output experiments/rxmesh-remesh/results/phase15_flat/reproduce_audit.json --target 7.623706340789795 --max-error 1.524741292 --threshold 0.026690566912293434
```

结果见 `diagonal_audit.json`、`diagonal_patch_quality.json`、`bounded_repair.json`。本阶段候选没有自动采用，待解决区域没有从统计中移除。
