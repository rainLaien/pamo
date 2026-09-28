# 第八轮：有限的困难区域候选比较

目标仍未完成。新增 `-ExploreProvisionalChildren` / `--explore-provisional-children`，默认关闭。只在多 patch 任务有合法、未达标候选时探索原有单 patch 恢复；不修改分区、GPU 算子、接缝调度或重建方法。

孩子候选分别通过原始参考几何和边界检查后，仅按不可变源锚点 ID 装配评估候选；不按坐标合并内点。候选须保持整体及每 patch 的均值、P05、面积加权均值、低质量面积和最大连通面积不退化，整体还须有可测进展，长边及短边数量不增加。失败保留合法当前候选；质量成功仍由最终区域验收决定。

报告含 `child_candidate_compared`、`child_candidate_selected`、`seconds_child_comparison`、`child_comparison_reason`。拒绝原因区分整体质量失败掩码、尺寸退化、首个退化 patch、几何边界失败、拓扑或评价失败。掩码定义见 `RegionQuality.h`。失败探索不会把整个区域退回原始劣质输入。

## 已完成的同输入阶段对照

使用第五轮已修复完整 CADPART，h=7.623706340789795，原始预算=1.524741292，Iterations=12、SmoothPasses=3，诊断阈值=0.026690566912293434。参数未变，仅开启上述候选探索。

| 指标 | 默认当前程序 | 候选探索 |
|---|---:|---:|
| batch 日志秒数 | 约 49.74 | 98.1456 |
| 三角形 | 499297 | 499297 |
| 平均质量 / P05 | 0.568807 / 0.031948 | 相同 |
| 面积加权均值 | 0.752271 | 相同 |
| 探索任务 / 采用任务 | 0 / 0 | 144 / 0 |
| fallback | 0 | 0 |

两份 PLY 字节完全相同（SHA256 对照）。因此没有质量、特征、拓扑或几何收益；增加近一倍 batch 耗时。这些数据只说明阶段代价，不能冒充完整流程耗时。当前不采用该配置为默认。

首轮阶段版本尚未记录拒绝原因；最新诊断版本的完整运行补充了原因。37 个任务整体质量掩码为零，但长边或短边数量增加；26 个任务缺乏可测质量进展；其余因整体质量、尺寸或逐 patch 退化拒绝。完整原因分布在 `results/phase8_candidates/full_comparison.json`。后续应依据缺陷区域和拒绝原因缩小探索范围，并考虑只拼入真正改善的区域，不能为了采用候选而放宽真实几何约束或改统计口径。

## 验证与复现

新增批处理测试确实进入候选比较分支，验证整体和逐 patch 非退化、尺寸缺陷不增加、边界保持且无原输入 fallback。3 个 C++ 和 21 个 Python 测试通过（24 个测试）。

```powershell
.\experiments\rxmesh-remesh\run_raw_partition.ps1 `
  -InputMesh .\examples\2.stl `
  -OutputDirectory .\experiments\rxmesh-remesh\results\child-candidates-reproduce `
  -ModelSeeds 250 -TargetLength 7.623706340789795 `
  -MaxError 1.524741292 -Iterations 12 -SmoothPasses 3 `
  -LowQualityThreshold 0.026690566912293434 -ExploreProvisionalChildren
```

最新诊断版本从原始 STL 的完整流程已结束：**162.9594 秒**，包含分区打包 47.4271 秒、首次 227 个窄带预处理 11.742 秒、batch 97.7339 秒、后续修复检查 2.360 秒以及读取和输出。输入参数与上表一致，fallback=0、unchanged=40、采用子候选=0；退出待改进状态。主输出与阶段候选及默认当前程序输出字节完全相同，不能宣称质量、几何或翻转检查新增收益。本轮没有额外重复全网格独立方向审查；既有方向问题仍未消除。

结果在 `results/phase8_candidates/full_explore`，保存了当前可执行程序及 tracked remesh diff 检查点。此检查点不包括未跟踪文件，也不包含无关壁厚修改；不要把它当作完整备份。未提交或推送。
