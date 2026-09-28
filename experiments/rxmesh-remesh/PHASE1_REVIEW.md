# 第一阶段：区域诊断与接受逻辑

本阶段已形成完整、可比较的代码改动。`examples/2.stl` 的候选没有通过最终区域验收，完整流程为 **102.65 秒**，未达到 90 秒目标。平均质量和 P05 上升，但面积加权质量下降、最大低质量连通区域没有缩小，不能把这些结果概括为全面质量改善。

没有改写分区、GPU 算子、接缝调度或条带重建。工作区已有的 ModelSeeds、奇数平滑缓冲区修正、条带缓存优化保持原状；壁厚分析修改经哈希复核保持不变。未提交、未推送。

## 代码范围和行为

| 文件 | 改动 |
|---|---|
| `include/cad_adaptive/RegionQuality.h`, `src/RegionQuality.cpp` | 正交约束角色、移动分类、质量评价、失败原因；跨面片边界的低质量连通区；尺寸缺陷及其面连通区 |
| `include/cad_adaptive/RawCudaBatch.h`, `src/RawCudaBatch.cpp` | 分开装配有效性和质量成功；已有恢复分支统一接入区域检查；保留未通过质量的候选；最终按原始区域检查退化和无改善的输入缺陷 |
| `tools/raw_partition_cli.cpp` | 输出区域指标、待处理区域和原因；未通过验收的候选仍保存，返回 4 |
| `run_raw_partition.ps1` | 接收 exit 4 并继续支持的修复；先比较质量状态/缺陷区域，再比较尺寸；取消仅因超长边为零就结束质量恢复 |
| `tools/audit_phase1.py` | 同一口径的独立比较、曲线/接缝链、拓扑、原始 STL 双向距离采样及参考法向检查 |
| `CMakeLists.txt`, `.gitignore`, `tests/` | 构建并保留针对性回归测试 |

任务级 `accepted` 只表示候选可装配；`quality_accepted` 表示该阶段的质量检查通过；`provisional` 表示保留候选供继续处理。它们不能互相代替。初始结果发生改变不能结束恢复尝试。现有恢复方案尝试后仍未达标时保留有效候选，质量失败不触发整片原始输入回退。硬拓扑或几何失败继续使用原有安全路径。

默认不再传递 `forcePatches`，因此没有为减少 unchanged 而强制插入质心的细分。健康且没有必要修改的区域可以保持不变。输入已有低质量面的区域还需有可测的质量或缺陷面积改善，否则继续尝试并标为待处理。

## 数据表达和验收范围

约束角色可以同时包含 persistent feature、open boundary、partition interface。现有固定/沿曲线/曲面移动分类保留，算子权限不变。快照没有真实几何边与计算切缝的可靠来源标签；也不能证明所有 persistent feature 都是真实 CAD 特征。因此分类明确保留“来源未知”，没有擅自开放接缝。

低质量截止值默认取不可变 batch 输入的 P05，本次为 `0.02669056691`，在边界细分前生成并贯穿外层修复轮次。也可显式设置 `-LowQualityThreshold`。这是定位输入低尾区域的诊断依据，不是通用工程质量合格线。本阶段没有通过重新定义或降低截止值来消除低质量面。

局部与最终质量检查要求 mean、P05、面积加权 mean、低质量面积及最大低质量连通区面积不退化；已有低质量输入还要求至少一项可测改善。质量比较容差为 `1e-6`，面积比较容差为源面积的 `1e-6`（最小 `1e-12`），用于浮点计算比较。最终检查同时作用于全网格和每个原始面片区域。独立的全网格连通检查跨越分区边界，避免把一块差网格拆成多个小统计单元。

`pending_patch_ids` 是待处理区域并集；`regressed_patch_ids` 表示指标退化；`unresolved_patch_ids` 表示输入已有缺陷而没有可测改善。后两者可重叠。`pending_patch_quality.failure_mask` 位定义：mean=1、P05=2、面积加权 mean=4、低质量面积=8、最大差区域=16、无效面=32、边绕向=64、输入缺陷无改善=128。JSON 只展示最大 32 个低质量连通区，但总数和面积包含全部区域。

短边以现有 collapse ratio `0.8` 检测，长边以现有 split ratio `4/3` 检测；短边未被全部当作可移除错误，特别是几何约束上的短边。报告同时提供尺寸缺陷面连通区和目标尺寸过渡。独立比较还计算相邻面 RMS 边长比，以观察实际网格过渡；均匀目标场的过渡比为 1 不代表实际网格过渡平滑。

这些检查是保守的非退化与进展检查，尚不是经过应用标定的绝对质量证书。原生几何检查仍相对各阶段输入/参考执行；相对原始 STL 的双向误差和参考方向由本轮独立验证检查，统一误差预算尚未实施。

## 同参数完整流程比较

两次完整计时都从原始 `examples/2.stl` 开始，包含重新识别、打包、条带修复、加载、GPU batch、报告和 PLY 输出。使用同一个分割器可执行文件、同一输入和同一组参数：

- ModelSeeds=250，Iterations=12，SmoothPasses=3。
- Workers=4，GpuConcurrency=4，CollapsePasses=8，FlipPasses=8。
- target=`7.623706340789795`，geometry budget=`1.524741292`；均为同样的默认推导值，没有放宽。
- FeatureRefine=false，StrictFlipQuality=true。
- 原始快照及修复快照分别 SHA256 一致，算法比较没有混入分区/预处理变化。

以下质量表统一使用独立工具的 float64 公式和相同的 order-statistic P05、相同截止值。原生 float32 指标另保存在各自 JSON 中；临近截止值的面可能因数值精度/PLY 坐标序列化而影响少量面积分类，不能混用两套统计来挑选较好结果。

| 指标 | 修改前 | 第一阶段最终版 |
|---|---:|---:|
| 完整流程秒数 | 84.16 | **102.65** |
| 面数 | 257,297 | 488,037 |
| mean | 0.448142 | 0.564172 |
| P05 | 0.023134 | 0.033410 |
| 面积加权 mean | **0.767257** | **0.751832** |
| 低质量面数 | 18,439 | 18,066 |
| 低质量面积 | 4,845.98 | 4,677.97 |
| 低质量连通区数 | 7,001 | 6,996 |
| 最大低质量连通区面积 | 55.5225 | **55.5225** |
| 超长边 | 0 | 0 |
| 过短边 | 311,108 | 659,414 |
| 相邻面尺寸比 P95 | 11.1264 | 6.1893 |
| 相邻面尺寸比最大值 | 9,079.49 | 9,079.49 |
| unchanged 任务（GPU 阶段） | 0 | 49 |
| fallback 任务 | 0 | 0 |
| 进入恢复尝试的任务 | 49 | 231 |
| 最终出口 | 0（旧策略） | **4（未验收）** |

最终原生区域检查：6810 个原始面片中，2050 个待处理，包含 1440 个退化区域及 792 个输入缺陷未改善区域（两者有重叠）。候选保留供分析；没有因这些质量失败新增整片 fallback。

本轮只改变算法控制和评价，没有参数优化实验。不能将较早的 2500-seed 默认流程时间与本次结果直接当成算法收益。

## 特征、拓扑、几何和方向验证

两版独立检查都得到：

- 源快照提供的 89,372 条特征段、88,592 条分区接缝段、81 条开边界段全部保留连通链，缺失锚点为 0。最大锚点误差约 `7.1e-7`；链检查数值容差约 `7.62e-5`，独立于几何预算。
- 1 个连通分量、81 条开边、Euler=-1；非流形边、绕向不一致、重复面及 float32 零面积面均为 0。
- 这证明源快照提供的特征与边界链被保留，没有额外 CAD 曲线真值可证明全部识别边都是真实特征。

双向原始 STL 距离采用固定随机种子：输出侧 2000 个面积采样点及至多 2000 个低质量面心，原始侧 2000 个面积采样点。它不能证明最大 Hausdorff 距离；本次检出的超预算采样点仍是需要处理的失败证据。

| 检查 | 修改前 | 第一阶段 |
|---|---:|---:|
| 输出 → 原始 STL 采样最大距离 | 1.3590 | 1.3331 |
| 原始 STL → 输出采样最大距离 | **2.1881** | **1.5398** |
| 同面片最近参考法向的非正点积面心数 | 475 | 464 |
| 上述面心所属面的面积总和 | 442.14 | 170.75 |

预算是 1.52474，因此新候选仍存在反向超预算样本。475/464 是在每个输出面心相对同面片原始三角形做的参考方向异常检查；附近有多张面或参考对应不唯一时，不能把数量全部直接认定为翻转。异常包含 plane、torus、cylinder 和 freeform 标签，尚未证明全部是数值噪声或错误对应。原生检查通过不能代替这些检查；本阶段不宣称“翻转为零”，也没有进行全网格自交证明。

## 窄带区域

同一组 227 条带全部通过预处理，没有修改其触发条件或重建算法，原始锚点和共享链保留。输出条带子集比较：

| 指标 | 修改前 | 第一阶段 |
|---|---:|---:|
| 面数 | 10,669 | 11,081 |
| mean | 0.034214 | 0.035282 |
| P05 | 0.020347 | 0.020351 |
| 低质量面积 | **2,218.50** | **2,225.76** |
| 参考方向异常面心数 | 0 | 0 |

条带质量没有明显改善，低质量面积略增。最大连通差区域跨 patch 3842/3843，共 203 面，面积约 55.52；bbox 的 x 跨度约 351，y/z 跨度约 0.194/0.201。第二大区域是 patch 3816，面积约 53.10。这些真实狭长几何在固定边界和统一 7.62 目标下仍困难，不能用更多沿长向细分宣称改善。后续需比较局部缩小尺寸与各向异性，并保留特征连接，而不是要求所有区域发生修改。

## 时间瓶颈和限制

| 阶段 | 修改前秒数 | 第一阶段秒数 |
|---|---:|---:|
| 分区和打包 | 44.880 | 44.583 |
| 首轮条带修复（内部计时） | 9.712 | 9.356 |
| batch | 26.643 | 43.021 |
| batch 内任务阶段 | 20.252 | 33.182 |
| batch 内装配阶段 | 3.614 | 6.506 |
| 最终按面片审计（含在 batch） | 无 | 0.781 |

恢复尝试增加、保留更密的候选和更大的装配输出是主要新增开销。质量未通过后还有一次候选集合未变化的残差修复检查，约 2.28 秒，计入完整时间。独立验证工具约 108 秒，在计时流程之后运行，没有混进上述生产流程时间；若以后将它集成进管线，也必须纳入时间预算。

早期探索运行及中间版本计时保存在同目录，不用于最终性能宣称。最终输出与已经独立审核的 `new/full_measured/remeshed.ply` 字节一致，SHA256 为 `258AACC4DE538A997D6F5EC1AA41A556AB7100ED773039B8B7DC383B83AC346C`，因此该审核适用于最终 `new/full_release` 输出。复用分区的 rollback 实验只证明输出一致，不是完整流程时间。

## 复现

从仓库根目录构建、测试：

```powershell
cmake --build experiments/rxmesh-remesh/build_rx --config Release --target cad_raw_partition_cli test_raw_batch test_region_quality -j 4
ctest --test-dir experiments/rxmesh-remesh/build_rx -C Release -R '^(region_quality|raw_batch_safety|raw_projection_safety)$' --output-on-failure
python experiments/rxmesh-remesh/tests/test_repair_narrow_strips.py
python experiments/rxmesh-remesh/tests/test_phase1_audit.py
python experiments/rxmesh-remesh/tests/test_phase1_cli.py
```

完整第一阶段流程（当前 `2.stl` 会返回 4，输出和诊断仍保存）：

```powershell
& experiments/rxmesh-remesh/run_raw_partition.ps1 `
  -InputMesh examples/2.stl -OutputDirectory experiments/rxmesh-remesh/results/phase1-reproduce `
  -ModelSeeds 250 -Iterations 12 -SmoothPasses 3
```

行为回退/对照：同一命令加 `-LegacyCoverageAcceptance`。该开关恢复旧的接受与覆盖细分决策，保留新审计；它不能证明结果合格。保存的旧二进制位于 `results/phase1_20260926/baseline/cad_raw_partition_cli.exe`。最终版 legacy 模式在相同修复快照和参数下的 PLY 与旧二进制逐字节一致（旧 PLY SHA256：`0E342F5BBF0A5498B9F4ADF44E8594F4338AC4F052C3F0D98563C6226ED4F416`）。

`results/phase1_20260926/phase1_only.patch` 仅包含本阶段改动，不包含接手前的性能优化或壁厚分析修改。已执行 `git apply --reverse --check` 验证其可独立反向应用，没有执行回退。需要源码回退时先重新检查工作区，再反向应用该补丁；若文件已被继续修改，检查失败应先审查差异。

独立验证：

```powershell
python experiments/rxmesh-remesh/tools/audit_phase1.py `
  --source examples/2.stl `
  --snapshot experiments/rxmesh-remesh/results/phase1_20260926/baseline/full_fast/input.cadpart `
  --repair-report experiments/rxmesh-remesh/results/phase1_20260926/baseline/full_fast/strip_repair/pass_01/repair.json `
  --baseline experiments/rxmesh-remesh/results/phase1_20260926/baseline/full_fast/remeshed.ply `
  --candidate experiments/rxmesh-remesh/results/phase1_20260926/new/full_release/remeshed.ply `
  --threshold 0.02669056691 --target 7.623706340789795 --max-error 1.524741292 `
  --output experiments/rxmesh-remesh/results/phase1_20260926/reproduced_audit.json
```

全部 12 个针对性测试通过。结果、原始日志、参数与 SHA256 证据在 `results/phase1_20260926/comparison.json`、`independent_audit.json`、`unrelated_preservation.json` 和两次完整流程目录中。

## 下一阶段的具体待办

先针对 3842/3843、3816 等最大的差区域建立有明确尺度依据的候选，并结合待处理区域列表检查被冻结约束和实际尺寸。区域比较需要继续保留面积加权、连通区域和尺寸指标；仅优化逐面 mean 或 P05 可能增加面数却不解决大片质量问题。

原始输入的双向几何误差和参考面对应需要统一，尤其是薄结构附近的来源对应与法向异常。之后再分别评估计算接缝来源分类和性能优化。第一阶段没有将这些问题解释成已经解决，也没有通过调整识别预算、几何预算或跳过困难区域满足计时目标。
