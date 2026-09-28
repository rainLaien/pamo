# 第十九阶段：约束证据与原生尺寸场

## 已完成与接受范围
这是可单独启用、可关闭的诊断数据改动。原始 STL 完整流程已执行，源快照、修复快照和最终 PLY 与 phase16 Workers8 对照的 SHA256 全部相同。没有改变边界移动权限、质量阈值、分区、GPU 算子或局部重建。本轮接受的是数据导出能力；网格的质量验收仍未通过。未经提交或推送。

## 约束证据
新的 ConstraintAudit 与操作 EdgeFlag 分开。CADPART1 第四列 hard 被保存在不可变输入身份域中。分类位可重叠；不会把 hard=0 全部视作可解锁接缝。

|输入来源类别|本次修复快照边数|解释及当前权限|
|---|---:|---|
|hard=1，同分区内部|699|生产者硬特征标记；未证明是 CAD 曲线；仍保护|
|hard=1，真实开边界|81|拓扑确认的孔或外轮廓边界；仍保护|
|hard=1，分区接口|8136|同时具有硬标记与接口身份；仍保护|
|hard=0，分区接口|83969|无法区分模型识别边界与计算接缝；仍保护|

角点认证和源曲线到输出曲线的连续身份映射仍未导出，清楚标为未知。source_constraints.tsv 使用输入快照顶点 ID；vertices.tsv/edges.tsv 使用对应保存 PLY 的顶点 ID。不得把两类 ID 直接连接。最终 edges.tsv 的 feature_id 是操作拓扑 ID，也不是 CAD 曲线认证。

## 实际尺寸验收
vertices.tsv 保存最终原生 targetLength 和 vertexConstraint；edges.tsv 保存有效目标尺寸、长短边比及 persistent_feature 布尔标记。有效尺寸计算与 RegionQuality 完全一致，包括必要的全局或特征尺寸回退。

独立工具校验了输出坐标的 float32 身份、全部边覆盖、尺寸与长度比、输入硬标记和特征记录身份，并精确复算原生长短边统计：

- 长边：0。
- 短边：676458，其中受约束短边 98300。
- 最终顶点目标尺寸范围：6.861335754394531 至 7.623706340789795。
- 旧全局 h 口径短边：678731。差异为2273，必须分别报告，不能混用。

低质量总面积仍为4727.792922，最大连通面积55.522459，平均质量0.568358022，P05=0.032024174。全局有 1945.845404 低质量面积直接接触受约束短边，约 41.16%。这是相关性，不是因果证明。

困难区3816的低质量面积87.418056，其中直接接触受约束短边仅0.334305；3842对应59.770193与1.005544。因此不能把整片低质量统一解释为受约束短边，也不能直接推出解锁边界就能解决问题。

## 完整性能与检查边界
从原始 examples/2.stl 开始，包含分区、预处理、重建、remesh、数据导出与最终输出，总耗时 **116.19601940秒**。分区及打包 44.40299680秒，首轮窄带8.840秒；native加载1.4751秒，batch53.5805秒，保存及报告3.66137秒，二次修复2.323秒。对照完整流程107.3933895秒。两次运行之差不能全部归因于导出开销；未做重复计时，不宣称性能改善。

最终 PLY 字节相同，先前同一 PLY 的特征链、拓扑、接缝与几何采样审查证据仍适用；未重新运行93秒独立几何审查。先前证据包括92885特征链、92105接缝链、81开边界保留，锚点位移0；1个连通分量、Euler=-1，无非流形、重复面、内部绕序冲突或float32零面积面。原始 STL 双向采样误差1.267034/1.304286，小于原误差预算1.524741292。这不是连续误差、自交或全部真实特征认证；先前456个最近源三角形法向非正样本仍待辨别，不能宣称翻转已全部消除。

## 修改文件
- include/cad_adaptive/ConstraintAudit.h 与 src/ConstraintAudit.cpp：来源数据与字段导出，失败导出标为 incomplete。
- PartitionInput.h/.cpp：可选保存 hard、原始特征 ID 与固有边界证据；原有保护行为保持。
- raw_partition_cli.cpp：--audit-fields。
- run_raw_partition.ps1：-AuditFields，并在 pipeline.json 中记录候选与最终选择的数据前缀。边数据保留在各轮目录，避免把回退到参考结果后的尺寸场误绑定到候选。
- RegionQuality.h：修正关于 CADPART1 完全没有来源的旧注释，明确 hard 与缺失信息的区别。
- tools/audit_native_fields.py：独立复算与输入/输出身份检查。
- CMakeLists.txt、test_constraint_audit.cpp、test_native_fields_audit.py：有针对性的验证。

5项 CTest 与4项 Python 检查通过。实验目录 diff --check 通过，9段无关修改与初始快照逐段一致。source_additions 与 remesh_current.patch 保存当前实验源码和差异，含 git 忽略的测试文件。

## 复现
```powershell
cmake --build experiments/rxmesh-remesh/build_rx --config Release --target cad_raw_partition_cli test_constraint_audit --parallel 8
ctest --test-dir experiments/rxmesh-remesh/build_rx -C Release -R '^(constraint_audit|region_quality|region_candidate_selection|raw_batch_safety|raw_projection_safety)$' --output-on-failure
python -m unittest discover -s experiments/rxmesh-remesh/tests -p test_native_fields_audit.py
./experiments/rxmesh-remesh/run_raw_partition.ps1 -InputMesh ./examples/2.stl -OutputDirectory ./experiments/rxmesh-remesh/results/phase19_constraints/reproduce -ModelSeeds 250 -Workers 8 -GpuConcurrency 4 -PatchesPerTask 16 -Iterations 12 -SmoothPasses 3 -CollapsePasses 8 -FlipPasses 8 -TargetLength 7.623706340789795 -MaxError 1.524741292 -LowQualityThreshold 0.026690566912293434 -SelectFinalRegions -AuditFields
python experiments/rxmesh-remesh/tools/audit_native_fields.py --mesh experiments/rxmesh-remesh/results/phase19_constraints/full/strip_repair/pass_01/remeshed.ply --snapshot experiments/rxmesh-remesh/results/phase19_constraints/full/strip_repair/pass_01/input.cadpart --output experiments/rxmesh-remesh/results/phase19_constraints/native_fields_audit.json
python experiments/rxmesh-remesh/results/phase19_constraints/diagnose_defects.py
```

完整流程的质量拒绝是预期结果，流水线返回失败状态；不得改成成功状态来声称达标。最后两条检查现有完整运行结果，其耗时不计入已记录完整流程。

## 下一步唯一目标
选3842及必要邻区，结合来源证据和实际尺寸场评估共享曲线站点的重新采样是否可行。先明确真实几何特征与计算接口的证据，并保留真实角点、孔连接关系及原始误差预算。任何不能证明可移动的边界继续保护；联合区域质量必须通过，不能用窄带单独改善换取邻区变差。最多两轮受控候选比较，失败停止该方向。
