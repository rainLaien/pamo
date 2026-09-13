# CAD-aware STL 曲面分区

独立于 PaMO 的 CAD 网格分区模块。使用 VCGLib 读取 STL，输出几何曲面实例及供 remesh 使用的共享边界拓扑；不依赖 OpenCASCADE。默认采用 `EnableModelFirst=true`，旧流程通过 `--legacy` 保留用于比较。

## 分区含义

每个 Patch 表示一个连通曲面实例。多个圆柱即使类型相同，也不会仅按类型合成一个 Patch。

- 几何类型：Plane、Cylinder、Cone、Sphere、Torus、Freeform。
- 形状角色：Ordinary 或 Fillet。圆角需要两侧母面、相切关系及带状几何支持，不能只凭曲率较大判断。
- 圆形端盖仍属于 Plane；圆和圆弧是边界曲线类型。本模块输出原网格边界折线，尚未拟合圆弧。
- Freeform 表示当前解析模型未能可靠解释的连续区域，后续投影目标为原三角网格。

## 默认处理流程

```text
STL -> 按尺度焊接/清理 -> 三角形拓扑
    -> 连通区域整体解析拟合
    -> 平面核心与多尺度曲面种子
    -> 候选解析模型比较、固定模型扩展及全成员验收
    -> 剩余连通区域拟合 / Freeform 回退
    -> 唯一归属和连通性验证
    -> 共享边界链、硬边/平滑交界、母面与圆角角色
    -> 实例色 / 类型色 / 角色色 PLY + VTK + JSON
```

局部临时单元和候选种子只是计算工具，不直接成为输出分区。原始曲率、法向变化等边界分数不能自动升级为不可跨越的硬约束；边界、非流形边、显式特征和超过配置二面角的折痕限制扩展。相切的不同曲面仍可拥有共享交界线，但不当作尖锐折痕。

拟合使用局部坐标归一化和面积权重；非线性细化有采样预算，最终输出和归属验收检查全部成员。候选模型不能依靠一块占主导面积的大平面加少量曲面条带来伪造超大半径圆柱。小倒角不会因三角形数量少被丢弃。

分区本身不移动顶点、不新增三角形。焊接/清理完成后，每个三角形必须且只能属于一个 Patch。`validatePartition()` 检查面归属、反向标签、连通性；错误会使 CLI 返回非零退出码。

## 构建与运行

默认依赖位置为 `D:/openSourceInstall/vcglib`。Windows 脚本默认构建 Release，支持同时跑回归：

```powershell
.\cad_mesh\run_segmentation.ps1 `
  -InputStl .\examples\Unnamed-Body.stl `
  -OutputDirectory .\cad_mesh\debug\my_run -RunTests
```

手工构建：

```powershell
cmake -S cad_mesh -B cad_mesh/build -G Ninja `
  -DVCGLIB_ROOT=D:/openSourceInstall/vcglib -DCMAKE_BUILD_TYPE=Release
cmake --build cad_mesh/build --parallel
ctest --test-dir cad_mesh/build --output-on-failure
.\cad_mesh\build\cad_mesh_segment.exe .\examples\Unnamed-Body.stl .\cad_mesh\debug\my_run
```

直接在 C++ 程序内完成分区、保型 remesh 和最终 PLY 输出：

```powershell
.\cad_mesh\build\cad_mesh_segment.exe `
  .\examples\3.stl .\examples\native_remesh `
  --remesh --target-edge-length 12 `
  --max-normal-deviation-degrees 10 `
  --target-mean-quality 0.8 --require-remesh-cuda
```

该路径只写入 `native_remesh/remesh_result.ply`，不生成
`patch_result.ply`、JSON 或 Python 交接文件。第二个参数也可以直接指定一个
`.ply` 文件。未指定目标边长时使用包围盒对角线的 5%；未指定最大几何偏差时
使用目标边长的 0.5%。折叠点和松弛点始终受同 Patch 原始参考面约束；解析
曲面投影只有在与原 STL 的距离不超过该偏差时才接受。拆分点保持在当前边上，
共享分区边界保持同一组顶点。每批修改通过 BVH 和空间哈希拒绝自相交候选。

解析 Patch 在共享边界完成统一采样后进入独立的整块参数域重建。Plane 使用平面
坐标，Cylinder/Cone 使用展开后的周期坐标，闭合 Sphere 使用球面细分，闭合
Torus 使用双周期参数网格；参数缝映射回三维时焊接为同一顶点。成功重建的解析
Patch 会被冻结，不再进入局部拓扑优化。无法形成合法参数图、拟合误差超过几何
偏差或重建质量下降的 Patch 才进入 Freeform 使用的保型局部回退路径。

Remesh CUDA 内核以源码嵌入可执行程序，并复用解析拟合已有的 CUDA Driver +
NVRTC 运行时动态编译。构建阶段不调用 `nvcc`，因此 MinGW 构建不要求 MSVC
`cl.exe`。默认优先使用 CUDA 批量筛选拆分边；CUDA 运行时不可用时回退到原生
CPU，`--require-remesh-cuda` 可改为直接报错。拓扑修改由 C++ 执行，因此运行阶段
不依赖 Python、PyTorch 或 LibTorch。

默认模式的 CLI 参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--fit-tolerance-ratio` | 3 | 相对于网格尺度拟合容差的最大顶点距离倍率 |
| `--normal-angle-deg` | 约 8.02 | 三角面法向与解析模型法向最大允许角度 |
| `--sharp-angle-deg` | 约 37.24 | 阻止初始跨边扩展的二面角阈值 |
| `--seed-radius-factor` | 8 | 局部种子物理半径相对于中位边长的倍率 |
| `--max-model-seeds` | 2500 | 局部候选种子的计算预算；耗尽后未解释区域保留为连续 Freeform |
| `--analytic-seed-backend` | 原生 CLI 为 `auto`，脚本为 `cuda` | 非线性解析种子拟合后端；`cuda` 不可用时明确报错，`auto` 可回退 CPU |
| `--legacy` | 关闭 | 使用旧边界分数、热扩散和闭合流程 |

`--strong`、`--weak`、`--rings` 仅影响 legacy 模式。长度容差由包围盒和边长统计派生，API 配置见 `include/CadMesh/Types.h`。提高容差可能将真实细小特征归入邻面；应结合输出误差和可视化验收，而不是只追求 Patch 数更少。

### CUDA analytic seeds

`--analytic-seed-backend cuda` 将 Cylinder、Cone、Sphere、Torus 的非线性拟合移到 GPU：同一曲面的多个初值一起提交，每个初值由一个 128 线程块完成残差、有限差分、Hessian、阻尼求解和回溯迭代，避免每轮 CPU/GPU 往返。仍使用 double、原采样预算和迭代上限；初始小矩阵拟合、全成员误差、模型选择、区域扩展和最终认证保持 CPU 原有流程。GPU 数值求解失败的单个初值从原初值重试原 CPU 求解器。

原生库通过 CUDA Driver + NVRTC 动态加载，不引入 nvcc/MSVC 构建要求，也不依赖 Warp。运行需要 NVIDIA 驱动和 NVRTC（可来自 `CUDA_PATH/bin` 或项目 PyTorch 的 `torch/lib`）。`run_segmentation.ps1 -AnalyticSeedBackend cuda` 会构建后运行；`remesh_file.py` / `remsh.ps1` 默认为 `cuda`，发现原生源码更新时会先对已有 CMake 配置增量构建 `cad_mesh_segment`，不运行测试。编译耗时与模型处理耗时需分开观察。

日志增加 `analytic seed search`、`residual seed search` 等分区子阶段耗时，以及 CUDA 初始化和种子批次统计；可以确认实际调用了 GPU。`--analytic-seed-backend cpu` 保留原 CPU 非线性求解，`auto` 在运行时不可用时记录原因并回退。CUDA 使用最多 8 个候选的前瞻窗口合并独立拟合，非线性种子数达到 256 后停止继续收集。实际候选仍根据当前面归属重新生成支持面，按原顺序评估和提交；仅复用完全相同的求解输入。718,710 面模型的局部对照中，原生分区从上一轮的 43.40 秒降到 30.65–30.87 秒，五个导出文件逐字节一致，原生测试通过。

CUDA 日志同时统计上传/回传字节数、样本上传缓存命中、CPU 打包与缓存耗时、上传耗时以及回读等待时间。回读等待包含内核执行和传输，不能视为纯 CPU 调度成本。设置 `CADMESH_CUDA_PROFILE=1` 并开启详细日志可额外记录 CUDA event 内核时间；默认不记录事件，计时利用已有阻塞回读完成后读取结果，不增加设备同步。

`remesh_file.py` 和 `remsh.ps1` 默认直接调用原生 `--remesh`：STL 导入、分区、参数域重建和最终 PLY 导出在同一个 C++ 进程内完成，只复制 `remesh_result.ply`，不再启动 `remesh_partition.py`，也不生成或读取 PLY/JSON 交接文件。显式使用 `--full-output` 时才保留旧的 Python 诊断流程。

分区缓存和 `--no-partition-cache` 只用于 `--full-output` 诊断流程。默认原生单文件流程不序列化分区，因此不需要缓存。末尾的 `[remesh-file] total wall time` 包含分区、重网格化及最终 PLY 复制。

默认模型优先流程在初始种子预算之外，增加 `ModelResidualSeedBudget=1250` 的空间轮询残余搜索；随后执行最多 `ModelMaximumMergePasses=4` 轮邻接归并。归并对整个合并组重拟合并认证，共享硬边会否决合并；小 Freeform 或局部平面残片仅在面积、法向和完整拟合检查通过时吸附到解析母面。两项配置可由 C++ `SegmentationConfig` 调整。

最终归并后还执行局部曲面补缝：沿实际邻接查找夹在同一圆柱/环面两侧的兼容 Plane 或 Freeform 走廊。双侧边界支撑、物理宽度、全体端片的共同曲面解释及联合重拟合均需通过；显式硬边、非流形和真实二面角屏障继续保留。它能从大 Freeform 中只接回局部走廊，并将剩余面重新拆为连通区域。短圆弧局部拟合的圆心/半径可能不稳定，因此补缝会检查实际三角形支持，而不会仅因参数略有差异就认定是不同曲面。`EnableModelSeamBridging` 控制该阶段，`ModelSeamMaximumFaces`、`ModelSeamEvaluationBudget` 和 `ModelSeamMaximumWidthRadiusRatio` 限制搜索范围与成本。

圆角角色判断将同一母平面的多个标签按一个物理支撑侧累计证据，输出仍以两个实际相邻的代表 `support_patch_ids` 指向两侧母面。补缝发生在分区器生成最终几何约束之前；使用旧分区报告直接 remesh 会继续遵守其中已有的硬边，应用此修正需要重新分区。

## 输出和 remesh 交接

- `patch_result.ply`：每个曲面实例独立着色。
- `surface_types.ply`：固定类型色（平面蓝、圆柱绿、圆锥橙、球面紫、环面粉、自由曲面灰）。
- `feature_roles.ply`：圆角橙、普通区域灰。
- `patch_report.json`：全部面成员、类型/角色/母面、参数、误差、共享边界及方向。
- `boundary_score.vtk`：边界拓扑及硬边/平滑交界。只有实际计算过的边界评分才输出为标量。
- legacy 模式另外输出四种曲率 VTK。默认模式不计算局部差分评分；JSON 的 `diagnostics` 显式标记未计算，相关值为 `null` 或省略。

三份 PLY 具有相同的顶点、面和标签，只改变颜色。面属性包含 `patch_id`、`primitive_type`、`feature_role`；读取器应按 PLY 属性名解析，不能假设最后三个字段是 RGB。

`ConstraintEdgeIds` 是所有需要共享采样的边界并集；`HardFeatureEdgeIds` 和 `SurfaceTransitionEdgeIds` 将它分成硬边及平滑曲面交界。平滑交界仍需共享采样以防裂缝，下游可以采用不同的法向/切向约束。

全局 `BoundaryChains` 提供有序边/顶点、两侧或全部关联 Patch、共享初始采样。闭环重复首顶点；链在端点、分叉、关联 Patch 或边界类别变化和明显转角处分段。相邻 Patch 引用同一条链，不能各自独立生成边界点。

`BoundaryChainRefs.Direction` 为 +1（顺序）、-1（逆序）、0（未解析）。非流形、多侧和绕序不一致会显式标记，不能将 0 当正向。ID 指向同次导出的清理后 PLY，不直接对应 STL 的重复顶点索引。

### 误差含义

解析 Patch 的 `rms` 是按三角形面积加权的顶点到拟合曲面距离；`max` 是全部成员顶点的最大距离。`normal_error` 是面积加权法向误差。最终默认分区会重新统计固定模型的全部成员，不混用种子或增长前的统计。

`sampled_mesh_deviation` 另外检查原三角形顶点、边中点和面心，用于显示离散平面三角形与恢复解析曲面的弦高差。它不是严格的 Hausdorff 上界，不能作为 remesh 全表面误差的证明。Freeform 的拟合误差仅是最佳平面的诊断基线，没有恢复出的自由曲面；其投影目标为 `ReferenceMesh`。

## 适用范围

Plane、Cylinder、Cone、Sphere、Torus 均有实际拟合。Cone 参数原点是顶点，轴指向单侧锥面，半角使用弧度；目前验收角度为 0.5°～89.5°。Torus 目前支持常规环面（主半径大于 1.005 倍管半径）。病态、证据不足或无法由解析面解释的区域回退为 Freeform。

圆角角色采用保守确认；复杂多母面交汇或变化半径圆角可能暂不标 Fillet。仅凭 STL 不能唯一恢复 CAD 建模历史，也无法区分产生完全相同三角网格的圆柱离散条带与刻意设计的多边形面。

周期解析曲面带 `NeedsParameterizationSeamAssessment` 标记。分区器没有生成参数域接缝或精确 CAD 边界曲线。后续 remesh 现已通过下述 Python 桥接接入 PaMO；网格尺寸由目标边长控制，Patch 数量不直接决定最终网格密度。

## 验证

回归覆盖完整圆柱在不同密度、非均匀采样及不同对角线下的稳定分区，带孔平面、浅窄倒角、不同半径/轴的相邻圆柱、断开的同类曲面，以及圆锥和环面。解析拟合测试包含单位缩放、大坐标偏移、面积权重和采样预算之外的异常顶点。原有单面倒角、非流形、热扩散回滚和共享链回归继续保留。

导出后用仅依赖 Python 标准库的检查器独立验证 JSON/PLY 一致性、全部边界、母面、共享采样与方向；大模型使用预索引避免逐 Patch 重扫全边：

```powershell
python .\cad_mesh\tests\verify_handoff.py .\cad_mesh\debug\my_run
```


## 接入 PaMO：对分区结果直接 remesh

默认 `surface` 模式采用整块曲面重建：兼容标签归并 → 共享几何边界统一采样 → 平面/圆柱/圆锥参数域重网格 → 其他区域的 PaMO 三维拓扑优化和整块参考曲面投影。保留几何类别及圆角角色，归并后的区域通过 `source_patch_ids` 记录来源编号。

平面支持带孔边界；圆柱和圆锥支持局部展开以及可验证的完整周向图，周期缝两侧共用同一组三维顶点。无法认证边界、参数图或误差的区域明确记录拒绝原因并回退到三维算法。三维路径允许原始内部顶点和新增点跨越原三角形边，在同一完整参考曲面内重新投影；不会把顶点限制在其最初的来源三角形内。

对已有分区结果运行（项目根目录）：

```powershell
.\cad_mesh\run_remesh.ps1 `
  -PartitionDirectory .\cad_mesh\debug\111_model_first_final_20260907
```

默认输出到输入分区目录的 `pamo_remesh` 子目录。指定世界坐标单位下的目标边长和输出位置：

```powershell
.\cad_mesh\run_remesh.ps1 `
  -PartitionDirectory .\cad_mesh\debug\111_model_first_final_20260907 `
  -OutputDirectory .\cad_mesh\debug\111_remesh_custom `
  -TargetEdgeLength 5 -SampleCount 2000
```

也可从 STL 一条命令完成分区及 remesh：

```powershell
.\cad_mesh\run_segmentation.ps1 `
  -InputStl .\examples\Unnamed-Body.stl `
  -OutputDirectory .\cad_mesh\debug\unnamed_partition `
  -Remesh -RemeshEdgeLength 10 -RunTests
```

Python 入口支持已有目录或其中的 `patch_result.ply`：

```powershell
.\.venv\Scripts\python.exe .\cad_mesh\remesh_partition.py `
  .\cad_mesh\debug\111_model_first_final_20260907\patch_result.ply `
  --target-edge-length 5 --sample-count 2000 --batch-face-limit 200000
```

运行环境沿用 PaMO 的 Python 环境；默认选择项目 `.venv/Scripts/python.exe`，依赖 NumPy、SciPy、trimesh、libigl、支持 CUDA 的 PyTorch 和 NVIDIA GPU。可用 `-PythonExecutable` 指定现有环境，不需要创建 PaMO 的 SDF/DMC 对象。

参数域重建另外使用 `triangle` 的约束三角化。`-Method surface` / `--method surface` 为默认入口；`-Method legacy` / `--method legacy` 保留原三角形细化算法用于对比。分区脚本可用 `-RemeshMethod` 选择 remesh 方法。

### 分区、约束和显存

先将通过整个合并组几何检查的同类曲面标签归入同一求解区域，其内部标签边不再固定。真实硬边、不同曲面的平滑交界、开口和非流形边保留为几何约束；同一共享边只生成一次新采样点，内部求解固定这些点和几何角点。当前共享采样保留原折线并补足过长边，不擅自删除曲线的原始折点。输出仍使用单一全局顶点表。

三维路径默认每批最多 75,000 个源三角形（边界预细分后可能略多）。小分区合批，过大曲面按连通遍历安排计算分块；投影查询仍使用该曲面的完整参考网格。分块接缝顶点暂时固定后在全局拼接，仍会限制接缝附近的优化自由度，但不会成为新的 `patch_id` 或语义硬边。已接受的解析参数图按完整曲面处理。可用 `-BatchFaceLimit` / `--batch-face-limit` 调节显存与批次成本；`remesh_file.py` 同样支持该参数。

`TargetEdgeLength` 是输出边长上限，省略时沿用 PaMO 原曲面约束模式的默认值：模型包围盒对角线的 5%。它不是最终面数目标，细化后面数可能增加。`SampleCount` 控制表面采样预算；各批按面积分配。`--max-deviation` 在默认模式下控制对所属完整参考曲面的采样偏差，默认输入模型包围盒对角线的 0.025%，与目标边长独立（legacy 模式仍为目标边长的 0.5%）；它不等价于经过证明的全局 Hausdorff 上界。

长边拆分优先使用解析曲面中点；若拟合曲面上的投影使任一侧子三角形违反约束，则依次尝试参考网格投影和原边中点。候选优先保留法向余量，必要时在用户指定的最终法向上限内重试；显式设置的候选法向限制仍然有效，方向及曲面距离检查始终保留。共享固定边仍由全局边界细分处理。报告中的 `stats.batches[].split_diagnostics` 记录拆分轮数、回退成功数及停止原因，区分轮数耗尽、几何拒绝和固定边阻塞；只有轮数耗尽才提示增加 `--split-passes`。

输出相对参考网格的法向偏差上限现默认 **10°**，由 `--max-normal-deviation-degrees` 或 PowerShell 的 `-MaxNormalDeviationDegrees` 控制；Python API 对应 `maximum_normal_deviation_degrees`。需要原来的严格上限时，显式传入 `5`。该选项传递到解析曲面检查和三维 remesh，几何距离上限与边长上限独立设置。调整法向上限会重新执行 remesh，但不使原生分区缓存失效。

完整曲面模式的预折叠和预翻边要求替换三角形的全部边已经满足目标边长，避免提前改变长三角形覆盖的参考曲面区域，造成后续子三角形无法满足法向约束。超长区域保留原始三角形进行细分，再执行后续质量优化；最终边长、曲面和拓扑校验不变。

### remesh 性能与耗时

2026-09-09 针对 718,710 面、目标边长 12 的日志继续优化：合并短边折叠重复的曲面检查，修复大模型缓存键溢出失效，布尔距离查询使用阈值内支撑搜索；CPU 分组、约束索引、边界细分、解析图映射及 PLY 读写减少重复扫描。每批保留 PyTorch 分配缓存，当前默认批次为 75,000 面。后续加入不可能参数图的提前判定、增大 CUDA 校验缓存及 CPU 边集合复用，并完成全部 13 个批次和全模型验证；实测范围及结果见 [性能验证记录](REMESH_PERFORMANCE_VERIFIED_20260909.md)。40 秒目标尚未达到。

最近一轮最小交接、单 PLY 输出和纯诊断计算裁剪后，同模型默认入口首次全流程从 144.46 秒降到 116.56 秒，命中分区缓存时从 95.63 秒降到 80.95 秒。两次均通过完整验证；输出面索引和标签完全一致，顶点坐标仅有约 10⁻¹² 的浮点差异，未改变质量策略。

默认 `surface` 路径启用以下计算复用，原有 `run_remesh.ps1` 命令直接生效：

- 短边折叠期间坐标固定，相同曲面标签和有序顶点组合复用完整曲面检查，包括被拒绝的组合；CUDA 缓存最多 2^22 个槽（约 148 MiB），CPU 缓存最多 1,000,000 项，每次折叠调用独立持有，坐标变化时失效。
- CUDA 将缓存未命中的查询集中到连续队列后检查，队列计数保留在 GPU。双向均被拒绝的折叠边在其相邻面未变化时复用结果；任何相关面修改或删除后立即重新评估。
- 顶点优化回退后，仅重查坐标实际变化的三角形。极细三角形的法向歧义通过按尺寸分组的空间索引筛选附近参考面，再执行原有严格几何判断。
- 锁定端点的无效折叠方向在曲面查询前排除；报告使用 C JSON 编码器批量写出。
- 局部细分使用整数边键进行保护边匹配；每批仅加载自身涉及的解析投影模型，纯自由曲面批次省去无用的解析投影计算。
- 最终 CUDA 曲面检查保留中心点最近面和法向歧义判定，其余六个采样点只查询是否存在误差阈值内的参考支撑。成功时返回相同的有效性及歧义标记；失败诊断和详细 tracing 仍计算精确距离、角度，检查规则不变。

采样预算、优化轮数、误差阈值、共享边保护和最终完整验证不变。默认跳过只用于报告的质量统计、全局采样偏差诊断，以及上层不使用的最终最近源面查询及专用局部索引；算法过程中的源面归属与曲面校验保持完整。解析参数图保留同一均值质量接受条件。日志保留每个 CUDA 批次及读取、计算、导出的耗时；完整报告模式还记录 `stats.batches[].batch_seconds`。性能对照见 [性能验证记录](REMESH_PERFORMANCE_VERIFIED_20260909.md)。

### remesh 输出

默认 CLI 和 PowerShell 入口仅生成 `remesh_result.ply`：完整共享索引网格，维持原 ASCII 格式、float64 坐标、颜色及 `patch_id`、`primitive_type`、`feature_role`。完整文件入口的位置为 `OUTPUT/remesh/remesh_result.ply`；已有分区入口则写到指定输出目录。

需要诊断时，添加 `--full-output`，或对 `remsh.ps1` / `run_remesh.ps1` 使用 `-FullOutput`，额外生成以下文件；完整文件入口同时保留两份内部交接文件：

- `surface_types.ply`、`feature_roles.ply`：相同几何及索引，仅改变颜色。
- `remesh_result.stl`：便于常规网格工具打开；STL 不携带分区标签，坐标按格式为 float32。
- `remesh_report.json`：新三角形成员和 `source_patch_ids`、原曲面参数来源、必要几何约束、已解除的标签边界、角点、几何约束到子边的连续来源映射、参数图接受/拒绝原因、质量和验证统计。schema 为 `cadmesh.partition_remesh`，与原始分区报告区分。

输出前检查求解区域全部保留、共享边两侧关联、几何角点及边界坐标、退化/重复三角形、新裂缝/非流形边、绕序以及目标边长。报告中的 `source_fit_diagnostics` 属于分区输入；`sampled_reference_deviation` 是 remesh 后采样点到参考网格的偏差诊断，也不是严格的全表面误差证明。质量报告同时包含平均质量与最差角度，平均质量改善不代表每个三角形都改善。

合并区域的 `source_fit_diagnostics` 仅属于 `representative_source_patch_id`，不是整个合并区域的重拟合误差；`source_fit_diagnostics_by_patch` 逐项保留全部原分区的诊断。共享几何边的来源映射以报告显式列出的 `source_constraint_edges` 为准，已解除的标签边单独记入 `boundary_policy`。

导入支持原 ASCII 和内部 binary little-endian 三角 PLY，按属性名称读取，不进行隐式焊接或重排。二进制输入先核对实际文件尺寸与声明记录数，再分配数组；成员、拓扑、角点和约束验证与 ASCII 一致。原输入不会被覆盖，文件不匹配或约束失效会明确报错。Python 库函数仍默认收集诊断并完整导出，可显式设置 `collect_diagnostics=False` / `full_output=False` 使用精简路径。

### remesh 回归

```powershell
$env:CADMESH_TEST_GPU = '1'
.\.venv\Scripts\python.exe -m unittest discover -s cad_mesh/tests -p test_remesh_pipeline.py
.\.venv\Scripts\python.exe -m unittest discover -s cad_mesh/tests -p test_surface_domains.py
.\.venv\Scripts\python.exe -m unittest discover -s cad_mesh/tests -p test_analytic_remesh.py
.\.venv\Scripts\python.exe -m unittest discover -s cad_mesh/tests -p test_surface_rebuild.py
.\.venv\Scripts\python.exe -m unittest discover -s simp_cuda/tests -p test_partition_boundary_lineage.py
.\.venv\Scripts\python.exe -m unittest discover -s simp_cuda/tests -p test_surface_sample.py
.\.venv\Scripts\python.exe -m unittest discover -s simp_cuda/tests -p test_whole_patch_surface.py
```

包含真实 C++ 分区交接、同平面不同标签的平滑接口、带孔平面、计算分块合并、无可采样面的密集批次，以及共享边来源和输出三种 PLY 的索引一致性。未设置 `CADMESH_TEST_GPU=1` 时，导入/导出验收仍运行，GPU 用例跳过。

2026-09-07 的 legacy 对比基线：`111_model_first_final_20260907` 默认目标边长 34.396983，718,710 面输出为 759,382 面，保留全部 2,734 个分区；最大边长 34.396352，无新增裂缝或非流形边。边长变异系数由 3.115 降至 1.949，但平均三角形质量由 0.608 降至 0.578，最差角度没有改善。该数据属于旧算法，新的整块曲面结果应单独比较。

2026-09-08 的完整 `surface` 结果位于 `debug/111_surface_rebuild_20260908/remesh`：同一 718,710 面输入输出为 805,464 面，2,593 个整块解析参数图接受；其余区域在完整参考曲面上优化，217,111 个原内部顶点发生移动。平均质量达到 0.724，边长变异系数 1.559，质量低于 0.1 的面占比由旧版的 13.44% 降至 4.53%。独立读回确认水密、绕序一致，无非流形、重复或退化面；最大边长 34.385706。分区器进行了 43 次邻接归并，但新增解析识别使分区总数达到 2,945；remesh 再合并为 2,922 个求解区域。因此本次改善了网格质量，面数和分区总数尚未下降，最差角度仍受保留边界上的极窄原始面限制。详见 [本次验证记录](debug/111_surface_rebuild_20260908/RESULTS.md)。

同日的圆角细缝修正版位于 `debug/111_fillet_seams_20260908`：52 次局部桥接使分区数从 2,945 降至 2,862；remesh 输出 2,840 个区域、808,562 面，平均质量 0.726。独立读回验证通过，仍保持水密。针对性候选中的 28 个平面条带与 3 个大型 Freeform 局部走廊已接回曲面，56 条候选误判硬边解除；一处较宽环面交汇因缺少对向支撑保留。见 [局部优化验证记录](debug/111_fillet_seams_20260908/RESULTS.md)。



普通 PLY/STL 的 `remesh_file.py` 同样支持 `--max-deviation`，PowerShell 入口使用 `-MaxDeviation`。仅调整 `--target-edge-length` 不再同步收紧默认几何误差；若需要更严格的几何保真，应显式指定误差值。
