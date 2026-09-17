#include "cad_adaptive/IRemeshBackend.h"
#include "cad_adaptive/RemeshMetrics.h"
#ifdef CAD_ADAPTIVE_RXMESH
#include "cad_adaptive/RxMeshBackend.h"
#endif
#ifdef CAD_ADAPTIVE_GLOBAL_TOPOLOGY
#include "cad_adaptive/global/GlobalSplitBackend.h"
#endif

#include "cad_adaptive/PartitionInput.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace cad_adaptive;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: cad_adaptive_cli INPUT.(obj|stl) OUTPUT.ply [target_length] [--gpu|--cpu|--global]\n"
              << "       cad_adaptive_cli INPUT.cadpart OUTPUT.ply [target_length] --partition [--gpu|--cpu|--global]\n"
              << "       cad_adaptive_cli --grid TRIS OUTPUT.ply [target_length] [--gpu|--cpu|--global]\n";
    return 2;
  }
  const bool grid = std::strcmp(argv[1], "--grid") == 0;
  bool useGpu = false;
  bool useGlobal = false;
  bool partitionInput = false;
  float target = 0;
  float maxError = 0;
  bool smooth=true, collapse=true, flip=true, cavity=true, split=true;
  float cavityRatio=2.5f;
  float smoothLambda=0.5f;
  int iters = 5;
  const int opt0 = grid ? 4 : 3;
  for (int i = opt0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--partition") == 0) partitionInput = true;
    else if (std::strcmp(argv[i], "--global") == 0) { useGlobal = true; useGpu = false; }
    else if (std::strcmp(argv[i], "--gpu") == 0) { useGpu = true; useGlobal = false; }
    else if (std::strcmp(argv[i], "--cpu") == 0) { useGpu = false; useGlobal = false; }
    else if (std::strcmp(argv[i], "--no-smooth") == 0) smooth=false;
    else if (std::strcmp(argv[i], "--no-collapse") == 0) collapse=false;
    else if (std::strcmp(argv[i], "--no-flip") == 0) flip=false;
    else if (std::strcmp(argv[i], "--no-split") == 0) split=false;
    else if (std::strcmp(argv[i], "--no-cavity") == 0) cavity=false;
    else if (std::strcmp(argv[i], "--cavity-ratio") == 0 && i+1<argc)
      cavityRatio=std::strtof(argv[++i],nullptr);
    else if (std::strcmp(argv[i], "--smooth-lambda") == 0 && i+1<argc)
      smoothLambda=std::strtof(argv[++i],nullptr);
    else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--max-error") == 0 && i + 1 < argc)
      maxError = std::strtof(argv[++i], nullptr);
    else target = std::strtof(argv[i], nullptr);
  }
  SemanticMesh mesh;
  std::string error;
  const char *outPath = grid ? argv[3] : argv[2];
  if (grid) {
    const int tris = std::max(2, std::atoi(argv[2]));
    const int n = std::max(2, int(std::ceil(std::sqrt(double(tris) * 0.5))));
    mesh = makeGrid(n, n, 0, 0, 1, 1, 0);
  } else if (!(partitionInput ? loadPartitionInput(argv[1], mesh, &error) : mesh.load(argv[1], &error))) {
    std::cerr << error << '\n';
    return 1;
  }
  lockMeshBoundary(mesh);
  std::cout << "geometric_patches=" << mesh.patches.size() << " partition_input=" << partitionInput << '\n';
  RemeshConfig cfg;
  cfg.adaptive = false;
  cfg.constantLength = target;
  const float modelScale = mesh.bboxDiagonal();
  const bool defaultTargetLength = !(cfg.constantLength > 0);
  if (defaultTargetLength) cfg.constantLength = modelScale * 0.01f;
  std::cout << "target_length=" << cfg.constantLength
            << " model_scale=" << modelScale
            << " target_source=" << (defaultTargetLength ? "bbox_diagonal*0.01" : "explicit")
            << '\n';
  cfg.maxGeometryError = cfg.constantLength * 0.2f;
  if (partitionInput) cfg.maxGeometryError = mesh.bboxDiagonal() * 0.001f;
  if (maxError > 0) cfg.maxGeometryError = maxError;
  cfg.maxIterations = iters > 0 ? iters : 5;
  cfg.enableSmooth=smooth; cfg.enableCollapse=collapse; cfg.enableFlip=flip;
  cfg.smoothLambda=smoothLambda;
  if (partitionInput) {
    const int boundarySplits=refinePartitionBoundary(mesh,cfg.splitRatio*cfg.constantLength);
    std::cout << "boundary_splits=" << boundarySplits << '\n';
  }
  if (!cfg.adaptive) {
    mesh.targetLength.assign(size_t(mesh.vertexCount()), cfg.constantLength);
  }
  mesh.rebuildTopology();
  const EdgeLengthAudit beforeAudit = mesh.edgeLengthAudit(
      cfg.constantLength, cfg.splitRatio, cfg.collapseRatio);
  std::cout << "edge_length_audit_before "
            << "target=" << beforeAudit.targetLength
            << " split_threshold=" << beforeAudit.splitThreshold
            << " collapse_threshold=" << beforeAudit.collapseThreshold
            << " min=" << beforeAudit.min
            << " p05=" << beforeAudit.p05
            << " median=" << beforeAudit.median
            << " p95=" << beforeAudit.p95
            << " p99=" << beforeAudit.p99
            << " max=" << beforeAudit.max
            << " editable_max=" << beforeAudit.editableMax
            << " protected_max=" << beforeAudit.protectedMax
            << " editable_above_split=" << beforeAudit.editableAboveSplit
            << " editable_below_collapse=" << beforeAudit.editableBelowCollapse
            << " protected_above_split=" << beforeAudit.protectedAboveSplit
            << " edge_count=" << beforeAudit.edgeCount << '\n';
  const AnalyticGeometryAudit beforeGeomAudit = mesh.analyticGeometryAudit();
  std::cout << "analytic_geometry_audit_before"
            << " mean=" << beforeGeomAudit.mean
            << " p95=" << beforeGeomAudit.p95
            << " max=" << beforeGeomAudit.max
            << " supported_vertices=" << beforeGeomAudit.supportedVertexCount
            << " unsupported_vertices=" << beforeGeomAudit.unsupportedVertexCount << '\n';
  RemeshReport report;
  SemanticMesh reference;
  if (partitionInput) reference = mesh;
  bool ok = false;
  const char *backendName = "cpu";
  if (useGlobal) {
#ifdef CAD_ADAPTIVE_GLOBAL_TOPOLOGY
    global::GlobalSplitBackend backend;
    // Global pools grow dynamically between passes; the factor is only initial headroom.
    ok = backend.Initialize(mesh, 64.0f, &error);
    for (int cycle = 0; ok && cycle < cfg.maxIterations; ++cycle) {
      global::GlobalTriangleRefineReport cavityReport;
      global::GlobalSplitReport splitReport;
      global::GlobalCollapseReport collapseReport;
      global::GlobalTopologyValidation cavityValidation;
      global::GlobalTopologyValidation splitValidation;
      global::GlobalTopologyValidation collapseValidation;
      if (cavity) {
        ok = backend.RunTriangleRefinePass(cavityRatio, cavityReport, &error);
        report.cavityRefines += int(cavityReport.acceptedCount);
        report.cavityCandidates += int(cavityReport.candidateCount);
        report.secondsCavity += cavityReport.totalMs * 0.001;
        if (ok) ok = backend.Validate(cavityValidation, &error);
      }
      if (ok && split) {
        ok = backend.RunSplitPass(cfg.splitRatio, splitReport, &error, cavityRatio);
        report.splits += int(splitReport.acceptedCount);
        report.splitCandidates += int(splitReport.candidateCount);
        report.secondsSplit += splitReport.totalMs * 0.001;
        if (ok) ok = backend.Validate(splitValidation, &error);
      }
      if (ok && collapse) {
        ok = backend.RunCollapsePass(cfg.collapseRatio, collapseReport, &error);
        report.collapses += int(collapseReport.acceptedCount);
        report.collapseCandidates += int(collapseReport.candidateCount);
        report.secondsCollapse += collapseReport.totalMs * 0.001;
        if (ok) ok = backend.Validate(collapseValidation, &error);
      }
      std::cout << "global_refine_cycle=" << cycle
                << " cavity_candidates=" << cavityReport.candidateCount
                << " cavity_accepted=" << cavityReport.acceptedCount
                << " cavity_projection_applied=" << cavityReport.projectionApplied
                << " cavity_projection_failed=" << cavityReport.projectionFailed
                << " split_candidates=" << splitReport.candidateCount
                << " split_accepted=" << splitReport.acceptedCount
                << " split_projection_applied=" << splitReport.projectionApplied
                << " split_projection_failed=" << splitReport.projectionFailed
                << " collapse_candidates=" << collapseReport.candidateCount
                << " collapse_accepted=" << collapseReport.acceptedCount
                << " collapse_topology_rejected=" << collapseReport.topologyRejected
                << " collapse_semantic_rejected=" << collapseReport.semanticRejected
                << " collapse_quality_rejected=" << collapseReport.qualityRejected
                << " collapse_adjacency_ms=" << collapseReport.adjacencyMs
                << " collapse_candidate_ms=" << collapseReport.candidateMs
                << " collapse_claim_ms=" << collapseReport.claimMs
                << " collapse_execute_ms=" << collapseReport.executeMs
                << " collapse_scheduler_rounds=" << collapseReport.schedulerRounds
                << " collapse_active_scanned=" << collapseReport.activeItemsScanned
                << " cavity_topology_ok=" << cavityValidation.ok()
                << " split_topology_ok=" << splitValidation.ok()
                << " collapse_topology_ok=" << collapseValidation.ok() << '\n';
      if (ok) {
        SemanticMesh cycleMesh;
        std::string auditError;
        if (backend.Export(cycleMesh, &auditError)) {
          cycleMesh.rebuildTopology();
          const EdgeLengthAudit passAudit = cycleMesh.edgeLengthAudit(
              cfg.constantLength, cfg.splitRatio, cfg.collapseRatio);
          uint32_t coarseAbove = 0;
          uint32_t residualAbove = 0;
          const float coarseThreshold = cavityRatio * cfg.constantLength;
          for (const auto &e : cycleMesh.edges) {
            if (e.flags & EdgeProtected) continue;
            const auto a = cycleMesh.position(int(e.v0));
            const auto b = cycleMesh.position(int(e.v1));
            const float dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
            const float len=std::sqrt(dx*dx+dy*dy+dz*dz);
            if (len > coarseThreshold) ++coarseAbove;
            else if (len > passAudit.splitThreshold) ++residualAbove;
          }
          const AnalyticGeometryAudit geomAudit = cycleMesh.analyticGeometryAudit();
          std::cout << "global_refine_sizing_cycle=" << cycle
                    << " median=" << passAudit.median
                    << " p95=" << passAudit.p95
                    << " p99=" << passAudit.p99
                    << " max=" << passAudit.max
                    << " coarse_above_cavity=" << coarseAbove
                    << " residual_above_split=" << residualAbove
                    << " editable_above_split=" << passAudit.editableAboveSplit
                    << " editable_below_collapse=" << passAudit.editableBelowCollapse
                    << " edge_count=" << passAudit.edgeCount
                    << " analytic_geom_mean=" << geomAudit.mean
                    << " analytic_geom_p95=" << geomAudit.p95
                    << " analytic_geom_max=" << geomAudit.max
                    << " analytic_supported_vertices=" << geomAudit.supportedVertexCount
                    << " analytic_unsupported_vertices=" << geomAudit.unsupportedVertexCount
                    << '\n';
        }
      }
      if (!ok) {
        std::cout << "global_refine_error=" << error << '\n';
        break;
      }
      if (cavityReport.acceptedCount == 0 && splitReport.acceptedCount == 0 &&
          collapseReport.acceptedCount == 0) break;
    }
    if (cfg.enableFlip) {
      for (int pass = 0; ok && pass < cfg.maxIterations; ++pass) {
        global::GlobalFlipReport flipReport;
        ok = backend.RunFlipPass(1e-4f, flipReport, &error);
        report.flips += int(flipReport.acceptedCount);
        report.flipCandidates += int(flipReport.candidateCount);
        report.secondsFlip += flipReport.totalMs * 0.001;
        if (flipReport.acceptedCount == 0) break;
      }
    }
    global::GlobalTopologyValidation validation;
    if (ok) ok = backend.Validate(validation, &error);
    if (ok) ok = backend.Export(mesh, &error);
    report.topologyValid = ok;
    report.constraintsHeld = ok;
    report.seconds = report.secondsCavity + report.secondsSplit + report.secondsCollapse + report.secondsFlip;
    backendName = "global-topology-poc";
#else
    std::cerr << "this binary was built without CAD_ADAPTIVE_GLOBAL_TOPOLOGY\n";
    return 1;
#endif
  } else if (useGpu) {
#ifdef CAD_ADAPTIVE_RXMESH
    if (!rxmeshAvailable()) {
      std::cerr << "RXMesh GPU backend unavailable\n";
      return 1;
    }
    RxMeshBackend backend;
    ok = backend.remesh(mesh, cfg, report);
    backendName = "gpu";
#else
    std::cerr << "this binary was built without CAD_ADAPTIVE_RXMESH\n";
    return 1;
#endif
  } else {
    CpuRemeshBackend backend;
    ok = backend.remesh(mesh, cfg, report);
  }
  if (ok && partitionInput && !validatePartitionOutput(reference, mesh, cfg, report, &error)) {
    std::cerr << error << '\n';
    ok = false;
  }
  std::cout << "backend=" << backendName << " verts=" << mesh.vertexCount()
            << " faces=" << mesh.faceCount() << " h=" << cfg.constantLength << '\n';
  if (ok) {
    mesh.rebuildTopology();
    const EdgeLengthAudit audit = mesh.edgeLengthAudit(
        cfg.constantLength, cfg.splitRatio, cfg.collapseRatio);
    std::cout << "edge_length_audit "
              << "target=" << audit.targetLength
              << " split_threshold=" << audit.splitThreshold
              << " collapse_threshold=" << audit.collapseThreshold
              << " min=" << audit.min
              << " p05=" << audit.p05
              << " median=" << audit.median
              << " p95=" << audit.p95
              << " p99=" << audit.p99
              << " max=" << audit.max
              << " editable_max=" << audit.editableMax
              << " protected_max=" << audit.protectedMax
              << " editable_above_split=" << audit.editableAboveSplit
              << " editable_below_collapse=" << audit.editableBelowCollapse
              << " protected_above_split=" << audit.protectedAboveSplit
              << " edge_count=" << audit.edgeCount << '\n';
    const AnalyticGeometryAudit geomAudit = mesh.analyticGeometryAudit();
    std::cout << "analytic_geometry_audit"
              << " mean=" << geomAudit.mean
              << " p95=" << geomAudit.p95
              << " max=" << geomAudit.max
              << " supported_vertices=" << geomAudit.supportedVertexCount
              << " unsupported_vertices=" << geomAudit.unsupportedVertexCount << '\n';
  }
  if (!ok) {
    std::cerr << "remesh failed\n" << remeshReportJson(report);
    return 1;
  }
  if (!mesh.save(outPath, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  const std::string json = remeshReportJson(report);
  std::cout << json;
  std::string jsonPath = outPath;
  const auto dot = jsonPath.find_last_of('.');
  if (dot != std::string::npos) jsonPath.resize(dot);
  jsonPath += ".json";
  std::ofstream js(jsonPath);
  if (js) js << json;
  return 0;
}
