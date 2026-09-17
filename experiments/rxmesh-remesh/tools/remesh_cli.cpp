#include "cad_adaptive/IRemeshBackend.h"
#include "cad_adaptive/RemeshMetrics.h"
#ifdef CAD_ADAPTIVE_RXMESH
#include "cad_adaptive/RxMeshBackend.h"
#endif
#ifdef CAD_ADAPTIVE_GLOBAL_TOPOLOGY
#include "cad_adaptive/global/GlobalSplitBackend.h"
#endif

#include "cad_adaptive/PartitionInput.h"
#include "cad_adaptive/GeometryProjector.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <chrono>

using namespace cad_adaptive;

int main(int argc, char **argv) {
  using Clock = std::chrono::steady_clock;
  const auto mainStart = Clock::now();
  const auto elapsed = [](Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
  };
  double secondsPreprocess=0, secondsPassValidation=0, secondsCycleAudit=0;
  double secondsFinalExport=0, secondsFinalValidation=0, secondsFinalAudit=0, secondsSave=0;

  if (argc < 3) {
    std::cerr << "usage: cad_adaptive_cli INPUT.(obj|stl) OUTPUT.ply [target_length] [--gpu|--cpu|--global]\n"
              << "       cad_adaptive_cli INPUT.cadpart OUTPUT.ply [target_length] --partition [--gpu|--cpu|--global]\n"
              << "       cad_adaptive_cli --grid TRIS OUTPUT.ply [target_length] [--gpu|--cpu|--global]\n";
    return 2;
  }
  const bool grid = std::strcmp(argv[1], "--grid") == 0;
  if(grid && argc<4) {std::cerr<<"--grid requires triangle count and output path\n";return 2;}
  bool useGpu = false;
  bool useGlobal = false;
  bool partitionInput = false;
  float target = 0;
  float maxError = 0;
  bool smooth=true, collapse=true, flip=true, cavity=false, split=true;
  float cavityRatio=2.5f;
  float smoothLambda=0.5f;
  int iters = 20;
  int splitPasses = 4;
  int collapsePasses = 8;
  float targetQualityP05 = 0.20f;
  float maxSizingOutlierFraction = 0.05f;
  float normalDegrees = 10.0f;
  float referenceSampleError = -1.0f;
  const char *termination = "not_assessed";
  int completedCycles = 0;
  float sizingOutlierFraction = 0.0f;
  int flipIters = 8;
  int smoothIters = 4; // passes per complete macro cycle, not a final tail
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
    else if (std::strcmp(argv[i], "--cavity") == 0) cavity=true;
    else if (std::strcmp(argv[i], "--split-passes") == 0 && i+1<argc)
      splitPasses=std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--collapse-passes") == 0 && i+1<argc)
      collapsePasses=std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--quality-p05") == 0 && i+1<argc)
      targetQualityP05=std::strtof(argv[++i],nullptr);
    else if (std::strcmp(argv[i], "--max-sizing-outliers") == 0 && i+1<argc)
      maxSizingOutlierFraction=std::strtof(argv[++i],nullptr);
    else if (std::strcmp(argv[i], "--normal-degrees") == 0 && i+1<argc)
      normalDegrees=std::strtof(argv[++i],nullptr);
    else if (std::strcmp(argv[i], "--cavity-ratio") == 0 && i+1<argc)
      cavityRatio=std::strtof(argv[++i],nullptr);
    else if (std::strcmp(argv[i], "--smooth-lambda") == 0 && i+1<argc)
      smoothLambda=std::strtof(argv[++i],nullptr);
    else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--flip-iters") == 0 && i + 1 < argc)
      flipIters = std::max(0, std::atoi(argv[++i]));
    else if (std::strcmp(argv[i], "--smooth-iters") == 0 && i + 1 < argc)
      smoothIters = std::max(0, std::atoi(argv[++i]));
    else if (std::strcmp(argv[i], "--max-error") == 0 && i + 1 < argc)
      maxError = std::strtof(argv[++i], nullptr);
    else {
      char *end=nullptr;
      const float value=std::strtof(argv[i],&end);
      if(end==argv[i] || *end!='\0' || !std::isfinite(value) || value<0) {
        std::cerr << "invalid option or target length: " << argv[i] << '\n'; return 2;
      }
      target=value;
    }
  }
  if(iters<1 || splitPasses<1 || collapsePasses<1 ||
     !std::isfinite(target) || target<0 || !std::isfinite(maxError) || maxError<0 ||
     !std::isfinite(cavityRatio) || cavityRatio<=4.0f/3.0f ||
     !std::isfinite(smoothLambda) || smoothLambda<0 || smoothLambda>1 ||
     !std::isfinite(normalDegrees) || normalDegrees<=0 || normalDegrees>=90 ||
     !std::isfinite(targetQualityP05) || targetQualityP05<=0 || targetQualityP05>1 ||
     !std::isfinite(maxSizingOutlierFraction) || maxSizingOutlierFraction<0 || maxSizingOutlierFraction>1) {
    std::cerr << "invalid remesh iteration, quality or constraint parameters\n"; return 2;
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
  cfg.normalDegrees=normalDegrees;
  if (partitionInput) {
    const int boundarySplits=refinePartitionBoundary(mesh,cfg.splitRatio*cfg.constantLength);
    std::cout << "boundary_splits=" << boundarySplits << '\n';
  }
  if(useGlobal && partitionInput && maxError==0.0f) {
    // Derive a conservative budget ONCE from the immutable handoff, never from
    // a later remeshed state. Explicit --max-error always wins.
    GeometryProjector projector;
    projector.build(mesh);
    referenceSampleError=0.0f;
    for(int f=0;f<mesh.faceCount();++f) {
      if(!mesh.faceAlive[f]) continue;
      const Vec3 a=mesh.facePoint(f,0),b=mesh.facePoint(f,1),c=mesh.facePoint(f,2);
      const Vec3 samples[7]={a,b,c,centroid3(a,b,c),(a+b)*0.5f,(b+c)*0.5f,(c+a)*0.5f};
      for(Vec3 sample:samples) {
        const auto hit=projector.projectSurface(mesh.facePatchId[f],sample);
        if(!hit.ok) {std::cerr<<"reference projection failed\n";return 1;}
        referenceSampleError=std::max(referenceSampleError,distance(sample,hit.position));
      }
    }
    const float numericalFloor=modelScale*1.0e-6f;
    cfg.maxGeometryError=std::max(numericalFloor,
        std::min(cfg.maxGeometryError,2.0f*referenceSampleError));
    std::cout << "geometry_budget_source=reference_samples reference_error=" << referenceSampleError
              << " max_geometry_error=" << cfg.maxGeometryError << '\n';
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
  secondsPreprocess=elapsed(mainStart);
  RemeshReport report;
  SemanticMesh reference;
  if (partitionInput) reference = mesh;
  bool ok = false;
  const char *backendName = "cpu";
  if (useGlobal) {
#ifdef CAD_ADAPTIVE_GLOBAL_TOPOLOGY
    global::GlobalSplitBackend backend;
    const auto setupStart=Clock::now();
    ok = backend.Initialize(mesh, 4.0f, &error);
    if (ok) ok = backend.configure(cfg, &error);
    report.secondsSetup=elapsed(setupStart);
    termination = "iteration_limit";
    int qualifiedCycles=0;
    std::cout << "global_policy=complete_cycle cavity=" << cavity
              << " split_passes=" << splitPasses << " collapse_passes=" << collapsePasses
              << " flip_passes=" << flipIters << " smooth_passes=" << smoothIters
              << " max_geometry_error=" << cfg.maxGeometryError
              << " normal_degrees=" << cfg.normalDegrees
              << " target_quality_p05=" << targetQualityP05
              << " max_sizing_outliers=" << maxSizingOutlierFraction << '\n';
    auto validatePass = [&]() {
      global::GlobalTopologyValidation validation;
      const auto validationStart=Clock::now();
      if (ok) ok=backend.Validate(validation,&error);
      secondsPassValidation+=elapsed(validationStart);
    };
    for (int cycle=0; ok && cycle<cfg.maxIterations; ++cycle) {
      uint64_t mutations=0;
      uint64_t splitAccepted=0,collapseAccepted=0,flipAccepted=0,smoothAccepted=0;
      if (cavity) {
        global::GlobalTriangleRefineReport r;
        ok=backend.RunTriangleRefinePass(cavityRatio,r,&error);
        report.cavityRefines+=int(r.acceptedCount); report.cavityCandidates+=int(r.candidateCount);
        report.secondsCavity+=r.totalMs*0.001; mutations+=r.acceptedCount;
        validatePass();
      }
      // GPU passes apply independent local sets. Regenerate candidates between
      // bounded sweeps so newly created topology participates in this cycle.
      for (int pass=0; ok && split && pass<splitPasses; ++pass) {
        global::GlobalSplitReport r;
        // No upper length cutoff: all long editable edges have a fallback.
        ok=backend.RunSplitPass(cfg.splitRatio,r,&error);
        report.splits+=int(r.acceptedCount); report.splitCandidates+=int(r.candidateCount);
        report.secondsSplit+=r.totalMs*0.001; splitAccepted+=r.acceptedCount;
        validatePass();
        if(r.acceptedCount==0) break;
      }
      for (int pass=0; ok && collapse && pass<collapsePasses; ++pass) {
        global::GlobalCollapseReport r;
        ok=backend.RunCollapsePass(cfg.collapseRatio,r,&error);
        report.collapses+=int(r.acceptedCount); report.collapseCandidates+=int(r.candidateCount);
        report.secondsCollapse+=r.totalMs*0.001; collapseAccepted+=r.acceptedCount;
        report.rejectTopology+=int(r.topologyRejected);
        report.rejectPatch+=int(r.semanticRejected); report.rejectQuality+=int(r.qualityRejected);
        validatePass();
        if(r.acceptedCount==0) break;
      }
      for (int pass=0; ok && cfg.enableFlip && pass<flipIters; ++pass) {
        global::GlobalFlipReport r;
        ok=backend.RunFlipPass(1.0e-4f,cfg.collapseRatio,r,&error);
        report.flips+=int(r.acceptedCount); report.flipCandidates+=int(r.candidateCount);
        report.secondsFlip+=r.totalMs*0.001; flipAccepted+=r.acceptedCount;
        report.rejectPatch+=int(r.semanticRejected);
        validatePass();
        if(r.acceptedCount==0) break;
      }
      for (int pass=0; ok && cfg.enableSmooth && pass<smoothIters; ++pass) {
        global::GlobalSmoothReport r;
        ok=backend.RunSmoothPass(cfg.smoothLambda,r,&error);
        report.smoothMoves+=int(r.acceptedCount); report.secondsSmooth+=r.totalMs*0.001;
        smoothAccepted+=r.acceptedCount; report.rejectPatch+=int(r.semanticRejected);
        report.rejectQuality+=int(r.qualityRejected);
        validatePass();
        if(r.acceptedCount==0) break;
      }
      mutations+=splitAccepted+collapseAccepted+flipAccepted+smoothAccepted;
      if(!ok) break;
      const auto auditStart=Clock::now();
      global::GlobalCycleMetrics audit;
      if (!(ok=backend.collectCycleMetrics(cfg.constantLength,cfg.splitRatio,
                                           cfg.collapseRatio,audit,&error))) break;
      report.qualityMean=audit.QualityMean;
      report.qualityP05=audit.QualityP05;
      report.qualityMin=audit.QualityMin;
      sizingOutlierFraction=audit.EditableCount==0 ? 0.0f :
          float(audit.EditableAboveSplit+audit.EditableBelowCollapse)/float(audit.EditableCount);
      secondsCycleAudit+=elapsed(auditStart);
      completedCycles=cycle+1;
      const bool meetsTarget=report.qualityP05>=targetQualityP05 &&
          report.qualityMin>=cfg.minQuality && sizingOutlierFraction<=maxSizingOutlierFraction;
      qualifiedCycles=meetsTarget ? qualifiedCycles+1 : 0;
      std::cout << "global_remesh_cycle=" << cycle
                << " faces=" << audit.FaceCount << " splits=" << splitAccepted
                << " collapses=" << collapseAccepted << " flips=" << flipAccepted
                << " smooth_moves=" << smoothAccepted << " quality_mean=" << report.qualityMean
                << " quality_p05=" << report.qualityP05 << " quality_min=" << report.qualityMin
                << " editable_above_split=" << audit.EditableAboveSplit
                << " editable_below_collapse=" << audit.EditableBelowCollapse
                << " sizing_outliers=" << sizingOutlierFraction << " target_met=" << meetsTarget << '\n';
      if(qualifiedCycles>=2 || (meetsTarget && mutations==0)) {termination="converged";break;}
      if(mutations==0) {termination="stalled";break;}
    }
    if(ok) validatePass();
    const auto finalExportStart=Clock::now();
    if(ok) ok=backend.Export(mesh,&error);
    secondsFinalExport=elapsed(finalExportStart);
    if(!ok) termination="constraint_failure";
    report.topologyValid=ok;
    report.constraintsHeld=ok;
    report.seconds=report.secondsCavity+report.secondsSplit+report.secondsCollapse+report.secondsFlip+report.secondsSmooth;
    backendName="global-constrained-isotropic";
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
  const auto finalValidationStart=Clock::now();
  if (ok && partitionInput && !validatePartitionOutput(reference, mesh, cfg, report, &error)) {
    std::cerr << error << '\n';
    ok = false;
  }
  secondsFinalValidation=elapsed(finalValidationStart);
  const auto finalAuditStart=Clock::now();
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
  // Populate final quality/sizing metrics for every backend. Preserve the stricter
  // CAD-contract geometry bound, which samples more than face centroids.
  const float validatedGeometryErrorMax = report.geometryErrorMax;
  fillMeshMetrics(mesh, cfg, report);
  report.geometryErrorMax = std::max(report.geometryErrorMax, validatedGeometryErrorMax);
  secondsFinalAudit=elapsed(finalAuditStart);
  const auto saveStart=Clock::now();
  if (!mesh.save(outPath, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  secondsSave=elapsed(saveStart);
  std::string json = remeshReportJson(report);
  if(useGlobal) {
    std::ostringstream extra;
    extra << ",\n  \"termination\": \"" << termination << "\",\n"
          << "  \"converged\": " << (std::strcmp(termination,"converged")==0 ? "true" : "false") << ",\n"
          << "  \"completed_cycles\": " << completedCycles << ",\n"
          << "  \"sizing_outlier_fraction\": " << sizingOutlierFraction << ",\n"
          << "  \"target_quality_p05\": " << targetQualityP05 << ",\n"
          << "  \"max_sizing_outlier_fraction\": " << maxSizingOutlierFraction << '\n';
    extra << ",  \"geometry_error_budget\": " << cfg.maxGeometryError << ",\n"
          << "  \"normal_degrees\": " << cfg.normalDegrees << ",\n"
          << "  \"rejection_counters_complete\": false\n";
    extra << ",\n";
    extra << "  \"seconds_preprocess\": " << secondsPreprocess << ",\n";
    extra << "  \"seconds_pass_validation\": " << secondsPassValidation << ",\n";
    extra << "  \"seconds_cycle_audit\": " << secondsCycleAudit << ",\n";
    extra << "  \"seconds_final_export\": " << secondsFinalExport << ",\n";
    extra << "  \"seconds_final_validation\": " << secondsFinalValidation << ",\n";
    extra << "  \"seconds_final_audit\": " << secondsFinalAudit << ",\n";
    extra << "  \"seconds_save\": " << secondsSave << ",\n";
    extra << "  \"seconds_operator_total\": " << report.seconds << ",\n";
    extra << "  \"seconds_total_before_report\": " << elapsed(mainStart) << "\n";
    json.insert(json.rfind('}'),extra.str());
  }
  std::cout << json;
  std::string jsonPath = outPath;
  const auto dot = jsonPath.find_last_of('.');
  if (dot != std::string::npos) jsonPath.resize(dot);
  jsonPath += ".json";
  std::ofstream js(jsonPath);
  if (js) js << json;
  if(useGlobal && std::strcmp(termination,"converged")!=0) {
    std::cerr << "candidate mesh saved, but targets not reached: " << termination << '\n';
    return 3;
  }
  return 0;
}
