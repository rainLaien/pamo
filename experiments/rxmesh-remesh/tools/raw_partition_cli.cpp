#include "cad_adaptive/RawCudaBatch.h"
#include "cad_adaptive/PartitionInput.h"
#include "cad_adaptive/BoundarySizingField.h"
#include <map>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace cad_adaptive;
namespace {
std::pair<double,double> qualitySummary(const SemanticMesh &mesh) {
  std::vector<float> quality;quality.reserve(mesh.faceCount());
  double sum=0;
  for(int f=0;f<mesh.faceCount();++f)if(mesh.faceAlive[f]) {
    const auto t=mesh.face(f);
    const float q=triangleQuality(mesh.position(t[0]),mesh.position(t[1]),mesh.position(t[2]));
    quality.push_back(q);sum+=q;
  }
  if(quality.empty())return {0,0};
  const size_t index=quality.size()/20;
  std::nth_element(quality.begin(),quality.begin()+index,quality.end());
  return {sum/double(quality.size()),quality[index]};
}
}
int main(int argc,char **argv) {
  if(argc<3){std::cerr<<"Usage: cad_raw_partition_cli INPUT.(cadpart|stl|obj|ply) OUTPUT.ply [--target h] [--iters n] [--workers n] [--gpu-concurrency n] [--patches-per-task n] [--smooth-passes n] [--collapse-passes n] [--flip-passes n] [--auto-partition] [--legacy-flip] [--no-idle-stop] [--memory-mb n] [--max-error e] [--feature-refine] [--feature-length-ratio r] [--feature-transition-width-ratio r]\n";return 2;}
  try {
    RawBatchOptions options;options.requireQualityImprovement=true;
    RemeshConfig cfg;cfg.adaptive=false;cfg.maxIterations=20;
    bool refine=false,auditFields=false,curvedSurfaceSizing=false;
    float errorBudget=0,featureEdgeRatio=.5f,featureTransitionWidthRatio=4.f;
    std::map<uint32_t,float> patchTargets;
    float sizingGradation=.5f;
    for(int i=3;i<argc;++i) {
      std::string flag=argv[i];if(flag=="--feature-refine"){refine=true;continue;}
      if(flag=="--curved-surface-sizing" || flag=="--cylinder-curvature-sizing"){curvedSurfaceSizing=true;continue;}
      if(flag=="--audit-fields"){auditFields=true;continue;}
      if(flag=="--no-idle-stop"){options.stopWhenIdle=false;continue;}
      if(flag=="--legacy-flip"){options.strictFlipQuality=false;continue;}
      if(flag=="--auto-partition"){options.autoPartitionSinglePatch=true;continue;}
      if(flag=="--legacy-coverage-acceptance"){options.legacyCoverageAcceptance=true;continue;}
      if(flag=="--global-quality-acceptance"){options.globalQualityAcceptance=true;continue;}
      if(flag=="--explore-provisional-children"){options.exploreProvisionalChildren=true;continue;}
      if(flag=="--track-region-candidates"){options.trackRegionCandidates=true;continue;}
      if(flag=="--size-feasible-final-refine"){options.sizeFeasibleFinalRefine=true;continue;}
      if(flag=="--select-final-regions"){options.selectFinalRegions=true;continue;}
      if(flag=="--disable-collapse"){cfg.enableCollapse=false;continue;}
      if(flag=="--disable-flip"){cfg.enableFlip=false;continue;}
      if(flag=="--disable-smooth"){cfg.enableSmooth=false;continue;}
      if(flag=="--patch-target") {
        if(i+1>=argc)throw std::runtime_error("missing patch:target");
        const std::string value=argv[++i];const auto colon=value.find(':');
        if(colon==std::string::npos)throw std::runtime_error("expected patch:target");
        size_t used=0;const auto id=std::stoul(value.substr(0,colon),&used);
        if(used!=colon || id>std::numeric_limits<uint32_t>::max())throw std::runtime_error("invalid sizing patch id");
        const auto target=std::stof(value.substr(colon+1),&used);
        if(used!=value.size()-colon-1 || !(target>0) || !std::isfinite(target))throw std::runtime_error("invalid patch target");
        if(!patchTargets.emplace(uint32_t(id),target).second)throw std::runtime_error("duplicate patch target");
        continue;
      }
      if(i+1>=argc)throw std::runtime_error("missing option value");
      std::string value=argv[++i];size_t used=0;double number=std::stod(value,&used);
      if(used!=value.size() || !std::isfinite(number) || number<=0)throw std::runtime_error("invalid positive option value");
      if(flag=="--target")cfg.constantLength=float(number);
      else if(flag=="--feature-length-ratio")featureEdgeRatio=float(number);
      else if(flag=="--feature-transition-width-ratio")featureTransitionWidthRatio=float(number);
      else if(flag=="--sizing-gradation")sizingGradation=float(number);
      else if(flag=="--low-quality-threshold")options.lowQualityThreshold=float(number);
      else if(flag=="--max-error")errorBudget=float(number);
      else if(flag=="--iters" || flag=="--workers" || flag=="--gpu-concurrency" || flag=="--patches-per-task" || flag=="--smooth-passes" || flag=="--collapse-passes" || flag=="--flip-passes") {
        if(number!=std::floor(number) || number>1000)throw std::runtime_error("invalid integer option");
        if(flag=="--iters")cfg.maxIterations=int(number);
        else if(flag=="--workers")options.workers=int(number);
        else if(flag=="--gpu-concurrency")options.gpuConcurrency=int(number);
        else if(flag=="--patches-per-task")options.patchesPerTask=int(number);
        else if(flag=="--smooth-passes")options.smoothPasses=int(number);
        else if(flag=="--collapse-passes")options.collapsePasses=int(number);
        else options.flipPasses=int(number);
      } else if(flag=="--memory-mb") {
        if(number>1024*1024)throw std::runtime_error("memory budget too large");
        options.memoryBytes=size_t(number*1024*1024);
      } else throw std::runtime_error("unknown option: "+flag);
    }
    if(options.legacyCoverageAcceptance && options.globalQualityAcceptance)
      throw std::runtime_error("global quality acceptance cannot be combined with legacy coverage acceptance");
    const auto clockStart=std::chrono::steady_clock::now();
    const auto elapsed=[&] {return std::chrono::duration<double>(std::chrono::steady_clock::now()-clockStart).count();};
    SemanticMesh source,output;std::string error;
    const std::string inputPath=argv[1];
    const bool isCadPart=inputPath.size()>=8 && inputPath.substr(inputPath.size()-8)==".cadpart";
    ConstraintAudit constraintAudit;
    if(auditFields && !isCadPart)throw std::runtime_error("constraint provenance export requires CADPART1 input");
    if(isCadPart) {
      if(!loadPartitionInput(inputPath,source,&error,true,auditFields?&constraintAudit:nullptr))throw std::runtime_error(error);
    } else {
      if(!source.load(inputPath,&error))throw std::runtime_error(error);
      source.rebuildTopology();
    }
    const double loadSeconds=elapsed();
    if(!(cfg.constantLength>0))cfg.constantLength=source.bboxDiagonal()*.01f;
    cfg.maxGeometryError=errorBudget>0?errorBudget:.2f*cfg.constantLength;
    // Feature vertices/edges remain constrained by the partition. Use a
    // moderate near-feature target so narrow STL tessellation creases do not
    // trigger a disproportionate split cascade.
    if(!(featureEdgeRatio>0.f && featureEdgeRatio<1.f))
      throw std::runtime_error("feature length ratio must be in (0,1)");
    if(!(featureTransitionWidthRatio>=1.f && featureTransitionWidthRatio<=10.f))
      throw std::runtime_error("feature transition width ratio must be in [1,10]");
    if(refine){cfg.featureEdgeLength=featureEdgeRatio*cfg.constantLength;
      cfg.featureBand=featureTransitionWidthRatio*cfg.constantLength;}
    if(curvedSurfaceSizing && (refine || !patchTargets.empty()))
      throw std::runtime_error("curved surface sizing requires no feature refine or explicit patch targets");
    if(!patchTargets.empty()) {
      if(refine)throw std::runtime_error("explicit patch targets cannot be combined with curvature refinement");
      std::vector<float> targets(source.patches.size(),cfg.constantLength);
      for(auto [patch,target]:patchTargets) {
        if(patch>=targets.size() || target>cfg.constantLength)
          throw std::runtime_error("patch target must address an existing patch and not exceed global length");
        targets[patch]=target;
      }
      // Existing short tessellation edges are not automatically new size
      // requirements. Only explicit region targets seed this experiment.
      source.LocalSizing=BoundarySizingField::create(source,cfg,sizingGradation,false,targets,false);
      source.LocalSizing->apply(source);
    } else if(curvedSurfaceSizing) {
      source.LocalSizing=BoundarySizingField::create(source,cfg,sizingGradation,true,{},false);
      source.LocalSizing->apply(source);
    }
    RawBatchReport report;
    const bool valid=remeshRawCudaPatches(source,output,cfg,options,report,&error);
    const double remeshEnd=elapsed();
    const auto sourceQuality=qualitySummary(source);
    const auto outputQuality=valid?qualitySummary(output):std::pair<double,double>{0,0};
    const bool qualityImproved=valid && outputQuality.first>=sourceQuality.first &&
        outputQuality.second>=sourceQuality.second;
    std::ofstream out(std::string(argv[2])+".json");
    auto writeArray=[&](const auto &values) {
      out<<'[';
      for(size_t j=0;j<values.size();++j)out<<(j?", ":"")<<values[j];
      out<<']';
    };
    out<<std::setprecision(10);
    out<<"{\n  \"patch_targets\": {";
    bool firstTarget=true;
    for(auto [patch,target]:patchTargets) {
      out<<(firstTarget?"":",")<<std::quoted(std::to_string(patch))<<":"<<target;
      firstTarget=false;
    }
    out<<"},\n  \"backend\": \"gpu-raw-cuda-patches\",\n  \"workers\": "<<options.workers
       <<",\n  \"immutable_local_sizing\": "<<((!patchTargets.empty()||curvedSurfaceSizing)?"true":"false")
       <<",\n  \"curved_surface_sizing\": "<<(curvedSurfaceSizing?"true":"false")
       <<",\n  \"sizing_gradation\": "<<sizingGradation
       <<",\n  \"collapse_enabled\": "<<(cfg.enableCollapse?"true":"false")
       <<",\n  \"flip_enabled\": "<<(cfg.enableFlip?"true":"false")
       <<",\n  \"smooth_enabled\": "<<(cfg.enableSmooth?"true":"false")
       <<",\n  \"gpu_concurrency\": "<<options.gpuConcurrency
       <<",\n  \"patches_per_task\": "<<report.patchesPerTask
       <<",\n  \"smooth_passes\": "<<options.smoothPasses
       <<",\n  \"collapse_passes\": "<<options.collapsePasses
       <<",\n  \"flip_passes\": "<<options.flipPasses
       <<",\n  \"strict_flip_quality\": "<<(options.strictFlipQuality?"true":"false")
       <<",\n  \"feature_refine\": "<<(refine?"true":"false")
       <<",\n  \"feature_length_ratio\": "<<featureEdgeRatio
       <<",\n  \"feature_edge_ratio\": "<<(refine?featureEdgeRatio:0.f)
       <<",\n  \"effective_target_length\": "<<cfg.constantLength
       <<",\n  \"max_geometry_error\": "<<cfg.maxGeometryError
       <<",\n  \"max_iterations\": "<<cfg.maxIterations
       <<",\n  \"source_faces\": "<<source.faceCount()
       <<",\n  \"output_faces\": "<<output.faceCount()
       <<",\n  \"auto_partition_single_patch\": "<<(options.autoPartitionSinglePatch?"true":"false")
       <<",\n  \"stop_when_idle\": "<<(options.stopWhenIdle?"true":"false")
       <<",\n  \"seconds\": "<<report.seconds<<",\n  \"memory_budget\": "<<report.memoryBudget
       <<",\n  \"source_quality_mean\": "<<sourceQuality.first
       <<",\n  \"source_quality_p05\": "<<sourceQuality.second
       <<",\n  \"output_quality_mean\": "<<outputQuality.first
       <<",\n  \"output_quality_p05\": "<<outputQuality.second
       <<",\n  \"global_quality_acceptance\": "<<(options.globalQualityAcceptance?"true":"false")
       <<",\n  \"quality_improved\": "<<(qualityImproved?"true":"false")
       <<",\n  \"quality_accepted\": "<<(report.qualityAccepted?"true":"false")
       <<",\n  \"legacy_coverage_acceptance\": "<<(options.legacyCoverageAcceptance?"true":"false")
       <<",\n  \"explore_provisional_children\": "<<(options.exploreProvisionalChildren?"true":"false")
       <<",\n  \"low_quality_threshold_basis\": "<<std::quoted(options.lowQualityThreshold>0?"explicit diagnostic cutoff":"immutable batch input P05; diagnostic, not universal quality target")
       <<",\n  \"constraint_provenance\": \"persistent feature, open boundary, partition interface (geometric versus computation provenance unavailable); movement constraints unchanged\""
       <<",\n  \"child_candidate_policy\": \"per_patch_then_connected_guard\""
       <<",\n  \"track_region_candidates\": "<<(options.trackRegionCandidates?"true":"false")
       <<",\n  \"size_feasible_final_refine\": "<<(options.sizeFeasibleFinalRefine?"true":"false")
       <<",\n  \"select_final_regions\": "<<(options.selectFinalRegions?"true":"false")
       <<",\n  \"source_region_quality\": ";
    writeRegionQualityJson(out,report.sourceQuality);
    out<<",\n  \"output_region_quality\": ";writeRegionQualityJson(out,report.outputQuality);
    std::vector<std::pair<float,int>> longSamples;
    for(size_t i=0;i<output.edges.size();++i) {
      const auto &edge=output.edges[i];const float target=.5f*(output.targetLength[edge.v0]+output.targetLength[edge.v1]);
      if(!(target>0))continue;
      const float ratio=distance(output.position(edge.v0),output.position(edge.v1))/target;
      if(ratio>cfg.splitRatio*(1.f+1.e-4f))longSamples.emplace_back(ratio,int(i));
    }
    std::stable_sort(longSamples.begin(),longSamples.end(),[](auto a,auto b){return a.first>b.first;});
    out<<",\n  \"local_long_edge_samples\": [";
    for(size_t i=0;i<std::min(size_t(32),longSamples.size());++i) {
      if(i)out<<',';const auto &edge=output.edges[longSamples[i].second];
      out<<"{\"ratio\":"<<longSamples[i].first<<",\"patch_left\":"<<edge.patchLeft
         <<",\"patch_right\":"<<edge.patchRight<<",\"flags\":"<<int(edge.flags)<<",\"ends\":[";
      for(int j=0;j<2;++j) {
        if(j)out<<',';const int v=j?edge.v1:edge.v0;const auto p=output.position(v);
        out<<"{\"position\":["<<p.x<<','<<p.y<<','<<p.z<<"],\"target\":"<<output.targetLength[v]
           <<",\"constraint\":"<<int(output.vertexConstraint[v])<<'}';
      }
      out<<"]}";
    }
    out<<']';
    out<<",\n  \"seconds_final_patch_quality_audit\": "<<report.secondsQualityAudit;
    out<<",\n  \"regressed_patch_ids\": ";writeArray(report.regressedPatchIds);
    out<<",\n  \"unresolved_patch_ids\": ";writeArray(report.unresolvedPatchIds);
    out<<",\n  \"pending_patch_ids\": ";writeArray(report.pendingPatchIds);
    out<<",\n  \"final_region_selected_patch_ids\": ";writeArray(report.finalRegionSelection.selectedPatchIds);
    out<<",\n  \"pending_patch_quality\": [";
    auto writeEndpoint=[&](const RegionQuality&q){out<<"{\"mean\":"<<q.mean<<",\"p05\":"<<q.p05
      <<",\"area_weighted_mean\":"<<q.areaWeightedMean<<",\"low_quality_area\":"<<q.lowQualityArea
      <<",\"largest_low_quality_area\":"<<q.largestLowQualityArea<<"}";};
    for(size_t j=0;j<report.pendingPatchIds.size();++j){const auto id=report.pendingPatchIds[j];
      out<<(j?",":"")<<"{\"patch_id\":"<<id<<",\"failure_mask\":"
        <<qualityEndpointMask(report.sourcePatchQuality[id],report.outputPatchQuality[id])<<",\"source\":";
      writeEndpoint(report.sourcePatchQuality[id]);out<<",\"output\":";
      writeEndpoint(report.outputPatchQuality[id]);out<<'}';}
    out<<"]";
    out
       <<",\n  \"seconds_boundary\": "<<report.secondsBoundary
       <<",\n  \"seconds_tasks\": "<<report.secondsTasks
       <<",\n  \"seconds_assembly\": "<<report.secondsAssembly
       <<",\n  \"peak_reserved\": "<<report.peakReserved<<",\n  \"peak_active\": "<<report.peakActive
       <<",\n  \"initial_tasks\": "<<report.patches.size()
       <<",\n  \"accepted\": "<<report.accepted
       <<",\n  \"unchanged\": "<<report.unchanged<<",\n  \"fallback\": "<<report.fallback
       <<",\n  \"uniform_regions\": "<<report.uniformRegions
       <<",\n  \"quality_split_regions\": "<<report.qualitySplitRegions
       <<",\n  \"mean_recovery_regions\": "<<report.meanRecoveryRegions
       <<",\n  \"retried\": "<<report.retried
       <<",\n  \"seam_repairs\": "<<report.seamRepairs
       <<",\n  \"seam_split_repairs\": "<<report.seamSplitRepairs
       <<",\n  \"boundary_splits\": "<<report.boundarySplits
       <<",\n  \"protected_edge_splits\": "<<report.boundarySplits
       <<",\n  \"boundary_edges_deferred\": "<<report.boundaryEdgesDeferred
       <<",\n  \"circular_plane_remeshed\": "<<report.circularPlaneRemeshed
       <<",\n  \"circular_plane_skipped\": "<<report.circularPlaneSkipped
       <<",\n  \"circular_plane_added_faces\": "<<report.circularPlaneAddedFaces
       <<",\n  \"final_size_refine_splits\": "<<report.finalSizeRefineSplits
       <<",\n  \"final_size_refine_levels\": "<<report.finalSizeRefineLevels
       <<",\n  \"final_size_refine_levels_basis\": "<<std::quoted(options.selectFinalRegions?"maximum generated refinement levels; mixed regional selection":"selected refinement levels")
       <<",\n  \"final_region_groups_compared\": "<<report.finalRegionSelection.groupsCompared
       <<",\n  \"final_region_groups_selected\": "<<report.finalRegionSelection.groupsSelected
       <<",\n  \"final_region_stop_reason\": "<<std::quoted(report.finalRegionSelection.stopReason)
       <<",\n  \"seconds_final_region_comparison\": "<<report.finalRegionSelection.seconds
       <<",\n  \"seconds_final_alternative_refine\": "<<report.secondsFinalAlternativeRefine
       <<",\n  \"final_size_unselected_long_edges\": "<<report.finalSizeUnselectedLongEdges
       <<",\n  \"final_size_rejected_splits\": "<<report.finalSizeRejectedSplits
       <<",\n  \"final_size_excess_before_last_candidate\": "<<report.finalSizeExcessBefore
       <<",\n  \"final_size_excess_after_last_candidate\": "<<report.finalSizeExcessAfter
       <<",\n  \"final_size_refine_stop_reason\": "<<std::quoted(report.finalSizeRefineStopReason)
       <<",\n  \"size_recovery_regions\": "<<report.sizeRecoveryRegions
       <<",\n  \"long_edges_after_refine\": "<<report.longEdgesAfterRefine
       <<",\n  \"max_output_edge_ratio\": "<<report.maxOutputEdgeRatio
       <<",\n  \"topology_valid\": "<<(report.topologyValid?"true":"false")
       <<",\n  \"boundaries_held\": "<<(report.boundariesHeld?"true":"false")<<",\n  \"patches\": [";
    for(size_t i=0;i<report.patches.size();++i){const auto &p=report.patches[i];
      out<<(i?",\n":"\n")<<"    {\"id\": "<<i<<", \"accepted\": "<<(p.accepted?"true":"false")
         <<", \"quality_accepted\": "<<(p.qualityAccepted?"true":"false")
         <<", \"provisional\": "<<(p.provisional?"true":"false")
         <<", \"unchanged\": "<<(p.unchanged?"true":"false")
         <<", \"unchanged_patch_ids\": [";
      for(size_t j=0;j<p.unchangedPatchIds.size();++j)
        out<<(j?", ":"")<<p.unchangedPatchIds[j];
      out<<"]"
         <<", \"uniform_sizing\": "<<(p.uniformSizing?"true":"false")
         <<", \"quality_split\": "<<(p.qualitySplit?"true":"false")
         <<", \"mean_recovery\": "<<(p.meanRecovery?"true":"false")
         <<", \"size_recovery\": "<<(p.sizeRecovery?"true":"false")
         <<", \"quality_split_ratio\": "<<p.qualitySplitRatio
         <<", \"retried\": "<<(p.retried?"true":"false")
         <<", \"input_faces\": "<<p.inputFaces<<", \"output_faces\": "<<p.outputFaces
         <<", \"input_quality_mean\": "<<p.inputQualityMean
         <<", \"input_quality_p05\": "<<p.inputQualityP05
         <<", \"output_quality_mean\": "<<p.report.qualityMean
         <<", \"output_quality_p05\": "<<p.report.qualityP05
         <<", \"sizing_error_mean\": "<<p.report.sizingErrorMean
         <<", \"sizing_error_p95\": "<<p.report.sizingErrorP95
         <<", \"max_edge_length_ratio\": "<<p.report.maxEdgeLengthRatio
         <<", \"overlong_edges\": "<<p.report.overlongEdges
         <<", \"cycles_executed\": "<<p.report.cyclesExecuted
         <<", \"recovered_cycle_failures\": "<<p.report.recoveredCycleFailures
         <<", \"selected_cycle\": "<<p.report.selectedCycle
         <<", \"candidate_geometry_checks\": "<<p.report.candidateGeometryChecks
         <<", \"candidate_geometry_rejected\": "<<p.report.candidateGeometryRejected
         <<", \"seconds_candidate_evaluation\": "<<p.report.secondsCandidateEvaluation
         <<", \"seconds\": "<<p.seconds<<", \"workspace_bytes\": "<<p.workspaceBytes
         <<", \"seconds_initial\": "<<p.secondsInitial
         <<", \"seconds_gentle\": "<<p.secondsGentle
         <<", \"seconds_child_comparison\": "<<p.secondsChildComparison
         <<", \"child_candidate_compared\": "<<(p.childCandidateCompared?"true":"false")
         <<", \"child_candidate_selected\": "<<(p.childCandidateSelected?"true":"false")
         <<", \"child_selected_patches\": "<<p.childSelectedPatches
         <<", \"child_generated_patches\": "<<p.childGeneratedPatches
         <<", \"child_boundary_target_changes\": "<<p.childBoundaryTargetChanges
         <<", \"child_candidate_policy\": \"per_patch_then_connected_guard\""
         <<", \"operation_counts_basis\": "<<std::quoted(p.childCandidateSelected?"incumbent plus selected child attempts":"incumbent attempt report")
         <<", \"child_comparison_reason\": "<<std::quoted(p.childComparisonReason)
         <<", \"seconds_uniform_gentle\": "<<p.secondsUniformGentle
         <<", \"seconds_uniform_strict\": "<<p.secondsUniformStrict
         <<", \"seconds_quality_split\": "<<p.secondsQualitySplit
         <<", \"seconds_mean_recovery\": "<<p.secondsMeanRecovery
         <<", \"seconds_setup\": "<<p.report.secondsSetup
         <<", \"seconds_split\": "<<p.report.secondsSplit
         <<", \"seconds_collapse\": "<<p.report.secondsCollapse
         <<", \"seconds_compact\": "<<p.report.secondsCompact
         <<", \"seconds_validate\": "<<p.report.secondsValidate
         <<", \"seconds_flip\": "<<p.report.secondsFlip
         <<", \"seconds_smooth\": "<<p.report.secondsSmooth
         <<", \"splits\": "<<p.report.splits<<", \"collapses\": "<<p.report.collapses
         <<", \"flips\": "<<p.report.flips<<", \"smooth_moves\": "<<p.report.smoothMoves
         <<", \"geometry_error_max\": "<<p.report.geometryErrorMax
         <<", \"geometry_error_reverse_max\": "<<p.report.geometryErrorReverseMax
         <<", \"error\": "<<std::quoted(p.error)
         <<", \"retry_reason\": "<<std::quoted(p.retryReason);
      out<<", \"collapse_pass_calls\": ";writeArray(p.report.collapsePassCalls);
      if(p.childCandidateEvaluated) {
        out<<", \"child_incumbent_quality\": ";writeRegionQualityJson(out,p.childIncumbentQuality);
        out<<", \"child_candidate_quality\": ";writeRegionQualityJson(out,p.childCandidateQuality);
      }
      out<<", \"collapse_pass_accepted\": ";writeArray(p.report.collapsePassAccepted);
      out<<", \"collapse_pass_seconds\": ";writeArray(p.report.collapsePassSeconds);
      out<<", \"flip_pass_calls\": ";writeArray(p.report.flipPassCalls);
      out<<", \"flip_pass_accepted\": ";writeArray(p.report.flipPassAccepted);
      out<<", \"flip_pass_seconds\": ";writeArray(p.report.flipPassSeconds);
      out<<'}';
    }
    out<<"\n  ]\n}\n";out.close();if(!out)throw std::runtime_error("cannot write batch report");
    if(!valid)throw std::runtime_error(error);
    if(!output.save(argv[2],&error))throw std::runtime_error(error);
    if(auditFields && !writeConstraintAudit(argv[2],constraintAudit,output,cfg,&error))throw std::runtime_error(error);
    std::cout<<"raw_batch_timing load_s="<<loadSeconds<<" batch_s="<<(remeshEnd-loadSeconds)
             <<" save_and_report_s="<<(elapsed()-remeshEnd)<<'\n';
    std::cout<<"raw_batch seconds="<<report.seconds<<" patches_per_task="<<report.patchesPerTask<<" accepted="<<report.accepted<<" unchanged="<<report.unchanged<<" uniform="<<report.uniformRegions<<" quality_split="<<report.qualitySplitRegions<<" fallback="<<report.fallback<<" retried="<<report.retried
             <<" peak_active="<<report.peakActive<<" vertices="<<output.vertexCount()<<" faces="<<output.faceCount()<<'\n';
    if(report.fallback) {
      std::cerr<<report.fallback<<" GPU tasks retained their valid source regions\n";
      // The whole-mesh policy judges the assembled endpoint. Retained source
      // regions remain explicit in the report, but do not reject a mesh that
      // passes the global topology, quality and size guards.
      if(!options.globalQualityAcceptance || !report.qualityAccepted || report.longEdgesAfterRefine>0)
        return 3;
    }
    if(report.boundaryEdgesDeferred>0) {
      std::cerr<<"boundary subdivision safety budget reached; candidate retained with "
               <<report.boundaryEdgesDeferred<<" shared edges left unsplit\n";
      return 3;
    }
    if(!options.legacyCoverageAcceptance && !report.qualityAccepted) {
      std::cerr<<"valid candidate retained, but endpoint quality/connected defect guard failed; output is not an accepted remesh\n";
      return 4;
    }
    if(!qualityImproved)
      std::cerr<<"output mesh quality traded off for remeshing coverage; topology and geometry validation passed\n";
    return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
