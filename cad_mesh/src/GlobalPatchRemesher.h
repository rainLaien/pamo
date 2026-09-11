// Flat registry construction followed by global remesh scheduling.
bool RemeshGlobalPatches(NativeRemeshResult &result,const CadMeshPatchSegmenter &segmenter,
    const NativeRemeshConfig &config,std::unordered_set<EdgeKey> &constraints,
    bool planesOnly,bool cylindersOnly,bool conesOnly,bool othersOnly,std::string &error){
  using Clock=std::chrono::steady_clock;
  const auto started=Clock::now();
  const auto &sourcePatches=segmenter.getPatches();
  const auto baseFaces=result.Triangles;
  std::vector<std::vector<int>> parents(sourcePatches.size());
  for(std::size_t f=0;f<baseFaces.size();++f){const int id=result.PatchIds[f];
    if(id<0 || std::size_t(id)>=parents.size()){error="invalid source patch ownership";return false;}
    parents[id].push_back(int(f));
  }
  std::vector<unsigned char> preserved;
  std::vector<double> inputAreas;
  const auto registerPatch=[&](MeshPatch model,int source,bool selected,bool keep,const std::vector<int>&ids){
    if(ids.empty())return;
    const int id=int(result.OutputPatches.size());model.Id=id;model.TriangleIds=ids;
    model.NeighborPatchIds.clear();model.SupportPatchIds.clear();model.BoundaryEdgeIds.clear();model.BoundaryChainRefs.clear();
    result.OutputPatches.push_back(std::move(model));result.SourcePatchIds.push_back(source);
    result.RequestedPatches.push_back(selected);preserved.push_back(keep);double area=0;
    for(int f:ids){result.PatchIds[f]=id;const auto &t=baseFaces[f];
      area+=.5*Norm(Cross(Sub(ToVec(result.Vertices[t[1]]),ToVec(result.Vertices[t[0]])),
                        Sub(ToVec(result.Vertices[t[2]]),ToVec(result.Vertices[t[0]]))));}
    inputAreas.push_back(area);
  };
  for(std::size_t source=0;source<parents.size();++source){
    if(parents[source].empty())continue;
    const auto &model=sourcePatches[source];const auto type=model.SurfaceType;
    const bool selected=(!planesOnly || type==PatchSurfaceType::Plane) &&
        (!cylindersOnly || type==PatchSurfaceType::Cylinder) && (!conesOnly || type==PatchSurfaceType::Cone) &&
        (!othersOnly || (type!=PatchSurfaceType::Plane && type!=PatchSurfaceType::Cylinder));
    if(!selected || type==PatchSurfaceType::Plane || type==PatchSurfaceType::Cylinder){
      registerPatch(model,int(source),selected,false,parents[source]);continue;
    }
    std::vector<std::array<int,3>> input;for(int id:parents[source])input.push_back(baseFaces[id]);
    auto prepared=PrepareRemeshRegions(result.Vertices,input,constraints,model,config,
        &segmenter.getConfig(),&segmenter.getMesh().getResolution());
    std::vector<unsigned char> owned(input.size(),0);
    const auto toGlobal=[&](const std::vector<int>&local){std::vector<int> ids;ids.reserve(local.size());
      for(int id:local){if(id<0 || std::size_t(id)>=input.size() || owned[id])
          throw std::runtime_error("invalid local repartition ownership");
        owned[id]=1;ids.push_back(parents[source][id]);}return ids;};
    for(const auto &island:prepared.Preserved)registerPatch(model,int(source),true,true,toGlobal(island));
    for(std::size_t r=0;r<prepared.Regions.size();++r)
      registerPatch(prepared.Models[r],int(source),true,false,toGlobal(prepared.Regions[r]));
    if(std::any_of(owned.begin(),owned.end(),[](unsigned char v){return !v;})){
      error="local repartition left unowned faces";return false;}
  }
  // Child interfaces use the original global edge endpoints on both sides.
  std::vector<std::set<int>> neighbors(result.OutputPatches.size());
  for(const auto &edge:BuildEdges(baseFaces)){
    if(edge.FaceCount!=2 || edge.NonManifold){constraints.insert(Key(edge.A,edge.B));continue;}
    const int a=result.PatchIds[edge.Faces[0]],b=result.PatchIds[edge.Faces[1]];
    if(a!=b){constraints.insert(Key(edge.A,edge.B));neighbors[a].insert(b);neighbors[b].insert(a);}
  }
  for(std::size_t id=0;id<neighbors.size();++id)
    result.OutputPatches[id].NeighborPatchIds.assign(neighbors[id].begin(),neighbors[id].end());
  const double partitionSeconds=std::chrono::duration<double>(Clock::now()-started).count();
  std::vector<std::vector<std::array<int,3>>> originals(result.OutputPatches.size()),replacements(result.OutputPatches.size());
  for(std::size_t f=0;f<baseFaces.size();++f)originals[result.PatchIds[f]].push_back(baseFaces[f]);
  auto chartModels=result.OutputPatches;
  for(std::size_t id=0;id<chartModels.size();++id)
    if(!result.RequestedPatches[id] || preserved[id])chartModels[id].ProjectionTarget=PatchProjectionTarget::ReferenceMesh;
  std::vector<unsigned char> excluded(chartModels.size(),0),rebuilt;
  AnalyticPatchRemeshReport report;
  RebuildAnalyticPatches(result.Vertices,result.Triangles,result.PatchIds,chartModels,config.TargetEdgeLength,
      config.MaximumDeviation,config.MaximumNormalDeviationDegrees,config.TargetMeanTriangleQuality,
      excluded,rebuilt,report,false,false,false,!config.DisableCuda);
  result.PatchReasons=report.PatchReasons;
  for(std::size_t id=0;id<chartModels.size();++id){
    if(!result.RequestedPatches[id])result.PatchReasons[id]=PatchRemeshReason::Unselected;
    else if(preserved[id])result.PatchReasons[id]=PatchRemeshReason::Preserved;
    else if(config.Verbose && !rebuilt[id] && chartModels[id].SurfaceType!=PatchSurfaceType::Freeform)
      std::clog << "[CadMesh] remesh failed: patch_id=" << id << ", source_patch_id=" << result.SourcePatchIds[id]
          << ", type=" << SurfaceTypeName(chartModels[id].SurfaceType) << ", source_faces=" << originals[id].size()
          << ", reason=" << PatchRemeshReasonName(result.PatchReasons[id])
          << ", " << report.FailureDetails[id] << std::endl;
  }
  // Immutable snapshot: assembly may append to result.Vertices while workers run.
  const auto sourceVertices=result.Vertices;
  std::vector<int> jobs;
  for(std::size_t id=0;id<chartModels.size();++id)
    if(result.RequestedPatches[id] && !preserved[id] && chartModels[id].SurfaceType==PatchSurfaceType::Freeform)jobs.push_back(int(id));
  std::stable_sort(jobs.begin(),jobs.end(),[&](int a,int b){return originals[a].size()>originals[b].size();});
  struct Output {std::vector<Point3> Points;std::vector<std::array<int,3>> Faces;std::vector<int> Aliases;
    PatchRemeshReason Reason=PatchRemeshReason::Unknown;std::string Detail;bool Ok=false;double Seconds=0;};
  const auto workers=std::min(jobs.size(),std::size_t(std::max(1,config.GenericRemeshWorkers)));
  std::vector<std::future<Output>> pending(jobs.size());
  RemeshWorkerPool pool(workers);
  std::size_t submitted=0,inFlight=0;
  const auto fill=[&]{while(submitted<jobs.size() && inFlight<workers){const auto position=submitted++;const int id=jobs[position];++inFlight;
      pending[position]=pool.submit([&,id]{Output out;auto localConfig=config;localConfig.Verbose=false;const auto tick=Clock::now();
        out.Ok=RemeshFreeformRegion(sourceVertices,originals[id],constraints,result.OutputPatches[id],localConfig,
            out.Points,out.Faces,out.Aliases,out.Reason,out.Detail);
        out.Seconds=std::chrono::duration<double>(Clock::now()-tick).count();return out;});}};
  const auto cpuStart=Clock::now();fill();
  for(std::size_t position=0;position<jobs.size();++position){const int id=jobs[position];
    auto out=pending[position].get();--inFlight;fill();result.PatchReasons[id]=out.Reason;
    if(out.Ok){std::vector<int> mapping=out.Aliases;
      for(auto &face:out.Faces)for(int &v:face){const int local=v;
        if(mapping[local]<0){mapping[local]=int(result.Vertices.size());result.Vertices.push_back(out.Points[local]);}
        v=mapping[local];}
      replacements[id]=std::move(out.Faces);rebuilt[id]=1;result.PatchReasons[id]=PatchRemeshReason::Rebuilt;
    }
    if(config.Verbose && !out.Ok)std::clog << "[CadMesh] remesh failed: patch_id=" << id
        << ", source_patch_id=" << result.SourcePatchIds[id] << ", type=Freeform"
        << ", reason=" << PatchRemeshReasonName(out.Reason) << ", detail=" << out.Detail << std::endl;
  }
  std::vector<unsigned char> generic(chartModels.size(),0);for(int id:jobs)generic[id]=1;
  std::vector<std::array<int,3>> assembled;std::vector<int> labels;
  for(std::size_t f=0;f<result.Triangles.size();++f){const int id=result.PatchIds[f];
    if(generic[id] && rebuilt[id])continue;assembled.push_back(result.Triangles[f]);labels.push_back(id);}
  for(int id:jobs)if(rebuilt[id])for(auto face:replacements[id]){assembled.push_back(face);labels.push_back(id);}
  result.Triangles.swap(assembled);result.PatchIds.swap(labels);result.RemeshedPatches=rebuilt;
  result.FaceRemeshed.clear();result.FaceReasons.clear();std::array<std::size_t,4> faceCounts{};
  for(auto &patch:result.OutputPatches)patch.TriangleIds.clear();
  for(std::size_t f=0;f<result.Triangles.size();++f){const int id=result.PatchIds[f];
    const unsigned char state=!result.RequestedPatches[id]?0:preserved[id]?3:rebuilt[id]?2:1;
    result.FaceRemeshed.push_back(state);result.FaceReasons.push_back(result.PatchReasons[id]);++faceCounts[state];
    result.OutputPatches[id].TriangleIds.push_back(int(f));}
  auto &stats=result.Statistics;
  stats.AnalyticPatchesAttempted=stats.AnalyticPatchesRebuilt=stats.AnalyticPatchesFallback=0;
  for(std::size_t id=0;id<chartModels.size();++id)if(result.RequestedPatches[id] && !preserved[id]){
    ++stats.AnalyticPatchesAttempted;if(rebuilt[id])++stats.AnalyticPatchesRebuilt;else ++stats.AnalyticPatchesFallback;}
  if(config.Verbose){
    std::array<std::size_t,256> counts{};std::array<double,256> areas{};double total=0;
    for(std::size_t id=0;id<inputAreas.size();++id){const auto reason=static_cast<unsigned char>(result.PatchReasons[id]);
      ++counts[reason];areas[reason]+=inputAreas[id];total+=inputAreas[id];}
    for(std::size_t reason=2;reason<counts.size();++reason)if(counts[reason] && reason!=9)std::clog << "[CadMesh] remesh reason area: reason="
        << PatchRemeshReasonName(static_cast<PatchRemeshReason>(reason)) << ", code=" << reason << ", patches=" << counts[reason]
        << ", input_area_percent=" << (total>0?100*areas[reason]/total:0) << std::endl;
    std::clog << "[CadMesh] remesh face status: unselected=" << faceCounts[0] << ", failed=" << faceCounts[1]
        << ", rebuilt=" << faceCounts[2] << ", preserved_good=" << faceCounts[3] << std::endl;
    std::clog << "[CadMesh] global patch remesh: partition_s=" << partitionSeconds << ", analytic_s=" << report.ChartSeconds
        << ", freeform_and_assembly_s=" << std::chrono::duration<double>(Clock::now()-cpuStart).count()
        << ", accepted=" << stats.AnalyticPatchesRebuilt << '/' << stats.AnalyticPatchesAttempted
        << ", fallback=" << stats.AnalyticPatchesFallback << std::endl;
  }
  return true;
}
