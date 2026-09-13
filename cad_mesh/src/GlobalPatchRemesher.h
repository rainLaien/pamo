// Flat registry construction followed by global remesh scheduling.
void ReconcileAnalyticBoundaries(std::vector<Point3> &vertices,
    std::vector<std::array<int,3>> &faces,std::vector<int> &labels,
    std::vector<std::array<int,3>> &referenceFaces,std::vector<int> &referenceLabels,
    std::unordered_set<EdgeKey> &constraints,const std::vector<AnalyticBoundarySample> &samples){
  std::unordered_map<EdgeKey,std::vector<AnalyticBoundarySample>> groups;
  for(auto sample:samples){if(sample.A>sample.B){std::swap(sample.A,sample.B);sample.Parameter=1-sample.Parameter;}
    groups[Key(sample.A,sample.B)].push_back(sample);}
  std::unordered_map<int,int> aliases;
  std::unordered_map<EdgeKey,std::vector<int>> chains;
  for(auto &entry:groups){auto &list=entry.second;const int a=list[0].A,b=list[0].B;
    std::sort(list.begin(),list.end(),[](const auto &x,const auto &y){
      return x.Parameter!=y.Parameter?x.Parameter<y.Parameter:x.Vertex<y.Vertex;});
    std::vector<int> unionIds{a};std::vector<double> parameters{0};
    std::unordered_map<int,std::vector<int>> owners;
    for(const auto &s:list){
      if(s.Parameter-parameters.back()>1e-10){unionIds.push_back(s.Vertex);parameters.push_back(s.Parameter);}
      aliases[s.Vertex]=unionIds.back();owners[s.Patch].push_back(int(unionIds.size())-1);
    }
    unionIds.push_back(b);
    const auto addInterval=[&](int low,int high){
      std::vector<int> chain(unionIds.begin()+low,unionIds.begin()+high+1);
      if(chain.front()>chain.back())std::reverse(chain.begin(),chain.end());
      chains[Key(chain.front(),chain.back())]=std::move(chain);
    };
    addInterval(0,int(unionIds.size())-1);
    for(auto &owner:owners){auto &path=owner.second;path.push_back(0);path.push_back(int(unionIds.size())-1);
      std::sort(path.begin(),path.end());path.erase(std::unique(path.begin(),path.end()),path.end());
      for(std::size_t j=1;j<path.size();++j)addInterval(path[j-1],path[j]);}
    constraints.erase(entry.first);
    for(std::size_t j=1;j<unionIds.size();++j)constraints.insert(Key(unionIds[j-1],unionIds[j]));
  }
  const auto conform=[&](std::vector<std::array<int,3>> &input,std::vector<int> &ids){
    std::vector<std::array<int,3>> output;std::vector<int> outputIds;
    output.reserve(input.size());outputIds.reserve(ids.size());
    for(std::size_t i=0;i<input.size();++i){auto f=input[i];
      for(int &v:f){auto found=aliases.find(v);if(found!=aliases.end())v=found->second;}
      std::vector<int> perimeter;
      for(int k=0;k<3;++k){const int a=f[k],b=f[(k+1)%3];perimeter.push_back(a);
        auto found=chains.find(Key(a,b));if(found==chains.end())continue;const auto &chain=found->second;
        if(chain.front()==a)for(std::size_t j=1;j+1<chain.size();++j)perimeter.push_back(chain[j]);
        else for(int j=int(chain.size())-2;j>0;--j)perimeter.push_back(chain[j]);
      }
      if(perimeter.size()==3){output.push_back(f);outputIds.push_back(ids[i]);continue;}
      const int center=int(vertices.size());vertices.push_back(ToPoint(Mul(Add(Add(ToVec(vertices[f[0]]),ToVec(vertices[f[1]])),ToVec(vertices[f[2]])),1.0/3)));
      for(std::size_t j=0;j<perimeter.size();++j){output.push_back({perimeter[j],perimeter[(j+1)%perimeter.size()],center});outputIds.push_back(ids[i]);}
    }
    input.swap(output);ids.swap(outputIds);
  };
  conform(faces,labels);conform(referenceFaces,referenceLabels);
}

bool RemeshGlobalPatches(NativeRemeshResult &result,const CadMeshPatchSegmenter &segmenter,
    const NativeRemeshConfig &config,std::unordered_set<EdgeKey> &constraints,
    bool planesOnly,bool cylindersOnly,bool conesOnly,bool othersOnly,std::string &error){
  using Clock=std::chrono::steady_clock;
  const auto started=Clock::now();
  const auto &sourcePatches=segmenter.getPatches();
  auto baseFaces=result.Triangles;
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
    const bool selected=(config.SelectedSourcePatches.empty() || config.SelectedSourcePatches[source]) &&
        (!planesOnly || type==PatchSurfaceType::Plane) &&
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
  auto referenceLabels=result.PatchIds;
  auto chartModels=result.OutputPatches;
  for(std::size_t id=0;id<chartModels.size();++id)
    if(!result.RequestedPatches[id] || preserved[id])chartModels[id].ProjectionTarget=PatchProjectionTarget::ReferenceMesh;
  std::vector<unsigned char> excluded(chartModels.size(),0),rebuilt;
  AnalyticPatchRemeshReport report;
  // Curved ruled domains first; adjacent analytic domains then consume the
  // same original boundary IDs. Physical seam subdivision is deferred until
  // both phases and generic remeshing finish.
  for(int phase=0;phase<2;++phase){
    auto models=chartModels;
    for(auto &model:models){const bool ruled=model.SurfaceType==PatchSurfaceType::Cylinder || model.SurfaceType==PatchSurfaceType::Cone;
      if(ruled!=(phase==0))model.ProjectionTarget=PatchProjectionTarget::ReferenceMesh;
    }
    AnalyticPatchRemeshReport current;std::vector<unsigned char> currentRebuilt;
    RebuildAnalyticPatches(result.Vertices,result.Triangles,result.PatchIds,models,config.TargetEdgeLength,
        config.MaximumDeviation,config.MaximumNormalDeviationDegrees,config.TargetMeanTriangleQuality,
        excluded,currentRebuilt,current,false,false,false,!config.DisableCuda);
    if(phase==0){
      if(!current.BoundarySamples.empty()){
        ReconcileAnalyticBoundaries(result.Vertices,result.Triangles,result.PatchIds,baseFaces,referenceLabels,constraints,current.BoundarySamples);
        for(auto &list:originals)list.clear();
        for(std::size_t f=0;f<baseFaces.size();++f)originals[referenceLabels[f]].push_back(baseFaces[f]);
      }
      if(config.Verbose)std::clog << "[CadMesh] ruled-first boundary handoff: proposals=" << current.BoundarySamples.size()
          << "; shared vertex sequences reconciled before adjacent charts" << std::endl;
      report=std::move(current);rebuilt=std::move(currentRebuilt);
    }
    else{report.ChartSeconds+=current.ChartSeconds;
      for(std::size_t id=0;id<models.size();++id){const auto type=chartModels[id].SurfaceType;
        if(type!=PatchSurfaceType::Cylinder && type!=PatchSurfaceType::Cone){
          report.PatchReasons[id]=current.PatchReasons[id];report.FailureDetails[id]=current.FailureDetails[id];rebuilt[id]=currentRebuilt[id];}
      }
    }
  }
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
  // Acceptance is relative to the immutable input subpatch, not just the
  // fitted analytic model. Reject only the affected output child region.
  std::vector<std::vector<std::array<int,3>>> candidates(chartModels.size());
  for(std::size_t f=0;f<result.Triangles.size();++f)candidates[result.PatchIds[f]].push_back(result.Triangles[f]);
  for(int id:jobs)if(rebuilt[id])candidates[id]=std::move(replacements[id]);
  size_t shapeRejected=0;
  for(std::size_t id=0;id<candidates.size();++id)if(rebuilt[id]){
    // The guard never accepts another patch as reference. Index only this
    // source region rather than traversing unrelated large planar boxes.
    const std::vector<int> localLabels(originals[id].size(),int(id));
    const SurfaceIndex reference(sourceVertices,originals[id],localLabels);
    ReferenceDeviationGuard deviation{reference,config.MaximumDeviation};
    bool valid=!candidates[id].empty();deviation.Hint=-1;
    const char *failure="empty_mesh";double measured=0;
    for(const auto &t:candidates[id]){
      const std::array<Point3,3> p{result.Vertices[t[0]],result.Vertices[t[1]],result.Vertices[t[2]]};
      if(!deviation.accepts(p,int(id))){failure="reference_distance";measured=deviation.LastDistance;valid=false;break;}
      SpatialTriangle hit;double distance=0;
      const Point3 center=ToPoint(Mul(Add(Add(ToVec(p[0]),ToVec(p[1])),ToVec(p[2])),1.0/3));
      const Vec3 normal=Normalize(Cross(Sub(ToVec(p[1]),ToVec(p[0])),Sub(ToVec(p[2]),ToVec(p[0]))));
      if(!reference.closestTriangle(center,int(id),hit,distance)){failure="reference_missing";valid=false;break;}
      const Vec3 sourceNormal=Normalize(Cross(Sub(ToVec(hit.Points[1]),ToVec(hit.Points[0])),Sub(ToVec(hit.Points[2]),ToVec(hit.Points[0]))));
      if(Dot(normal,sourceNormal)<std::cos(config.MaximumNormalDeviationDegrees*std::acos(-1.0)/180)){
        failure="reference_normal";measured=std::acos(std::clamp(Dot(normal,sourceNormal),-1.0,1.0))*180/std::acos(-1.0);valid=false;break;}
    }
    if(!valid){++shapeRejected;rebuilt[id]=0;candidates[id]=originals[id];
      result.PatchReasons[id]=PatchRemeshReason::Deviation;
      if(config.Verbose)std::clog << "[CadMesh] remesh shape rejected: patch_id=" << id
          << ", source_patch_id=" << result.SourcePatchIds[id] << ", failure=" << failure << ", measured=" << measured << "; retained_original_child" << std::endl;
    }
  }
  if(config.Verbose)std::clog << "[CadMesh] reference shape guard: rejected_children=" << shapeRejected
      << ", distance_limit=" << config.MaximumDeviation << ", normal_limit_deg=" << config.MaximumNormalDeviationDegrees << std::endl;
  std::vector<std::array<int,3>> assembled;std::vector<int> labels;
  for(std::size_t id=0;id<candidates.size();++id)for(auto face:candidates[id]){assembled.push_back(face);labels.push_back(int(id));}
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
