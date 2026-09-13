// Implementation fragment: isolated reference-mesh remeshing for one patch.
bool RemeshFreeformRegion(
    const std::vector<Point3> &sourceVertices,
    const std::vector<std::array<int,3>> &sourceFaces,
    const std::unordered_set<EdgeKey> &globalConstraints,
    const MeshPatch &patch,const NativeRemeshConfig &config,
    std::vector<Point3> &points,std::vector<std::array<int,3>> &faces,
    std::vector<int> &globalIds,PatchRemeshReason &reason,std::string &detail,
    const std::vector<Box3> *repairZones=nullptr) {
  std::unordered_map<int,int> localIds;
  for(auto face:sourceFaces){
    for(int &v:face){
      auto inserted=localIds.emplace(v,int(points.size()));
      if(inserted.second){points.push_back(sourceVertices[v]);globalIds.push_back(v);}
      v=inserted.first->second;
    }
    faces.push_back(face);
  }
  std::unordered_set<EdgeKey> constraints,boundary;
  for(const auto &edge:BuildEdges(faces)){
    const auto key=Key(edge.A,edge.B);
    if(edge.FaceCount==1)boundary.insert(key);
    if(edge.NonManifold || edge.FaceCount!=2 || globalConstraints.count(Key(globalIds[edge.A],globalIds[edge.B])))constraints.insert(key);
  }
  // Conservative replay: preserve original triangles inside conflict zones.
  // Lock all their edges/vertices so split/collapse/flip/relax cannot modify
  // those triangles. This is a spatial replay, not arbitrary triangle deletion.
  std::size_t protectedFaces=0;
  if(repairZones)for(const auto &face:faces){
    Box3 box;for(int v:face)box.include(points[v]);
    bool protect=false;for(const auto &zone:*repairZones)if(BoxesOverlap(box,zone)){protect=true;break;}
    if(!protect)continue;++protectedFaces;
    for(int side=0;side<3;++side)constraints.insert(Key(face[side],face[(side+1)%3]));
  }
  if(config.Verbose && repairZones)std::clog << "[CadMesh] freeform local replay: zones="
      << repairZones->size() << ", protected_source_faces=" << protectedFaces
      << ", total_source_faces=" << faces.size() << std::endl;
  RemeshIsotropicPatch(points,faces,globalIds,constraints,patch,config);
  reason=PatchRemeshReason::Rebuilt;return true;
}

struct PreparedRemeshRegions {
  std::vector<std::vector<int>> Preserved,Regions;
  std::vector<MeshPatch> Models;
};
PreparedRemeshRegions PrepareRemeshRegions(
    const std::vector<Point3> &sourceVertices,
    const std::vector<std::array<int,3>> &sourceFaces,
    const std::unordered_set<EdgeKey> &globalConstraints,
    const MeshPatch &patch,const NativeRemeshConfig &config,
    const SegmentationConfig *partitionConfig,
    const MeshResolutionInfo *partitionResolution){
  const auto start=std::chrono::steady_clock::now();
  const std::size_t count=sourceFaces.size();
  if(!count)return {};
  // An analytic cone is one trimmed domain. Keeping short input triangles
  // creates artificial holes and destroys its periodic sidewall topology.
  if(patch.SurfaceType==PatchSurfaceType::Cone &&
     patch.ProjectionTarget==PatchProjectionTarget::AnalyticSurface){
    std::vector<int> faces(count);for(std::size_t i=0;i<count;++i)faces[i]=int(i);
    return {{},{std::move(faces)},{patch}};
  }
  std::vector<std::vector<int>> neighbors(count);
  for(const auto &edge:BuildEdges(sourceFaces)){
    if(!edge.NonManifold && edge.FaceCount==2 && !globalConstraints.count(Key(edge.A,edge.B))){neighbors[edge.Faces[0]].push_back(edge.Faces[1]);neighbors[edge.Faces[1]].push_back(edge.Faces[0]);}
  }
  std::vector<unsigned char> candidate(count,0),keep(count,0),seen(count,0);
  std::vector<double> area(count,0);
  double totalArea=0,keptArea=0;
  // Preserve input solely by edge length, irrespective of shape or island size.
  const double preservationLength=config.TargetEdgeLength*(1+1e-6);
  for(std::size_t i=0;i<count;++i){
    const auto &f=sourceFaces[i];const auto &a=sourceVertices[f[0]],&b=sourceVertices[f[1]],&c=sourceVertices[f[2]];
    area[i]=.5*Norm(Cross(Sub(ToVec(b),ToVec(a)),Sub(ToVec(c),ToVec(a))));totalArea+=area[i];
    const std::array<double,3> lengths{Distance(a,b),Distance(b,c),Distance(c,a)};
    candidate[i]=std::all_of(lengths.begin(),lengths.end(),[&](double length){
      return std::isfinite(length) && length<=preservationLength;
    });
  }
  std::vector<std::vector<int>> preserved;
  std::vector<int> queue;std::size_t islands=0,keptFaces=0;
  for(std::size_t seed=0;seed<count;++seed)if(candidate[seed] && !seen[seed]){
    queue.clear();queue.push_back(int(seed));seen[seed]=1;
    for(std::size_t head=0;head<queue.size();++head)for(int n:neighbors[queue[head]])
      if(candidate[n] && !seen[n]){seen[n]=1;queue.push_back(n);}
    preserved.push_back(queue);
    ++islands;for(int id:queue){keep[id]=1;++keptFaces;keptArea+=area[id];}
  }
  std::fill(seen.begin(),seen.end(),0);
  std::vector<std::vector<int>> regions;
  // Kept triangles are barriers. Repartition only the remaining face graph.
  for(std::size_t seed=0;seed<count;++seed)if(!keep[seed] && !seen[seed]){
    std::vector<int> region{int(seed)};seen[seed]=1;
    for(std::size_t head=0;head<region.size();++head)for(int n:neighbors[region[head]])
      if(!keep[n] && !seen[n]){seen[n]=1;region.push_back(n);}
    regions.push_back(std::move(region));
  }
  // A clipped subset of an already classified cone inherits its surface.
  // Removing preserved triangles destroys the support needed to estimate an
  // apex again; failed rediscovery must not silently relabel it Freeform.
  if(patch.SurfaceType==PatchSurfaceType::Cone &&
     patch.ProjectionTarget==PatchProjectionTarget::AnalyticSurface){
    std::vector<MeshPatch> models(regions.size(),patch);
    return {std::move(preserved),std::move(regions),std::move(models)};
  }
  // Reuse the initial model-first partitioner, including independent nonlinear
  // seeds. Preserve indexed source face IDs and the original fitting resolution.
  std::vector<MeshPatch> regionModels;
  MeshPatch residualModel=patch;
  residualModel.SurfaceType=PatchSurfaceType::Freeform;
  residualModel.ProjectionTarget=PatchProjectionTarget::ReferenceMesh;
  regionModels.assign(regions.size(),residualModel);
  if(!regions.empty() && partitionConfig && partitionResolution){
    const auto partitionStart=std::chrono::steady_clock::now();
    TriangleSoup soup;std::vector<int> sourceIds,vertexAliases;
    std::unordered_map<int,int> vertexLookup;
    for(const auto &region:regions)for(int id:region){
      auto face=sourceFaces[id];
      for(int &v:face){const int original=v;
        auto inserted=vertexLookup.emplace(original,int(soup.Vertices.size()));
        if(inserted.second){soup.Vertices.push_back(sourceVertices[original]);vertexAliases.push_back(original);}
        v=inserted.first->second;
      }
      soup.Triangles.push_back(face);sourceIds.push_back(id);
    }
    MeshTopology localMesh;
    if(localMesh.buildIndexed(soup,*partitionResolution)){
      for(auto &edge:localMesh.getEdges())
        if(globalConstraints.count(Key(vertexAliases[edge.Vertex0],vertexAliases[edge.Vertex1])))
          edge.IsConstrainedFeature=true;
      auto fitting=*partitionConfig;fitting.EnableModelFirst=true;fitting.Verbose=false;
      if(config.DisableCuda)fitting.ModelAnalyticSeedBackend=AnalyticSeedBackend::Cpu;
      auto children=PartitionBySurfaceModels(localMesh,fitting);
      std::vector<std::vector<int>> fittedRegions;
      std::vector<MeshPatch> fittedModels;
      std::vector<unsigned char> owned(sourceIds.size(),0);
      bool complete=true;
      for(auto &child:children){
        std::vector<int> ids;
        for(int face:child.TriangleIds){
          if(face<0 || std::size_t(face)>=sourceIds.size() || owned[face]){complete=false;break;}
          owned[face]=1;ids.push_back(sourceIds[face]);
        }
        if(!complete)break;
        if(ids.empty())continue;
        if(child.SurfaceType==PatchSurfaceType::Unknown || child.ProjectionTarget!=PatchProjectionTarget::AnalyticSurface){
          child.SurfaceType=PatchSurfaceType::Freeform;
          child.ProjectionTarget=PatchProjectionTarget::ReferenceMesh;
        }
        fittedRegions.push_back(std::move(ids));fittedModels.push_back(std::move(child));
      }
      complete=complete && std::all_of(owned.begin(),owned.end(),[](unsigned char v){return v!=0;});
      if(complete){regions.swap(fittedRegions);regionModels.swap(fittedModels);}
      else if(config.Verbose)std::clog << "[CadMesh] local model partition: incomplete ownership; using connected reference regions" << std::endl;
    }else if(config.Verbose)std::clog << "[CadMesh] local model partition: indexed topology unavailable; using connected reference regions" << std::endl;
  }
  return {std::move(preserved),std::move(regions),std::move(regionModels)};
}
