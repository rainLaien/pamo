#include "cad_adaptive/BoundarySizingField.h"
#include "cad_adaptive/SemanticMesh.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cad_adaptive {
uint32_t BoundarySizingField::BuildTree(std::vector<uint32_t> &ids, size_t begin, size_t end) {
  const uint32_t index=uint32_t(mNodes.size());
  BoundarySizingNode node;
  node.MinX=node.MinY=node.MinZ=node.MinSize=std::numeric_limits<float>::max();
  node.MaxX=node.MaxY=node.MaxZ=-std::numeric_limits<float>::max();
  for(size_t i=begin;i<end;++i) {
    const auto &s=mSeeds[ids[i]];
    node.MinX=std::min(node.MinX,std::min(s.Ax,s.Bx));
    node.MinY=std::min(node.MinY,std::min(s.Ay,s.By));
    node.MinZ=std::min(node.MinZ,std::min(s.Az,s.Bz));
    node.MaxX=std::max(node.MaxX,std::max(s.Ax,s.Bx));
    node.MaxY=std::max(node.MaxY,std::max(s.Ay,s.By));
    node.MaxZ=std::max(node.MaxZ,std::max(s.Az,s.Bz));
    node.MinSize=std::min(node.MinSize,s.Size);
  }
  mNodes.push_back(node);
  if(end-begin==1) mNodes[index].Seed=ids[begin];
  else {
    const float extent[3]={node.MaxX-node.MinX,node.MaxY-node.MinY,node.MaxZ-node.MinZ};
    const int axis=int(std::max_element(extent,extent+3)-extent);
    const auto center=[&](uint32_t id) {
      const auto &s=mSeeds[id];
      return axis==0 ? s.Ax+s.Bx : (axis==1 ? s.Ay+s.By : s.Az+s.Bz);
    };
    const size_t middle=begin+(end-begin)/2;
    std::nth_element(ids.begin()+begin,ids.begin()+middle,ids.begin()+end,
        [&](uint32_t a,uint32_t b) {const float ca=center(a),cb=center(b);return ca!=cb?ca<cb:a<b;});
    BuildTree(ids,begin,middle); BuildTree(ids,middle,end);
  }
  mNodes[index].Skip=uint32_t(mNodes.size());
  return index;
}

std::shared_ptr<const BoundarySizingField> BoundarySizingField::create(
    const SemanticMesh &reference,const RemeshConfig &config,float gradation,bool curvatureSizing,
    const std::vector<float> &patchTargetLengths) {
  if(!(gradation>0.0f) || !std::isfinite(gradation) ||
     !(config.constantLength>0.0f) || !std::isfinite(config.constantLength) ||
     !(config.maxGeometryError>0.0f) || !std::isfinite(config.maxGeometryError) ||
     !(config.splitRatio>1.0f) || !std::isfinite(config.splitRatio))
    throw std::runtime_error("invalid boundary sizing parameters");
  if(!patchTargetLengths.empty() && patchTargetLengths.size()!=reference.patches.size())
    throw std::runtime_error("patch sizing override count mismatch");
  auto field=std::make_shared<BoundarySizingField>();
  field->mGradation=gradation; field->mFallbackLength=config.constantLength;
  field->mPatches.resize(reference.patches.size());
  for(size_t i=0;i<reference.patches.size();++i) {
    const auto &p=reference.patches[i];
    float base=config.constantLength;
    if(curvatureSizing && p.type==PatchType::Cylinder) {
      if(!(p.radius>0.0f) || !std::isfinite(p.radius) ||
         !(config.normalDegrees>0.0f) || !(config.normalDegrees<90.0f))
        throw std::runtime_error("invalid cylinder sizing geometry");
      // Target length is below the permissible chord, allowing for splitRatio.
      const double radius=p.radius, epsilon=std::min(double(config.maxGeometryError),radius);
      const double chord=2.0*std::sqrt(std::max(0.0,2.0*radius*epsilon-epsilon*epsilon));
      const double normalChord=2.0*radius*std::sin(double(config.normalDegrees)*0.017453292519943295);
      base=std::min(base,float(0.95*std::min(chord,normalChord)/config.splitRatio));
    }
    if(!patchTargetLengths.empty()) {
      const float requested=patchTargetLengths[i];
      if(!(requested>0) || !std::isfinite(requested)) throw std::runtime_error("invalid patch sizing override");
      base=std::min(base,requested);
    }
    if(!(base>0.0f) || !std::isfinite(base)) throw std::runtime_error("invalid patch sizing length");
    field->mPatches[i].BaseLength=base;
  }
  std::vector<std::vector<uint32_t>> patchSeeds(reference.patches.size());
  for(const auto &e:reference.edges) {
    if(!(e.flags & uint8_t(EdgeProtected|EdgeMeshBoundary|EdgePatchBoundary|EdgeSharp))) continue;
    const Vec3 a=reference.position(int(e.v0)),b=reference.position(int(e.v1));
    const float edgeLength=distance(a,b);
    if(!(edgeLength>0.0f) || !std::isfinite(edgeLength)) throw std::runtime_error("invalid sizing seed edge");
    float target=std::min(config.constantLength,edgeLength);
    const uint32_t supports[2]={e.patchLeft,e.patchRight};
    for(uint32_t p:supports) if(p<field->mPatches.size()) target=std::min(target,field->mPatches[p].BaseLength);
    const uint32_t seedId=uint32_t(field->mSeeds.size());
    field->mSeeds.push_back({a.x,a.y,a.z,b.x,b.y,b.z,target});
    for(int k=0;k<2;++k) {
      const uint32_t p=supports[k];
      if(p>=patchSeeds.size() || (k==1 && p==supports[0])) continue;
      if(target<field->mPatches[p].BaseLength) patchSeeds[p].push_back(seedId);
    }
  }
  for(size_t p=0;p<patchSeeds.size();++p) {
    field->mPatches[p].Begin=uint32_t(field->mNodes.size());
    if(!patchSeeds[p].empty()) field->BuildTree(patchSeeds[p],0,patchSeeds[p].size());
    field->mPatches[p].End=uint32_t(field->mNodes.size());
  }
  return field;
}

float BoundarySizingField::evaluate(uint32_t patchId,Vec3 point) const {
  if(patchId>=mPatches.size()) return mFallbackLength;
  return evaluateBoundarySizing(point.x,point.y,point.z,mPatches[patchId],
                                mNodes.data(),mSeeds.data(),mGradation);
}
void BoundarySizingField::apply(SemanticMesh &mesh) const {
  mesh.targetLength.resize(mesh.vertexCount());
  for(int v=0;v<mesh.vertexCount();++v)
    mesh.targetLength[v]=evaluate(mesh.vertexPatchId[v],mesh.position(v));
  // Boundary vertices have multiple supports; take the most restrictive one.
  for(int f=0;f<mesh.faceCount();++f) if(mesh.faceAlive[f])
    for(int v:mesh.face(f)) mesh.targetLength[v]=std::min(mesh.targetLength[v],evaluate(mesh.facePatchId[f],mesh.position(v)));
}
} // namespace cad_adaptive
