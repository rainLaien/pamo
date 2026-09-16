#include "cad_adaptive/PartitionInput.h"
#include "cad_adaptive/GeometryProjector.h"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <set>

namespace cad_adaptive {
int refinePartitionBoundary(SemanticMesh &mesh, float maxLength) {
  if (!(maxLength>0) || !std::isfinite(maxLength)) throw std::runtime_error("invalid boundary target length");
  int splits=0;
  for (;;) {
    mesh.rebuildTopology();
    EdgeRec selected; bool found=false;
    for(const auto &e:mesh.edges) {
      if ((e.flags & (EdgePatchBoundary|EdgeMeshBoundary)) &&
          distance(mesh.position(e.v0),mesh.position(e.v1))>maxLength) {
        selected=e; found=true; break;
      }
    }
    if(!found) break;
    if(++splits>100000) throw std::runtime_error("boundary refinement budget exceeded");
    const auto &e=selected;
    const int middle=mesh.addVertex((mesh.position(e.v0)+mesh.position(e.v1))*0.5f,
                                    e.patchLeft,VertexConstraint::Locked);
    for(int f:{e.face0,e.face1}) {
      if(f<0) continue;
      auto t=mesh.face(f); auto pid=mesh.facePatchId[f];
      for(int k=0;k<3;++k) {
        int a=t[k],b=t[(k+1)%3],c=t[(k+2)%3];
        if((a==int(e.v0) && b==int(e.v1)) || (a==int(e.v1) && b==int(e.v0))) {
          mesh.killFace(f);
          mesh.addFace(a,middle,c,pid,mesh.patches[pid].type);
          mesh.addFace(middle,b,c,pid,mesh.patches[pid].type);
          break;
        }
      }
    }
  }
  mesh.compact(); mesh.computeVertexNormals();
  return splits;
}

bool validatePartitionOutput(const SemanticMesh &source, const SemanticMesh &output,
                             const RemeshConfig &config, RemeshReport &report, std::string *error) {
  const float tolerance=std::max(1e-6f,source.bboxDiagonal()*1e-7f);
  std::vector<int> mapped(source.vertexCount(),-1);
  report.movedLockedVertices=0;
  for(int v=0;v<source.vertexCount();++v) {
    if(source.vertexConstraint[v]!=uint8_t(VertexConstraint::Locked)) continue;
    float best=tolerance*tolerance;
    for(int w=0;w<output.vertexCount();++w) {
      float d=length2(source.position(v)-output.position(w));
      if(d<=best) { best=d; mapped[v]=w; }
    }
    if(mapped[v]<0) ++report.movedLockedVertices;
  }
  std::set<std::pair<int,int>> edges;
  for(const auto &e:output.edges) edges.emplace(e.v0,e.v1);
  report.missingBoundaryEdges=0;
  for(const auto &e:source.edges) {
    if(!(e.flags & (EdgePatchBoundary|EdgeMeshBoundary))) continue;
    int a=mapped[e.v0],b=mapped[e.v1];
    if(a<0 || b<0 || !edges.count({std::min(a,b),std::max(a,b)})) ++report.missingBoundaryEdges;
  }
  GeometryProjector projector; projector.build(source);
  std::vector<float> orientation(source.patches.size(),0);
  for(int f=0;f<source.faceCount();++f) {
    auto pid=source.facePatchId[f];
    auto a=source.facePoint(f,0),b=source.facePoint(f,1),c=source.facePoint(f,2);
    orientation[pid]+=dot(cross(b-a,c-a),projector.analyticNormal(pid,centroid3(a,b,c)));
  }
  bool normalsValid=true;
  std::vector<int> patchFaces(source.patches.size(),0);
  for(int f=0;f<output.faceCount();++f) {
    auto pid=output.facePatchId[f];
    if(pid>=orientation.size()) { normalsValid=false; continue; }
    ++patchFaces[pid];
    auto a=output.facePoint(f,0),b=output.facePoint(f,1),c=output.facePoint(f,2);
    auto mid=centroid3(a,b,c);
    if(dot(cross(b-a,c-a),projector.analyticNormal(pid,mid))*orientation[pid]<=0) normalsValid=false;
    Vec3 samples[]={a,b,c,mid,(a+b)*0.5f,(b+c)*0.5f,(c+a)*0.5f};
    for(auto p:samples) {
      auto projected=projector.projectSurface(pid,p);
      if(!projected.ok) { normalsValid=false; continue; }
      report.geometryErrorMax=std::max(report.geometryErrorMax,distance(p,projected.position));
    }
  }
  for(int count:patchFaces) if(!count) normalsValid=false;
  report.constraintsHeld=report.movedLockedVertices==0 && report.missingBoundaryEdges==0;
  if(!report.constraintsHeld || !normalsValid || report.geometryErrorMax>config.maxGeometryError+tolerance) {
    if(error) *error=!report.constraintsHeld ? "CAD feature vertices/edges were lost" :
      (!normalsValid ? "CAD patch orientation or ownership invalid" : "CAD surface deviation exceeds tolerance");
    return false;
  }
  return true;
}

bool loadPartitionInput(const std::string &path, SemanticMesh &mesh, std::string *error) {
  try {
    std::ifstream in(path, std::ios::binary);
    auto read = [&](auto &value) {
      if (!in.read(reinterpret_cast<char *>(&value), sizeof(value)))
        throw std::runtime_error("truncated partition snapshot");
    };
    char magic[8]; read(magic);
    if (std::memcmp(magic, "CADPART1", 8)) throw std::runtime_error("expected CADPART1 snapshot");
    uint32_t nv, nf, np, ne; read(nv); read(nf); read(np); read(ne);
    double resolution[6]; read(resolution);
    if (!nv || !nf || !np || np > nf || nv > 100000000 || nf > 100000000 || ne > uint64_t(nf)*3)
      throw std::runtime_error("invalid partition counts");
    SemanticMesh result;
    for (uint32_t v=0; v<nv; ++v) {
      double p[3]; read(p);
      for (double x : p) if (!std::isfinite(x) || std::abs(x) > 1e30)
        throw std::runtime_error("invalid partition coordinate");
      result.addVertex({float(p[0]),float(p[1]),float(p[2])}, 0, VertexConstraint::Surface);
    }
    std::vector<uint32_t> counts(np, 0);
    for (uint32_t f=0; f<nf; ++f) {
      uint32_t t[4]; read(t);
      if (t[0]>=nv || t[1]>=nv || t[2]>=nv || t[3]>=np ||
          t[0]==t[1] || t[1]==t[2] || t[2]==t[0]) throw std::runtime_error("invalid partition face");
      result.addFace(t[0],t[1],t[2],t[3],PatchType::Unknown);
      ++counts[t[3]];
      for (int k=0;k<3;++k) result.vertexPatchId[t[k]]=t[3];
    }
    for (uint32_t p=0;p<np;++p) {
      uint32_t h[5]; read(h);
      if (h[3]!=counts[p] || h[4]>np) throw std::runtime_error("partition ownership mismatch");
      for (uint32_t s=0;s<h[4];++s) { uint32_t id; read(id); if(id>=np) throw std::runtime_error("invalid support patch"); }
      double params[9]; read(params);
      if (h[1]!=1 || (h[0]!=uint32_t(PatchType::Plane) && h[0]!=uint32_t(PatchType::Cylinder)))
        throw std::runtime_error("partition requires unsupported projection (only analytic plane/cylinder supported)");
      for (double x:params) if (!std::isfinite(x)) throw std::runtime_error("invalid surface parameters");
      PatchRecord rec;
      rec.type=PatchType(h[0]);
      rec.origin={float(params[0]),float(params[1]),float(params[2])};
      rec.axis={float(params[3]),float(params[4]),float(params[5])};
      rec.radius=float(params[6]);
      if (length2(rec.axis)<1e-12f || (rec.type==PatchType::Cylinder && !(rec.radius>0)))
        throw std::runtime_error("invalid analytic surface");
      result.patches.push_back(rec);
    }
    std::vector<std::array<uint32_t, 2>> featureEdges;
    for (uint32_t e=0;e<ne;++e) {
      uint32_t r[4]; read(r);
      if (r[1]>=nv || r[2]>=nv || r[1]==r[2] || r[3]>1) throw std::runtime_error("invalid feature edge");
      // Preserve the supplied feature polyline and its corners exactly.
      result.vertexConstraint[r[1]]=uint8_t(VertexConstraint::Locked);
      result.vertexConstraint[r[2]]=uint8_t(VertexConstraint::Locked);
      featureEdges.push_back({r[1],r[2]});
    }
    if (in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing partition data");
    for(uint32_t f=0;f<nf;++f) result.facePatchType[f]=uint8_t(result.patches[result.facePatchId[f]].type);
    result.rebuildTopology();
    for (auto ends : featureEdges) {
      bool found=false;
      for (const auto &e : result.edges) {
        if (e.v0!=std::min(ends[0],ends[1]) || e.v1!=std::max(ends[0],ends[1])) continue;
        found=true;
        if (!(e.flags & (EdgePatchBoundary|EdgeMeshBoundary)))
          throw std::runtime_error("feature within one patch requires an explicit curve constraint; refusing unprotected remesh");
        break;
      }
      if (!found) throw std::runtime_error("constraint references a missing mesh edge");
    }
    for(const auto &e:result.edges) if(e.flags & (EdgePatchBoundary|EdgeMeshBoundary)) {
      result.vertexConstraint[e.v0]=uint8_t(VertexConstraint::Locked);
      result.vertexConstraint[e.v1]=uint8_t(VertexConstraint::Locked);
    }
    result.computeVertexNormals();
    if (!result.validate(error)) return false;
    mesh=std::move(result);
    return true;
  } catch(const std::exception &e) { if(error) *error=e.what(); return false; }
}
}
