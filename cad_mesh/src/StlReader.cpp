#include "CadMesh/StlReader.h"
#include <vcg/complex/complex.h>
#include <wrap/io_trimesh/import_stl.h>
#include <vector>

namespace CadMesh { namespace {
class VcgVertex;class VcgFace;
struct VcgUsedTypes:public vcg::UsedTypes<vcg::Use<VcgVertex>::AsVertexType,vcg::Use<VcgFace>::AsFaceType>{};
class VcgVertex:public vcg::Vertex<VcgUsedTypes,vcg::vertex::Coord3d,vcg::vertex::Normal3d,vcg::vertex::BitFlags>{};
class VcgFace:public vcg::Face<VcgUsedTypes,vcg::face::VertexRef,vcg::face::Normal3d,vcg::face::BitFlags>{};
class VcgMesh:public vcg::tri::TriMesh<std::vector<VcgVertex>,std::vector<VcgFace>>{};
}
bool StlReader::read(const std::filesystem::path&path,TriangleSoup&soup,std::string&error){VcgMesh mesh;int mask=0;int code=vcg::tri::io::ImporterSTL<VcgMesh>::Open(mesh,path.string().c_str(),mask);if(code!=0){error=vcg::tri::io::ImporterSTL<VcgMesh>::ErrorMsg(code);return false;}soup={};soup.Vertices.reserve(mesh.face.size()*3);soup.Triangles.reserve(mesh.face.size());for(const auto&face:mesh.face){if(face.IsD())continue;std::array<int,3>triangle{};for(int k=0;k<3;++k){triangle[k]=int(soup.Vertices.size());const auto&p=face.P(k);soup.Vertices.emplace_back(p.X(),p.Y(),p.Z());}soup.Triangles.push_back(triangle);}if(soup.Triangles.empty()){error="STL contains no triangles";return false;}return true;}
}
