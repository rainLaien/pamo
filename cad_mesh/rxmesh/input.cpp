#include "input.h"
#include "CadMesh/StlReader.h"
#include "CadMesh/CadMeshPatchSegmenter.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>
namespace pamo_rx {
using Tri=std::array<int,3>;
using Key=uint64_t;
static Key key(int a,int b){if(a>b)std::swap(a,b);return (uint64_t(a)<<32)|uint32_t(b);}
static Vec vec(const CadMesh::Point3& p){return {p.X(),p.Y(),p.Z()};}
// Convert a polygon soup with non-manifold connections to manifold sheets.
// Corner unions cross only two-face, consistently oriented edges. No triangle
// is deleted or displaced; the introduced coincident seams become constraints.
static size_t manifoldSheets(const CadMesh::MeshTopology& source,CadMesh::MeshTopology& result){
    const auto& fs=source.getTriangles();std::vector<bool> isolated(fs.size());
    for(int attempt=0;attempt<3;++attempt){
        std::vector<int> parent(3*fs.size());std::iota(parent.begin(),parent.end(),0);
        auto root=[&](int a){while(parent[a]!=a){parent[a]=parent[parent[a]];a=parent[a];}return a;};
        auto join=[&](int a,int b){a=root(a);b=root(b);if(a!=b)parent[std::max(a,b)]=std::min(a,b);};
        for(const auto&e:source.getEdges())if(e.IncidentTriangleIds.size()==2){
            int a=e.Triangle0,b=e.Triangle1;if(isolated[a]||isolated[b])continue;
            int av=-1,bv=-1,aw=-1,bw=-1;
            for(int k=0;k<3;++k){if(fs[a].VertexIds[k]==e.Vertex0)av=k;if(fs[a].VertexIds[k]==e.Vertex1)aw=k;if(fs[b].VertexIds[k]==e.Vertex0)bv=k;if(fs[b].VertexIds[k]==e.Vertex1)bw=k;}
            if(((av+1)%3==aw)==((bv+1)%3==bw))continue;
            join(3*a+av,3*b+bv);join(3*a+aw,3*b+bw);
        }
        CadMesh::TriangleSoup soup;std::vector<int> remap(parent.size(),-1);soup.Triangles.resize(fs.size());
        for(size_t f=0;f<fs.size();++f)for(int k=0;k<3;++k){int r=root(int(3*f+k));if(remap[r]<0){remap[r]=int(soup.Vertices.size());soup.Vertices.push_back(source.getVertices()[fs[f].VertexIds[k]].Position);}soup.Triangles[f][k]=remap[r];}
        if(!result.buildIndexed(soup,source.getResolution()))throw std::runtime_error("manifold sheet construction failed");
        for(size_t f=0;f<fs.size();++f)result.getTriangles()[f].PatchId=fs[f].PatchId;
        if(!result.getCleanupReport().NonManifoldEdges)return soup.Vertices.size()>source.getVertices().size()?soup.Vertices.size()-source.getVertices().size():0;
        // Rare multi-edge identifications require cutting the affected faces
        // from their local fan, still preserving all their geometry.
        for(const auto&e:result.getEdges())if(e.IsNonManifold)for(int f:e.IncidentTriangleIds)isolated[f]=true;
    }
    throw std::runtime_error("could not separate non-manifold sheets");
}
void buildReference(std::vector<RefTriangle>& ts,std::vector<BvhNode>& ns,std::vector<int>& roots){
    ns.clear();roots.clear();if(ts.empty())return;
    int maxLabel=0;for(const auto& t:ts)maxLabel=std::max(maxLabel,t.region);
    roots.assign(maxLabel+1,-1);
    std::stable_sort(ts.begin(),ts.end(),[](const auto&a,const auto&b){return a.region<b.region;});
    ns.reserve(ts.size()/2);
    std::function<int(int,int)> build=[&](int a,int b){
        BvhNode n;n.lo={1e100,1e100,1e100};n.hi={-1e100,-1e100,-1e100};
        for(int i=a;i<b;++i)for(Vec p:ts[i].p)for(int k=0;k<3;++k){n.lo[k]=std::min(n.lo[k],p[k]);n.hi[k]=std::max(n.hi[k],p[k]);}
        int id=int(ns.size());ns.push_back(n);
        if(b-a<=8){ns[id].begin=a;ns[id].count=b-a;return id;}
        Vec d=n.hi-n.lo;int axis=d.y>d.x?1:0;if(d.z>d[axis])axis=2;
        int mid=a+(b-a)/2;
        std::nth_element(ts.begin()+a,ts.begin()+mid,ts.begin()+b,[axis](const auto&x,const auto&y){return x.p[0][axis]+x.p[1][axis]+x.p[2][axis]<y.p[0][axis]+y.p[1][axis]+y.p[2][axis];});
        int l=build(a,mid),r=build(mid,b);ns[id].left=l;ns[id].right=r;return id;
    };
    for(int a=0;a<int(ts.size());){int b=a+1;while(b<int(ts.size())&&ts[b].region==ts[a].region)++b;roots[ts[a].region]=build(a,b);a=b;}
}
Input prepare(const Options& o){
    const auto started=std::chrono::steady_clock::now();
    CadMesh::TriangleSoup soup;CadMesh::MeshTopology topology;CadMesh::CadMeshPatchSegmenter segmenter;
    std::string error;const CadMesh::MeshTopology* mesh=&topology;
    if(o.snapshot){if(!segmenter.loadRemeshSnapshot(o.input,error))throw std::runtime_error(error);mesh=&segmenter.getMesh();}
    else {if(!CadMesh::StlReader::read(o.input,soup,error))throw std::runtime_error(error);
        if(o.segment){if(!segmenter.segment(soup))throw std::runtime_error("CAD partition failed");mesh=&segmenter.getMesh();}
        else if(!topology.build(soup))throw std::runtime_error("STL cleanup failed");}
    if(mesh->getTriangles().empty()||mesh->getVertices().empty())throw std::runtime_error("empty input mesh");
    Input in;in.sourceFaces=mesh->getTriangles().size();in.inputNonmanifoldEdges=mesh->getCleanupReport().NonManifoldEdges;
    std::unordered_set<Key> partitionConstraints;
    if(o.snapshot||o.segment)for(int id:segmenter.getRemeshConstraint().ConstraintEdgeIds){const auto&e=mesh->getEdges().at(id);partitionConstraints.insert(key(e.Vertex0,e.Vertex1));}
    CadMesh::MeshTopology manifold;
    // Also separates disconnected vertex fans in edge-manifold inputs.
    const auto* original=mesh;in.manifoldVertexCopies=manifoldSheets(*original,manifold);mesh=&manifold;
    std::unordered_set<Key> mappedConstraints;
    for(size_t f=0;f<in.sourceFaces;++f)for(int k=0;k<3;++k){const auto&a=original->getTriangles()[f].VertexIds;const auto&b=mesh->getTriangles()[f].VertexIds;if(partitionConstraints.count(key(a[k],a[(k+1)%3])))mappedConstraints.insert(key(b[k],b[(k+1)%3]));}
    std::clog<<"[RX] input_nonmanifold_edges="<<in.inputNonmanifoldEdges<<", manifold_vertex_copies="<<in.manifoldVertexCopies<<", sheet_boundary_edges="<<mesh->getCleanupReport().BoundaryEdges<<'\n';
    Vec lo=vec(mesh->getVertices().front().Position),hi=lo;
    for(const auto& v:mesh->getVertices())for(int k=0;k<3;++k){lo[k]=std::min(lo[k],v.Position[k]);hi[k]=std::max(hi[k],v.Position[k]);}
    in.diagonal=sqrt(norm2(hi-lo));if(!(in.diagonal>0))throw std::runtime_error("zero-size mesh");
    in.origin=(lo+hi)*.5;in.target=o.target>0?o.target/in.diagonal:o.ratio;in.deviation=o.deviation/in.diagonal;
    for(const auto& v:mesh->getVertices())in.points.push_back((vec(v.Position)-in.origin)*(1/in.diagonal));
    std::vector<Tri> faces;std::vector<int> labels;
    for(const auto& f:mesh->getTriangles()){faces.push_back(f.VertexIds);labels.push_back((o.snapshot||o.segment)?f.PatchId:-1);}
    std::unordered_set<Key> hard;
    double cosine=cos(o.featureDegrees*acos(-1.)/180);
    for(const auto& e:mesh->getEdges())if(e.IsBoundary||e.IsNonManifold||
        (e.Triangle0>=0&&e.Triangle1>=0&&CadMesh::Dot(CadMesh::ToVec(mesh->getTriangles()[e.Triangle0].Normal),CadMesh::ToVec(mesh->getTriangles()[e.Triangle1].Normal))<cosine))hard.insert(key(e.Vertex0,e.Vertex1));
    if(o.snapshot||o.segment){hard.insert(mappedConstraints.begin(),mappedConstraints.end());}
    else {int region=0;for(int seed=0;seed<int(faces.size());++seed)if(labels[seed]<0){
        std::vector<int> todo{seed};labels[seed]=region;
        for(size_t i=0;i<todo.size();++i)for(int eid:mesh->getTriangles()[todo[i]].EdgeIds){const auto&e=mesh->getEdges()[eid];if(hard.count(key(e.Vertex0,e.Vertex1)))continue;
            int next=e.Triangle0==todo[i]?e.Triangle1:e.Triangle0;if(next>=0&&labels[next]<0){labels[next]=region;todo.push_back(next);}}
        ++region;
    }}
    for(const auto&e:mesh->getEdges())if(e.Triangle0>=0&&e.Triangle1>=0&&labels[e.Triangle0]!=labels[e.Triangle1])hard.insert(key(e.Vertex0,e.Vertex1));
    for(size_t i=0;i<faces.size();++i){if(labels[i]<0)throw std::runtime_error("invalid source region");const auto&f=faces[i];in.reference.push_back({{in.points[f[0]],in.points[f[1]],in.points[f[2]]},labels[i]});}
    // All shared edges are split once globally. Each incident face consumes
    // the same midpoint ID; subdivision leaves the original PL surface intact.
    for(int round=0;round<32;++round){std::unordered_map<Key,int> mids;
        for(Key e:hard){int a=int(e>>32),b=int(uint32_t(e));if(norm2(in.points[a]-in.points[b])>in.target*in.target*(1+1e-12)){mids[e]=int(in.points.size());in.points.push_back((in.points[a]+in.points[b])*.5);}}
        if(mids.empty())break;
        if(faces.size()+2*mids.size()>size_t(o.maxFaces))throw std::runtime_error("shared-boundary refinement exceeds --max-faces");
        std::vector<Tri> next;std::vector<int> nextLabels;next.reserve(faces.size()+2*mids.size());nextLabels.reserve(next.capacity());
        for(size_t f=0;f<faces.size();++f){std::vector<Tri> children{faces[f]};
            for(int k=0;k<3;++k){Key e=key(faces[f][k],faces[f][(k+1)%3]);auto it=mids.find(e);if(it==mids.end())continue;
                for(size_t j=0;j<children.size();++j){Tri t=children[j];bool split=false;for(int s=0;s<3;++s)if(key(t[s],t[(s+1)%3])==e){children[j]={t[s],it->second,t[(s+2)%3]};children.push_back({it->second,t[(s+1)%3],t[(s+2)%3]});split=true;break;}if(split)break;}}
            for(auto t:children){next.push_back(t);nextLabels.push_back(labels[f]);}}
        for(auto [e,m]:mids){hard.erase(e);hard.insert(key(int(e>>32),m));hard.insert(key(m,int(uint32_t(e))));}
        faces.swap(next);labels.swap(nextLabels);in.boundarySplits+=mids.size();
        if(round==31)throw std::runtime_error("boundary refinement did not converge");
    }
    in.fixed.assign(in.points.size(),{0});in.vertexRegion.assign(in.points.size(),{-2});in.sizes.assign(in.points.size(),{in.target});
    for(Key e:hard){int a=int(e>>32),b=int(uint32_t(e));in.fixed[a][0]=in.fixed[b][0]=1;in.constraints.push_back({a,b});}
    for(size_t i=0;i<faces.size();++i){const auto&f=faces[i];in.faces.push_back({uint32_t(f[0]),uint32_t(f[1]),uint32_t(f[2])});in.labels.push_back({labels[i]});
        for(int v:f){int&r=in.vertexRegion[v][0];if(r==-2)r=labels[i];else if(r!=labels[i])r=-1;}}
    for(Vec p:in.points)in.coordinates.push_back({p.x,p.y,p.z});
    // Keep the requested sizing target independent of triangulation quality.
    // Curvature is enforced by projection/error guards during coarsening:
    // rejected edits retain the source's finer sampling. Dihedral/triangle-height
    // curvature estimates on skinny STL triangles would spuriously refine them.
    // Welding/degenerate-face cleanup can leave unused vertex IDs. RXMesh
    // attribute initialization requires a dense, used input index range.
    std::vector<int> remap(in.points.size(),-1);int used=0;
    for(const auto&f:in.faces)for(auto v:f)if(remap[v]<0)remap[v]=used++;
    if(size_t(used)!=in.points.size()){
        auto points=in.points;auto coords=in.coordinates;auto sizes=in.sizes;auto fixed=in.fixed;auto regions=in.vertexRegion;
        in.points.resize(used);in.coordinates.resize(used);in.sizes.resize(used);in.fixed.resize(used);in.vertexRegion.resize(used);
        for(size_t i=0;i<remap.size();++i)if(remap[i]>=0){int j=remap[i];in.points[j]=points[i];in.coordinates[j]=coords[i];in.sizes[j]=sizes[i];in.fixed[j]=fixed[i];in.vertexRegion[j]=regions[i];}
        for(auto&f:in.faces)for(auto&v:f)v=remap[v];
        for(auto&e:in.constraints)for(auto&v:e)v=remap[v];
    }
    buildReference(in.reference,in.nodes,in.roots);
    in.planes.resize(in.roots.size());std::vector<double> largest(in.roots.size());
    for(auto&t:in.reference){double ar=norm2(cross(t.p[1]-t.p[0],t.p[2]-t.p[0]));t.gpuFast=quality(t.p[0],t.p[1],t.p[2])>1e-6;if(ar>largest[t.region]){largest[t.region]=ar;in.planes[t.region].point=t.p[0];in.planes[t.region].n=normal(t.p[0],t.p[1],t.p[2]);}}
    for(const auto&t:in.reference){auto&p=in.planes[t.region];p.minimumNormalDot=std::min(p.minimumNormalDot,dot(p.n,normal(t.p[0],t.p[1],t.p[2])));for(Vec v:t.p)p.error=std::max(p.error,std::abs(dot(v-p.point,p.n)));}
    for(auto&p:in.planes)p.enabled=p.error<=in.deviation*.01&&p.minimumNormalDot>=.999999;
    size_t planeFaces=0;for(const auto&t:in.reference)planeFaces+=in.planes[t.region].enabled;
    std::clog<<"[RX] plane_projection_faces="<<planeFaces<<'\n';
    std::clog<<"[RX] input_faces="<<in.sourceFaces<<", seed_faces="<<in.faces.size()<<", regions="<<in.roots.size()<<", boundary_splits="<<in.boundarySplits<<", target="<<in.target*in.diagonal<<", prepare_s="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<'\n';
    return in;
}
void writeResult(const Options&o,const Input&in,const std::vector<Vec>&points,
                 const std::vector<Tri>&faces,const std::vector<int>&labels,const std::vector<int>&anchors,
                 double elapsed,double gpu,double prep,bool topologyValid){
    auto started=std::chrono::steady_clock::now();
    std::vector<RefTriangle> candidate;candidate.reserve(faces.size());
    size_t degenerate=0,small10=0,small28=0;double minimum=180,angleSum=0,area=0,smallArea=0;
    std::unordered_map<Key,int> incidence;std::unordered_set<Key> preservedEdges;
    for(size_t i=0;i<faces.size();++i){const auto&f=faces[i];Vec a=points.at(f[0]),b=points.at(f[1]),c=points.at(f[2]);
        candidate.push_back({{a,b,c},labels[i]});double ar=.5*sqrt(norm2(cross(b-a,c-a)));if(!(ar>0)||!std::isfinite(ar))++degenerate;
        double angle=180;Vec p[3]={a,b,c};for(int k=0;k<3;++k){Vec u=p[(k+1)%3]-p[k],v=p[(k+2)%3]-p[k];angle=std::min(angle,atan2(sqrt(norm2(cross(u,v))),dot(u,v))*180/acos(-1.));}
        minimum=std::min(minimum,angle);angleSum+=angle;small10+=angle<10;small28+=angle<28;area+=ar;if(angle<28)smallArea+=ar;
        for(int k=0;k<3;++k){int x=f[k],y=f[(k+1)%3];++incidence[key(x,y)];if(anchors[x]>=0&&anchors[y]>=0)preservedEdges.insert(key(anchors[x],anchors[y]));}
    }
    size_t missingConstraints=0,movedAnchors=0,nonmanifold=0,boundaries=0;
    for(const auto&e:in.constraints)missingConstraints+=!preservedEdges.count(key(e[0],e[1]));
    for(size_t i=0;i<points.size();++i)if(anchors[i]>=0)movedAnchors+=norm2(points[i]-in.points.at(anchors[i]))>1e-24;
    for(auto [k,n]:incidence){nonmanifold+=n>2;boundaries+=n==1;}
    Reference source{in.reference.data(),in.nodes.data(),in.roots.data(),int(in.roots.size())};
    size_t forwardFailed=0,reverseFailed=0;double maxForward=0,maxReverse=0;
    std::vector<double> forwardMax(8*omp_get_max_threads()),reverseMax(forwardMax.size());
    #pragma omp parallel for reduction(+:forwardFailed)
    for(int i=0;i<int(candidate.size());++i){const auto&t=candidate[i];Vec a=t.p[0],b=t.p[1],c=t.p[2],q;int hit;
        Vec sample[7]={a,b,c,(a+b)*.5,(b+c)*.5,(c+a)*.5,(a+b+c)*(1./3.)};
        double& maximum=forwardMax[8*omp_get_thread_num()];
        bool valid=true;for(Vec p:sample){if(!source.nearest(p,t.region,in.deviation,q,hit))valid=false;else maximum=std::max(maximum,sqrt(norm2(p-q)));}
        valid=valid&&source.accepts(a,b,c,t.region,in.deviation,cos(o.normalDegrees*acos(-1.)/180));forwardFailed+=!valid;
    }
    std::vector<BvhNode> nodes;std::vector<int> roots;buildReference(candidate,nodes,roots);
    Reference target{candidate.data(),nodes.data(),roots.data(),int(roots.size())};
    #pragma omp parallel for reduction(+:reverseFailed)
    for(int i=0;i<int(in.reference.size());++i){const auto&t=in.reference[i];Vec sample[4]={t.p[0],t.p[1],t.p[2],(t.p[0]+t.p[1]+t.p[2])*(1./3.)};Vec q;int hit;bool valid=true;
        double& maximum=reverseMax[8*omp_get_thread_num()];
        for(Vec p:sample){if(!target.nearest(p,t.region,in.deviation,q,hit))valid=false;else maximum=std::max(maximum,sqrt(norm2(p-q)));}reverseFailed+=!valid;
    }
    maxForward=*std::max_element(forwardMax.begin(),forwardMax.end());maxReverse=*std::max_element(reverseMax.begin(),reverseMax.end());
    bool passed=topologyValid&&!degenerate&&!nonmanifold&&!missingConstraints&&!movedAnchors&&!forwardFailed&&!reverseFailed;
    std::filesystem::create_directories(o.output);auto out=std::filesystem::path(o.output);
    std::ofstream ply(out/((passed||o.skipAudit)?"rxmesh_result.ply":"rxmesh_candidate.ply"),std::ios::binary);
    ply<<"ply\nformat binary_little_endian 1.0\ncomment RXMesh reference-surface remeshing\nelement vertex "<<points.size()<<"\nproperty double x\nproperty double y\nproperty double z\nelement face "<<faces.size()<<"\nproperty list uchar int vertex_indices\nproperty int source_patch_id\nend_header\n";
    for(Vec p:points){p=p*in.diagonal+in.origin;for(int k=0;k<3;++k)ply.write(reinterpret_cast<const char*>(&p[k]),8);}
    for(size_t i=0;i<faces.size();++i){uint8_t n=3;ply.write(reinterpret_cast<const char*>(&n),1);ply.write(reinterpret_cast<const char*>(faces[i].data()),12);ply.write(reinterpret_cast<const char*>(&labels[i]),4);}
    ply.close();if(!ply)throw std::runtime_error("PLY write failed");
    double total=elapsed+std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::ofstream report(out/"rxmesh_report.json");report<<std::setprecision(12)<<std::boolalpha
        <<"{\n  \"backend\":\"RXMesh\",\n  \"rxmesh_commit\":\"e468c34ffabc70cd207309bce662d4821a9ed3b7\","
        <<"\n  \"input_faces\":"<<in.sourceFaces<<",\n  \"output_faces\":"<<faces.size()<<",\n  \"diagonal\":"<<in.diagonal
        <<",\n  \"target_edge_length\":"<<in.target*in.diagonal<<",\n  \"maximum_deviation\":"<<o.deviation
        <<",\n  \"source_regions\":"<<in.roots.size()<<",\n  \"boundary_splits\":"<<in.boundarySplits
        <<",\n  \"input_nonmanifold_edges\":"<<in.inputNonmanifoldEdges<<",\n  \"manifold_vertex_copies\":"<<in.manifoldVertexCopies
        <<",\n  \"prepare_seconds\":"<<prep<<",\n  \"gpu_remesh_seconds\":"<<gpu<<",\n  \"total_seconds\":"<<total
        <<",\n  \"within_60_seconds\":"<<(total<=60)<<",\n  \"sampled_shape_and_topology_passed\":"<<passed
        <<",\n  \"strict_hausdorff_certified\":false,\n  \"self_intersections_checked\":false"
        <<",\n  \"forward_failed_faces\":"<<forwardFailed<<",\n  \"reverse_failed_faces\":"<<reverseFailed
        <<",\n  \"sampled_forward_max_within_search_radius\":"<<maxForward*in.diagonal<<",\n  \"sampled_reverse_max_within_search_radius\":"<<maxReverse*in.diagonal
        <<",\n  \"missing_constraint_edges\":"<<missingConstraints<<",\n  \"moved_constraint_vertices\":"<<movedAnchors
        <<",\n  \"nonmanifold_edges\":"<<nonmanifold<<",\n  \"boundary_edges\":"<<boundaries<<",\n  \"degenerate_faces\":"<<degenerate
        <<",\n  \"minimum_angle_degrees\":"<<minimum<<",\n  \"mean_minimum_angle_degrees\":"<<angleSum/faces.size()
        <<",\n  \"below_10_faces\":"<<small10<<",\n  \"below_28_faces\":"<<small28
        <<",\n  \"below_28_area_fraction\":"<<smallArea/std::max(1e-30,area)<<"\n}\n";
    report.close();if(!report)throw std::runtime_error("report write failed");
    std::clog<<"[RX] shape_passed="<<passed<<", below_28_faces="<<small28<<", total_s="<<total<<'\n';
    if(!passed&&!o.skipAudit)throw std::runtime_error("final sampled shape/topology audit failed; retained candidate and report");
}
}
