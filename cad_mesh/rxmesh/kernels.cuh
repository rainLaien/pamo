#pragma once
#include "geometry.h"
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"
namespace pamo_rx {
using namespace rxmesh;
using Coordinates=VertexAttribute<double>;
using VInfo=VertexAttribute<int>; // region, fixed source vertex ID, last move pass
using EInfo=EdgeAttribute<int>;   // common incident region, last topology pass
__device__ inline Vec position(Coordinates c,VertexHandle v){return {c(v,0),c(v,1),c(v,2)};}
__device__ inline void position(Coordinates c,VertexHandle v,Vec p){c(v,0)=p.x;c(v,1)=p.y;c(v,2)=p.z;}
struct Policy {Reference ref;double limit,cosine;int stage;};

// Kind: 0 split, 1 flip, 2 collapse. All live attributes participate in
// migration; geometric region IDs never serve as RXMesh scheduling IDs.
template<int Kind,uint32_t Threads=256>
__global__ void edit(Context context,Coordinates coords,VertexAttribute<double> sizes,
                     VInfo info,FaceAttribute<int> labels,EInfo edges,Policy policy){
    auto block=cooperative_groups::this_thread_block();ShmemAllocator shmem;
    constexpr CavityOp op=Kind==2?CavityOp::EV:CavityOp::E;
    CavityManager<Threads,op> cavity(block,context,shmem,true,Kind!=1);
    if(cavity.patch_id()==INVALID32)return;
    const auto before=shmem.get_allocated_size_bytes();
    Query<Threads> neighbors(context,cavity.patch_id());
    if constexpr(Kind!=0)neighbors.template prologue<Op::VV>(block,shmem);
    Query<Threads> query(context,cavity.patch_id());
    query.template dispatch<Op::EVDiamond>(block,shmem,[&](EdgeHandle e,const VertexIterator& v){
        if(edges(e,1)==policy.stage)return;
        if(v.size()!=4)return;for(int i=0;i<4;++i)if(!v[i].is_valid())return;
        for(int i=0;i<4;++i)for(int j=i+1;j<4;++j)if(v[i]==v[j])return;
        const int region=edges(e,0);if(region<0)return;
        // The ribbon guarantees complete vertex queries for owned vertices.
        // A non-owned vertex's local ring may be truncated, which is not a
        // valid basis for a topology link test.
        if constexpr(Kind==1)if(v[1].patch_id()!=cavity.patch_id())return;
        if constexpr(Kind==2)if(v[0].patch_id()!=cavity.patch_id()||v[2].patch_id()!=cavity.patch_id())return;
        Vec a=position(coords,v[0]),b=position(coords,v[2]),c=position(coords,v[1]),d=position(coords,v[3]);
        const double h=minv(sizes(v[0]),sizes(v[2]));bool selected=false;
        if constexpr(Kind==0){selected=norm2(a-b)>h*h*(16./9.);}
        if constexpr(Kind==1){
            auto ring=neighbors.template get_iterator<VertexIterator>(v.local(1));
            for(int j=0;j<ring.size();++j)if(!ring[j].is_valid()||ring[j]==v[3])return;
            selected=norm2(c-d)<=maxv(norm2(a-b),h*h*16./9.) &&
                minv(quality(c,d,b),quality(d,c,a))>minv(quality(a,c,b),quality(a,b,d))*1.001+1e-12;
        }
        if constexpr(Kind==2){
            if(info(v[0],1)>=0||info(v[2],1)>=0||info(v[0],0)!=region||info(v[2],0)!=region)return;
            if(norm2(a-b)>=h*h*.64)return;
            auto ra=neighbors.template get_iterator<VertexIterator>(v.local(0));
            auto rb=neighbors.template get_iterator<VertexIterator>(v.local(2));
            if(ra.size()<=3||rb.size()<=3)return;
            for(int i=0;i<ra.size();++i)if(!ra[i].is_valid())return;
            for(int i=0;i<rb.size();++i)if(!rb[i].is_valid())return;
            int common=0;for(int i=0;i<ra.size();++i)for(int j=0;j<rb.size();++j)if(ra[i]==rb[j])++common;
            selected=common==2;
        }
        if(selected)cavity.create(e);else edges(e,1)=policy.stage;
    });
    block.sync();shmem.dealloc(shmem.get_allocated_size_bytes()-before);
    if(cavity.prologue(block,shmem,coords,sizes,info,labels,edges)){
        cavity.for_each_cavity(block,[&](uint16_t c,uint16_t n){
            EdgeHandle seed=cavity.template get_creator<EdgeHandle>(c);const int region=edges(seed,0);
            VertexHandle va,vb;cavity.get_vertices(seed,va,vb);
            Vec pa=position(coords,va),pb=position(coords,vb);
            double h=minv(sizes(va),sizes(vb));
            if(n<3||n>128||region<0){cavity.recover(seed);edges(seed,1)=policy.stage;return;}
            if constexpr(Kind==1){
                if(n!=4){cavity.recover(seed);edges(seed,1)=policy.stage;return;}
                Vec p[4];for(int i=0;i<4;++i)p[i]=position(coords,cavity.get_cavity_vertex(c,i));
                // Cavity boundary ordering is authoritative after migration.
                bool good=policy.ref.accepts(p[0],p[1],p[3],region,policy.limit,policy.cosine)&&
                          policy.ref.accepts(p[1],p[2],p[3],region,policy.limit,policy.cosine);
                Vec oldSamples[3]={(pa+pb)*.5,(p[0]+p[1]+p[2])*(1./3.),(p[0]+p[2]+p[3])*(1./3.)};
                for(Vec s:oldSamples)good=good&&minv(norm2(s-closest(s,p[0],p[1],p[3])),norm2(s-closest(s,p[1],p[2],p[3])))<=policy.limit*policy.limit;
                if(!good){cavity.recover(seed);edges(seed,1)=policy.stage;return;}
                auto diagonal=cavity.add_edge(cavity.get_cavity_vertex(c,1),cavity.get_cavity_vertex(c,3));
                if(diagonal.is_valid()){
                    edges(diagonal.get_edge_handle(),0)=region;edges(diagonal.get_edge_handle(),1)=policy.stage;
                    auto f=cavity.add_face(cavity.get_cavity_edge(c,0),diagonal,cavity.get_cavity_edge(c,3));if(f.is_valid())labels(f)=region;
                    f=cavity.add_face(cavity.get_cavity_edge(c,1),cavity.get_cavity_edge(c,2),diagonal.get_flip_dedge());if(f.is_valid())labels(f)=region;
                }
            }else{
                Vec proposal=(pa+pb)*.5,q;int hit;
                if(!policy.ref.nearest(proposal,region,policy.limit,q,hit)){cavity.recover(seed);edges(seed,1)=policy.stage;return;}
                proposal=q;bool good=true;double da=1e100,db=1e100,dm=1e100;
                for(int i=0;i<n;++i){Vec a=position(coords,cavity.get_cavity_vertex(c,i)),b=position(coords,cavity.get_cavity_vertex(c,(i+1)%n));
                    good=good&&policy.ref.accepts(proposal,a,b,region,policy.limit,policy.cosine);
                    if constexpr(Kind==2)good=good&&norm2(a-proposal)<=h*h*16./9.&&quality(proposal,a,b)>1e-5;
                    da=minv(da,norm2(pa-closest(pa,proposal,a,b)));db=minv(db,norm2(pb-closest(pb,proposal,a,b)));
                    Vec m=(pa+pb)*.5;dm=minv(dm,norm2(m-closest(m,proposal,a,b)));
                }
                if constexpr(Kind==2)good=good&&maxv(da,maxv(db,dm))<=policy.limit*policy.limit;
                if(!good){cavity.recover(seed);edges(seed,1)=policy.stage;return;}
                auto v=cavity.add_vertex();if(v.is_valid()){
                    position(coords,v,proposal);sizes(v)=h;info(v,0)=region;info(v,1)=-1;info(v,2)=-1;
                    auto e0=cavity.add_edge(v,cavity.get_cavity_vertex(c,0));const auto first=e0;
                    if(e0.is_valid()){edges(e0.get_edge_handle(),0)=region;edges(e0.get_edge_handle(),1)=policy.stage;
                        for(int i=0;i<n;++i){auto e1=i==n-1?first.get_flip_dedge():cavity.add_edge(cavity.get_cavity_vertex(c,i+1),v);
                            if(!e1.is_valid())break;edges(e1.get_edge_handle(),0)=region;edges(e1.get_edge_handle(),1)=policy.stage;
                            auto f=cavity.add_face(e0,cavity.get_cavity_edge(c,i),e1);if(!f.is_valid())break;labels(f)=region;e0=e1.get_flip_dedge();}
                    }
                }
            }
        });
    }
    cavity.epilogue(block);
}

// Work through incident triangles rather than sorting a vertex ring. The
// immutable snapshot and independent set permit at most one moving corner
// per triangle, including triangles spanning computational patch boundaries.
template<uint32_t Threads=256>
__global__ void relocate(Context context,Coordinates coords,Coordinates next,VInfo info,Policy policy){
    auto block=cooperative_groups::this_thread_block();ShmemAllocator shmem;
    Query<Threads> triangles(context);triangles.template prologue<Op::FV>(block,shmem);
    Query<Threads> query(context);
    query.template dispatch<Op::VF>(block,shmem,[&](VertexHandle seed,const FaceIterator& incident){
        Vec old=position(coords,seed);position(next,seed,old);
        int n=incident.size(),region=info(seed,0);
        if(info(seed,1)>=0||region<0||n<3||n>128)return;
        auto corners=[&](int i,Vec& a,Vec& b){
            auto f=triangles.template get_iterator<VertexIterator>(incident.local(i));
            // Relocation is deliberately patch-local.  A VF query can expose
            // ghost/neighbor faces at a computational-patch boundary; do not
            // dereference or optimize such a mixed one-ring here.
            for(int k=0;k<3;++k)if(f[k]==seed){
                auto va=f[(k+1)%3],vb=f[(k+2)%3];if(!va.is_valid()||!vb.is_valid())return false;
                if(va.patch_id()!=seed.patch_id()||vb.patch_id()!=seed.patch_id())return false;
                if((info(va,1)<0&&info(va,0)>=0&&info(va,2)>=info(seed,2))||
                   (info(vb,1)<0&&info(vb,0)>=0&&info(vb,2)>=info(seed,2)))return false;
                a=position(coords,va);b=position(coords,vb);return true;
            }return false;
        };
        Vec mean{},norm{};double before=1;
        for(int i=0;i<n;++i){Vec a,b;if(!corners(i,a,b))return;mean=mean+a+b;norm=norm+cross(a-old,b-old);before=minv(before,quality(old,a,b));}
        mean=mean*(.5/n);norm=norm*(1./sqrt(maxv(norm2(norm),1e-60)));Vec delta=mean-old;delta=delta-norm*dot(delta,norm);
        for(int attempt=0;attempt<4;++attempt){Vec q;int hit;Vec candidate=old+delta*(.5/(1<<attempt));
            if(!policy.ref.nearest(candidate,region,policy.limit,q,hit))continue;
            double after=1,reverse=1e100;bool valid=true;
            for(int i=0;i<n;++i){Vec a,b;if(!corners(i,a,b)){valid=false;break;}after=minv(after,quality(q,a,b));valid=valid&&policy.ref.accepts(q,a,b,region,policy.limit,policy.cosine);reverse=minv(reverse,norm2(old-closest(old,q,a,b)));}
            if(valid&&after>before*1.0001+1e-12&&reverse<=policy.limit*policy.limit){position(next,seed,q);break;}
        }
    });
}
}
