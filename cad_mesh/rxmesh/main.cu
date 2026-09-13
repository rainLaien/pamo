#include "input.h"
#include "kernels.cuh"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
using namespace pamo_rx;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static void check(cudaError_t e){if(e!=cudaSuccess){std::cerr<<"[RX] CUDA operation failed: "<<cudaGetErrorString(e)<<'\n';throw std::runtime_error(cudaGetErrorString(e));}}
template<class T> struct Device {
    T* p=nullptr;explicit Device(const std::vector<T>& v){check(cudaMalloc((void**)&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;Device& operator=(const Device&)=delete;
};
static Options parse(int argc,char**argv){
    if(argc<3)throw std::runtime_error("usage: cad_mesh_rxmesh INPUT.stl OUTPUT_DIRECTORY [...options] [--skip-audit]");
    Options o;o.input=argv[1];o.output=argv[2];
    for(int i=3;i<argc;++i){std::string k=argv[i];if(k=="--partition-snapshot"){o.snapshot=true;continue;}if(k=="--segment"){o.segment=true;continue;}if(k=="--skip-audit"){o.skipAudit=true;continue;}if(k=="--skip-relocation"){o.skipRelocation=true;continue;}if(k=="--skip-collapse"){o.skipCollapse=true;continue;}
        if(++i==argc)throw std::runtime_error("missing value for "+k);size_t used=0;double v=std::stod(argv[i],&used);
        if(used!=std::string(argv[i]).size()||!std::isfinite(v)||v<=0)throw std::runtime_error("invalid positive value for "+k);
        if(k=="--target-edge-ratio"&&v<=1)o.ratio=v;
        else if(k=="--target-edge-length")o.target=v;
        else if(k=="--max-deviation")o.deviation=v;
        else if(k=="--normal-degrees"&&v<90)o.normalDegrees=v;
        else if(k=="--feature-angle"&&v<180)o.featureDegrees=v;
        else if(k=="--iterations"&&v<=30&&v==floor(v))o.iterations=int(v);
        else if(k=="--max-faces"&&v<=20000000&&v==floor(v))o.maxFaces=int(v);
        else throw std::runtime_error("unknown option or out-of-range value: "+k);
    }
    if(o.snapshot&&o.segment)throw std::runtime_error("choose snapshot or STL segmentation");
    if(std::filesystem::exists(o.output)&&!std::filesystem::is_empty(o.output))throw std::runtime_error("output directory must be empty or new");
    return o;
}
int main(int argc,char**argv){
    const auto started=Clock::now();
    try{
        Options o=parse(argc,argv);rxmesh::rx_init(0);Input in=prepare(o);double prep=seconds(started);
        if(in.faces.size()>size_t(o.maxFaces))throw std::runtime_error("input exceeds --max-faces");
        Device<RefTriangle> rt(in.reference);Device<BvhNode> bn(in.nodes);Device<int> roots(in.roots);
        Device<RegionPlane> planes(in.planes);
        Reference reference{rt.p,bn.p,roots.p,int(in.roots.size()),planes.p};
        // Tiny seeds still need room for several independently sliced patches.
        float patchReserve=std::max(3.0f,64.0f/std::max(1.0f,float(in.faces.size())/256.0f));
        rxmesh::RXMeshDynamic rx(in.faces,"",256,3.5f,patchReserve);
        auto coords=rx.add_vertex_attribute<double>(in.coordinates,"reference_coords");
        auto scratch=rx.add_vertex_attribute<double>("relocation_scratch",3);
        auto sizes=rx.add_vertex_attribute<double>(in.sizes,"target_size");
        std::vector<std::vector<int>> metadata(in.points.size());
        for(int i=0;i<int(metadata.size());++i)metadata[i]={in.vertexRegion[i][0],in.fixed[i][0]?i:-1,-1};
        auto info=rx.add_vertex_attribute<int>(metadata,"source_region_anchor_epoch");
        auto labels=rx.add_face_attribute<int>(in.labels,"source_patch_id");
        auto edges=rx.add_edge_attribute<int>("region_epoch",2);
        auto e=*edges;auto fl=*labels;
        rx.for_each<rxmesh::Op::EF,256>([=] __device__(rxmesh::EdgeHandle h,const rxmesh::FaceIterator& fs) mutable {
            int r=-1;if(fs.size()==2&&fs[0].is_valid()&&fs[1].is_valid()&&fl(fs[0])==fl(fs[1]))r=fl(fs[0]);e(h,0)=r;e(h,1)=-1;
        });
        // A crease may end inside a connected region; labels alone cannot
        // identify every protected edge.
        std::vector<uint64_t> hard;
        for(auto pair:in.constraints){int a=std::min(pair[0],pair[1]),b=std::max(pair[0],pair[1]);hard.push_back((uint64_t(a)<<32)|uint32_t(b));}
        std::sort(hard.begin(),hard.end());
        if(!hard.empty()){
            Device<uint64_t> protectedKeys(hard);const auto keys=protectedKeys.p;const int count=int(hard.size());auto vi=*info;
            rx.for_each<rxmesh::Op::EV,256>([=] __device__(rxmesh::EdgeHandle h,const rxmesh::VertexIterator& vs) mutable {
                int a=vi(vs[0],1),b=vi(vs[1],1);if(a<0||b<0)return;if(a>b){int t=a;a=b;b=t;}
                uint64_t k=(uint64_t(a)<<32)|uint32_t(b);int lo=0,hi=count;
                while(lo<hi){int m=(lo+hi)/2;if(keys[m]<k)lo=m+1;else hi=m;}
                if(lo<count&&keys[lo]==k)e(h,0)=-1;
            });
            check(cudaDeviceSynchronize());
        }
        check(cudaDeviceSynchronize());
        if(!rx.validate())throw std::runtime_error("RXMesh input topology validation failed");
        std::clog<<"[RX] gpu_setup_s="<<seconds(started)-prep<<", compute_patches="<<rx.get_num_patches()<<'\n';
        const auto remeshStart=Clock::now();int stage=0;
        const bool trace=std::getenv("PAMO_RX_TRACE")!=nullptr;
        auto run=[&](auto kernel,std::vector<rxmesh::Op> ops,bool concurrent){
            const auto began=Clock::now();
            Policy policy{reference,in.deviation*.5,cos(o.normalDegrees*acos(-1.)/180),++stage};
            rxmesh::LaunchBox<256> lb;
            rx.reset_scheduler();int launches=0;
            while(!rx.is_queue_empty()){
                if(++launches>2048)throw std::runtime_error("RXMesh scheduler made insufficient progress");
                rx.update_launch_box(ops,lb,(const void*)kernel,true,false,false,concurrent);
                auto ctx=rx.get_context();auto c=*coords;auto s=*sizes;auto v=*info;auto f=*labels;auto e=*edges;
                void* args[]={&ctx,&c,&s,&v,&f,&e,&policy};
                if(trace)std::clog<<"[RX trace] stage="<<stage<<" launch="<<launches<<" patches="<<rx.get_num_patches()<<" kernel\n";
                check(cudaLaunchKernel((const void*)kernel,dim3(lb.blocks),dim3(lb.num_threads),args,lb.smem_bytes_dyn,nullptr));
                check(cudaDeviceSynchronize());if(trace)std::clog<<"[RX trace] cleanup\n";
                rx.cleanup();check(cudaDeviceSynchronize());
                // Upstream slicing does not safely handle an exhausted patch
                // pool. Reserve room for the worst case of one split per patch.
                if(2*rx.get_num_patches(true)>rx.get_max_num_patches())throw std::runtime_error("RXMesh patch pool exhausted before slicing; reduce refinement or increase seed density");
                if(trace)std::clog<<"[RX trace] slice\n";
                rx.slice_patches(*coords,*sizes,*info,*labels,*edges,*scratch);
                rx.cleanup();check(cudaDeviceSynchronize());
                if(rx.get_num_faces(true)>uint32_t(o.maxFaces))throw std::runtime_error("GPU growth exceeds --max-faces");
                if(trace)std::clog<<"[RX trace] complete faces="<<rx.get_num_faces()<<'\n';
            }
            std::clog<<"[RX] stage="<<stage<<", launches="<<launches<<", faces="<<rx.get_num_faces()<<", seconds="<<seconds(began)<<'\n';
        };
        for(int iteration=0;iteration<o.iterations;++iteration){const auto pass=Clock::now();
            // Refinement is driven by the local sizing field, not source edges.
            // Keep scheduler execution conservative until patch-local edits
            // are proven stable on all input topologies.
            if(!o.skipCollapse)run(edit<2>,{rxmesh::Op::VV,rxmesh::Op::EVDiamond},false);
            run(edit<1>,{rxmesh::Op::VV,rxmesh::Op::EVDiamond},false);
            run(edit<0>,{rxmesh::Op::EVDiamond},false);
            run(edit<1>,{rxmesh::Op::VV,rxmesh::Op::EVDiamond},false);
            if(trace){rx.update_host();if(!rx.validate())throw std::runtime_error("topology failed before relocation");}
            for(int smooth=0; smooth<3 && !o.skipRelocation; ++smooth){
                const auto smoothStart=Clock::now();
                auto vi=*info;uint32_t salt=uint32_t(++stage)*747796405u;
                rx.for_each_vertex(rxmesh::DEVICE,[=] __device__(rxmesh::VertexHandle v) mutable {
                    uint64_t id=v.unique_id();uint32_t h=uint32_t(id)^uint32_t(id>>32)^salt;
                    h=(h^(h>>16))*2246822519u;h=(h^(h>>13))*3266489917u;vi(v,2)=int((h^(h>>16))&0x7fffffffu);
                });
                Policy policy{reference,in.deviation*.5,cos(o.normalDegrees*acos(-1.)/180),stage};rxmesh::LaunchBox<256> lb;
                rx.update_launch_box({rxmesh::Op::FV,rxmesh::Op::VF},lb,(const void*)relocate<>,false,false,false,true);
                relocate<><<<lb.blocks,lb.num_threads,lb.smem_bytes_dyn>>>(rx.get_context(),*coords,*scratch,*info,policy);
                check(cudaDeviceSynchronize());std::swap(coords,scratch);
                std::clog<<"[RX] relocation="<<smooth+1<<", seconds="<<seconds(smoothStart)<<'\n';
            }
            std::clog<<"[RX] iteration="<<iteration+1<<", faces="<<rx.get_num_faces()<<", seconds="<<seconds(pass)<<'\n';
        }
        double gpu=seconds(remeshStart);rx.update_host();bool topologyValid=rx.validate();
        if(!topologyValid)throw std::runtime_error("RXMesh output topology validation failed");
        coords->move(rxmesh::DEVICE,rxmesh::HOST);info->move(rxmesh::DEVICE,rxmesh::HOST);labels->move(rxmesh::DEVICE,rxmesh::HOST);
        std::vector<Vec> points(rx.get_num_vertices());std::vector<int> anchors(points.size()),outLabels(rx.get_num_faces());
        rx.for_each_vertex(rxmesh::HOST,[&](rxmesh::VertexHandle v){size_t id=rx.linear_id(v);points[id]={(*coords)(v,0),(*coords)(v,1),(*coords)(v,2)};anchors[id]=(*info)(v,1);},nullptr,false);
        rx.for_each_face(rxmesh::HOST,[&](rxmesh::FaceHandle f){outLabels[rx.linear_id(f)]=(*labels)(f);},nullptr,false);
        std::vector<uint32_t> raw(3*rx.get_num_faces());rx.create_face_list(raw.data(),false);
        std::vector<std::array<int,3>> faces(rx.get_num_faces());for(size_t i=0;i<faces.size();++i)for(int k=0;k<3;++k)faces[i][k]=int(raw[3*i+k]);
        writeResult(o,in,points,faces,outLabels,anchors,seconds(started),gpu,prep,topologyValid);
        std::clog<<"[RX] total_wall_s="<<seconds(started)<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<"[RX] failed: "<<e.what()<<"; wall_s="<<seconds(started)<<'\n';return 1;}
}
