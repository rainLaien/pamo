// Native local surface remeshing. No primitive classification or parameter charts.
namespace GenericSurface {
using Tri=std::array<int,3>;
double angle(const std::array<Point3,3>&p){double smallest=180;
  for(int i=0;i<3;++i){const auto a=Sub(ToVec(p[(i+1)%3]),ToVec(p[i])),b=Sub(ToVec(p[(i+2)%3]),ToVec(p[i]));
    const double d=Norm(a)*Norm(b);if(!(d>0))return 0;
    smallest=std::min(smallest,std::atan2(Norm(Cross(a,b)),Dot(a,b))*180/std::acos(-1.0));}return smallest;
}
#include "GenericAnalyticGuide.h"
struct Engine {
  const NativeRemeshConfig &C;
  std::vector<Point3> P,Original;
  std::vector<Tri> F,OriginalFaces;
  std::vector<Vec3> N,OriginalNormals;
  std::vector<unsigned char> Alive,Fixed,Rejected;
  std::vector<int> Hints;
  std::vector<double> Size;
  std::vector<std::vector<int>> Incident;
  std::unordered_map<EdgeKey,EdgeRecord> Edges;
  std::unordered_map<EdgeKey,int> Features;
  std::vector<std::pair<int,int>> SourceFeatures;
  std::unique_ptr<SurfaceIndex> Reference;
  std::unique_ptr<AnalyticGuide> Guide;
  std::size_t Splits=0,Collapses=0,Flips=0,Moves=0,GeometryRejects=0,TopologyRejects=0;
  double Cosine;
  Engine(const MeshTopology &mesh,const NativeRemeshConfig &config):C(config){
    Cosine=std::cos(C.MaximumNormalDeviationDegrees*std::acos(-1.0)/180);
    for(const auto &v:mesh.getVertices())P.push_back(v.Position);
    for(const auto &f:mesh.getTriangles())F.push_back(f.VertexIds);
    Original=P;OriginalFaces=F;Incident.resize(P.size());Fixed.assign(P.size(),0);Size.assign(P.size(),C.TargetEdgeLength);
    Alive.assign(F.size(),1);Rejected.assign(F.size(),0);Hints.resize(F.size());N.resize(F.size());
    Edges.reserve(F.size()*2);
    for(int f=0;f<int(F.size());++f){Hints[f]=f;N[f]=normal(F[f]);attach(f);}OriginalNormals=N;
    const double featureCos=std::cos(C.GenericFeatureAngleDegrees*std::acos(-1.0)/180);
    for(const auto &entry:Edges){const auto &e=entry.second;
      if(e.FaceCount!=2 || Dot(N[e.Faces[0]],N[e.Faces[1]])<featureCos){
        Features[entry.first]=int(SourceFeatures.size());SourceFeatures.push_back({e.A,e.B});Fixed[e.A]=Fixed[e.B]=1;
        if(e.FaceCount>2)for(int v:{e.A,e.B})for(int f:Incident[v])for(int w:F[f])Fixed[w]=1;
      }else{
        const double turn=std::acos(std::clamp(Dot(N[e.Faces[0]],N[e.Faces[1]]),-1.0,1.0));
        if(turn<.01)continue;
        double width=0;for(int f:e.Faces){const auto &t=F[f];width+=Norm(Cross(Sub(ToVec(P[t[1]]),ToVec(P[t[0]])),Sub(ToVec(P[t[2]]),ToVec(P[t[0]]))));}
        const double radius=width/std::max(1e-20,Distance(P[e.A],P[e.B])*turn);
        const double h=std::max(C.MaximumDeviation,std::min({C.TargetEdgeLength,std::sqrt(4*radius*C.MaximumDeviation),2*radius*C.MaximumNormalDeviationDegrees*std::acos(-1.0)/180}));
        for(int f:e.Faces)for(int v:F[f])Size[v]=std::min(Size[v],h);
      }
    }
    if(C.GenericAnalyticGuides){
      Guide=std::make_unique<AnalyticGuide>(mesh,C,Original,OriginalFaces,OriginalNormals,Edges,Features);
      // Only vertices whose entire incident support agrees receive analytic sizing.
      // No guide boundary is added to Features or Fixed.
      for(std::size_t v=0;v<P.size();++v){if(Incident[v].empty())continue;
        const int model=Guide->FaceModel[Incident[v][0]];if(model<0)continue;
        bool agrees=true;for(int id:Incident[v])if(Guide->FaceModel[id]!=model){agrees=false;break;}
        if(!agrees)continue;const double r=Guide->radius(Guide->Models[model],P[v]);
        Size[v]=std::max(C.MaximumDeviation,std::min({C.TargetEdgeLength,std::sqrt(4*r*C.MaximumDeviation),2*r*C.MaximumNormalDeviationDegrees*std::acos(-1.0)/180}));
      }
    }
    // A spatial growth bound avoids transmitting a short edge's size globally.
    for(int pass=0;pass<3;++pass)for(const auto &entry:Edges){const auto &e=entry.second;
      const double growth=Distance(P[e.A],P[e.B])*.35;
      Size[e.A]=std::min(Size[e.A],Size[e.B]+growth);Size[e.B]=std::min(Size[e.B],Size[e.A]+growth);}
    const std::vector<int> labels(F.size(),0);Reference=std::make_unique<SurfaceIndex>(Original,OriginalFaces,labels);
    auto sizes=Size;std::sort(sizes.begin(),sizes.end());
    if(!sizes.empty())std::clog<<"[Generic] local size p10="<<sizes[sizes.size()/10]<<", p50="<<sizes[sizes.size()/2]<<", p90="<<sizes[sizes.size()*9/10]<<std::endl;
  }
  std::array<Point3,3> points(const Tri &f)const{return {P[f[0]],P[f[1]],P[f[2]]};}
  Vec3 normal(const Tri &f)const{return Normalize(Cross(Sub(ToVec(P[f[1]]),ToVec(P[f[0]])),Sub(ToVec(P[f[2]]),ToVec(P[f[0]]))));}
  void attach(int id){for(int v:F[id])Incident[v].push_back(id);
    for(int k=0;k<3;++k){const int a=F[id][k],b=F[id][(k+1)%3];auto &e=Edges[Key(a,b)];e.A=std::min(a,b);e.B=std::max(a,b);e.addFace(id);}}
  void detach(int id){for(int v:F[id]){auto &list=Incident[v];list.erase(std::remove(list.begin(),list.end(),id),list.end());}
    for(int k=0;k<3;++k){auto found=Edges.find(Key(F[id][k],F[id][(k+1)%3]));auto &e=found->second;
      if(e.Faces[0]==id){e.Faces[0]=e.Faces[1];e.Faces[1]=-1;}else e.Faces[1]=-1;
      if(--e.FaceCount==0)Edges.erase(found);}}
  int append(Tri f,int hint){const int id=int(F.size());F.push_back(f);N.push_back(normal(f));Hints.push_back(hint);Alive.push_back(1);Rejected.push_back(0);attach(id);return id;}
  std::vector<int> neighbors(int v)const{std::vector<int> out;for(int f:Incident[v])for(int w:F[f])if(w!=v)out.push_back(w);
    std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;}
  bool unsafeFace(int id)const{
    for(int k=0;k<3;++k){const auto edge=Edges.find(Key(F[id][k],F[id][(k+1)%3]));
      if(edge==Edges.end()||edge->second.FaceCount>2)return true;}
    return false;
  }
  Point3 project(const Point3 &p,int &hint){
    if(Guide&&hint>=0&&Guide->FaceModel[hint]>=0){const int model=Guide->FaceModel[hint];auto q=p;int checked=hint;
      if(ProjectAnalytic(Guide->Models[model],q)&&Distance(p,q)<=C.MaximumDeviation
          &&Reference->within(q,0,C.MaximumDeviation*.5,checked)&&Guide->FaceModel[checked]==model){
        hint=checked;++Guide->Projections;return q;}
      ++Guide->Fallbacks;
    }
    if(!Reference->within(p,0,C.MaximumDeviation,hint))return p;
    const auto &f=OriginalFaces[hint];return ClosestPointOnTriangle(p,Original[f[0]],Original[f[1]],Original[f[2]]);
  }
  bool accepts(const std::array<Point3,3> &p,const Vec3 &previous,int hint){
    const auto cross=Cross(Sub(ToVec(p[1]),ToVec(p[0])),Sub(ToVec(p[2]),ToVec(p[0])));
    if(!(Norm(cross)>1e-18) || Dot(Normalize(cross),previous)<Cosine)return false;
    const auto a=ToVec(p[0]),b=ToVec(p[1]),c=ToVec(p[2]);
    const std::array<Point3,7> samples{p[0],p[1],p[2],ToPoint(Mul(Add(a,b),.5)),ToPoint(Mul(Add(b,c),.5)),ToPoint(Mul(Add(c,a),.5)),ToPoint(Mul(Add(Add(a,b),c),1.0/3))};
    // Reserve error for unsampled locations and subsequent local operations.
    for(const auto &sample:samples)if(!Reference->within(sample,0,C.MaximumDeviation*.5,hint))return false;
    return Dot(Normalize(cross),OriginalNormals[hint])>=Cosine;
  }
  bool covered(const Point3 &p,const std::vector<Tri> &faces)const{
    for(const auto &f:faces){const auto q=ClosestPointOnTriangle(p,P[f[0]],P[f[1]],P[f[2]]);
      if(Distance(p,q)<=C.MaximumDeviation*.25)return true;}return false;
  }
  bool collapse(EdgeKey key){auto found=Edges.find(key);if(found==Edges.end())return false;const auto e=found->second;
    if(e.FaceCount!=2 || Features.count(key) || (Fixed[e.A]&&Fixed[e.B]))return false;
    int keep=e.A,remove=e.B;if(Fixed[remove])std::swap(keep,remove);
    const auto a=neighbors(keep),b=neighbors(remove);std::vector<int> common;
    std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(common));
    if(common.size()!=2){++TopologyRejects;return false;}
    std::vector<int> star=Incident[keep];star.insert(star.end(),Incident[remove].begin(),Incident[remove].end());
    std::sort(star.begin(),star.end());star.erase(std::unique(star.begin(),star.end()),star.end());
    for(int id:star)if(unsafeFace(id)){++TopologyRejects;return false;}
    std::vector<Tri> proposal;std::vector<int> surviving;std::set<Tri> unique;
    for(int id:star){auto f=F[id];for(int &v:f)if(v==remove)v=keep;
      if(f[0]==f[1]||f[1]==f[2]||f[2]==f[0])continue;
      auto sorted=f;std::sort(sorted.begin(),sorted.end());if(!unique.insert(sorted).second){++TopologyRejects;return false;}
      proposal.push_back(f);surviving.push_back(id);}
    if(proposal.empty())return false;
    const auto oldKeep=P[keep],oldRemove=P[remove];int hint=Hints[star[0]];
    if(!Fixed[keep])P[keep]=project(ToPoint(Mul(Add(ToVec(oldKeep),ToVec(oldRemove)),.5)),hint);
    bool good=true;
    for(std::size_t i=0;i<proposal.size()&&good;++i)good=accepts(points(proposal[i]),N[surviving[i]],Hints[surviving[i]]);
    // Reverse local coverage protects material removed by a collapse.
    if(good)good=covered(oldKeep,proposal)&&covered(oldRemove,proposal);
    if(good)for(int id:e.Faces){const auto &f=F[id];Vec3 center{};
      for(int v:f)center=Add(center,ToVec(v==keep?oldKeep:v==remove?oldRemove:P[v]));
      if(!covered(ToPoint(Mul(center,1.0/3)),proposal)){good=false;break;}}
    if(!good){P[keep]=oldKeep;++GeometryRejects;for(int id:star)Rejected[id]|=1;return false;}
    for(int id:star){detach(id);Alive[id]=0;}
    for(std::size_t i=0;i<proposal.size();++i){const int id=surviving[i];F[id]=proposal[i];N[id]=normal(F[id]);Alive[id]=1;attach(id);}
    Size[keep]=std::min({C.TargetEdgeLength,Size[keep]+Distance(P[keep],oldKeep)*.35,Size[remove]+Distance(P[keep],oldRemove)*.35});++Collapses;return true;
  }
  bool flip(EdgeKey key){auto found=Edges.find(key);if(found==Edges.end())return false;const auto e=found->second;
    if(e.FaceCount!=2||Features.count(key))return false;
    const int f=e.Faces[0],g=e.Faces[1];int c=-1,d=-1;
    if(unsafeFace(f)||unsafeFace(g)){++TopologyRejects;return false;}
    for(int v:F[f])if(v!=e.A&&v!=e.B)c=v;for(int v:F[g])if(v!=e.A&&v!=e.B)d=v;
    if(c<0||d<0||c==d||Edges.count(Key(c,d)))return false;
    const auto cd=Sub(ToVec(P[d]),ToVec(P[c]));
    if(Dot(Cross(cd,Sub(ToVec(P[e.A]),ToVec(P[c]))),Cross(cd,Sub(ToVec(P[e.B]),ToVec(P[c]))))>=0)return false;
    Tri x{c,d,e.A},y{d,c,e.B};if(Dot(normal(x),N[f])<0)std::swap(x[0],x[1]);if(Dot(normal(y),N[g])<0)std::swap(y[0],y[1]);
    const double before=std::min(angle(points(F[f])),angle(points(F[g]))),after=std::min(angle(points(x)),angle(points(y)));
    if(after<=before+.1)return false;
    if(!accepts(points(x),N[f],Hints[f])||!accepts(points(y),N[g],Hints[g])){++GeometryRejects;Rejected[f]|=1;Rejected[g]|=1;return false;}
    detach(f);detach(g);F[f]=x;F[g]=y;N[f]=normal(x);N[g]=normal(y);attach(f);attach(g);++Flips;return true;
  }
  bool split(EdgeKey key){auto found=Edges.find(key);if(found==Edges.end())return false;const auto e=found->second;
    if(e.FaceCount>2)return false;
    for(int id:e.Faces)if(id>=0&&unsafeFace(id)){++TopologyRejects;return false;}
    const int v=int(P.size());
    const double growth=Distance(P[e.A],P[e.B])*.175;
    P.push_back(ToPoint(Mul(Add(ToVec(P[e.A]),ToVec(P[e.B])),.5)));Fixed.push_back(Features.count(key)?1:0);Size.push_back(std::min({C.TargetEdgeLength,Size[e.A]+growth,Size[e.B]+growth}));Incident.emplace_back();
    std::vector<std::pair<Tri,int>> added;
    for(int id:e.Faces)if(id>=0){const auto f=F[id];for(int k=0;k<3;++k)if(Key(f[k],f[(k+1)%3])==key){
      added.push_back({{f[k],v,f[(k+2)%3]},Hints[id]});added.push_back({{v,f[(k+1)%3],f[(k+2)%3]},Hints[id]});break;}}
    for(int id:e.Faces)if(id>=0){detach(id);Alive[id]=0;}
    for(const auto &item:added)append(item.first,item.second);
    auto feature=Features.find(key);if(feature!=Features.end()){const int origin=feature->second;Features.erase(feature);Features[Key(e.A,v)]=origin;Features[Key(v,e.B)]=origin;}
    ++Splits;return true;
  }
  bool smooth(int v){if(Fixed[v]||Incident[v].empty())return false;const auto adjacent=neighbors(v);if(adjacent.size()<3)return false;
    Vec3 average{},normalSum{};for(int w:adjacent)average=Add(average,ToVec(P[w]));average=Mul(average,1.0/adjacent.size());
    for(int id:Incident[v])normalSum=Add(normalSum,N[id]);normalSum=Normalize(normalSum);
    auto delta=Sub(average,ToVec(P[v]));delta=Sub(delta,Mul(normalSum,Dot(delta,normalSum)));delta=Mul(delta,.35);
    const double length=Norm(delta);if(length<1e-10)return false;if(length>Size[v]*.25)delta=Mul(delta,Size[v]*.25/length);
    const auto old=P[v];int hint=Hints[Incident[v][0]];
    double before=180,oldScore=0;for(int id:Incident[v]){const double a=angle(points(F[id]));before=std::min(before,a);oldScore+=std::min(a,28.0);}
    P[v]=project(ToPoint(Add(ToVec(old),delta)),hint);double after=180,newScore=0;std::vector<Tri> ring;
    for(int id:Incident[v]){ring.push_back(F[id]);const double a=angle(points(F[id]));after=std::min(after,a);newScore+=std::min(a,28.0);}
    bool good=after>=std::min(before,10.0)*.8 && newScore>oldScore+1e-5;
    if(good)for(int id:Incident[v])if(!accepts(points(F[id]),N[id],Hints[id])){good=false;break;}
    good=good && covered(old,ring);
    if(!good){P[v]=old;return false;}for(int id:Incident[v])N[id]=normal(F[id]);++Moves;return true;
  }
  void run(){for(int iteration=0;iteration<C.GenericRemeshIterations;++iteration){const auto start=std::chrono::steady_clock::now();
    const auto s=Splits,c=Collapses,f=Flips,m=Moves;std::vector<std::pair<double,EdgeKey>> shortEdges;
    for(const auto &entry:Edges){const auto &e=entry.second;const double length=Distance(P[e.A],P[e.B]);
      if(length<.8*std::min(Size[e.A],Size[e.B]))shortEdges.push_back({length,entry.first});}
    std::sort(shortEdges.begin(),shortEdges.end());for(const auto &entry:shortEdges)collapse(entry.second);
    std::vector<EdgeKey> keys;keys.reserve(Edges.size());for(const auto &entry:Edges)keys.push_back(entry.first);
    for(auto key:keys)flip(key);
    std::vector<std::pair<double,EdgeKey>> longEdges;
    for(const auto &entry:Edges){const auto &e=entry.second;const double ratio=Distance(P[e.A],P[e.B])/std::min(Size[e.A],Size[e.B]);
      if(ratio>4.0/3)longEdges.push_back({-ratio,entry.first});}
    std::sort(longEdges.begin(),longEdges.end());const std::size_t budget=200000;
    const bool cleanup=iteration>=std::max(1,C.GenericRemeshIterations-2);
    if(!cleanup)for(std::size_t i=0;i<std::min(budget,longEdges.size());++i)split(longEdges[i].second);
    keys.clear();for(const auto &entry:Edges)keys.push_back(entry.first);for(auto key:keys)flip(key);
    for(int v=0;v<int(P.size());++v)smooth(v);
    std::clog<<"[Generic] iteration="<<iteration+1<<", collapse="<<Collapses-c<<", flip="<<Flips-f<<", split="<<Splits-s<<", moved="<<Moves-m
      <<", cleanup_only="<<cleanup<<", remaining_long_edges="<<longEdges.size()<<", split_budget_limited="<<(!cleanup && longEdges.size()>budget)<<", seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
    if(Splits==s&&Collapses==c&&Flips==f&&Moves==m)break;
  }}
};
struct QualityReport {std::size_t Faces=0,Small5=0,Small10=0,Small20=0,Small28=0,FeatureSmall28=0,OrdinarySmall28=0,Degenerate=0;double Minimum=180,Sum=0,Area=0,SmallArea=0;};
QualityReport quality(const std::vector<Point3> &p,const std::vector<Tri>&f,const std::vector<unsigned char>&fixed){QualityReport r;r.Faces=f.size();
  for(const auto &t:f){const double a=angle({p[t[0]],p[t[1]],p[t[2]]});r.Minimum=std::min(r.Minimum,a);r.Sum+=a;
    const double area=.5*Norm(Cross(Sub(ToVec(p[t[1]]),ToVec(p[t[0]])),Sub(ToVec(p[t[2]]),ToVec(p[t[0]]))));r.Area+=area;r.Degenerate+=area==0;
    r.Small5+=a<5;r.Small10+=a<10;r.Small20+=a<20;r.Small28+=a<28;
    if(a<28){r.SmallArea+=area;if(fixed[t[0]]||fixed[t[1]]||fixed[t[2]])++r.FeatureSmall28;else ++r.OrdinarySmall28;}}
  return r;
}
void jsonQuality(std::ostream &o,const QualityReport&r){o<<"{\"faces\":"<<r.Faces<<",\"minimum_angle\":"<<r.Minimum<<",\"mean_minimum_angle\":"<<r.Sum/std::max<std::size_t>(1,r.Faces)
  <<",\"below_5\":"<<r.Small5<<",\"below_10\":"<<r.Small10<<",\"below_20\":"<<r.Small20<<",\"below_28\":"<<r.Small28
  <<",\"below_28_percent\":"<<100.0*r.Small28/std::max<std::size_t>(1,r.Faces)<<",\"below_28_area_percent\":"<<100*r.SmallArea/std::max(1e-30,r.Area)
  <<",\"feature_adjacent_below_28\":"<<r.FeatureSmall28<<",\"ordinary_below_28\":"<<r.OrdinarySmall28<<",\"degenerate\":"<<r.Degenerate<<"}";}
bool execute(const TriangleSoup &soup,const NativeRemeshConfig &config,const std::filesystem::path &directory,std::string &error){
  using Clock=std::chrono::steady_clock;const auto start=Clock::now();MeshTopology topology;
  if(std::filesystem::exists(directory/"generic_result.ply")||std::filesystem::exists(directory/"generic_candidate.ply")){
    error="generic output already exists; use a new output directory";return false;}
  if(!topology.build(soup)){error="generic topology build failed";return false;}
  if(topology.getTriangles().empty()){error="generic input contains no usable triangles";return false;}
  const auto prepared=Clock::now();Engine engine(topology,config);const auto initial=quality(engine.P,engine.F,engine.Fixed);const auto indexed=Clock::now();
  std::clog<<"[Generic] classification=disabled, faces="<<engine.F.size()<<", features="<<engine.Features.size()<<", topology_s="<<std::chrono::duration<double>(prepared-start).count()<<", index_s="<<std::chrono::duration<double>(indexed-prepared).count()<<std::endl;
  engine.run();const auto remeshed=Clock::now();std::vector<Tri> output;std::vector<unsigned char> rejected;
  for(std::size_t i=0;i<engine.F.size();++i)if(engine.Alive[i]){output.push_back(engine.F[i]);rejected.push_back(engine.Rejected[i]);}
  const auto final=quality(engine.P,output,engine.Fixed);
  std::filesystem::create_directories(directory);
  // Never overwrite an earlier result: each invocation owns a fresh output directory.
  if(std::filesystem::exists(directory/"generic_result.ply")||std::filesystem::exists(directory/"generic_candidate.ply")){
    error="generic output already exists; use a new output directory";return false;}
  std::ofstream ply(directory/"generic_candidate.ply",std::ios::binary);
  if(!ply){error="cannot create generic PLY";return false;}
  std::vector<int> mapping(engine.P.size(),-1);int count=0;for(const auto &f:output)for(int v:f)if(mapping[v]<0)mapping[v]=count++;
  std::vector<Point3> compact(count);for(std::size_t v=0;v<mapping.size();++v)if(mapping[v]>=0)compact[mapping[v]]=engine.P[v];
  ply<<"ply\nformat binary_little_endian 1.0\ncomment issue_flags 1=angle_below_28 2=feature_adjacent 4=geometry_rejection_history\nelement vertex "<<count<<"\nproperty double x\nproperty double y\nproperty double z\nelement face "<<output.size()<<"\nproperty list uchar int vertex_indices\nproperty float minimum_angle\nproperty uchar issue_flags\nend_header\n";
  for(const auto &p:compact){const double values[]{p.X(),p.Y(),p.Z()};ply.write(reinterpret_cast<const char*>(values),sizeof(values));}
  for(std::size_t i=0;i<output.size();++i){const auto &f=output[i];const unsigned char n=3;const int ids[]{mapping[f[0]],mapping[f[1]],mapping[f[2]]};const float a=float(angle({engine.P[f[0]],engine.P[f[1]],engine.P[f[2]]}));
    const unsigned char flags=(a<28?1:0)|((engine.Fixed[f[0]]||engine.Fixed[f[1]]||engine.Fixed[f[2]])?2:0)|(rejected[i]?4:0);
    ply.write(reinterpret_cast<const char*>(&n),1);ply.write(reinterpret_cast<const char*>(ids),sizeof(ids));ply.write(reinterpret_cast<const char*>(&a),4);ply.write(reinterpret_cast<const char*>(&flags),1);}
  ply.close();if(!ply){error="PLY write failed";return false;}
  // Deterministic bidirectional sample check. This is not a Hausdorff proof.
  const std::vector<int> labels(output.size(),0);SurfaceIndex resultIndex(engine.P,output,labels);
  double forward=0,reverse=0;std::size_t samples=0;
  const auto measure=[&](const std::vector<Point3>&p,const std::vector<Tri>&faces,const SurfaceIndex&index,double &maximum){
    const std::size_t step=std::max<std::size_t>(1,faces.size()/10000);
    for(std::size_t i=0;i<faces.size();i+=step){const auto &f=faces[i];const auto center=ToPoint(Mul(Add(Add(ToVec(p[f[0]]),ToVec(p[f[1]])),ToVec(p[f[2]])),1.0/3));
      for(const Point3 &q:std::array<Point3,2>{p[f[0]],center}){Point3 hit;double d=0;
        if(index.closest(q,0,hit,d))maximum=std::max(maximum,d);else throw std::runtime_error("generic audit reference query failed");++samples;}}
  };
  measure(engine.P,output,*engine.Reference,forward);measure(engine.Original,engine.OriginalFaces,resultIndex,reverse);
  std::vector<double> lengths(engine.SourceFeatures.size(),0);double featureDeviation=0;
  for(const auto &entry:engine.Features){const int a=KeyFirst(entry.first),b=KeySecond(entry.first);const auto source=engine.SourceFeatures[entry.second];
    const auto origin=ToVec(engine.Original[source.first]),direction=Sub(ToVec(engine.Original[source.second]),origin);const double l2=Dot(direction,direction);
    for(int v:{a,b}){const double t=l2>0?std::clamp(Dot(Sub(ToVec(engine.P[v]),origin),direction)/l2,0.0,1.0):0;featureDeviation=std::max(featureDeviation,Distance(engine.P[v],ToPoint(Add(origin,Mul(direction,t)))));}
    lengths[entry.second]+=Distance(engine.P[a],engine.P[b]);
  }
  std::size_t preserved=0;for(std::size_t i=0;i<lengths.size();++i){const auto s=engine.SourceFeatures[i];if(std::abs(lengths[i]-Distance(engine.Original[s.first],engine.Original[s.second]))<1e-7)++preserved;}
  const bool auditPassed=std::max(forward,reverse)<=config.MaximumDeviation&&final.Degenerate==0
    &&preserved==lengths.size()&&featureDeviation<=1e-7;
  std::size_t boundaryEdges=0,nonmanifoldEdges=0;
  for(const auto &entry:engine.Edges){boundaryEdges+=entry.second.FaceCount==1;nonmanifoldEdges+=entry.second.FaceCount>2;}
  const auto done=Clock::now();std::ofstream report(directory/"generic_report.json");report<<std::setprecision(12)<<"{\n\"input\":";jsonQuality(report,initial);report<<",\n\"output\":";jsonQuality(report,final);
  report<<",\n\"target\":"<<config.TargetEdgeLength<<",\"distance_limit\":"<<config.MaximumDeviation<<",\"normal_limit_deg\":"<<config.MaximumNormalDeviationDegrees
    <<",\"topology_seconds\":"<<std::chrono::duration<double>(prepared-start).count()<<",\"setup_seconds\":"<<std::chrono::duration<double>(indexed-prepared).count()
    <<",\"remesh_seconds\":"<<std::chrono::duration<double>(remeshed-indexed).count()<<",\"export_and_audit_seconds\":"<<std::chrono::duration<double>(done-remeshed).count()
    <<",\"total_seconds_without_stl_read\":"<<std::chrono::duration<double>(done-start).count()<<",\"splits\":"<<engine.Splits<<",\"collapses\":"<<engine.Collapses<<",\"flips\":"<<engine.Flips<<",\"moves\":"<<engine.Moves
    <<",\"geometry_rejections\":"<<engine.GeometryRejects<<",\"topology_rejections\":"<<engine.TopologyRejects<<",\"sampled_forward_distance\":"<<forward<<",\"sampled_reverse_distance\":"<<reverse<<",\"distance_samples\":"<<samples
    <<",\"source_feature_segments\":"<<lengths.size()<<",\"preserved_feature_segments\":"<<preserved<<",\"maximum_feature_vertex_distance\":"<<featureDeviation
    <<",\"analytic_guides_enabled\":"<<(config.GenericAnalyticGuides?"true":"false")
    <<",\"analytic_fit_seconds\":"<<(engine.Guide?engine.Guide->Seconds:0)
    <<",\"analytic_cylinders\":"<<(engine.Guide?engine.Guide->Cylinders:0)<<",\"analytic_cones\":"<<(engine.Guide?engine.Guide->Cones:0)
    <<",\"analytic_supported_faces\":"<<(engine.Guide?engine.Guide->SupportedFaces:0)
    <<",\"analytic_projection_proposals\":"<<(engine.Guide?engine.Guide->Projections:0)<<",\"analytic_projection_fallbacks\":"<<(engine.Guide?engine.Guide->Fallbacks:0)
    <<",\"sampled_shape_audit_passed\":"<<(auditPassed?"true":"false")<<",\"quality_target_reached\":"<<(final.OrdinarySmall28==0?"true":"false")
    <<",\"output_boundary_edges\":"<<boundaryEdges<<",\"output_nonmanifold_edges\":"<<nonmanifoldEdges
    <<",\"constraint_note\":\"feature adjacency is not proof that a small angle is unavoidable\",\"distance_note\":\"deterministic vertex/centroid samples, not a certified Hausdorff bound\"}\n";
  std::clog<<"[Generic] complete: faces="<<initial.Faces<<" -> "<<final.Faces<<", below28="<<100.0*initial.Small28/initial.Faces<<"% -> "<<100.0*final.Small28/final.Faces<<"%, sampled_deviation="<<std::max(forward,reverse)<<", features="<<preserved<<'/'<<lengths.size()<<", total_s="<<std::chrono::duration<double>(done-start).count()<<std::endl;
  report.close();if(!report){error="generic report write failed";return false;}
  if(!auditPassed){error="generic sampled shape audit failed; see generic_report.json and generic_candidate.ply";return false;}
  std::filesystem::rename(directory/"generic_candidate.ply",directory/"generic_result.ply");
  if(final.OrdinarySmall28)std::clog<<"[Generic] quality target NOT reached: ordinary_below28="<<final.OrdinarySmall28<<"; inspect minimum_angle and issue_flags in PLY"<<std::endl;
  return true;
}
}
