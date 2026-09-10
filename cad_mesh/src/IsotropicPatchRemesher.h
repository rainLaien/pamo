// SPDX-License-Identifier: GPL-2.0-or-later
// Algorithm port from VCGLib isotropic_remeshing.h, refine.h, triangle3.h,
// face/topology.h and smooth.h. Original copyright (C) 2004-2017
// Visual Computing Lab, ISTI - Italian National Research Council.
// Distributed without warranty under GNU GPL version 2 or (at your option) later.
// Native indexed storage; no VCGLib types or remeshing API calls.
namespace NativeVcgRules {
using Triangle=std::array<int,3>;
double Quality(const Point3 &a,const Point3 &b,const Point3 &c){
  const auto u=Sub(ToVec(b),ToVec(a)),v=Sub(ToVec(c),ToVec(a)),w=Sub(ToVec(c),ToVec(b));
  const double longest=std::max({Dot(u,u),Dot(v,v),Dot(w,w)});
  return longest>0?Norm(Cross(u,v))/longest:0;
}
double Radii(const Point3 &p,const Point3 &q,const Point3 &r){
  const double a=Distance(p,q),b=Distance(p,r),c=Distance(q,r),s=(a+b+c)*.5;
  const double area=s*(a+b-s)*(a+c-s)*(b+c-s);
  return area>0?8*area/(a*b*c*s):0;
}
struct Mesh {
  std::vector<Point3> &P;std::vector<Triangle> &F;
  std::vector<unsigned char> dead;
  std::vector<std::set<int>> incident;
  std::map<EdgeKey,std::set<int>> edges;
  std::unordered_set<EdgeKey> hard,crease;
  std::unordered_set<int> fixed;
  Mesh(std::vector<Point3>&p,std::vector<Triangle>&f,const std::unordered_set<EdgeKey>&boundary):P(p),F(f),hard(boundary){
    for(auto key:hard){fixed.insert(KeyFirst(key));fixed.insert(KeySecond(key));}
    rebuild();
  }
  void attach(int id){for(int v:F[id])incident[v].insert(id);
    for(int k=0;k<3;++k)edges[Key(F[id][k],F[id][(k+1)%3])].insert(id);}
  void detach(int id){for(int v:F[id])incident[v].erase(id);
    for(int k=0;k<3;++k){const auto key=Key(F[id][k],F[id][(k+1)%3]);auto it=edges.find(key);
      if(it!=edges.end()){it->second.erase(id);if(it->second.empty())edges.erase(it);}}}
  void rebuild(){edges.clear();incident.assign(P.size(),{});dead.assign(F.size(),0);for(int i=0;i<int(F.size());++i)attach(i);}
  std::set<int> neighbors(int v)const{std::set<int> out;for(int id:incident[v])for(int w:F[id])if(w!=v)out.insert(w);return out;}
  bool border(int v)const{for(int w:neighbors(v)){auto it=edges.find(Key(v,w));if(it->second.size()!=2)return true;}return false;}
  bool manifold(int v)const{
    if(incident[v].empty())return true;
    std::set<int> visited;std::vector<int> queue{*incident[v].begin()};visited.insert(queue[0]);
    for(std::size_t i=0;i<queue.size();++i)for(int w:F[queue[i]])if(w!=v){
      const auto &adjacent=edges.at(Key(v,w));if(adjacent.size()>2)return false;
      for(int f:adjacent)if(visited.insert(f).second)queue.push_back(f);}
    return visited.size()==incident[v].size();
  }
  bool featureVertex(int v)const{if(fixed.count(v))return true;for(int w:neighbors(v))if(crease.count(Key(v,w)))return true;return false;}
  Vec3 normal(int id)const{return TriangleNormal(P,F[id]);}
  double quality(int id)const{const auto &t=F[id];return Quality(P[t[0]],P[t[1]],P[t[2]]);}
  void compact(){std::vector<Triangle> out;for(std::size_t i=0;i<F.size();++i)if(!dead[i])out.push_back(F[i]);F.swap(out);rebuild();}
  void tag(double angle){crease=hard;const double cosine=std::cos(angle*std::acos(-1.0)/180);
    for(const auto &e:edges){if(e.second.size()!=2){crease.insert(e.first);continue;}
      auto it=e.second.begin();const int f=*it++,g=*it;
      const double d=Dot(normal(f),normal(g));const auto &a=F[f],&b=F[g];
      if(d<=cosine && d>=-.98 && Radii(P[a[0]],P[a[1]],P[a[2]])>.0001 && Radii(P[b[0]],P[b[1]],P[b[2]])>.0001)crease.insert(e.first);
    }
  }
  // One simultaneous RefineMidpoint pass, including VCGLib's 2-edge diagonal rule.
  std::size_t split(double maximum){
    std::map<EdgeKey,int> mids;
    for(std::size_t f=0;f<F.size();++f)if(!dead[f])for(int k=0;k<3;++k){const int a=F[f][k],b=F[f][(k+1)%3];const auto key=Key(a,b);
      if(hard.count(key) || mids.count(key) || edges[key].size()>2 || Distance(P[a],P[b])<=maximum)continue;
      const auto midpoint=ToPoint(Mul(Add(ToVec(P[a]),ToVec(P[b])),.5));mids[key]=int(P.size());P.push_back(midpoint);
    }
    if(mids.empty())return 0;
    static const int count[8]={1,2,2,3,2,3,3,4};
    static const int table[8][4][3]={{{0,1,2}},{{0,3,2},{3,1,2}},{{0,1,4},{0,4,2}},
      {{3,1,4},{0,3,2},{4,2,3}},{{0,1,5},{5,1,2}},{{0,3,5},{3,1,5},{2,5,1}},
      {{2,5,4},{0,1,5},{4,5,1}},{{3,4,5},{0,3,5},{3,1,4},{5,4,2}}};
    static const int diagonal[8][4]={{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,4,3,2},
      {0,0,0,0},{3,2,5,1},{0,4,5,1},{0,0,0,0}};
    // RefineE retains the first child in its original face slot, appending others.
    auto output=F;std::vector<Triangle> extra;
    for(std::size_t f=0;f<F.size();++f){int v[6]={F[f][0],F[f][1],F[f][2],-1,-1,-1},mask=0;
      for(int k=0;k<3;++k){auto it=mids.find(Key(v[k],v[(k+1)%3]));if(it!=mids.end()){v[k+3]=it->second;mask|=1<<k;}}
      Triangle children[4];for(int j=0;j<count[mask];++j)for(int k=0;k<3;++k)children[j][k]=v[table[mask][j][k]];
      if(count[mask]==3){const auto *d=diagonal[mask];if(Distance(P[v[d[0]]],P[v[d[1]]])<Distance(P[v[d[2]]],P[v[d[3]]])){
        children[2][1]=children[1][0];children[1][1]=children[2][0];}}
      output[f]=children[0];for(int j=1;j<count[mask];++j)extra.push_back(children[j]);
    }
    output.insert(output.end(),extra.begin(),extra.end());F.swap(output);
    for(const auto &m:mids)if(crease.erase(m.first)){crease.insert(Key(KeyFirst(m.first),m.second));crease.insert(Key(m.second,KeySecond(m.first)));}
    rebuild();return mids.size();
  }
  bool movable(int a,int b)const{
    if(fixed.count(a) || !manifold(a))return false;
    const auto direction=Normalize(Sub(ToVec(P[a]),ToVec(P[b])));int features=0;
    for(int w:neighbors(a)){const auto key=Key(a,w);if(edges.at(key).size()>2)return false;
      if(crease.count(key)){++features;if(!crease.count(Key(a,b)) || std::abs(Dot(Normalize(Sub(ToVec(P[w]),ToVec(P[a]))),direction))<.9)return false;}}
    return features<=2;
  }
  bool collapse(int face,int side,double minimum,double maximum,bool relaxed){
    const int a=F[face][side],b=F[face][(side+1)%3];const auto key=Key(a,b);
    if(hard.count(key) || edges.at(key).size()!=2)return false;
    const auto &t=F[face];const double area=.5*Norm(Cross(Sub(ToVec(P[t[1]]),ToVec(P[t[0]])),Sub(ToVec(P[t[2]]),ToVec(P[t[0]]))));
    if(!relaxed && Distance(P[a],P[b])>=minimum && area>=minimum*minimum/100)return false;
    const bool ma=movable(a,b),mb=movable(b,a);if(!ma && !mb)return false;
    const int drop=ma?a:b,keep=ma?b:a;
    const Point3 midpoint=ma&&mb?ToPoint(Mul(Add(ToVec(P[a]),ToVec(P[b])),.5)):P[keep];
    auto na=neighbors(a),nb=neighbors(b);std::vector<int> common;
    std::set_intersection(na.begin(),na.end(),nb.begin(),nb.end(),std::back_inserter(common));
    if(common.size()!=2)return false;
    std::set<int> affected=incident[a];affected.insert(incident[b].begin(),incident[b].end());
    std::set<Triangle> unique;
    for(int id:affected){auto tri=F[id];for(int &v:tri)if(v==drop)v=keep;
      if(tri[0]==tri[1] || tri[1]==tri[2] || tri[2]==tri[0])continue;
      auto canonical=tri;std::sort(canonical.begin(),canonical.end());if(!unique.insert(canonical).second)return false;
      const auto pos=[&](int v)->const Point3&{return v==keep?midpoint:P[v];};
      if(Quality(pos(tri[0]),pos(tri[1]),pos(tri[2]))<=.5*quality(id))return false;
      const auto cross=Cross(Sub(ToVec(pos(tri[1])),ToVec(pos(tri[0]))),Sub(ToVec(pos(tri[2])),ToVec(pos(tri[0]))));
      if(Norm(cross)<=0 || Dot(Normalize(cross),normal(id))<.7)return false;
      if(!relaxed)for(int v:tri)if(v!=keep && Distance(midpoint,P[v])>maximum)return false;
    }
    std::vector<EdgeKey> oldCreases,newCreases;
    for(int w:neighbors(drop))if(crease.count(Key(drop,w))){oldCreases.push_back(Key(drop,w));if(w!=keep)newCreases.push_back(Key(keep,w));}
    for(int id:affected)detach(id);
    P[keep]=midpoint;
    for(int id:affected){for(int &v:F[id])if(v==drop)v=keep;const auto &tri=F[id];
      if(tri[0]==tri[1] || tri[1]==tri[2] || tri[2]==tri[0])dead[id]=1;else attach(id);}
    for(auto e:oldCreases)crease.erase(e);for(auto e:newCreases)crease.insert(e);
    return true;
  }
  std::size_t collapsePass(double minimum,double maximum,bool crosses){std::size_t total=0;
    for(int id=0;id<int(F.size());++id)if(!dead[id])for(int side=0;side<3;++side){
      const int v=F[id][side];if(crosses && (border(v) || (incident[v].size()!=3 && incident[v].size()!=4)))continue;
      if(collapse(id,side,minimum,maximum,crosses)){++total;break;}}
    return total;
  }
  std::size_t flip(){std::size_t total=0;const double normalLimit=std::cos(double(float(5*std::acos(-1.0)/180)));
    for(int id=0;id<int(F.size());++id)if(!dead[id])for(int side=0;side<3;++side){
      const auto face=F[id];const int a=face[side],b=face[(side+1)%3],c=face[(side+2)%3];const auto key=Key(a,b);
      auto it=edges.find(key);if(it==edges.end() || it->second.size()!=2 || crease.count(key))continue;
      int other=*it->second.begin();if(other==id)other=*it->second.rbegin();if(other>=id)continue;
      int d=-1;for(int v:F[other])if(v!=a && v!=b)d=v;
      if(d<0 || d==c || edges.count(Key(c,d)))continue;
      const Triangle t0{a,d,c},t1{b,c,d};
      const auto error=[&](int v,int delta){return std::abs(int(neighbors(v).size())+delta-(border(v)?4:6));};
      const int before=error(a,0)+error(b,0)+error(c,0)+error(d,0),after=error(a,-1)+error(b,-1)+error(c,1)+error(d,1);
      const double oldQ=std::min(quality(id),quality(other)),newQ=std::min(Quality(P[a],P[d],P[c]),Quality(P[b],P[c],P[d]));
      if(!((after<before && newQ>=oldQ*.5)||(after==before && newQ>oldQ)||newQ>1.5*oldQ))continue;
      const auto n0=TriangleNormal(P,t0),n1=TriangleNormal(P,t1),old0=normal(id),old1=normal(other);
      if(Dot(old0,n0)<normalLimit || Dot(old0,n1)<normalLimit || Dot(old1,n0)<normalLimit || Dot(old1,n1)<normalLimit)continue;
      detach(id);detach(other);F[id]=t0;F[other]=t1;attach(id);attach(other);++total;break;
    }return total;
  }
  void accumulate(std::vector<Vec3>&sums,std::vector<int>&counts)const{
    sums.assign(P.size(),{0,0,0});counts.assign(P.size(),0);
    // Smooth::AccumulateLaplacianInfo visits each face edge: interior edges twice.
    for(const auto &e:edges)if(e.second.size()==2){const int a=KeyFirst(e.first),b=KeySecond(e.first);
      sums[a]=Add(sums[a],Mul(ToVec(P[b]),2));sums[b]=Add(sums[b],Mul(ToVec(P[a]),2));counts[a]+=2;counts[b]+=2;}
    for(const auto &e:edges)if(e.second.size()==1)for(int v:{KeyFirst(e.first),KeySecond(e.first)}){sums[v]=ToVec(P[v]);counts[v]=1;}
    for(const auto &e:edges)if(e.second.size()==1){const int a=KeyFirst(e.first),b=KeySecond(e.first);
      sums[a]=Add(sums[a],ToVec(P[b]));sums[b]=Add(sums[b],ToVec(P[a]));++counts[a];++counts[b];}
  }
  std::size_t smooth(){std::vector<Vec3>sums;std::vector<int>counts;accumulate(sums,counts);std::size_t moved=0;
    for(int v=0;v<int(P.size());++v)if(counts[v] && !featureVertex(v) && !border(v) && manifold(v)){
      const auto average=Mul(Add(ToVec(P[v]),sums[v]),1.0/(counts[v]+1));
      const auto next=ToPoint(Add(Mul(ToVec(P[v]),.8),Mul(average,.2)));moved+=Distance(next,P[v])>0;P[v]=next;}
    std::vector<unsigned char> folded(P.size(),0);const double foldCos=std::cos(140*std::acos(-1.0)/180);
    for(const auto &e:edges)if(e.second.size()==2){auto it=e.second.begin();const int a=*it++,b=*it;
      if(Dot(normal(a),normal(b))<=foldCos)for(int f:{a,b})for(int v:F[f])if(!featureVertex(v) && manifold(v))folded[v]=1;}
    for(int step=0;step<2;++step){accumulate(sums,counts);
      // VCGLib FoldRelax commits in face order using a sweep-start accumulator.
      for(int id=0;id<int(F.size());++id)if(!dead[id]){auto next=std::array<Point3,3>{P[F[id][0]],P[F[id][1]],P[F[id][2]]};
        for(int k=0;k<3;++k){const int v=F[id][k];if(folded[v] && counts[v])next[k]=ToPoint(Mul(Add(ToVec(P[v]),sums[v]),1.0/(counts[v]+1)));}
        for(int k=0;k<3;++k){const int v=F[id][k];moved+=Distance(next[k],P[v])>0;P[v]=next[k];}}
    }return moved;
  }
};
}
void RemeshIsotropicPatch(std::vector<Point3> &points,std::vector<std::array<int,3>> &faces,
    std::vector<int> &aliases,std::unordered_set<EdgeKey> constraints,const MeshPatch &patch,const NativeRemeshConfig &config){
  (void)patch;const auto start=std::chrono::steady_clock::now();
  const auto originalPoints=points;const auto originalFaces=faces;
  std::vector<int> labels(faces.size(),0);const SurfaceIndex reference(originalPoints,originalFaces,labels);
  NativeVcgRules::Mesh mesh(points,faces,constraints);mesh.tag(config.GenericFeatureAngleDegrees);
  const double minimum=config.TargetEdgeLength*.8,maximum=config.TargetEdgeLength*4/3;
  for(int iteration=0;iteration<std::max(1,config.GenericRemeshIterations);++iteration){
    const auto tick=std::chrono::steady_clock::now();const auto split=mesh.split(maximum);
    const auto collapse=mesh.collapsePass(minimum,maximum,false),crosses=mesh.collapsePass(minimum,maximum,true);
    mesh.compact();const auto flips=mesh.flip(),moved=mesh.smooth();
    // Projection belongs after all local operators, not inside candidate tests.
    for(int v=0;v<int(points.size());++v)if(!mesh.incident[v].empty() && !mesh.fixed.count(v)){
      Point3 nearest;double distance=0;if(reference.closest(points[v],0,nearest,distance))points[v]=nearest;}
    if(config.Verbose)std::clog << "[CadMesh] native vcg-rules iteration: " << iteration+1
        << ", split=" << split << ", collapse=" << collapse << ", cross_collapse=" << crosses
        << ", flips=" << flips << ", smooth_updates=" << moved << ", faces=" << faces.size()
        << ", seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count() << std::endl;
  }
  aliases.resize(points.size(),-1);for(int v=0;v<int(aliases.size());++v)if(!mesh.fixed.count(v))aliases[v]=-1;
  if(config.Verbose)std::clog << "[CadMesh] native vcg-rules complete: feature_angle_deg=" << config.GenericFeatureAngleDegrees
      << ", iterations=" << config.GenericRemeshIterations << ", surf_dist_check=off, cleanup=off, shared_boundaries=fixed"
      << ", seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << std::endl;
}
