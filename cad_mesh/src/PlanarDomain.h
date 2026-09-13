// Implementation fragment included after Chart/UV inside the remesher's
// private namespace. No external triangulation dependency is required.
struct ChartBuildDiagnostics {
  std::string Stage,Failure;
  double LegalizeSeconds=0,LatticeSeconds=0,RepairSeconds=0;
  std::size_t Work=0,Points=0,Faces=0;
  int EdgeA=-1,EdgeB=-1,GlobalEdgeA=-1,GlobalEdgeB=-1;
  double EdgeLength=0,EdgeTarget=0;
  int RingRows=0,RingColumns=0;
  double RingInteriorSpacing=0;
};
class PlanarDomain {
  Chart &C;
  struct Edge { int A=-1,B=-1; };
  std::unordered_map<EdgeKey,Edge> Adj;
  std::unordered_set<EdgeKey> Fixed,Pending;
  std::deque<EdgeKey> Queue;
  std::vector<EdgeKey> Changed;
  std::size_t Work=0;
  int Hint=0;
  bool Good=true,Track=false,MultipleLoops=false;
  double FlipLimit=0;
  const char *Failure="";
  int ProblemA=-1,ProblemB=-1;
  double ProblemLength=0,ProblemTarget=0;
  bool failEdge(const char *reason,EdgeKey key,double length,double target){
    Failure=reason;Good=false;ProblemA=int(key>>32);ProblemB=int(std::uint32_t(key));
    ProblemLength=length;ProblemTarget=target;return false;
  }
  bool tick(){if(!Good)return false;if(++Work>20000000){Failure="work_budget";Good=false;}return Good;}
  void enqueue(EdgeKey k){if(Pending.insert(k).second)Queue.push_back(k);}
  void add(int id){
    const auto f=C.Faces[id];
    for(int i=0;i<3;++i){const auto k=Key(f[i],f[(i+1)%3]);auto &e=Adj[k];
      if(e.A<0)e.A=id;else if(e.B<0)e.B=id;else Good=false;
      enqueue(k);if(Track)Changed.push_back(k);
    }
  }
  void remove(int id){
    const auto f=C.Faces[id];
    for(int i=0;i<3;++i){auto it=Adj.find(Key(f[i],f[(i+1)%3]));
      if(it==Adj.end()){Good=false;continue;}auto &e=it->second;
      if(e.A==id){e.A=e.B;e.B=-1;}else if(e.B==id)e.B=-1;else Good=false;
      if(e.A<0)Adj.erase(it);
    }
  }
  void assign(int id,std::array<int,3> f){
    if(Cross2(C.Points[f[0]],C.Points[f[1]],C.Points[f[2]])<0)std::swap(f[1],f[2]);
    if(id==int(C.Faces.size()))C.Faces.push_back(f);else C.Faces[id]=f;add(id);
  }
  int opposite(int id,int a,int b)const{
    for(int v:C.Faces[id])if(v!=a&&v!=b)return v;return -1;
  }
  void flip(EdgeKey key){
    if(Fixed.count(key))return;
    auto it=Adj.find(key);if(it==Adj.end()||it->second.B<0)return;
    int a=int(key>>32),b=int(std::uint32_t(key)),t=it->second.A,u=it->second.B;
    int c=opposite(t,a,b),d=opposite(u,a,b);
    if(c<0||d<0||c==d||Adj.count(Key(c,d)))return;
    const UV pa=C.Points[a],pb=C.Points[b],pc=C.Points[c],pd=C.Points[d];
    if(FlipLimit>0 && std::hypot(pc.X-pd.X,pc.Y-pd.Y)>
       std::max(FlipLimit,std::hypot(pa.X-pb.X,pa.Y-pb.Y))*(1+1e-6))return;
    if(Cross2(pc,pd,pa)*Cross2(pc,pd,pb)>=0)return;
    const long double ax=(long double)pa.X-pd.X,ay=(long double)pa.Y-pd.Y;
    const long double bx=(long double)pb.X-pd.X,by=(long double)pb.Y-pd.Y;
    const long double cx=(long double)pc.X-pd.X,cy=(long double)pc.Y-pd.Y;
    const long double aa=ax*ax+ay*ay,bb=bx*bx+by*by,cc=cx*cx+cy*cy;
    long double det=aa*(bx*cy-by*cx)-bb*(ax*cy-ay*cx)+cc*(ax*by-ay*bx);
    if(Cross2(pa,pb,pc)<0)det=-det;
    const long double scale=std::max({aa,bb,cc});
    // Do not oscillate on cocircular configurations. MSVC long double is
    // double precision: this is a filtered numerical predicate, not exact.
    if(det<=scale*scale*1e-13L)return;
    remove(t);remove(u);assign(t,{c,d,a});assign(u,{d,c,b});
  }
public:
  PlanarDomain(Chart &chart,const std::vector<std::array<int,2>> &segments,bool multipleLoops,double flipLimit=0):C(chart),MultipleLoops(multipleLoops),FlipLimit(flipLimit){
    for(const auto &edge:segments)Fixed.insert(Key(edge[0],edge[1]));
    for(const auto &edge:C.InteriorConstraints)Fixed.insert(Key(edge[0],edge[1]));
    for(int i=0;i<int(C.Faces.size());++i)add(i);
    for(const auto &edge:Adj)if(edge.second.B<0 && !Fixed.count(edge.first))Good=false;
    for(auto key:Fixed)if(!Adj.count(key))Good=false;
    if(!Good)Failure="seed_topology";
    if(Good && FlipLimit>0)for(auto key:Fixed){
      const UV a=C.Points[int(key>>32)],b=C.Points[int(std::uint32_t(key))];
      const double length=std::hypot(a.X-b.X,a.Y-b.Y);
      if(length>FlipLimit*(1+1e-6)){
        failEdge("fixed_boundary_requires_sampling",key,length,FlipLimit);break;
      }
    }
  }
  void diagnose(ChartBuildDiagnostics *diagnostics)const{
    if(!diagnostics)return;
    diagnostics->Work=Work;diagnostics->Points=C.Points.size();diagnostics->Faces=C.Faces.size();
    if(ProblemA>=0){
      diagnostics->EdgeA=ProblemA;diagnostics->EdgeB=ProblemB;
      diagnostics->GlobalEdgeA=C.Aliases[ProblemA];diagnostics->GlobalEdgeB=C.Aliases[ProblemB];
      diagnostics->EdgeLength=ProblemLength;diagnostics->EdgeTarget=ProblemTarget;
    }
    if(!Good)diagnostics->Failure=*Failure?Failure:"adjacency_update";
  }
  bool legalize(){
    while(!Queue.empty()&&tick()){auto k=Queue.front();Queue.pop_front();Pending.erase(k);flip(k);}return Good;
  }
  int locate(UV point){
    int id=std::min(Hint,int(C.Faces.size())-1);
    for(std::size_t step=0;step<=C.Faces.size()&&tick();++step){
      const auto f=C.Faces[id];int side=-1;double worst=0;
      for(int k=0;k<3;++k){double cross=Cross2(C.Points[f[k]],C.Points[f[(k+1)%3]],point);
        if(cross<worst){worst=cross;side=k;}}
      if(side<0){Hint=id;return id;}
      const auto &e=Adj.at(Key(f[side],f[(side+1)%3]));
      int next=e.A==id?e.B:e.A;
      if(next<0){
        if(!MultipleLoops)return -1;
        // A walk may hit a hole even when the sample is in another valid
        // part of the domain. The caller's parity test already excluded
        // holes; perform a bounded fallback location instead of dropping it.
        for(int candidate=0;candidate<int(C.Faces.size())&&tick();++candidate){
          const auto f=C.Faces[candidate];
          if(Cross2(C.Points[f[0]],C.Points[f[1]],point)>=0&&
             Cross2(C.Points[f[1]],C.Points[f[2]],point)>=0&&
             Cross2(C.Points[f[2]],C.Points[f[0]],point)>=0){Hint=candidate;return candidate;}
        }
        return -1;
      }id=next;
    }
    Failure="point_location_cycle";Good=false;return -1;
  }
  bool insert(UV point,int face=-1,EdgeKey onEdge=0){
    if(onEdge){
      const UV a=C.Points[int(onEdge>>32)],b=C.Points[int(std::uint32_t(onEdge))];
      if(!std::isfinite(point.X)||!std::isfinite(point.Y)||
         (point.X==a.X&&point.Y==a.Y)||(point.X==b.X&&point.Y==b.Y))
        return failEdge("midpoint_precision_stall",onEdge,std::hypot(a.X-b.X,a.Y-b.Y),FlipLimit);
    }
    if(face<0)face=locate(point);
    if(face<0)return Good;
    const auto old=C.Faces[face];
    if(C.Points.size()>=100000||C.Faces.size()+4>200000){Failure="mesh_size_budget";Good=false;return false;}
    if(!onEdge)for(int k=0;k<3;++k){
      const UV a=C.Points[old[k]],b=C.Points[old[(k+1)%3]];
      const double dx=a.X-b.X,dy=a.Y-b.Y,len2=dx*dx+dy*dy,cross=Cross2(a,b,point);
      // Skip almost-on-edge lattice sites; explicit midpoint insertions use
      // the two-sided edge split below and never create hanging vertices.
      if(cross*cross<=len2*len2*1e-24)return Good;
    }
    const int v=int(C.Points.size());
    C.Points.push_back(point);C.Aliases.push_back(-1);C.Fixed.push_back(0);
    if(onEdge){
      auto it=Adj.find(onEdge);
      if(it==Adj.end()||it->second.B<0||Fixed.count(onEdge)){Good=false;return false;}
      const int a=int(onEdge>>32),b=int(std::uint32_t(onEdge)),t=it->second.A,u=it->second.B;
      const int c=opposite(t,a,b),d=opposite(u,a,b);
      remove(t);remove(u);assign(t,{a,v,c});assign(u,{b,v,d});
      assign(int(C.Faces.size()),{v,b,c});assign(int(C.Faces.size()),{v,a,d});Hint=t;
    }else{
      remove(face);assign(face,{old[0],old[1],v});
      assign(int(C.Faces.size()),{old[1],old[2],v});
      assign(int(C.Faces.size()),{old[2],old[0],v});Hint=face;
    }
    return legalize();
  }
  bool finish(double target){
    for(const auto &entry:Adj)enqueue(entry.first);
    std::size_t splitCount=0,previousLongCount=std::numeric_limits<std::size_t>::max();
    double previousMaximum=std::numeric_limits<double>::infinity();
    int stalledWindows=0;
    while(!Queue.empty()&&tick()){
      auto key=Queue.front();Queue.pop_front();Pending.erase(key);
      auto it=Adj.find(key);if(it==Adj.end()||it->second.B<0||Fixed.count(key))continue;
      const UV a=C.Points[int(key>>32)],b=C.Points[int(std::uint32_t(key))];
      const double dx=a.X-b.X,dy=a.Y-b.Y;
      if(dx*dx+dy*dy<=target*target*(1+2.000001e-6))continue;
      const int face=it->second.A;
      auto waiting=std::move(Queue);auto marked=std::move(Pending);
      Queue.clear();Pending.clear();Changed.clear();Track=true;
      if(!insert(Mul2(Add2(a,b),.5),face,key))return false;
      Track=false;Queue=std::move(waiting);Pending=std::move(marked);
      for(auto changed:Changed)enqueue(changed);
      // A bounded stagnation detector, not a proof of infeasibility. Retain
      // the source patch when four 256-split windows make no measurable
      // progress instead of spending the full 100k-vertex budget.
      if(FlipLimit>0 && ++splitCount%256==0){
        std::size_t longCount=0;double maximum=0;EdgeKey worst=0;
        for(const auto &entry:Adj){
          if(!tick())return false;
          if(entry.second.B<0||Fixed.count(entry.first))continue;
          const UV x=C.Points[int(entry.first>>32)],y=C.Points[int(std::uint32_t(entry.first))];
          const double length=std::hypot(x.X-y.X,x.Y-y.Y);
          if(length>target*(1+1e-6)){++longCount;if(length>maximum){maximum=length;worst=entry.first;}}
        }
        if(longCount && maximum>=previousMaximum*(1-1e-5) && longCount>=previousLongCount)++stalledWindows;
        else stalledWindows=0;
        previousMaximum=maximum;previousLongCount=longCount;
        if(stalledWindows>=4)return failEdge("edge_repair_stalled",worst,maximum,target);
      }
    }return Good;
  }
};

bool PopulateSeededDomain(Chart &chart,const std::vector<std::array<int,2>> &segments,
                          double target,bool multipleLoops,ChartBuildDiagnostics *diagnostics=nullptr);

bool MakeSeededPlanarChart(const Frame &frame,const std::vector<std::vector<int>> &loops,
                         const std::vector<std::array<int,3>> &sourceFaces,
                         const std::vector<Point3> &vertices,double target,Chart &chart){
  if(!(target>0)||loops.empty())return false;
  chart.Surface=frame;
  std::unordered_map<int,int> local;
  std::vector<std::array<int,2>> segments;
  for(const auto &loop:loops){
    if(loop.size()<3)return false;
    for(int id:loop){
      if(local.count(id))return false; // touching/duplicated trim vertices
      UV uv;if(!frame.parameter(vertices[id],uv))return false;
      local[id]=int(chart.Points.size());
      chart.Points.push_back(uv);chart.Aliases.push_back(id);chart.Fixed.push_back(1);
    }
    for(std::size_t i=0;i<loop.size();++i)segments.push_back({local.at(loop[i]),local.at(loop[(i+1)%loop.size()])});
  }
  if(loops.size()==1){
    if(!EarClip(chart.Points,chart.Faces,2000000))return false;
  }else{
    // The source patch supplies a conforming multiply-connected topology
    // seed. No bridges or duplicated seam aliases are introduced. Existing
    // interior sites are retained; all unconstrained diagonals are legalized.
    if(sourceFaces.empty()||sourceFaces.size()>200000)return false;
    for(const auto &source:sourceFaces){std::array<int,3> triangle;
      for(int k=0;k<3;++k){int id=source[k];auto found=local.find(id);
        if(found==local.end()){
          UV uv;if(!frame.parameter(vertices[id],uv))return false;
          int index=int(chart.Points.size());local[id]=index;
          chart.Points.push_back(uv);chart.Aliases.push_back(-1);chart.Fixed.push_back(0);
          triangle[k]=index;
        }else triangle[k]=found->second;
      }
      const double area=Cross2(chart.Points[triangle[0]],chart.Points[triangle[1]],chart.Points[triangle[2]]);
      if(!std::isfinite(area)||area==0)return false;
      if(area<0)std::swap(triangle[1],triangle[2]);chart.Faces.push_back(triangle);
    }
    if(chart.Points.size()>100000)return false;
  }
  return PopulateSeededDomain(chart,segments,target,loops.size()>1);
}

bool PopulateSeededDomain(Chart &chart,const std::vector<std::array<int,2>> &segments,
                          double target,bool multipleLoops,ChartBuildDiagnostics *diagnostics){
  using Timer=std::chrono::steady_clock;
  PlanarDomain domain(chart,segments,multipleLoops,diagnostics?target:0);
  struct Record {PlanarDomain &Domain;ChartBuildDiagnostics *D;~Record(){Domain.diagnose(D);}} record{domain,diagnostics};
  auto start=Timer::now();
  if(diagnostics)diagnostics->Stage="initial_legalization";
  const bool legalized=domain.legalize();
  if(diagnostics)diagnostics->LegalizeSeconds=std::chrono::duration<double>(Timer::now()-start).count();
  if(!legalized)return false;
  if(diagnostics)diagnostics->Stage="lattice_setup";
  UV lo=chart.Points[0],hi=lo;
  for(UV p:chart.Points){lo.X=std::min(lo.X,p.X);lo.Y=std::min(lo.Y,p.Y);
    hi.X=std::max(hi.X,p.X);hi.Y=std::max(hi.Y,p.Y);}
  const double spacing=target*.85,height=spacing*std::sqrt(3.0)*.5;
  const double columns=std::ceil((hi.X-lo.X)/spacing),rows=std::ceil((hi.Y-lo.Y)/height);
  if(!std::isfinite(columns)||!std::isfinite(rows)||columns>200000||rows>200000||columns*rows>200000)return false;
  // Boundary proximity buckets prevent the interior lattice from creating
  // arbitrarily thin strips next to densely sampled shared boundaries.
  std::unordered_map<EdgeKey,std::vector<int>> boundaryCells;
  const auto cellKey=[](int x,int y){return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);};
  const auto cellX=[&](double x){return int(std::floor((x-lo.X)/spacing));};
  const auto cellY=[&](double y){return int(std::floor((y-lo.Y)/spacing));};
  std::size_t bucketWork=0;
  for(int i=0;i<int(segments.size());++i){const UV a=chart.Points[segments[i][0]],b=chart.Points[segments[i][1]];
    for(int x=cellX(std::min(a.X,b.X));x<=cellX(std::max(a.X,b.X));++x)
      for(int y=cellY(std::min(a.Y,b.Y));y<=cellY(std::max(a.Y,b.Y));++y){
        if(++bucketWork>2000000)return false;
        boundaryCells[cellKey(x,y)].push_back(i);
      }
  }
  start=Timer::now();if(diagnostics)diagnostics->Stage="lattice_insertion";
  struct LatticeTimer {Timer::time_point Start;ChartBuildDiagnostics *D;bool Finished=false;
    void finish(){if(D&&!Finished)D->LatticeSeconds=std::chrono::duration<double>(Timer::now()-Start).count();Finished=true;}
    ~LatticeTimer(){finish();}} latticeTimer{start,diagnostics};
  for(int row=0;row<int(rows);++row){
    const double sampleY=lo.Y+(row+.5)*height;
    std::vector<double> crossings;
    for(const auto &segment:segments){
      if(++bucketWork>20000000)return false;
      const UV a=chart.Points[segment[0]],b=chart.Points[segment[1]];
      if((a.Y>sampleY)!=(b.Y>sampleY))crossings.push_back(a.X+(sampleY-a.Y)*(b.X-a.X)/(b.Y-a.Y));
    }
    std::sort(crossings.begin(),crossings.end());
    for(int column=0;column<int(columns);++column){
    const int col=(row&1)?int(columns)-1-column:column;
    const UV point{lo.X+(col+.5+(row&1)*.5)*spacing,lo.Y+(row+.5)*height};
    // Even-odd domain membership supports holes and disconnected islands,
    // independently of boundary loop orientation.
    if((std::size_t(std::upper_bound(crossings.begin(),crossings.end(),point.X)-crossings.begin())&1)==0)continue;
    bool nearBoundary=false;
    for(int x=cellX(point.X)-1;x<=cellX(point.X)+1&&!nearBoundary;++x)
      for(int y=cellY(point.Y)-1;y<=cellY(point.Y)+1&&!nearBoundary;++y){
        auto found=boundaryCells.find(cellKey(x,y));if(found==boundaryCells.end())continue;
        for(int id:found->second){const UV a=chart.Points[segments[id][0]],b=chart.Points[segments[id][1]];
          const double dx=b.X-a.X,dy=b.Y-a.Y,length=dx*dx+dy*dy;
          const double t=length>0?std::clamp(((point.X-a.X)*dx+(point.Y-a.Y)*dy)/length,0.0,1.0):0;
          const double ex=point.X-a.X-t*dx,ey=point.Y-a.Y-t*dy;
          if(ex*ex+ey*ey<spacing*spacing*.09){nearBoundary=true;break;}
        }
      }
    if(nearBoundary)continue;
    if(!domain.insert(point))return false;
    }
  }
  latticeTimer.finish();start=Timer::now();if(diagnostics)diagnostics->Stage="local_edge_repair";
  const bool finished=domain.finish(target);
  if(diagnostics)diagnostics->RepairSeconds=std::chrono::duration<double>(Timer::now()-start).count();
  return finished;
}

// Build a ruled periodic strip between two angularly monotone trims. Heights
// may vary around either trim; neither boundary is flattened to an end plane.
bool MakeDirectCylinderRing(const Frame &frame,
                            const std::vector<std::vector<int>> &loops,
                            const std::vector<Point3> &vertices,double target,Chart &chart,
                            double axialScale,ChartBuildDiagnostics *diagnostics){
  if(diagnostics)diagnostics->Stage="ring_trim_parameterization";
  const auto fail=[&](const char *reason){if(diagnostics)diagnostics->Failure=reason;return false;};
  struct Trim {std::vector<UV> P;std::vector<int> Id;std::vector<double> T;};
  Trim trim[2];const double period=frame.UPeriod;
  const auto load=[&](Trim &out,const std::vector<int> &ids){
    if(ids.size()<3)return false;
    out.Id=ids;for(int id:ids){UV p;if(!frame.parameter(vertices[id],p))return false;p.Y*=axialScale;out.P.push_back(p);}
    double winding=0;for(std::size_t i=0;i<out.P.size();++i)
      winding+=std::remainder(out.P[(i+1)%out.P.size()].X-out.P[i].X,period);
    if(std::abs(std::abs(winding)-period)>period*.01)return fail("ring_winding");
    if(winding<0){std::reverse(out.P.begin(),out.P.end());std::reverse(out.Id.begin(),out.Id.end());}
    return true;
  };
  if(!load(trim[0],loops[0])||!load(trim[1],loops[1]))return false;
  const auto mean=[](const Trim &t){double sum=0;for(UV p:t.P)sum+=p.Y;return sum/t.P.size();};
  if(mean(trim[0])>mean(trim[1]))std::swap(trim[0],trim[1]);
  // Choose nearby existing boundary vertices for the seam. No new vertex is
  // independently inserted into either shared trim.
  std::size_t closest=0;double nearest=period;
  for(std::size_t i=0;i<trim[1].P.size();++i){
    double d=std::abs(std::remainder(trim[1].P[i].X-trim[0].P[0].X,period));
    if(d<nearest){nearest=d;closest=i;}
  }
  std::rotate(trim[1].P.begin(),trim[1].P.begin()+closest,trim[1].P.end());
  std::rotate(trim[1].Id.begin(),trim[1].Id.begin()+closest,trim[1].Id.end());
  for(auto &t:trim){
    const auto original=t.P;
    for(std::size_t i=1;i<t.P.size();++i){
      const double advance=std::remainder(original[i].X-original[i-1].X,period);
      if(advance<=period*1e-12)return fail("ring_angular_backtracking");
      t.P[i].X=t.P[i-1].X+advance;
    }
    if(t.P.back().X-t.P.front().X>=period*(1-1e-12))return fail("ring_period_overlap");
    t.P.push_back({t.P.front().X+period,t.P.front().Y});t.Id.push_back(t.Id.front());
    for(UV p:t.P)t.T.push_back((p.X-t.P.front().X)/period);
  }
  const double shift=period*std::round((trim[0].P.front().X-trim[1].P.front().X)/period);
  for(UV &p:trim[1].P)p.X+=shift;
  const auto sample=[](const Trim &t,double u){
    auto it=std::upper_bound(t.T.begin(),t.T.end(),u);
    const std::size_t i=std::min(t.T.size()-2,std::size_t(it-t.T.begin()-1));
    const double f=(u-t.T[i])/(t.T[i+1]-t.T[i]);
    return Add2(Mul2(t.P[i],1-f),Mul2(t.P[i+1],f));
  };
  std::vector<double> breaks=trim[0].T;breaks.insert(breaks.end(),trim[1].T.begin(),trim[1].T.end());
  std::sort(breaks.begin(),breaks.end());breaks.erase(std::unique(breaks.begin(),breaks.end()),breaks.end());
  // Layout in physical surface distance, even though edge repair retains the
  // anisotropic acceptance metric used by globally sampled boundaries.
  const auto length=[&](UV a,UV b){return std::hypot(a.X-b.X,(a.Y-b.Y)/axialScale);};
  if(diagnostics)diagnostics->Stage="ring_layout";
  double gap=0;
  for(double u:breaks){const UV low=sample(trim[0],u),high=sample(trim[1],u);
    if(high.Y<=low.Y)return fail("ring_trim_order");gap=std::max(gap,length(low,high));}
  const double rowEstimate=std::ceil(gap/(target*.55));
  if(!std::isfinite(rowEstimate)||rowEstimate>50000)return fail("ring_row_budget");
  const int rows=std::max(2,int(rowEstimate));
  // Integrate trim length, then resample independently of input vertices.
  // Input breakpoints describe the trims but must not become full-height columns.
  std::vector<double> cumulative{0};
  for(std::size_t i=1;i<breaks.size();++i){
    const double a=breaks[i-1],b=breaks[i];
    const UV l0=sample(trim[0],a),l1=sample(trim[0],b),h0=sample(trim[1],a),h1=sample(trim[1],b);
    // Reject a folding parameterization rather than generating overlapping
    // triangles where highly irregular trims cannot form a ruled strip.
    if(Cross2(l0,l1,h0)<=0||Cross2(h0,l1,h1)<=0)return fail("ring_folded_parameterization");
    cumulative.push_back(cumulative.back()+std::max(length(l0,l1),length(h0,h1)));
  }
  const double columnEstimate=std::ceil(cumulative.back()/(target*.65));
  if(!std::isfinite(columnEstimate)||columnEstimate>100000)return fail("ring_column_budget");
  const int columnCount=std::max(3,int(columnEstimate));
  if(std::uint64_t(rows-1)*(columnCount+2)+trim[0].P.size()+trim[1].P.size()>100000)return fail("ring_mesh_size_budget");
  if(diagnostics){diagnostics->RingRows=rows;diagnostics->RingColumns=columnCount;
    diagnostics->RingInteriorSpacing=target;}
  const auto columnParameter=[&](double fraction){
    const double distance=fraction*cumulative.back();
    auto it=std::upper_bound(cumulative.begin(),cumulative.end(),distance);
    const std::size_t i=std::min(cumulative.size()-2,std::size_t(it-cumulative.begin()-1));
    const double f=(distance-cumulative[i])/(cumulative[i+1]-cumulative[i]);
    return breaks[i]+f*(breaks[i+1]-breaks[i]);
  };
  if(diagnostics)diagnostics->Stage="ring_grid_generation";
  chart.Surface=frame;
  const auto append=[&](UV p,int alias,bool fixed){const int id=int(chart.Points.size());
    chart.Points.push_back(p);chart.Aliases.push_back(alias);chart.Fixed.push_back(fixed?1:0);return id;};
  std::vector<std::array<int,2>> constraints;
  const auto addFace=[&](int a,int b,int c){
    if(Cross2(chart.Points[a],chart.Points[b],chart.Points[c])<0)std::swap(b,c);
    chart.Faces.push_back({a,b,c});
  };
  const auto connect=[&](const std::vector<int> &lower,const std::vector<double> &lt,
                         const std::vector<int> &upper,const std::vector<double> &ut){
    std::size_t i=0,j=0;
    while(i+1<lower.size()||j+1<upper.size()){
      if(i+1<lower.size()&&(j+1==upper.size()||lt[i+1]<=ut[j+1])){
        addFace(lower[i],lower[i+1],upper[j]);++i;
      }else{addFace(lower[i],upper[j+1],upper[j]);++j;}
    }
  };
  std::vector<int> previous;std::vector<double> previousT=trim[0].T;
  for(std::size_t i=0;i<trim[0].P.size();++i)previous.push_back(append(trim[0].P[i],trim[0].Id[i],true));
  for(std::size_t i=1;i<previous.size();++i)constraints.push_back({previous[i-1],previous[i]});
  for(int row=1;row<=rows;++row){
    std::vector<double> columns;
    if(row!=rows){
      columns.push_back(0);
      // Offset alternating rows to avoid long aligned diagonal strips. Both
      // copies of the periodic seam still share exactly the same vertex alias.
      const double offset=(row&1)?.5:1.0;
      for(double k=offset;k<columnCount;k+=1)columns.push_back(columnParameter(k/columnCount));
      columns.push_back(1);
    }
    std::vector<int> current;const auto &t=row==rows?trim[1].T:columns;
    for(std::size_t i=0;i<t.size();++i){
      if(row==rows)current.push_back(append(trim[1].P[i],trim[1].Id[i],true));
      else{
        const double v=double(row)/rows;
        const UV p=Add2(Mul2(sample(trim[0],t[i]),1-v),Mul2(sample(trim[1],t[i]),v));
        const bool seam=i==0||i+1==t.size();
        current.push_back(append(p,seam?-1-row:-1,seam));
      }
    }
    connect(previous,previousT,current,t);
    constraints.push_back({previous.front(),current.front()});constraints.push_back({previous.back(),current.back()});
    if(row==rows)for(std::size_t i=1;i<current.size();++i)constraints.push_back({current[i-1],current[i]});
    previous=std::move(current);previousT=t;
  }
  if(chart.Faces.size()>200000)return false;
  // Interior cells were generated directly. Only the remaining long edges
  // (typically trim transitions) need local insertions; no lattice location
  // and no initial whole-domain Delaunay legalization are performed.
  PlanarDomain domain(chart,constraints,false,target);
  if(diagnostics)diagnostics->Stage="ring_edge_repair";
  const auto repairStart=std::chrono::steady_clock::now();
  const bool finished=domain.finish(target);domain.diagnose(diagnostics);
  if(diagnostics)diagnostics->RepairSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-repairStart).count();
  return finished;
}

// Recover the trimmed domain from the input connectivity. In particular, a
// second loop can be a hole in a trimmed wall, rather than a periodic ring.
// Unwrap along ALL source edges so crossing atan2's branch cut is harmless.
bool MakeSourceCylinderChart(const Frame &frame,
                            const std::vector<std::array<int,3>> &sourceFaces,
                            const std::vector<Point3> &vertices,double target,
                            double axialScale,Chart &chart,ChartBuildDiagnostics &diagnostics){
  diagnostics.Stage="source_cylinder_unwrap";
  const auto fail=[&](const char *reason){diagnostics.Failure=reason;return false;};
  if(sourceFaces.empty())return fail("empty_source_mesh");
  if(sourceFaces.size()>200000)return fail("source_mesh_size_budget");
  Chart seed;seed.Surface=frame;
  std::unordered_map<int,int> local;
  std::vector<int> globals;
  std::unordered_map<EdgeKey,int> counts;
  std::vector<std::vector<int>> neighbors;
  for(const auto &source:sourceFaces){
    std::array<int,3> face;
    for(int k=0;k<3;++k){
      const int id=source[k];
      if(id<0||std::size_t(id)>=vertices.size())return fail("source_vertex_index");
      auto found=local.find(id);
      if(found==local.end()){
        UV uv;if(!frame.parameter(vertices[id],uv)||!std::isfinite(uv.X)||!std::isfinite(uv.Y))
          return fail("source_parameterization");
        const int index=int(seed.Points.size());local.emplace(id,index);globals.push_back(id);
        uv.Y*=axialScale;seed.Points.push_back(uv);neighbors.emplace_back();
        seed.Aliases.push_back(-1);seed.Fixed.push_back(0);face[k]=index;
      }else face[k]=found->second;
    }
    if(face[0]==face[1]||face[1]==face[2]||face[2]==face[0])return fail("source_degenerate_face");
    seed.Faces.push_back(face);
    for(int k=0;k<3;++k){
      const int a=face[k],b=face[(k+1)%3];
      int &count=counts[Key(a,b)];
      if(++count>2)return fail("source_nonmanifold_edge");
      if(count==1){neighbors[a].push_back(b);neighbors[b].push_back(a);}
    }
  }
  if(seed.Points.size()>100000)return fail("source_mesh_size_budget");
  const std::vector<UV> wrapped=seed.Points;
  std::vector<unsigned char> visited(seed.Points.size(),0);
  std::vector<int> queue;
  const double angularTolerance=frame.UPeriod*1e-8;
  for(int root=0;root<int(seed.Points.size());++root){
    if(visited[root])continue;
    queue.clear();queue.push_back(root);visited[root]=1;
    for(std::size_t head=0;head<queue.size();++head){
      const int a=queue[head];
      for(int b:neighbors[a]){
        const double x=seed.Points[a].X+std::remainder(wrapped[b].X-wrapped[a].X,frame.UPeriod);
        if(visited[b]){
          // A nonzero cycle winding needs a cut, not an overlapping disk.
          if(std::abs(seed.Points[b].X-x)>angularTolerance)return fail("source_periodic_cut_required");
        }else{seed.Points[b].X=x;visited[b]=1;queue.push_back(b);}
      }
    }
  }
  diagnostics.Stage="source_cylinder_seed";
  int orientation=0;
  for(auto &face:seed.Faces){
    const double area=Cross2(seed.Points[face[0]],seed.Points[face[1]],seed.Points[face[2]]);
    if(!std::isfinite(area)||area==0)return fail("source_uv_degenerate_face");
    const int sign=area>0?1:-1;
    if(orientation && orientation!=sign)return fail("source_uv_fold_or_inconsistent_orientation");
    orientation=sign;
  }
  if(orientation<0)for(auto &face:seed.Faces)std::swap(face[1],face[2]);
  std::vector<std::array<int,2>> segments;
  for(const auto &entry:counts)if(entry.second==1){
    const int a=int(entry.first>>32),b=int(std::uint32_t(entry.first));
    segments.push_back({a,b});
    seed.Fixed[a]=seed.Fixed[b]=1;seed.Aliases[a]=globals[a];seed.Aliases[b]=globals[b];
  }
  if(segments.empty())return fail("source_periodic_cut_required");
  std::sort(segments.begin(),segments.end());
  // Even-odd filling uses the real boundary edges, including every hole.
  // Internal source diagonals are free to flip and split.
  chart=std::move(seed);
  return PopulateSeededDomain(chart,segments,target,true,&diagnostics);
}

bool MakeBoundaryCylinderChart(const Frame &frame,
                             const std::vector<std::vector<int>> &loops,
                             const std::vector<Point3> &vertices,double target,Chart &chart,
                             double axialTarget,ChartBuildDiagnostics &diagnostics){
  diagnostics.Stage="cylinder_input";
  // Curvature constrains angular travel, not axial travel. Work in scaled
  // UV coordinates and restore physical height before mapping back to 3D.
  if(!(axialTarget>0)||!(target>0)||!std::isfinite(axialTarget)||!std::isfinite(target))return false;
  const double axialScale=target/axialTarget;
  struct RestoreHeight {Chart &C;double Scale;~RestoreHeight(){for(UV &p:C.Points)p.Y/=Scale;}} restore{chart,axialScale};
  if(frame.Type!=PatchSurfaceType::Cylinder || !(frame.Radius>0) || !(target>0))return false;
  std::size_t boundaryCount=0;double minHeight=std::numeric_limits<double>::infinity(),maxHeight=-minHeight;
  for(const auto &loop:loops)for(int id:loop){
    ++boundaryCount;UV uv;if(!frame.parameter(vertices[id],uv))return false;
    minHeight=std::min(minHeight,uv.Y);maxHeight=std::max(maxHeight,uv.Y);
  }
  if(boundaryCount>100000)return false;
  diagnostics.Stage="boundary_compatibility";
  for(const auto &loop:loops)for(std::size_t i=0;i<loop.size();++i){
    const int a=loop[i],b=loop[(i+1)%loop.size()];UV pa,pb;
    if(!frame.parameter(vertices[a],pa)||!frame.parameter(vertices[b],pb))return false;
    const double arc=std::remainder(pb.X-pa.X,frame.UPeriod),height=(pb.Y-pa.Y)*axialScale;
    const double length=std::hypot(arc,height);
    if(length>target*(1+1e-6)){
      diagnostics.Failure="fixed_boundary_requires_sampling";
      diagnostics.GlobalEdgeA=a;diagnostics.GlobalEdgeB=b;
      diagnostics.EdgeLength=length;diagnostics.EdgeTarget=target;
      return false;
    }
  }
  if(loops.size()==2){
    const double seamEstimate=std::ceil((maxHeight-minHeight)/std::max(axialTarget*.9,1e-12));
    if(!std::isfinite(seamEstimate)||boundaryCount+2*seamEstimate+2>100000)return false;
  }
  if(loops.size()==1){
    diagnostics.Stage="single_loop_unwrap";
    if(loops[0].size()<3)return false;
    chart.Surface=frame;
    for(int id:loops[0]){UV uv;if(!frame.parameter(vertices[id],uv))return false;
      uv.Y*=axialScale;chart.Points.push_back(uv);chart.Aliases.push_back(id);chart.Fixed.push_back(1);}
    double winding=0;
    for(std::size_t i=0;i<chart.Points.size();++i)
      winding+=std::remainder(chart.Points[(i+1)%chart.Points.size()].X-chart.Points[i].X,frame.UPeriod);
    // A single loop with nonzero winding does not bound a disk chart.
    if(std::abs(winding)>frame.UPeriod*.05){diagnostics.Failure="single_loop_winding";return false;}
    Unwrap(chart.Points,frame.UPeriod,0);
    double lo=chart.Points[0].X,hi=lo;
    for(UV p:chart.Points){lo=std::min(lo,p.X);hi=std::max(hi,p.X);}
    if(hi-lo>=frame.UPeriod*(1-1e-10)){diagnostics.Failure="single_loop_period_overlap";return false;}
    diagnostics.Stage="single_loop_seed_triangulation";
    if(!EarClip(chart.Points,chart.Faces,2000000))return false;
  }else if(loops.size()==2){
    return MakeDirectCylinderRing(frame,loops,vertices,target,chart,axialScale,&diagnostics);
  }else {diagnostics.Failure="unsupported_boundary_loop_count";return false;}
  std::vector<std::array<int,2>> segments;
  for(int i=0;i<int(chart.Points.size());++i)segments.push_back({i,(i+1)%int(chart.Points.size())});
  // u=radius*angle and v=axial height preserve the cylinder surface metric.
  // No iterative 3D projection or quality-threshold loop is required.
  return PopulateSeededDomain(chart,segments,target,false,&diagnostics);
}

#include "PeriodicCylinderDomain.h"

bool MakeSeededCylinderChart(const Frame &frame,
                             const std::vector<std::vector<int>> &loops,
                             const std::vector<std::array<int,3>> &sourceFaces,
                             const std::vector<Point3> &vertices,double target,Chart &chart,
                             double axialTarget,ChartBuildDiagnostics &diagnostics){
  // Source topology supplies the trimmed domain. Boundary subdivision stays
  // local until its proposals are reconciled with the neighboring patches.
  if(frame.Type==PatchSurfaceType::Cylinder || frame.Type==PatchSurfaceType::Cone){
    const double scale=target/axialTarget;
    const bool made=MakePeriodicCylinderChart(frame,sourceFaces,vertices,target,scale,chart,diagnostics);
    for(UV &point:chart.Points)point.Y/=scale;
    return made;
  }
  // Preserve the inexpensive boundary-only path when it succeeds. Retry from
  // source topology when a trimmed cylinder violates the ring/ear-clip assumptions.
  if(MakeBoundaryCylinderChart(frame,loops,vertices,target,chart,axialTarget,diagnostics))return true;
  const std::string boundaryFailure=diagnostics.Failure.empty()?diagnostics.Stage+"_failed":diagnostics.Failure;
  chart={};diagnostics={};
  if(!(target>0)||!(axialTarget>0)||!std::isfinite(target)||!std::isfinite(axialTarget)||
     !(frame.UPeriod>0)||!std::isfinite(frame.UPeriod)){
    diagnostics.Stage="source_cylinder_input";diagnostics.Failure="invalid_cylinder_metric";return false;
  }
  const double scale=target/axialTarget;
  bool made=MakeSourceCylinderChart(frame,sourceFaces,vertices,target,scale,chart,diagnostics);
  if(!made && diagnostics.Failure=="source_periodic_cut_required")
    made=MakePeriodicCylinderChart(frame,sourceFaces,vertices,target,scale,chart,diagnostics);
  for(UV &point:chart.Points)point.Y/=scale;
  if(!made){
    if(diagnostics.Failure.empty())diagnostics.Failure=diagnostics.Stage+"_failed";
    diagnostics.Failure+="; boundary_attempt="+boundaryFailure;
  }
  return made;
}
