// Private implementation fragment: UV, Chart and PlanarDomain are in scope.
// Immutable per-chart boundary index; no shared mutable worker state.
class GradedPlanarBoundary {
  struct Segment { UV A,B; double Length; };
  struct Node { UV Lo,Hi; double Minimum; int Begin,End,Left=-1,Right=-1; };
  std::vector<Segment> Segments;
  std::vector<Node> Nodes;
  int build(int begin,int end){
    Node n{Segments[begin].A,Segments[begin].A,
           std::numeric_limits<double>::infinity(),begin,end};
    for(int i=begin;i<end;++i){const auto &s=Segments[i];
      n.Lo.X=std::min({n.Lo.X,s.A.X,s.B.X});n.Lo.Y=std::min({n.Lo.Y,s.A.Y,s.B.Y});
      n.Hi.X=std::max({n.Hi.X,s.A.X,s.B.X});n.Hi.Y=std::max({n.Hi.Y,s.A.Y,s.B.Y});
      n.Minimum=std::min(n.Minimum,s.Length);
    }
    const int id=int(Nodes.size());Nodes.push_back(n);
    if(end-begin>8){
      const bool x=n.Hi.X-n.Lo.X>=n.Hi.Y-n.Lo.Y;const int middle=(begin+end)/2;
      std::nth_element(Segments.begin()+begin,Segments.begin()+middle,Segments.begin()+end,
        [x](const Segment &a,const Segment &b){return x?a.A.X+a.B.X<b.A.X+b.B.X:a.A.Y+a.B.Y<b.A.Y+b.B.Y;});
      const int left=build(begin,middle),right=build(middle,end);
      Nodes[id].Left=left;Nodes[id].Right=right;
    }
    return id;
  }
  static double boxDistance(const Node &n,UV p){
    return std::hypot(std::max({n.Lo.X-p.X,0.0,p.X-n.Hi.X}),
                      std::max({n.Lo.Y-p.Y,0.0,p.Y-n.Hi.Y}));
  }
  void measure(int id,UV p,double &size,double &distance)const{
    const Node &n=Nodes[id];const double bound=boxDistance(n,p);
    if(1.25*n.Minimum+.3*bound>=size && bound>=distance)return;
    if(n.Left>=0){
      int a=n.Left,b=n.Right;
      if(boxDistance(Nodes[b],p)<boxDistance(Nodes[a],p))std::swap(a,b);
      measure(a,p,size,distance);measure(b,p,size,distance);return;
    }
    for(int i=n.Begin;i<n.End;++i){const auto &s=Segments[i];
      const double dx=s.B.X-s.A.X,dy=s.B.Y-s.A.Y;
      const double t=std::clamp(((p.X-s.A.X)*dx+(p.Y-s.A.Y)*dy)/(s.Length*s.Length),0.0,1.0);
      const double d=std::hypot(p.X-s.A.X-t*dx,p.Y-s.A.Y-t*dy);
      distance=std::min(distance,d);size=std::min(size,1.25*s.Length+.3*d);
    }
  }
  bool parity(int id,UV p)const{
    const Node &n=Nodes[id];
    if(p.Y<n.Lo.Y||p.Y>=n.Hi.Y||p.X>=n.Hi.X)return false;
    if(n.Left>=0)return parity(n.Left,p)!=parity(n.Right,p);
    bool inside=false;
    for(int i=n.Begin;i<n.End;++i){const auto &s=Segments[i];
      if((s.A.Y>p.Y)!=(s.B.Y>p.Y))
        if(p.X<s.A.X+(p.Y-s.A.Y)*(s.B.X-s.A.X)/(s.B.Y-s.A.Y))inside=!inside;
    }
    return inside;
  }
public:
  GradedPlanarBoundary(const Chart &chart,const std::vector<std::array<int,2>> &edges){
    for(const auto &e:edges){const UV a=chart.Points[e[0]],b=chart.Points[e[1]];
      const double length=std::hypot(a.X-b.X,a.Y-b.Y);
      if(length>0 && std::isfinite(length))Segments.push_back({a,b,length});
    }
    if(!Segments.empty())build(0,int(Segments.size()));
  }
  void sample(UV p,double &size,double &distance)const{
    if(!Nodes.empty())measure(0,p,size,distance);
  }
  bool inside(UV p)const{return !Nodes.empty() && parity(0,p);}
};

bool PopulateGradedPlane(Chart &chart,PlanarDomain &domain,
                         const std::vector<std::array<int,2>> &segments,double target){
  GradedPlanarBoundary boundary(chart,segments);
  UV lo=chart.Points[0],hi=lo;
  for(UV p:chart.Points){lo.X=std::min(lo.X,p.X);lo.Y=std::min(lo.Y,p.Y);
    hi.X=std::max(hi.X,p.X);hi.Y=std::max(hi.Y,p.Y);}
  struct Cell {double X,Y,Side;int Depth;};
  const double root=std::max(hi.X-lo.X,hi.Y-lo.Y);
  if(!(root>0)||!std::isfinite(root))return false;
  // Breadth-first traversal covers the whole patch before spending a budget
  // on the finest boundary cells of any one corner.
  std::deque<Cell> pending{{lo.X,lo.Y,root,0}};
  std::size_t visited=0;
  // Optional sampling ends with the current chart at a budget, not a rollback.
  // Long-edge repair and the existing angle request follow this stage.
  while(!pending.empty() && visited++<200000 && chart.Points.size()<80000 && chart.Faces.size()<160000){
    const Cell c=pending.front();pending.pop_front();
    const UV center{c.X+c.Side*.5,c.Y+c.Side*.5};
    double size=target,distance=std::numeric_limits<double>::infinity();
    boundary.sample(center,size,distance);
    const bool inside=boundary.inside(center);
    // Do not traverse empty cells outside the polygon or inside its holes.
    if(!inside && distance>c.Side*.708)continue;
    // Use a conservative size bound across the cell, not just its center.
    const double local=std::max(size-.3*c.Side*.708,target*1e-8);
    if(c.Side>local*.85 && c.Depth<28){
      const double half=c.Side*.5;
      for(int y=1;y>=0;--y)for(int x=1;x>=0;--x)
        pending.push_back({c.X+x*half,c.Y+y*half,half,c.Depth+1});
      continue;
    }
    if(!inside || distance<size*.3)continue;
    const int face=domain.locate(center);
    if(face<0)continue;
    bool close=false;
    for(int v:chart.Faces[face]){const UV p=chart.Points[v];
      if(std::hypot(p.X-center.X,p.Y-center.Y)<c.Side*.3){close=true;break;}}
    if(!close && !domain.insert(center,face))return false;
  }
  return domain.legalize();
}
