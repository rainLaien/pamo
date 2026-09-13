// Private chart cleanup. Collapse only ordinary interior vertices; shared
// boundary aliases and both copies of a periodic seam are immutable.
void CollapseCylinderInterior(Chart &chart,double target,double angularLimit){
  if(chart.PreserveCylinderColumns)return; // Regular rows are sized at creation.
  for(int pass=0;pass<3;++pass){
    std::vector<std::vector<int>> incident(chart.Points.size());
    std::unordered_map<EdgeKey,std::vector<int>> edges;
    for(int i=0;i<int(chart.Faces.size());++i){const auto &f=chart.Faces[i];
      for(int k=0;k<3;++k){incident[f[k]].push_back(i);edges[Key(f[k],f[(k+1)%3])].push_back(i);}}
    std::vector<unsigned char> locked(chart.Points.size(),0),dead(chart.Faces.size(),0);
    for(std::size_t v=0;v<locked.size();++v)locked[v]=chart.Fixed[v]||chart.Aliases[v]!=-1;
    for(const auto &e:edges)if(e.second.size()!=2){locked[int(e.first>>32)]=1;locked[int(std::uint32_t(e.first))]=1;}
    std::vector<std::pair<double,EdgeKey>> candidates;
    for(const auto &e:edges){const int a=int(e.first>>32),b=int(std::uint32_t(e.first));
      if(locked[a]||locked[b]||e.second.size()!=2)continue;
      const UV p=chart.Points[a],q=chart.Points[b];const double d=std::hypot(p.X-q.X,p.Y-q.Y);
      if(d<target*.35)candidates.push_back({d,e.first});}
    std::sort(candidates.begin(),candidates.end());
    int changed=0;
    const auto quality=[&](const std::array<int,3> &f){
      const UV a=chart.Points[f[0]],b=chart.Points[f[1]],c=chart.Points[f[2]];
      const double sum=std::pow(a.X-b.X,2)+std::pow(a.Y-b.Y,2)+std::pow(b.X-c.X,2)+std::pow(b.Y-c.Y,2)+std::pow(c.X-a.X,2)+std::pow(c.Y-a.Y,2);
      return sum>0?std::abs(Cross2(a,b,c))/sum:0.0;
    };
    for(const auto &candidate:candidates){
      const int a=int(candidate.second>>32),b=int(std::uint32_t(candidate.second));
      if(locked[a]||locked[b])continue;
      std::set<int> neighborsA,neighborsB,faces;
      for(int f:incident[a]){faces.insert(f);for(int v:chart.Faces[f])if(v!=a)neighborsA.insert(v);}
      for(int f:incident[b]){faces.insert(f);for(int v:chart.Faces[f])if(v!=b)neighborsB.insert(v);}
      int common=0;for(int v:neighborsA)if(neighborsB.count(v))++common;
      if(common!=2)continue; // Interior manifold link condition.
      std::vector<double> lengths;
      for(int v:neighborsA)if(v!=b)lengths.push_back(std::hypot(chart.Points[v].X-chart.Points[a].X,chart.Points[v].Y-chart.Points[a].Y));
      for(int v:neighborsB)if(v!=a)lengths.push_back(std::hypot(chart.Points[v].X-chart.Points[b].X,chart.Points[v].Y-chart.Points[b].Y));
      if(lengths.empty())continue;
      std::sort(lengths.begin(),lengths.end());
      if(candidate.first>=.35*std::min(target,lengths[lengths.size()/2]))continue;
      double oldQuality=1;for(int f:faces)oldQuality=std::min(oldQuality,quality(chart.Faces[f]));
      int keep=-1;
      // Retain an existing endpoint instead of moving the fitted surface or
      // creating a new angular sample. Try both directions deterministically.
      for(int choice:{a,b}){
        const int drop=choice==a?b:a;bool valid=true;
        for(int id:faces){auto f=chart.Faces[id];
          const bool hasA=std::find(f.begin(),f.end(),a)!=f.end(),hasB=std::find(f.begin(),f.end(),b)!=f.end();
          if(hasA&&hasB)continue;
          double oldMax=0;for(int k=0;k<3;++k){const UV p=chart.Points[f[k]],q=chart.Points[f[(k+1)%3]];oldMax=std::max(oldMax,std::hypot(p.X-q.X,p.Y-q.Y));}
          for(int &v:f)if(v==drop)v=choice;
          const UV p=chart.Points[f[0]],q=chart.Points[f[1]],r=chart.Points[f[2]];
          if(Cross2(p,q,r)<=0||quality(f)+1e-12<oldQuality ||
             std::max({p.X,q.X,r.X})-std::min({p.X,q.X,r.X})>angularLimit*(1+1e-8)){valid=false;break;}
          for(int k=0;k<3;++k){const UV u=chart.Points[f[k]],v=chart.Points[f[(k+1)%3]];
            if(std::hypot(u.X-v.X,u.Y-v.Y)>std::max(target,oldMax)*(1+1e-8)){valid=false;break;}}
          if(!valid)break;
        }
        if(valid){keep=choice;break;}
      }
      if(keep<0)continue;
      const int drop=keep==a?b:a;
      // Disjoint one-rings keep this sweep's incidence index valid.
      for(int id:faces){auto &f=chart.Faces[id];for(int v:f)locked[v]=1;
        for(int &v:f)if(v==drop)v=keep;
        if(f[0]==f[1]||f[1]==f[2]||f[2]==f[0])dead[id]=1;}
      ++changed;
    }
    if(!changed)break;
    std::size_t out=0;for(std::size_t i=0;i<chart.Faces.size();++i)if(!dead[i])chart.Faces[out++]=chart.Faces[i];
    chart.Faces.resize(out);
  }
}
