// Optional geometric guides. These supports never create fixed patch seams.
class AnalyticGuide {
public:
  const std::vector<Point3>& P;
  const std::vector<Tri>& F;
  const std::vector<Vec3>& N;
  std::vector<int> FaceModel;
  std::vector<MeshPatch> Models;
  std::size_t Attempts=0,Cylinders=0,Cones=0,SupportedFaces=0,Projections=0,Fallbacks=0;
  std::size_t FitSucceeded=0,DistancePassed=0,SupportPassed=0;
  double Seconds=0,Tolerance;

  // Check the entire original triangle's vertices and its centroid normal.
  // This is stronger than accepting a fitter's aggregate RMS alone.
  bool supports(const MeshPatch& model,int id)const {
    Vec3 center{};
    for(int v:F[id]){auto q=P[v];if(!ProjectAnalytic(model,q)||Distance(q,P[v])>Tolerance)return false;
      center=Add(center,ToVec(P[v]));}
    center=Mul(center,1.0/3);
    Vec3 axis{},origin{},normal{};double slope=0;
    if(const auto* c=std::get_if<CylinderParameters>(&model.Parameters)){
      axis=Normalize(ToVec(c->Axis.Direction));origin=ToVec(c->Axis.Origin);
    }else if(const auto* c=std::get_if<ConeParameters>(&model.Parameters)){
      axis=Normalize(ToVec(c->Axis.Direction));origin=ToVec(c->Axis.Origin);slope=std::tan(c->SemiAngle);
    }else return false;
    const auto delta=Sub(center,origin),radial=Sub(delta,Mul(axis,Dot(delta,axis)));
    if(Norm(radial)<1e-15)return false;
    normal=Normalize(Sub(Normalize(radial),Mul(axis,slope)));
    return std::abs(Dot(normal,N[id]))>=std::cos(5*std::acos(-1.0)/180);
  }
  double radius(const MeshPatch& model,const Point3& p)const {
    if(const auto* c=std::get_if<CylinderParameters>(&model.Parameters))return c->Radius;
    const auto& c=std::get<ConeParameters>(model.Parameters);
    return Dot(Sub(ToVec(p),ToVec(c.Axis.Origin)),Normalize(ToVec(c.Axis.Direction)))*std::tan(c.SemiAngle);
  }
  bool observable(const MeshPatch& model,const std::vector<int>& support)const {
    const auto axis=std::holds_alternative<CylinderParameters>(model.Parameters)
      ?std::get<CylinderParameters>(model.Parameters).Axis:std::get<ConeParameters>(model.Parameters).Axis;
    const auto direction=Normalize(ToVec(axis.Direction));
    double lo=1e100,hi=-1e100,r=0;bool spread=false;
    for(int id:support){if(Dot(N[id],N[support.front()])<.94)spread=true;
      for(int v:F[id]){const double t=Dot(Sub(ToVec(P[v]),ToVec(axis.Origin)),direction);
        lo=std::min(lo,t);hi=std::max(hi,t);r=std::max(r,radius(model,P[v]));}}
    return spread&&r>Tolerance*10&&hi-lo>.5*r;
  }
  AnalyticGuide(const MeshTopology& mesh,const NativeRemeshConfig& config,
      const std::vector<Point3>& p,const std::vector<Tri>& f,const std::vector<Vec3>& n,
      const std::unordered_map<EdgeKey,EdgeRecord>& edges,
      const std::unordered_map<EdgeKey,int>& features):P(p),F(f),N(n),FaceModel(f.size(),-1){
    const auto start=std::chrono::steady_clock::now();Tolerance=std::min(.02,config.MaximumDeviation*.25);
    std::vector<std::array<int,3>> adjacent(F.size(),std::array<int,3>{-1,-1,-1});
    for(std::size_t id=0;id<F.size();++id)for(int k=0;k<3;++k){const auto key=Key(F[id][k],F[id][(k+1)%3]);
      const auto& e=edges.at(key);if(e.FaceCount!=2||features.count(key))continue;
      const int other=e.Faces[0]==int(id)?e.Faces[1]:e.Faces[0];
      if(Dot(N[id],N[other])>.94)adjacent[id][k]=other;
    }
    std::vector<std::pair<double,int>> seeds;seeds.reserve(F.size());
    std::vector<Vec3> generators(F.size());std::vector<double> aspect(F.size());
    for(int id=0;id<int(F.size());++id){double shortest=1e100,longest=0;
      for(int k=0;k<3;++k){const auto delta=Sub(ToVec(P[F[id][k]]),ToVec(P[F[id][(k+1)%3]]));const double d=Norm(delta);shortest=std::min(shortest,d);
        if(d>longest){longest=d;generators[id]=Normalize(delta);}}
      aspect[id]=longest/std::max(shortest,1e-15);seeds.push_back({-aspect[id],id});}
    std::sort(seeds.begin(),seeds.end());std::vector<unsigned char> tried(F.size(),0);std::vector<int> stamps(F.size(),-1);
    for(const auto& seed:seeds){if(Attempts>=4096)break;const int root=seed.second;if(FaceModel[root]>=0||tried[root]||aspect[root]<3)continue;
      std::vector<int> support{root};const int stamp=root;stamps[root]=stamp;
      for(std::size_t i=0;i<support.size()&&support.size()<64;++i)for(int other:adjacent[support[i]]){
        if(other>=0&&FaceModel[other]<0&&stamps[other]!=stamp&&aspect[other]>=3
            &&std::abs(Dot(generators[other],generators[root]))>.94){stamps[other]=stamp;support.push_back(other);}}
      for(int id:support)tried[id]=1;
      bool spread=false;for(int id:support)if(Dot(N[id],N[root])<.94)spread=true;
      if(support.size()<12||!spread)continue;
      ++Attempts;MeshPatch best;double score=1e100;
      const auto consider=[&](ISurfaceFitter& fitter){if(!fitter.fit(mesh,support))return;++FitSucceeded;
        if(fitter.computeMaxError()>Tolerance)return;++DistancePassed;
        MeshPatch model;model.SurfaceType=fitter.getType();model.Parameters=fitter.getParameters();
        if(!observable(model,support))return;for(int id:support)if(!supports(model,id))return;
        ++SupportPassed;if(fitter.computeMaxError()<score){score=fitter.computeMaxError();best=model;}};
      CylinderSurfaceFitter cylinder;consider(cylinder);ConeSurfaceFitter cone;consider(cone);
      if(score==1e100)continue;
      const int index=int(Models.size());Models.push_back(best);
      Cylinders+=best.SurfaceType==PatchSurfaceType::Cylinder;Cones+=best.SurfaceType==PatchSurfaceType::Cone;
      std::vector<int> grow;for(int id:support){FaceModel[id]=index;grow.push_back(id);}
      for(std::size_t i=0;i<grow.size();++i)for(int other:adjacent[grow[i]])if(other>=0&&FaceModel[other]<0&&supports(best,other)){
        FaceModel[other]=index;grow.push_back(other);}
      SupportedFaces+=grow.size();
    }
    Seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::clog<<"[Generic] analytic_guides: attempts="<<Attempts<<", cylinders="<<Cylinders<<", cones="<<Cones
      <<", supported_faces="<<SupportedFaces<<", fit_succeeded="<<FitSucceeded<<", distance_passed="<<DistancePassed
      <<", support_passed="<<SupportPassed<<", seconds="<<Seconds<<std::endl;
  }
};
