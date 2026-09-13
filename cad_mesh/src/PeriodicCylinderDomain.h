// Included after PlanarDomain. Port of analytic_remesh.py's trimmed periodic
// cylinder cut and paired seam sampling. The native CDT is seeded from source
// connectivity; source interior diagonals remain unconstrained.
bool MakePeriodicCylinderSeed(const Frame &frame,
    const std::vector<std::array<int,3>> &sourceFaces,
    const std::vector<Point3> &vertices,double target,double axialScale,
    double cut,Chart &chart,std::vector<std::array<int,2>> &segments,
    ChartBuildDiagnostics &diagnostics){
  diagnostics.Stage="periodic_cylinder_cut";
  const auto fail=[&](const char *reason){diagnostics.Failure=reason;return false;};
  chart={};chart.Surface=frame;segments.clear();
  if(sourceFaces.empty()||sourceFaces.size()>200000)return fail("source_mesh_size_budget");
  const double period=frame.UPeriod;
  std::unordered_map<int,UV> parameters;
  std::unordered_map<EdgeKey,int> copies,physicalCounts;
  std::vector<int> globals;
  std::unordered_map<EdgeKey,Vec3> sourceNormals;
  std::unordered_set<EdgeKey> featureEdges;
  for(const auto &f:sourceFaces){
    const Vec3 normal=Normalize(Cross(Sub(ToVec(vertices[f[1]]),ToVec(vertices[f[0]])),Sub(ToVec(vertices[f[2]]),ToVec(vertices[f[0]]))));
    for(int k=0;k<3;++k){const auto key=Key(f[k],f[(k+1)%3]);const auto inserted=sourceNormals.emplace(key,normal);
      if(!inserted.second && Dot(inserted.first->second,normal)<std::cos(5*Pi/180))featureEdges.insert(key);
    }
  }
  for(const auto &source:sourceFaces){
    double angles[3],lifted[3];
    for(int k=0;k<3;++k){
      const int id=source[k];
      if(id<0||std::size_t(id)>=vertices.size())return fail("source_vertex_index");
      auto found=parameters.find(id);
      if(found==parameters.end()){
        UV uv;if(!frame.parameter(vertices[id],uv)||!std::isfinite(uv.X)||!std::isfinite(uv.Y))
          return fail("source_parameterization");
        found=parameters.emplace(id,uv).first;
      }
      angles[k]=std::fmod(found->second.X-cut,period);
      if(angles[k]<0)angles[k]+=period;
    }
    for(int k=0;k<3;++k)lifted[k]=angles[0]+std::remainder(angles[k]-angles[0],period);
    const double shift=period*std::floor((lifted[0]+lifted[1]+lifted[2])/(3*period));
    for(double &x:lifted)x-=shift;
    if(*std::max_element(lifted,lifted+3)-*std::min_element(lifted,lifted+3)>=period*.5)
      return fail("periodic_ambiguous_angular_triangle");
    std::array<int,3> face;
    for(int k=0;k<3;++k){
      const int winding=int(std::llround((lifted[k]-angles[k])/period));
      const EdgeKey key=(EdgeKey(std::uint32_t(source[k]))<<32)|std::uint32_t(winding);
      auto found=copies.find(key);
      if(found==copies.end()){
        face[k]=int(chart.Points.size());copies.emplace(key,face[k]);globals.push_back(source[k]);
        chart.Points.push_back({lifted[k]+cut,parameters.at(source[k]).Y*axialScale});
        chart.Aliases.push_back(-1);chart.Fixed.push_back(0);
      }else face[k]=found->second;
      if(++physicalCounts[Key(source[k],source[(k+1)%3])]>2)return fail("source_nonmanifold_edge");
    }
    chart.Faces.push_back(face);
  }
  if(chart.Points.size()>100000)return fail("mesh_size_budget");
  int orientation=0;
  struct Incidence {int Count=0,A=-1,B=-1;};
  std::unordered_map<EdgeKey,Incidence> edges;
  for(auto &face:chart.Faces){
    const double area=Cross2(chart.Points[face[0]],chart.Points[face[1]],chart.Points[face[2]]);
    if(!std::isfinite(area)||area==0)return fail("periodic_uv_degenerate_face");
    const int sign=area>0?1:-1;
    if(orientation&&orientation!=sign)return fail("periodic_uv_fold_or_inconsistent_orientation");
    orientation=sign;
    if(sign<0)std::swap(face[1],face[2]);
    for(int k=0;k<3;++k){
      const int a=face[k],b=face[(k+1)%3];auto &edge=edges[Key(a,b)];
      if(++edge.Count>2)return fail("periodic_nonmanifold_edge");
      if(edge.Count==2 && (edge.A!=b||edge.B!=a))return fail("periodic_inconsistent_edge_orientation");
      if(edge.Count==1){edge.A=a;edge.B=b;}
    }
  }
  std::vector<EdgeKey> boundary;
  std::unordered_map<EdgeKey,int> cutCounts;
  for(const auto &entry:edges)if(entry.second.Count==1){
    boundary.push_back(entry.first);
    ++cutCounts[Key(globals[entry.second.A],globals[entry.second.B])];
  }
  std::sort(boundary.begin(),boundary.end());
  if(boundary.empty())return fail("periodic_empty_cut_boundary");
  struct Samples {int Divisions=1,FirstAlias=-2;double Length=0;};
  std::unordered_map<EdgeKey,Samples> seamSamples;
  // Each source-internal edge exposed by the cut must have exactly two copies.
  for(EdgeKey key:boundary){
    const auto edge=edges.at(key);const EdgeKey physical=Key(globals[edge.A],globals[edge.B]);
    if(cutCounts.at(physical)!=physicalCounts.at(physical))return fail("periodic_unpaired_cut_edge");
  }
  diagnostics.Stage="periodic_cylinder_seam_sampling";
  for(const auto &entry:edges)if(entry.second.Count==2 && featureEdges.count(Key(globals[entry.second.A],globals[entry.second.B])))
    boundary.push_back(entry.first);
  std::sort(boundary.begin(),boundary.end());
  std::unordered_map<EdgeKey,std::vector<int>> chains;
  int nextAlias=-2;
  for(EdgeKey key:boundary){
    const int a=int(key>>32),b=int(std::uint32_t(key));
    const UV pa=chart.Points[a],pb=chart.Points[b];
    const double length=std::hypot(pb.X-pa.X,pb.Y-pa.Y);
    const EdgeKey physical=Key(globals[a],globals[b]);
    chart.Aliases[a]=globals[a];chart.Aliases[b]=globals[b];chart.Fixed[a]=chart.Fixed[b]=1;
    std::vector<int> chain{a};
    if(physicalCounts.at(physical)==2){
      auto found=seamSamples.find(physical);
      if(found==seamSamples.end()){
        const double count=std::max(1.0,std::ceil(length/(target*.8)));
        if(!std::isfinite(count)||count>100000)return fail("periodic_seam_sample_budget");
        Samples samples;samples.Divisions=int(count);samples.FirstAlias=nextAlias;samples.Length=length;
        nextAlias-=samples.Divisions-1;found=seamSamples.emplace(physical,samples).first;
      }
      const Samples samples=found->second;
      if(std::abs(length-samples.Length)>std::max(target,length)*1e-8)
        return fail("periodic_seam_metric_mismatch");
      if(chart.Points.size()+samples.Divisions-1>100000)return fail("periodic_seam_sample_budget");
      for(int j=1;j<samples.Divisions;++j){
        const double t=double(j)/samples.Divisions;
        const int order=globals[a]<globals[b]?j:samples.Divisions-j;
        chain.push_back(int(chart.Points.size()));
        chart.Points.push_back({pa.X+(pb.X-pa.X)*t,pa.Y+(pb.Y-pa.Y)*t});
        chart.Aliases.push_back(samples.FirstAlias-(order-1));chart.Fixed.push_back(1);
      }
    }else if(length>target*(1+1e-6)){
      const double divisions=std::ceil(length/(target*.8));
      if(!std::isfinite(divisions)||divisions>100000||chart.Points.size()+divisions>100000)return fail("boundary_sample_budget");
      for(int j=1;j<int(divisions);++j){const double t=double(j)/divisions;
        const int v=int(chart.Points.size());chain.push_back(v);
        chart.Points.push_back({pa.X+(pb.X-pa.X)*t,pa.Y+(pb.Y-pa.Y)*t});
        chart.Aliases.push_back(-1);chart.Fixed.push_back(1);
        chart.BoundarySamples.push_back({globals[a],globals[b],v,t});
      }
    }
    chain.push_back(b);
    for(std::size_t j=1;j<chain.size();++j){
      if(edges.at(key).Count==1)segments.push_back({chain[j-1],chain[j]});
      else chart.InteriorConstraints.push_back({chain[j-1],chain[j]});
    }
    chains.emplace(key,std::move(chain));
  }
  // Insert paired samples into their incident seed faces. A center fan handles
  // multiple subdivided edges without zero-area triangles on collinear sites.
  std::vector<std::array<int,3>> refined;
  for(const auto &face:chart.Faces){
    std::vector<int> perimeter;
    for(int k=0;k<3;++k){
      const int a=face[k],b=face[(k+1)%3];perimeter.push_back(a);
      auto found=chains.find(Key(a,b));if(found==chains.end())continue;
      const auto &chain=found->second;
      if(chain.front()==a)for(std::size_t j=1;j+1<chain.size();++j)perimeter.push_back(chain[j]);
      else for(int j=int(chain.size())-2;j>0;--j)perimeter.push_back(chain[j]);
    }
    if(perimeter.size()==3){refined.push_back(face);continue;}
    if(chart.Points.size()>=100000||refined.size()+perimeter.size()>200000)return fail("mesh_size_budget");
    const UV a=chart.Points[face[0]],b=chart.Points[face[1]],c=chart.Points[face[2]];
    const int center=int(chart.Points.size());
    chart.Points.push_back({(a.X+b.X+c.X)/3,(a.Y+b.Y+c.Y)/3});
    chart.Aliases.push_back(-1);chart.Fixed.push_back(0);
    for(std::size_t j=0;j<perimeter.size();++j)
      refined.push_back({perimeter[j],perimeter[(j+1)%perimeter.size()],center});
  }
  if(refined.size()>200000)return fail("mesh_size_budget");
  chart.Faces=std::move(refined);
  return true;
}

bool MakePeriodicCylinderChart(const Frame &frame,
    const std::vector<std::array<int,3>> &sourceFaces,
    const std::vector<Point3> &vertices,double target,double axialScale,
    Chart &chart,ChartBuildDiagnostics &diagnostics){
  // Like the Python route, try alternative meridians. Shared boundary vertices
  // stay shared in the seed connectivity; no simple-loop tracing is required.
  for(int route=0;route<12;++route){
    diagnostics={};chart={};std::vector<std::array<int,2>> segments;
    if(!MakePeriodicCylinderSeed(frame,sourceFaces,vertices,target,axialScale,
                                frame.UPeriod*route/12,chart,segments,diagnostics)){
      if(diagnostics.Failure=="fixed_boundary_requires_sampling"||
         diagnostics.Failure=="source_mesh_size_budget"||diagnostics.Failure=="source_nonmanifold_edge"||
         diagnostics.Failure=="source_vertex_index"||diagnostics.Failure=="source_parameterization")break;
      continue;
    }
    if(PopulateSeededDomain(chart,segments,target,true,&diagnostics))return true;
    if(diagnostics.Failure.empty())diagnostics.Failure=diagnostics.Stage+"_failed";
    // Resource exhaustion and a physical boundary mismatch cannot be repaired
    // by repeating the same expensive refinement at eleven other meridians.
    if(diagnostics.Failure=="work_budget"||diagnostics.Failure=="mesh_size_budget"||
       diagnostics.Failure=="edge_repair_stalled"||diagnostics.Failure=="fixed_boundary_requires_sampling")break;
  }
  diagnostics.Failure+="; periodic_cut_attempt_failed";
  return false;
}
