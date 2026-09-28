#pragma once

namespace CadMesh {
inline constexpr const char *WallThicknessRayKernelSource = R"cuda(
struct cm_thickness_node {
  double bounds[6];
  int children[2];
  int firstFace,faceCount;
};
struct cm_thickness_seed { int face; double distance,u,v; };
__device__ bool cm_ray_box(const cm_thickness_node &n,const double *o,const double *d,double limit) {
  double lo=0.0,hi=limit;
  for(int k=0;k<3;++k){
    const double a=n.bounds[k],b=n.bounds[k+3];
    if(d[k]==0.0){if(o[k]<a||o[k]>b)return false;continue;}
    double x=(a-o[k])/d[k],y=(b-o[k])/d[k];if(x>y){double t=x;x=y;y=t;}
    lo=fmax(lo,x);hi=fmin(hi,y);if(lo>hi)return false;
  }
  return true;
}
__device__ bool cm_ray_triangle(const double *o,const double *d,const double *a,
                                const double *b,const double *c,double &t,double &u,double &v) {
  const double e1[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
  const double e2[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
  const double p[3]={d[1]*e2[2]-d[2]*e2[1],d[2]*e2[0]-d[0]*e2[2],d[0]*e2[1]-d[1]*e2[0]};
  const double det=e1[0]*p[0]+e1[1]*p[1]+e1[2]*p[2];
  if(fabs(det)<=1e-30)return false;
  const double inv=1.0/det,q[3]={o[0]-a[0],o[1]-a[1],o[2]-a[2]};
  u=(q[0]*p[0]+q[1]*p[1]+q[2]*p[2])*inv;
  if(u<0.0||u>1.0)return false;
  const double r[3]={q[1]*e1[2]-q[2]*e1[1],q[2]*e1[0]-q[0]*e1[2],q[0]*e1[1]-q[1]*e1[0]};
  v=(d[0]*r[0]+d[1]*r[1]+d[2]*r[2])*inv;
  if(v<0.0||u+v>1.0)return false;
  t=(e2[0]*r[0]+e2[1]*r[1]+e2[2]*r[2])*inv;
  return t>0.0&&isfinite(t);
}
extern "C" __global__ void cadmesh_wall_thickness_ray_seeds(
    const double *vertices,const double *normals,const int *triangles,
    const cm_thickness_node *nodes,const int *leafFaceIds,int vertexCount,
    cm_thickness_seed *seeds) {
  const int id=int(blockIdx.x*blockDim.x+threadIdx.x);if(id>=vertexCount)return;
  const double *o=vertices+3*id,*d=normals+3*id;
  int stack[64],top=0;stack[top++]=0;double closest=1.7976931348623157e+308;int hit=-1;double hu=0,hv=0;
  while(top>0){
    const int ni=stack[--top];const auto &node=nodes[ni];
    if(!cm_ray_box(node,o,d,closest))continue;
    if(node.children[0]>=0){
      if(top+2>64)continue;
      stack[top++]=node.children[0];stack[top++]=node.children[1];continue;
    }
    for(int k=0;k<node.faceCount;++k){
      const int fi=leafFaceIds[node.firstFace+k];const int *f=triangles+3*fi;
      if(f[0]==id||f[1]==id||f[2]==id)continue;
      double t,u,v;
      if(cm_ray_triangle(o,d,vertices+3*f[0],vertices+3*f[1],vertices+3*f[2],t,u,v)&&t<closest){
        closest=t;hit=fi;hu=u;hv=v;
      }
    }
  }
  seeds[id].face=hit;seeds[id].distance=hit>=0?closest:0.0;seeds[id].u=hu;seeds[id].v=hv;
}
)cuda";
} // namespace CadMesh
