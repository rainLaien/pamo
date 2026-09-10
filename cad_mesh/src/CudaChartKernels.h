#pragma once
namespace CadMesh {
inline constexpr const char *ChartKernelSource = R"cuda(
// Each block owns one chart. All topology writes are confined to that chart;
// vertex-disjoint flips/moves and face-disjoint splits need no global lock.
struct cm_uv { double x,y; };
__device__ double cm_area(cm_uv a,cm_uv b,cm_uv c) {
  return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
}
__device__ double cm_len(cm_uv a,cm_uv b) {
  double x=a.x-b.x,y=a.y-b.y;return x*x+y*y;
}
__device__ double cm_quality(cm_uv a,cm_uv b,cm_uv c) {
  double d=cm_len(a,b)+cm_len(b,c)+cm_len(c,a);
  return d>0?3.464101615137754587* fabs(cm_area(a,b,c))/d:0;
}
__device__ unsigned long long cm_key(int a,int b) {
  if(a>b){int t=a;a=b;b=t;}
  return ((unsigned long long)(unsigned int)a<<32)|(unsigned int)b;
}
__device__ int cm_slot(unsigned long long key,unsigned long long *keys,int cap,bool insert) {
  unsigned long long h=key;h^=h>>33;h*=0xff51afd7ed558ccdULL;h^=h>>33;
  int s=int(h)&(cap-1);
  for(int k=0;k<cap;++k){
    unsigned long long old=insert?atomicCAS(keys+s,~0ULL,key):keys[s];
    if(old==key||(insert&&old==~0ULL))return s;
    if(!insert&&old==~0ULL)return -1;
    s=(s+1)&(cap-1);
  }return -1;
}
__device__ int cm_other(const int *f,int a,int b) {
  for(int k=0;k<3;++k)if(f[k]!=a&&f[k]!=b)return f[k];return -1;
}
__device__ bool cm_flip(int e,unsigned long long *keys,int *first,int *second,
                       int *faces,cm_uv *p,int hc,double target,int &a,int &b,int &c,int &d) {
  if(keys[e]==~0ULL||first[e]<0||second[e]<0)return false;
  a=int(keys[e]>>32);b=int(keys[e]&0xffffffffULL);
  c=cm_other(faces+3*first[e],a,b);d=cm_other(faces+3*second[e],a,b);
  if(c<0||d<0||c==d||cm_slot(cm_key(c,d),keys,hc,false)>=0)return false;
  if(cm_area(p[c],p[d],p[a])*cm_area(p[c],p[d],p[b])>=-1e-20)return false;
  if(cm_len(p[c],p[d])>fmax(target*target,cm_len(p[a],p[b]))*(1+2.000001e-6))return false;
  double before=fmin(cm_quality(p[a],p[b],p[c]),cm_quality(p[b],p[a],p[d]));
  double after=fmin(cm_quality(p[c],p[d],p[a]),cm_quality(p[d],p[c],p[b]));
  return after>before+1e-8;
}
extern "C" __global__ void cadmesh_chart_step(
    const int *jobs,const double *targets,cm_uv *points,int *triangles,
    unsigned long long *allKeys,int *scratch,int *states,int operation) {
  int j=int(blockIdx.x),lane=int(threadIdx.x),stride=int(blockDim.x);
  const int *job=jobs+8*j;
  int vo=job[0],fo=job[1],eo=job[2],so=job[3],vc=job[4],fc=job[5],hc=job[6],fixed=job[7];
  int *state=states+8*j;
  // state: points, faces, status, changed, splits, flips, moves, reserved
  if(state[2]!=0)return;
  int nv=state[0],nf=state[1];double target=targets[j];
  cm_uv *p=points+vo;int *f=triangles+3*fo;
  unsigned long long *keys=allKeys+eo;
  int *first=scratch+so,*second=first+hc,*work=second+hc;
  int *claim=work+hc,*head=claim+vc,*next=head+vc;
  __shared__ int additions,changes;
  if(lane==0){additions=0;changes=0;state[3]=0;}
  // GPU hash adjacency: no CPU edge maps or transfers between iterations.
  for(int e=lane;e<hc;e+=stride){keys[e]=~0ULL;first[e]=-1;second[e]=-1;work[e]=-1;}
  for(int v=lane;v<nv;v+=stride){claim[v]=2147483647;head[v]=-1;}
  __syncthreads();
  for(int t=lane;t<nf;t+=stride)for(int k=0;k<3;++k){
    int a=f[3*t+k],b=f[3*t+(k+1)%3];
    if(a<0||b<0||a>=nv||b>=nv||a==b){atomicExch(state+2,3);continue;}
    int e=cm_slot(cm_key(a,b),keys,hc,true);
    if(e<0){atomicExch(state+2,2);continue;}
    int old=atomicCAS(first+e,-1,t);
    if(old!=-1&&atomicCAS(second+e,-1,t)!=-1)atomicExch(state+2,3);
    if(operation==2)next[3*t+k]=atomicExch(head+a,3*t+k);
  }
  __syncthreads();
  if(state[2])return;
  if(operation==0){
    // Score before committing anything; all four vertices must grant the edge.
    for(int e=lane;e<hc;e+=stride){int a,b,c,d;
      if(cm_flip(e,keys,first,second,f,p,hc,target,a,b,c,d)){
        work[e]=1;atomicMin(claim+a,e);atomicMin(claim+b,e);atomicMin(claim+c,e);atomicMin(claim+d,e);
      }
    }
    __syncthreads();
    // Store opposite vertices before any face is modified (avoid read/write
    // races even for losing candidates sharing a face with a winner).
    for(int e=lane;e<hc;e+=stride)if(work[e]==1){
      int a=int(keys[e]>>32),b=int(keys[e]&0xffffffffULL);
      int c=cm_other(f+3*first[e],a,b),d=cm_other(f+3*second[e],a,b);
      work[e]=(claim[a]==e&&claim[b]==e&&claim[c]==e&&claim[d]==e)?c:-1;
      // second face's opposite is read later only by winning disjoint edges.
    }
    __syncthreads();
    for(int e=lane;e<hc;e+=stride)if(work[e]>=0){
      int a=int(keys[e]>>32),b=int(keys[e]&0xffffffffULL),c=work[e];
      int t=first[e],u=second[e],d=cm_other(f+3*u,a,b);
      f[3*t]=c;f[3*t+1]=d;f[3*t+2]=a;
      f[3*u]=d;f[3*u+1]=c;f[3*u+2]=b;
      if(cm_area(p[c],p[d],p[a])<0){f[3*t+1]=a;f[3*t+2]=d;}
      if(cm_area(p[d],p[c],p[b])<0){f[3*u+1]=b;f[3*u+2]=c;}
      atomicAdd(&changes,1);
    }
    __syncthreads();
    if(lane==0){state[3]=changes;state[5]+=changes;}
  }else if(operation==1){
    // Every face votes for its longest eligible edge. Split only edges that
    // win on both sides: a face is touched at most once, no hanging vertices.
    for(int t=lane;t<nf;t+=stride){int best=-1;double longest=target*target*(1+2.000001e-6);
      for(int k=0;k<3;++k){int a=f[3*t+k],b=f[3*t+(k+1)%3];
        int e=cm_slot(cm_key(a,b),keys,hc,false);if(second[e]<0)continue;
        double length=cm_len(p[a],p[b]);
        if(length>longest||(best>=0&&length==longest&&keys[e]<keys[best])){longest=length;best=e;}
      }next[t]=best;
    }
    __syncthreads();
    for(int e=lane;e<hc;e+=stride)if(keys[e]!=~0ULL&&second[e]>=0&&next[first[e]]==e&&next[second[e]]==e){
      work[e]=atomicAdd(&additions,1);
    }
    __syncthreads();
    if(nv+additions>vc||nf+2*additions>fc){if(lane==0)state[2]=2;return;}
    for(int e=lane;e<hc;e+=stride)if(work[e]>=0){
      int a=int(keys[e]>>32),b=int(keys[e]&0xffffffffULL),m=nv+work[e];
      p[m]={(p[a].x+p[b].x)*.5,(p[a].y+p[b].y)*.5};
      int adjacent[2]={first[e],second[e]};
      for(int side=0;side<2;++side){int t=adjacent[side];int x=-1,y=-1,c=-1;
        for(int k=0;k<3;++k)if(cm_key(f[3*t+k],f[3*t+(k+1)%3])==keys[e]){
          x=f[3*t+k];y=f[3*t+(k+1)%3];c=f[3*t+(k+2)%3];break;}
        int out=nf+2*work[e]+side;
        f[3*t]=x;f[3*t+1]=m;f[3*t+2]=c;
        f[3*out]=m;f[3*out+1]=y;f[3*out+2]=c;
      }
    }
    __syncthreads();
    if(lane==0){state[0]=nv+additions;state[1]=nf+2*additions;state[3]=additions;state[4]+=additions;}
  }else{
    // A local minimum vertex ID among neighbors may move. Adjacent vertices
    // cannot move together; quality checks observe a single immutable mesh.
    for(int v=lane;v<nv;v+=stride)if(v>=fixed){
      cm_uv sum={0,0};int count=0;
      for(int h=head[v];h>=0;h=next[h]){int t=h/3,k=h%3;
        for(int q=1;q<=2;++q){int b=f[3*t+(k+q)%3];
          sum.x+=p[b].x;sum.y+=p[b].y;++count;}
      }
      // Rotate priorities with a phase supplied in state[7], so vertices that
      // lose one sweep are not permanently starved by lower IDs.
      bool selected=true;
      int priority=(v+state[7])%nv;
      for(int h=head[v];h>=0&&selected;h=next[h]){int t=h/3,k=h%3;
        for(int q=1;q<=2;++q){int b=f[3*t+(k+q)%3];if(b>=fixed&&(b+state[7])%nv<priority)selected=false;}
      }
      if(!selected||!count)continue;
      cm_uv candidate={(p[v].x+sum.x/count)*.5,(p[v].y+sum.y/count)*.5};
      double before=1,after=1;bool valid=true;
      for(int h=head[v];h>=0&&valid;h=next[h]){int t=h/3;cm_uv a[3],b[3];
        for(int k=0;k<3;++k){int id=f[3*t+k];a[k]=p[id];b[k]=id==v?candidate:p[id];}
        valid=cm_area(a[0],a[1],a[2])*cm_area(b[0],b[1],b[2])>0;
        before=fmin(before,cm_quality(a[0],a[1],a[2]));after=fmin(after,cm_quality(b[0],b[1],b[2]));
        for(int k=0;k<3;++k)valid=valid&&cm_len(b[k],b[(k+1)%3])<=target*target*(1+2.000001e-6);
      }
      if(valid&&after>before+1e-8){claim[v]=1;
        // Scratch two integer words per coordinate are avoided by staging in
        // unused point capacity, reserved by the host for exactly this purpose.
        p[vc+v]=candidate;
      }
    }
    __syncthreads();
    for(int v=lane;v<nv;v+=stride)if(claim[v]==1){p[v]=p[vc+v];atomicAdd(&changes,1);}
    __syncthreads();
    if(lane==0){state[3]=changes;state[6]+=changes;state[7]=(state[7]+nv/8+1)%nv;}
  }
}
)cuda";
}
