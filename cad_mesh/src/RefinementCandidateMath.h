#pragma once
// Original implementation from elementary circumcenter / triangle formulas.
// The same function body is compiled for host and device; no Triangle code.
#define CADMESH_CANDIDATE_FUNCTION(QUALIFIER) \
QUALIFIER void cadmesh_refinement_candidate(const double *p,double *out){ \
  for(int i=0;i<6;++i)out[i]=0; \
  double x[3]={p[0],p[2],p[4]},y[3]={p[1],p[3],p[5]},s[3]; \
  int shortest=0; \
  for(int k=0;k<3;++k){int j=(k+1)%3;s[k]=(x[k]-x[j])*(x[k]-x[j])+(y[k]-y[j])*(y[k]-y[j]);if(s[k]<s[shortest])shortest=k;} \
  double bx=x[1]-x[0],by=y[1]-y[0],cx=x[2]-x[0],cy=y[2]-y[0]; \
  double cross=bx*cy-by*cx,bl=bx*bx+by*by,cl=cx*cx+cy*cy; \
  if(!(s[shortest]>0)||!(cross!=0)||!(p[6]>0)||!(p[7]>0&&p[7]<1))return; \
  double length=sqrt(s[shortest]),height=.5*length*sqrt(1-p[7]*p[7])/p[7]; \
  int next=(shortest+1)%3;double sign=cross>0?1:-1; \
  out[0]=x[0]+(bl*cy-cl*by)/(2*cross);out[1]=y[0]+(bx*cl-cx*bl)/(2*cross); \
  out[2]=.5*(x[shortest]+x[next])-sign*(y[next]-y[shortest])*height/length; \
  out[3]=.5*(y[shortest]+y[next])+sign*(x[next]-x[shortest])*height/length;out[4]=length; \
  double a=sqrt(s[(shortest+1)%3])*sqrt(s[(shortest+2)%3]); \
  double angleScore=p[7]*a/fabs(cross),longest=s[0]>s[1]?s[0]:s[1];if(s[2]>longest)longest=s[2]; \
  double sizeScore=sqrt(longest)/p[6];out[5]=angleScore>sizeScore?angleScore:sizeScore; \
}

#define CADMESH_STRINGIFY_INNER(...) #__VA_ARGS__
#define CADMESH_STRINGIFY(...) CADMESH_STRINGIFY_INNER(__VA_ARGS__)
inline const char *RefinementCandidateKernelSource =
  CADMESH_STRINGIFY(CADMESH_CANDIDATE_FUNCTION(__device__))
  "\nextern \"C\" __global__ void cadmesh_refinement_candidates(const double *p,double *out,int n){"
  "int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)cadmesh_refinement_candidate(p+8*i,out+6*i);}";
