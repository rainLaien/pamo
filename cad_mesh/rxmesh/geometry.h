#pragma once
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define RX_HD __host__ __device__
#else
#define RX_HD
#endif
namespace pamo_rx {
struct Vec {
    double x=0,y=0,z=0;
    RX_HD double& operator[](int i) { return i==0?x:i==1?y:z; }
    RX_HD double operator[](int i) const { return i==0?x:i==1?y:z; }
};
RX_HD inline Vec operator+(Vec a,Vec b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
RX_HD inline Vec operator-(Vec a,Vec b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
RX_HD inline Vec operator*(Vec a,double b){return {a.x*b,a.y*b,a.z*b};}
RX_HD inline double dot(Vec a,Vec b){return a.x*b.x+a.y*b.y+a.z*b.z;}
RX_HD inline Vec cross(Vec a,Vec b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
RX_HD inline double norm2(Vec a){return dot(a,a);}
RX_HD inline double minv(double a,double b){return a<b?a:b;}
RX_HD inline double maxv(double a,double b){return a>b?a:b;}
RX_HD inline Vec normal(Vec a,Vec b,Vec c){Vec n=cross(b-a,c-a);return n*(1.0/sqrt(maxv(norm2(n),1e-60)));}
RX_HD inline double quality(Vec a,Vec b,Vec c){double l=maxv(norm2(a-b),maxv(norm2(b-c),norm2(c-a)));return norm2(cross(b-a,c-a))/maxv(l*l,1e-60);}
RX_HD inline Vec closest(Vec p,Vec a,Vec b,Vec c){
    Vec ab=b-a,ac=c-a,ap=p-a;double d1=dot(ab,ap),d2=dot(ac,ap);
    if(d1<=0&&d2<=0)return a;
    Vec bp=p-b;double d3=dot(ab,bp),d4=dot(ac,bp);if(d3>=0&&d4<=d3)return b;
    double vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0)return a+ab*(d1/(d1-d3));
    Vec cp=p-c;double d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0&&d5<=d6)return c;
    double vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0)return a+ac*(d2/(d2-d6));
    double va=d3*d6-d5*d4;if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0)return b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)));
    double inv=1.0/(va+vb+vc);return a+ab*(vb*inv)+ac*(vc*inv);
}
// Float broad-phase projection is used only at tolerances well above float
// roundoff. The selected triangle is projected again in double precision.
RX_HD inline Vec closestFast(Vec p,Vec a,Vec b,Vec c){
    Vec ab=b-a,ac=c-a,ap=p-a,bp=p-b,cp=p-c;
    auto fdot=[] RX_HD (Vec u,Vec v){return float(u.x)*float(v.x)+float(u.y)*float(v.y)+float(u.z)*float(v.z);};
    float d1=fdot(ab,ap),d2=fdot(ac,ap);if(d1<=0&&d2<=0)return a;
    float d3=fdot(ab,bp),d4=fdot(ac,bp);if(d3>=0&&d4<=d3)return b;
    float vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0)return a+ab*double(d1/(d1-d3));
    float d5=fdot(ab,cp),d6=fdot(ac,cp);if(d6>=0&&d5<=d6)return c;
    float vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0)return a+ac*double(d2/(d2-d6));
    float va=d3*d6-d5*d4;if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0)return b+(c-b)*double((d4-d3)/((d4-d3)+(d5-d6)));
    float inv=1.f/(va+vb+vc);return a+ab*double(vb*inv)+ac*double(vc*inv);
}
struct RefTriangle { Vec p[3]; int region=0,gpuFast=0; };
struct RegionPlane {Vec point,n;double error=0,minimumNormalDot=1;int enabled=0;};
struct BvhNode {Vec lo,hi;int left=-1,right=-1,begin=0,count=0;};
struct Reference {
    const RefTriangle* triangles=nullptr;const BvhNode* nodes=nullptr;const int* roots=nullptr;int regions=0;
    const RegionPlane* planes=nullptr;
    RX_HD double boxDistance(Vec p,const BvhNode& n,bool fast=false) const {
        #ifdef __CUDA_ARCH__
        if(fast){float d=0;for(int k=0;k<3;++k){float q=fmaxf(float(n.lo[k])-float(p[k])-4e-7f,fmaxf(0.f,float(p[k])-float(n.hi[k])-4e-7f));d+=q*q;}return double(d)*.999999;}
        #endif
        double d=0;for(int k=0;k<3;++k){double q=maxv(n.lo[k]-p[k],maxv(0,p[k]-n.hi[k]));d+=q*q;}return d;
    }
    RX_HD bool nearest(Vec p,int region,double limit,Vec& q,int& hit) const {
        if(region<0||region>=regions||roots[region]<0)return false;
        if(planes&&planes[region].enabled){const auto& plane=planes[region];double d=dot(p-plane.point,plane.n);q=p-plane.n*d;hit=-1;return fabs(d)+plane.error<=limit;}
        int stack[64],top=0;stack[top++]=roots[region];double best=limit*limit;hit=-1;
        while(top){const auto& n=nodes[stack[--top]];if(boxDistance(p,n,limit>=1e-5)>best)continue;
            if(n.count){for(int j=n.begin;j<n.begin+n.count;++j){const auto& t=triangles[j];Vec c;
                #ifdef __CUDA_ARCH__
                c=(limit>=1e-5&&t.gpuFast)?closestFast(p,t.p[0],t.p[1],t.p[2]):closest(p,t.p[0],t.p[1],t.p[2]);
                #else
                c=closest(p,t.p[0],t.p[1],t.p[2]);
                #endif
                double d=norm2(p-c);if(d<=best){best=d;q=c;hit=j;}}}
            else {if(top+2>64)return false;stack[top++]=n.left;stack[top++]=n.right;}
        }
        #ifdef __CUDA_ARCH__
        if(hit>=0){const auto&t=triangles[hit];q=closest(p,t.p[0],t.p[1],t.p[2]);return norm2(p-q)<=limit*limit;}
        #endif
        return hit>=0;
    }
    RX_HD bool accepts(Vec a,Vec b,Vec c,int region,double limit,double cosine) const {
        if(norm2(cross(b-a,c-a))<1e-40)return false;
        if(planes&&region>=0&&region<regions&&planes[region].enabled){
            const auto& p=planes[region];double margin=limit-p.error;if(margin<=0)return false;
            if(fabs(dot(a-p.point,p.n))>margin||fabs(dot(b-p.point,p.n))>margin||fabs(dot(c-p.point,p.n))>margin)return false;
            double required=cosine*p.minimumNormalDot+sqrt(maxv(0,1-cosine*cosine))*sqrt(maxv(0,1-p.minimumNormalDot*p.minimumNormalDot));
            return dot(normal(a,b,c),p.n)>=required;
        }
        Vec samples[7]={a,b,c,(a+b)*.5,(b+c)*.5,(c+a)*.5,(a+b+c)*(1.0/3.0)};
        int hit=-1;Vec q;for(Vec p:samples)if(!nearest(p,region,limit,q,hit))return false;
        const auto& t=triangles[hit];Vec candidateNormal=normal(a,b,c);
        if(dot(candidateNormal,normal(t.p[0],t.p[1],t.p[2]))>=cosine)return true;
        // At coincident facets/creases a nearest point can have more than one
        // valid source normal. Resolve numerical distance ties explicitly.
        Vec center=samples[6];double radius=minv(limit,sqrt(norm2(center-q))+1e-11),radius2=radius*radius;
        int stack[64],top=0;stack[top++]=roots[region];
        while(top){const auto& node=nodes[stack[--top]];if(boxDistance(center,node)>radius2)continue;
            if(node.count){for(int j=node.begin;j<node.begin+node.count;++j){const auto&r=triangles[j];if(dot(candidateNormal,normal(r.p[0],r.p[1],r.p[2]))>=cosine&&norm2(center-closest(center,r.p[0],r.p[1],r.p[2]))<=radius2)return true;}}
            else{if(top+2>64)return false;stack[top++]=node.left;stack[top++]=node.right;}
        }return false;
    }
};
}
