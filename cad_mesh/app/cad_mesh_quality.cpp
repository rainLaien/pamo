#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
int main(int argc,char **argv){
  if(argc!=2)return 2;std::ifstream in(argv[1],std::ios::binary);std::string line;std::size_t nv=0,nf=0;bool ascii=false;
  while(std::getline(in,line)){if(line.find("format ascii")==0)ascii=true;
    if(line.find("element vertex ")==0)nv=std::stoull(line.substr(15));
    if(line.find("element face ")==0)nf=std::stoull(line.substr(13));if(line=="end_header"||line=="end_header\r")break;}
  if(!ascii){std::cerr<<"This evaluator reads the existing ASCII native PLY export.\n";return 2;}
  using V=std::array<double,3>;std::vector<V> p(nv);
  for(auto &v:p){std::getline(in,line);std::istringstream row(line);row>>v[0]>>v[1]>>v[2];if(!row)return 1;}
  std::size_t below5=0,below10=0,below20=0,below28=0,zero=0;double minimum=180,sum=0,areaSum=0,smallArea=0;
  for(std::size_t f=0;f<nf;++f){std::getline(in,line);std::istringstream row(line);int n;std::array<int,3> t;row>>n>>t[0]>>t[1]>>t[2];if(!row||n!=3)return 1;
    double angle=180,area=0;
    for(int k=0;k<3;++k){V u,v,c;for(int j=0;j<3;++j){u[j]=p.at(t[(k+1)%3])[j]-p.at(t[k])[j];v[j]=p.at(t[(k+2)%3])[j]-p.at(t[k])[j];}
      c={u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
      double cross=std::sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);if(k==0)area=.5*cross;
      angle=std::min(angle,std::atan2(cross,u[0]*v[0]+u[1]*v[1]+u[2]*v[2])*180/std::acos(-1.0));}
    minimum=std::min(minimum,angle);sum+=angle;below5+=angle<5;below10+=angle<10;below20+=angle<20;below28+=angle<28;zero+=area==0;areaSum+=area;if(angle<28)smallArea+=area;
  }
  std::cout<<"{\"faces\":"<<nf<<",\"minimum_angle\":"<<minimum<<",\"mean_minimum_angle\":"<<sum/nf<<",\"below_5\":"<<below5<<",\"below_10\":"<<below10<<",\"below_20\":"<<below20<<",\"below_28\":"<<below28<<",\"below_28_percent\":"<<100.0*below28/nf<<",\"below_28_area_percent\":"<<100*smallArea/areaSum<<",\"degenerate\":"<<zero<<"}\n";
}
