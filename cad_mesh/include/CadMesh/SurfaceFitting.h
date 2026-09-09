#pragma once
#include "CadMesh/MeshTopology.h"
#include <memory>
namespace CadMesh {
class ISurfaceFitter {
public: virtual ~ISurfaceFitter()=default;
    virtual bool fit(const MeshTopology&,const std::vector<int>&)=0;
    virtual double computeRmsError() const=0; virtual double computeMaxError() const=0;
    virtual double computeNormalError() const=0; virtual PatchSurfaceType getType() const=0;
    virtual SurfaceParameters getParameters() const{return{};}
};
class PlaneSurfaceFitter final:public ISurfaceFitter { public: bool fit(const MeshTopology&,const std::vector<int>&) override; double computeRmsError()const override{return mRms;} double computeMaxError()const override{return mMax;} double computeNormalError()const override{return mNormal;} PatchSurfaceType getType()const override{return PatchSurfaceType::Plane;} SurfaceParameters getParameters()const override{return PlaneParameters{mPlane};} private:PlaneEquation mPlane;double mRms=0,mMax=0,mNormal=0;};
class CylinderSurfaceFitter final:public ISurfaceFitter { public: bool fit(const MeshTopology&,const std::vector<int>&) override; double computeRmsError()const override{return mRms;} double computeMaxError()const override{return mMax;} double computeNormalError()const override{return mNormal;} PatchSurfaceType getType()const override{return PatchSurfaceType::Cylinder;} SurfaceParameters getParameters()const override{return CylinderParameters{mAxis,mRadius};} private:AxisLine mAxis;double mRadius=0,mRms=0,mMax=0,mNormal=0;};
class SphereSurfaceFitter final:public ISurfaceFitter { public: bool fit(const MeshTopology&,const std::vector<int>&) override; double computeRmsError()const override{return mRms;} double computeMaxError()const override{return mMax;} double computeNormalError()const override{return mNormal;} PatchSurfaceType getType()const override{return PatchSurfaceType::Sphere;} SurfaceParameters getParameters()const override{return SphereParameters{mCenter,mRadius};} private:Point3 mCenter;double mRadius=0,mRms=0,mMax=0,mNormal=0;};
// Axis.Origin is the apex; Direction points into the fitted single nappe.
// SemiAngle is in radians, strictly between zero and pi/2.
class ConeSurfaceFitter final:public ISurfaceFitter { public:bool fit(const MeshTopology&,const std::vector<int>&)override;double computeRmsError()const override{return mRms;}double computeMaxError()const override{return mMax;}double computeNormalError()const override{return mNormal;}PatchSurfaceType getType()const override{return PatchSurfaceType::Cone;}SurfaceParameters getParameters()const override{return ConeParameters{mAxis,mAngle};}private:AxisLine mAxis;double mAngle=0,mRms=0,mMax=0,mNormal=0;};
// Axis.Origin is the torus center. Only regular ring tori (R > r > 0)
// are certified; horn/spindle degeneracies fall back to the reference mesh.
class TorusSurfaceFitter final:public ISurfaceFitter { public:bool fit(const MeshTopology&,const std::vector<int>&)override;double computeRmsError()const override{return mRms;}double computeMaxError()const override{return mMax;}double computeNormalError()const override{return mNormal;}PatchSurfaceType getType()const override{return PatchSurfaceType::Torus;}SurfaceParameters getParameters()const override{return TorusParameters{mAxis,mMajor,mMinor};}private:AxisLine mAxis;double mMajor=0,mMinor=0,mRms=0,mMax=0,mNormal=0;};
// Classification fallback only: errors describe the best-plane baseline,
// not a fitted freeform surface. getParameters() intentionally stays empty.
class FreeformSurfaceFitter final:public ISurfaceFitter { public:bool fit(const MeshTopology&,const std::vector<int>&)override;double computeRmsError()const override{return mRms;}double computeMaxError()const override{return mMax;}double computeNormalError()const override{return mNormal;}PatchSurfaceType getType()const override{return PatchSurfaceType::Freeform;}private:double mRms=0,mMax=0,mNormal=0;};
struct SurfaceFitResult { PatchSurfaceType Type=PatchSurfaceType::Unknown;double Rms=0,Max=0,Normal=0,Score=1e100;SurfaceParameters Parameters; };
class SurfaceModelSelector { public: static SurfaceFitResult fitBest(const MeshTopology&,const std::vector<int>&,const MeshResolutionInfo&,double complexityPenalty=0.025); };
}
