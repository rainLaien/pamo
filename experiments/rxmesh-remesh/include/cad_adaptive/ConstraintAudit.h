#pragma once
#include "cad_adaptive/SemanticMesh.h"

namespace cad_adaptive {
// Evidence about the immutable snapshot, independent of operational EdgeFlag.
// InputHard is a producer assertion, not a certified CAD curve. CADPART1 cannot
// distinguish a model-recognition interface from a computation-only interface.
enum SourceConstraintEvidence : uint8_t {
  SuppliedRecord=1, InputHard=2, IntrinsicOpen=4, PatchInterface=8
};
struct SourceConstraint {
  uint32_t v0=0,v1=0,featureId=0;
  uint32_t patchLeft=kInvalidId,patchRight=kInvalidId;
  uint8_t evidence=0;
};
struct SourcePatchEvidence {
  uint32_t patchId=0;
  PatchType type=PatchType::Unknown;
  uint8_t producerFeatureRole=0; // CADPART1: 0 ordinary, 1 fillet; not CAD certification.
  std::vector<uint32_t> supportPatchIds;
};
struct ConstraintAudit {
  uint32_t sourceVertices=0,sourceFaces=0;
  std::vector<SourcePatchEvidence> sourcePatches;
  std::vector<SourceConstraint> sourceEdges;
};
// Output IDs index the saved mesh; source IDs index the input CADPART1. These
// are separate domains, not a guessed source/output lineage.
bool writeConstraintAudit(const std::string& prefix,const ConstraintAudit&,
                          const SemanticMesh& output,const RemeshConfig&,
                          std::string* error=nullptr);
}
