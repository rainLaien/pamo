#include "cad_adaptive/ConstraintAudit.h"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace cad_adaptive {
bool writeConstraintAudit(const std::string& prefix,const ConstraintAudit& source,
                          const SemanticMesh& mesh,const RemeshConfig& cfg,std::string* error) {
  try {
    // A failed/partial export must never leave a manifest claiming completeness.
    {std::ofstream pending(prefix+".fields.json");pending<<"{\"complete\":false}\n";
      pending.close();if(!pending)throw std::runtime_error("cannot initialize audit manifest");}
    if(mesh.targetLength.size()!=mesh.px.size() || mesh.vertexConstraint.size()!=mesh.px.size() ||
       mesh.py.size()!=mesh.px.size() || mesh.pz.size()!=mesh.px.size() || mesh.vertexPatchId.size()!=mesh.px.size())
      throw std::runtime_error("audit vertex field count mismatch");
    for(int v=0;v<mesh.vertexCount();++v) {
      const auto p=mesh.position(v);const float h=mesh.targetLength[v];
      if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||!std::isfinite(h)||h<0)
        throw std::runtime_error("invalid audit vertex field");
    }
    for(const auto& e:source.sourceEdges)
      if(e.v0>=source.sourceVertices||e.v1>=source.sourceVertices||e.v0>=e.v1||!e.evidence)
        throw std::runtime_error("invalid audit source identity");
    if(source.sourcePatches.empty())throw std::runtime_error("missing source patch evidence");
    for(size_t i=0;i<source.sourcePatches.size();++i) {
      const auto& p=source.sourcePatches[i];
      if(p.patchId!=i||p.producerFeatureRole>1)throw std::runtime_error("invalid source patch evidence");
      for(uint32_t support:p.supportPatchIds)
        if(support>=source.sourcePatches.size())throw std::runtime_error("invalid source support identity");
    }
    std::ofstream vertices(prefix+".vertices.tsv"),constraints(prefix+".source_constraints.tsv"),
                  edges(prefix+".edges.tsv"),patches(prefix+".source_patches.tsv");
    if(!vertices||!constraints||!edges||!patches)throw std::runtime_error("cannot open audit field files");
    vertices<<std::setprecision(std::numeric_limits<float>::max_digits10)
            <<"output_vertex\tx\ty\tz\ttarget\tconstraint\tpatch\n";
    for(int v=0;v<mesh.vertexCount();++v) {
      const auto p=mesh.position(v);
      vertices<<v<<'\t'<<p.x<<'\t'<<p.y<<'\t'<<p.z<<'\t'<<mesh.targetLength[v]
              <<'\t'<<int(mesh.vertexConstraint[v])<<'\t'<<mesh.vertexPatchId[v]<<'\n';
    }
    constraints<<"source_v0\tsource_v1\tfeature_id\tevidence\tpatch_left\tpatch_right\n";
    std::array<size_t,16> counts{};
    for(const auto& e:source.sourceEdges) {
      if(e.evidence>=counts.size())throw std::runtime_error("unknown source evidence");
      ++counts[e.evidence];
      constraints<<e.v0<<'\t'<<e.v1<<'\t'<<e.featureId<<'\t'<<int(e.evidence)
                 <<'\t'<<e.patchLeft<<'\t'<<e.patchRight<<'\n';
    }
    patches<<"source_patch\tpatch_type\tproducer_feature_role\tsupport_patch_ids\n";
    for(const auto& p:source.sourcePatches) {
      patches<<p.patchId<<'\t'<<uint32_t(p.type)<<'\t'<<int(p.producerFeatureRole)<<'\t';
      for(size_t i=0;i<p.supportPatchIds.size();++i){if(i)patches<<',';patches<<p.supportPatchIds[i];}
      patches<<'\n';
    }
    edges<<std::setprecision(std::numeric_limits<float>::max_digits10)
         <<"output_v0\toutput_v1\tflags\tfeature_id\teffective_target\tlength_ratio\tpersistent_feature\n";
    for(const auto& e:mesh.edges) {
      if(e.v0>=e.v1||e.v1>=mesh.px.size())throw std::runtime_error("invalid audit output edge identity");
      const uint64_t key=(uint64_t(e.v0)<<32)|e.v1;
      const bool feature=mesh.featureEdges.count(key)!=0;
      const float stored=.5f*(mesh.targetLength[e.v0]+mesh.targetLength[e.v1]);
      const float fallback=cfg.featureEdgeLength>0&&feature?cfg.featureEdgeLength:cfg.constantLength;
      const float target=stored>0?stored:fallback;
      if(!(target>0)||!std::isfinite(target))throw std::runtime_error("invalid effective audit target");
      edges<<e.v0<<'\t'<<e.v1<<'\t'<<int(e.flags)<<'\t'<<e.featureCurveId<<'\t'<<target
           <<'\t'<<distance(mesh.position(e.v0),mesh.position(e.v1))/target<<'\t'<<int(feature)<<'\n';
    }
    vertices.close();constraints.close();edges.close();patches.close();
    if(!vertices||!constraints||!edges||!patches)throw std::runtime_error("cannot finish audit field files");
    std::ofstream manifest(prefix+".fields.json");
    manifest<<std::setprecision(std::numeric_limits<float>::max_digits10)
      <<"{\"complete\":true,\"version\":2,\"source_vertices\":"<<source.sourceVertices
      <<",\"source_faces\":"<<source.sourceFaces<<",\"output_vertices\":"<<mesh.vertexCount()
      <<",\"output_faces\":"<<mesh.faceCount()<<",\"output_edges\":"<<mesh.edges.size()
      <<",\"source_patches\":"<<source.sourcePatches.size()
      <<",\"source_constraint_edges\":"<<source.sourceEdges.size()
      <<",\"evidence_bits\":{\"supplied\":1,\"input_hard\":2,\"intrinsic_open\":4,\"patch_interface\":8}"
      <<",\"interface_origin\":\"unknown_model_or_compute\",\"input_hard_is_cad_certification\":false"
      <<",\"producer_fillet_support_is_cad_certification\":false"
      <<",\"source_output_lineage_exported\":false,\"corner_identity_exported\":false"
      <<",\"source_id_domain\":\"input_snapshot\",\"output_id_domain\":\"saved_ply\""
      <<",\"split_ratio\":"<<cfg.splitRatio<<",\"collapse_ratio\":"<<cfg.collapseRatio
      <<",\"long_edge_tolerance\":0.0001,\"max_geometry_error\":"<<cfg.maxGeometryError
      <<",\"constant_length\":"<<cfg.constantLength<<",\"evidence_counts\":[";
    for(size_t i=0;i<counts.size();++i){if(i)manifest<<',';manifest<<counts[i];}
    manifest<<"]}\n";manifest.close();if(!manifest)throw std::runtime_error("cannot finish audit manifest");
    return true;
  }catch(const std::exception& e){if(error)*error=e.what();return false;}
}
}
