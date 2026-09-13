#include "CadMesh/DebugVisualizer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <set>

namespace CadMesh {
namespace {
std::array<int, 3> Color(int id) {
  unsigned x = unsigned(id + 1) * 2654435761u;
  x ^= x >> 16;
  return {64 + int(x & 127), 64 + int((x >> 8) & 127),
          64 + int((x >> 16) & 127)};
}
std::array<int, 3> SurfaceColor(PatchSurfaceType type) {
  // Fixed category colors across files and runs. Instance colors deliberately
  // remain separate so a single category cannot conceal fragmented patches.
  switch (type) {
  case PatchSurfaceType::Plane: return {74, 144, 226};
  case PatchSurfaceType::Cylinder: return {53, 183, 121};
  case PatchSurfaceType::Cone: return {239, 155, 56};
  case PatchSurfaceType::Sphere: return {160, 108, 213};
  case PatchSurfaceType::Torus: return {230, 105, 158};
  case PatchSurfaceType::Freeform: return {145, 153, 164};
  default: return {74, 79, 86};
  }
}
template <class T> void AppendLittleEndian(std::vector<char> &buffer, T value) {
  const std::uint16_t one = 1;
  char bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(value));
  if (*reinterpret_cast<const unsigned char *>(&one) != 1)
    std::reverse(bytes, bytes + sizeof(T));
  buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

bool ExportBinaryHandoffPly(const CadMeshPatchSegmenter &s,
                            const std::filesystem::path &path) {
  static_assert(sizeof(double) == 8, "Handoff PLY requires float64 coordinates");
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;
  const auto &mesh = s.getMesh();
  out << "ply\nformat binary_little_endian 1.0\ncomment color_by surface_instance"
         "\ncomment surface_type_ids 0=Unknown 1=Plane 2=Cylinder 3=Cone 4=Sphere 5=Torus 6=Freeform"
         "\ncomment feature_role_ids 0=Ordinary 1=Fillet 2=FilletCandidate\nelement vertex "
      << mesh.getVertices().size()
      << "\nproperty double x\nproperty double y\nproperty double z\nelement face "
      << mesh.getTriangles().size()
      << "\nproperty list uchar int vertex_indices\nproperty int patch_id"
         "\nproperty int primitive_type\nproperty uchar red\nproperty uchar green"
         "\nproperty uchar blue\nproperty int feature_role\nend_header\n";
  // Standard packed records: 24 bytes per vertex and 28 bytes per face.
  // Buffer blocks explicitly; native struct padding must never enter the PLY.
  constexpr std::size_t BlockRows = 16384;
  std::vector<char> buffer;
  buffer.reserve(BlockRows * 28);
  const auto flush = [&]() {
    out.write(buffer.data(), std::streamsize(buffer.size()));
    buffer.clear();
  };
  for (const auto &vertex : mesh.getVertices()) {
    for (int axis = 0; axis < 3; ++axis)
      AppendLittleEndian(buffer, double(vertex.Position[axis]));
    if (buffer.size() >= BlockRows * 24)
      flush();
  }
  if (!buffer.empty())
    flush();
  for (const auto &triangle : mesh.getTriangles()) {
    const auto *patch = triangle.PatchId >= 0 && triangle.PatchId < int(s.getPatches().size())
                            ? &s.getPatches()[triangle.PatchId] : nullptr;
    const auto type = patch ? patch->SurfaceType : PatchSurfaceType::Unknown;

    const auto color = Color(triangle.PatchId);
    AppendLittleEndian(buffer, std::uint8_t(3));
    for (int vertex : triangle.VertexIds)
      AppendLittleEndian(buffer, std::int32_t(vertex));
    AppendLittleEndian(buffer, std::int32_t(triangle.PatchId));
    AppendLittleEndian(buffer, std::int32_t(SurfaceTypeId(type)));
    for (int channel : color)
      AppendLittleEndian(buffer, std::uint8_t(channel));
    AppendLittleEndian(buffer, std::int32_t(patch ? static_cast<int>(patch->FeatureRole) : 0));
    if (buffer.size() >= BlockRows * 28)
      flush();
  }
  if (!buffer.empty())
    flush();
  out.flush();
  return bool(out);
}
const char *RoleName(PatchFeatureRole role) {
  return role == PatchFeatureRole::FilletCandidate ? "FilletCandidate" : role == PatchFeatureRole::Fillet ? "Fillet" : "Ordinary";
}
bool ExportColoredPly(const CadMeshPatchSegmenter &s,
                      const std::filesystem::path &path, int colorMode) {
  std::ofstream out(path);
  if (!out)
    return false;
  const auto &mesh = s.getMesh();
  out << "ply\nformat ascii 1.0\ncomment color_by "
      << (colorMode == 1 ? "surface_type" : colorMode == 2 ? "feature_role" : "surface_instance")
      << "\ncomment surface_type_ids 0=Unknown 1=Plane 2=Cylinder 3=Cone 4=Sphere 5=Torus 6=Freeform"
      << "\ncomment feature_role_ids 0=Ordinary 1=Fillet 2=FilletCandidate\nelement vertex "
      << mesh.getVertices().size()
      << "\nproperty double x\nproperty double y\nproperty double z\nelement face "
      << mesh.getTriangles().size()
      << "\nproperty list uchar int vertex_indices\nproperty int patch_id"
         "\nproperty int primitive_type\nproperty uchar red\nproperty uchar green"
         "\nproperty uchar blue\nproperty int feature_role\nend_header\n"
      << std::setprecision(17);
  for (const auto &v : mesh.getVertices())
    out << v.Position.X() << ' ' << v.Position.Y() << ' ' << v.Position.Z() << '\n';
  for (const auto &t : mesh.getTriangles()) {
    const auto *patch = t.PatchId >= 0 && t.PatchId < int(s.getPatches().size())
                            ? &s.getPatches()[t.PatchId] : nullptr;
    auto type = patch ? patch->SurfaceType : PatchSurfaceType::Unknown;
    bool fillet = patch && patch->FeatureRole == PatchFeatureRole::Fillet;
    auto color = colorMode == 1 ? SurfaceColor(type)
                 : colorMode == 2 ? (fillet ? std::array<int, 3>{245, 145, 45}
                                            : std::array<int, 3>{150, 161, 177})
                                  : Color(t.PatchId);
    out << "3 " << t.VertexIds[0] << ' ' << t.VertexIds[1] << ' '
        << t.VertexIds[2] << ' ' << t.PatchId << ' ' << SurfaceTypeId(type)
        << ' ' << color[0] << ' ' << color[1] << ' ' << color[2]
        << ' ' << (patch ? static_cast<int>(patch->FeatureRole) : 0) << '\n';
  }
  out.flush();
  return bool(out);
}
void Number(std::ostream &out, double value) {
  if (std::isfinite(value))
    out << value;
  else
    out << "null";
}
void String(std::ostream &out, const std::string &value) {
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32) {
      const char *hex = "0123456789abcdef";
      out << "\\u00" << hex[c >> 4] << hex[c & 15];
    } else
      out << c;
  }
  out << '"';
}
template <typename Values> void Ids(std::ostream &out, const Values &values) {
  out << '[';
  bool first = true;
  for (int value : values) {
    if (!first)
      out << ',';
    first = false;
    out << value;
  }
  out << ']';
}
void Point(std::ostream &out, const Point3 &point) {
  out << '[';
  for (int i = 0; i < 3; ++i) {
    if (i)
      out << ',';
    Number(out, point[i]);
  }
  out << ']';
}
void Axis(std::ostream &out, const AxisLine &axis) {
  out << "\"axis_origin\":";
  Point(out, axis.Origin);
  out << ",\"axis_direction\":";
  Point(out, axis.Direction);
}
void Parameters(std::ostream &out, const MeshPatch &patch) {
  if (patch.ProjectionTarget == PatchProjectionTarget::ReferenceMesh) {
    out << "null";
    return;
  }
  out << '{';
  if (const auto *p = std::get_if<PlaneParameters>(&patch.Parameters)) {
    out << "\"origin\":";
    Point(out, p->Plane.Origin);
    out << ",\"normal\":";
    Point(out, p->Plane.Normal);
  } else if (const auto *p =
                 std::get_if<CylinderParameters>(&patch.Parameters)) {
    Axis(out, p->Axis);
    out << ",\"radius\":";
    Number(out, p->Radius);
  } else if (const auto *p = std::get_if<ConeParameters>(&patch.Parameters)) {
    Axis(out, p->Axis);
    out << ",\"semi_angle_radians\":";
    Number(out, p->SemiAngle);
  } else if (const auto *p = std::get_if<SphereParameters>(&patch.Parameters)) {
    out << "\"center\":";
    Point(out, p->Center);
    out << ",\"radius\":";
    Number(out, p->Radius);
  } else if (const auto *p = std::get_if<TorusParameters>(&patch.Parameters)) {
    Axis(out, p->Axis);
    out << ",\"major_radius\":";
    Number(out, p->MajorRadius);
    out << ",\"minor_radius\":";
    Number(out, p->MinorRadius);
  }
  out << '}';
}

bool ExportHandoffReport(const CadMeshPatchSegmenter &s,
                         const std::filesystem::path &path) {
  std::ofstream out(path);
  if (!out)
    return false;
  const auto &mesh = s.getMesh();
  const auto &r = mesh.getResolution();
  const auto &constraint = s.getRemeshConstraint();
  out << std::setprecision(17) << std::boolalpha;
  out << "{\"schema\":\"cadmesh.remesh_handoff\",\"schema_version\":1,"
         "\"indexing\":{\"base\":0,\"vertices\":\"clean_mesh_ply_vertex_order\","
         "\"triangles\":\"clean_mesh_ply_face_order\",\"edges\":\"explicit_constraint_edge_ids\"},"
         "\"partition_valid\":" << s.validatePartition()
      << ",\"mesh\":{\"vertex_count\":" << mesh.getVertices().size()
      << ",\"triangle_count\":" << mesh.getTriangles().size()
      << ",\"edge_count\":" << mesh.getEdges().size() << "},\"resolution\":{\"bbox_diagonal\":";
  Number(out, r.BoundingBoxDiagonal);
  out << ",\"median_edge\":"; Number(out, r.MedianEdgeLength);
  out << ",\"weld_tolerance\":"; Number(out, r.WeldTolerance);
  out << ",\"fitting_tolerance\":"; Number(out, r.FittingTolerance);
  out << ",\"curvature_tolerance\":"; Number(out, r.CurvatureTolerance);
  out << ",\"angular_tolerance\":"; Number(out, r.AngularTolerance);
  out << "},\"patches\":[";
  for (std::size_t i = 0; i < s.getPatches().size(); ++i) {
    const auto &patch = s.getPatches()[i];
    if (i) out << ',';
    out << "{\"id\":" << patch.Id << ",\"type\":\"" << SurfaceTypeName(patch.SurfaceType)
        << "\",\"feature_role\":\"" << RoleName(patch.FeatureRole) << "\",\"support_patch_ids\":";
    Ids(out, patch.SupportPatchIds);
    out << ",\"triangle_count\":" << patch.TriangleIds.size() << ",\"triangle_ids\":";
    Ids(out, patch.TriangleIds);
    out << ",\"parameters\":"; Parameters(out, patch);
    out << ",\"max_sampled_surface_deviation\":";
    // Legacy partitioning did not compute this measurement. Preserve that
    // distinction so native snapshot loading can compute it instead of using 0.
    Number(out, s.usesModelFirstPartitioning() ? patch.MaxSampledSurfaceDeviation : -1.0);
    out << ",\"projection_target\":\""
        << (patch.ProjectionTarget == PatchProjectionTarget::AnalyticSurface
                ? "analytic_surface" : "reference_mesh") << "\"}";
  }
  out << "],\"constraints\":{\"edge_ids\":";
  Ids(out, constraint.ConstraintEdgeIds);
  out << ",\"hard_feature_edge_ids\":"; Ids(out, constraint.HardFeatureEdgeIds);
  out << ",\"smooth_surface_transition_edge_ids\":"; Ids(out, constraint.SurfaceTransitionEdgeIds);
  out << ",\"corner_vertex_ids\":"; Ids(out, constraint.CornerVertexIds);
  out << "},\"constraint_edges\":[";
  for (std::size_t i = 0; i < constraint.ConstraintEdgeIds.size(); ++i) {
    const int id = constraint.ConstraintEdgeIds[i];
    const auto &edge = mesh.getEdges()[id];
    std::set<int> patches;
    for (int face : edge.IncidentTriangleIds)
      patches.insert(mesh.getTriangles()[face].PatchId);
    if (i) out << ',';
    out << "{\"id\":" << id << ",\"vertex_ids\":[" << edge.Vertex0 << ',' << edge.Vertex1
        << "],\"incident_triangle_ids\":";
    Ids(out, edge.IncidentTriangleIds);
    out << ",\"incident_patch_ids\":"; Ids(out, patches);
    out << ",\"open_boundary\":" << edge.IsBoundary
        << ",\"non_manifold\":" << edge.IsNonManifold
        << ",\"constrained_feature\":" << edge.IsConstrainedFeature
        << ",\"hard_feature\":" << (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature)
        << '}';
  }
  out << "]}\n";
  out.flush();
  return bool(out);
}
void VtkPoints(std::ostream &out, const MeshTopology &mesh) {
  out << std::setprecision(17) << "POINTS " << mesh.getVertices().size()
      << " double\n";
  for (const auto &v : mesh.getVertices())
    out << v.Position.X() << ' ' << v.Position.Y() << ' ' << v.Position.Z()
        << '\n';
}
void VtkMeshHeader(std::ofstream &out, const MeshTopology &mesh) {
  out << "# vtk DataFile Version 3.0\nCadMesh debug\nASCII\nDATASET POLYDATA\n";
  VtkPoints(out, mesh);
  out << "POLYGONS " << mesh.getTriangles().size() << ' '
      << mesh.getTriangles().size() * 4 << '\n';
  for (const auto &t : mesh.getTriangles())
    out << "3 " << t.VertexIds[0] << ' ' << t.VertexIds[1] << ' '
        << t.VertexIds[2] << '\n';
}
} // namespace

bool DebugVisualizer::exportRemeshHandoff(const CadMeshPatchSegmenter &s,
                                         const std::filesystem::path &directory) {
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error)
    return false;
  return ExportBinaryHandoffPly(s, directory / "patch_result.ply") &&
         ExportHandoffReport(s, directory / "patch_report.json");
}

bool DebugVisualizer::exportAll(const CadMeshPatchSegmenter &s,
                                const std::filesystem::path &directory) {
  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  if (ec)
    return false;
  bool ok = exportPatchPly(s, directory / "patch_result.ply");
  ok = exportSurfaceTypePly(s, directory / "surface_types.ply") && ok;
  ok = exportFeatureRolePly(s, directory / "feature_roles.ply") && ok;
  ok = exportBoundaryVtk(s, directory / "boundary_score.vtk") && ok;
  if (s.hasComputedDifferentialGeometry()) {
    ok = exportCurvatureVtk(s, directory / "mean_curvature.vtk", "mean_curvature", 0) && ok;
    ok = exportCurvatureVtk(s, directory / "gaussian_curvature.vtk", "gaussian_curvature", 1) && ok;
    ok = exportCurvatureVtk(s, directory / "k1.vtk", "k1", 2) && ok;
    ok = exportCurvatureVtk(s, directory / "k2.vtk", "k2", 3) && ok;
  } else {
    // Refreshing an existing export directory must not leave stale measured
    // curvature from an earlier run beside the current uncomputed metadata.
    for (const char *file : {"mean_curvature.vtk", "gaussian_curvature.vtk", "k1.vtk", "k2.vtk"}) {
      std::error_code removeError;
      std::filesystem::remove(directory / file, removeError);
      // Older libstdc++ filesystem implementations report ENOENT here;
      // absence is the intended result when refreshing a new export directory.
      if (removeError == std::errc::no_such_file_or_directory)
        removeError.clear();
      ok = !removeError && ok;
    }
  }
  return exportReportJson(s, directory / "patch_report.json") && ok;
}

bool DebugVisualizer::exportPatchPly(const CadMeshPatchSegmenter &s,
                                     const std::filesystem::path &path) {
  return ExportColoredPly(s, path, 0);
}
bool DebugVisualizer::exportSurfaceTypePly(const CadMeshPatchSegmenter &s,
                                           const std::filesystem::path &path) {
  return ExportColoredPly(s, path, 1);
}
bool DebugVisualizer::exportFeatureRolePly(const CadMeshPatchSegmenter &s,
                                          const std::filesystem::path &path) {
  return ExportColoredPly(s, path, 2);
}

bool DebugVisualizer::exportBoundaryVtk(const CadMeshPatchSegmenter &s,
                                        const std::filesystem::path &path) {
  std::ofstream out(path);
  if (!out)
    return false;
  const auto &mesh = s.getMesh();
  out << "# vtk DataFile Version 3.0\nCadMesh boundary topology; scores_computed="
      << (s.hasComputedBoundaryScores() ? 1 : 0)
      << "\nASCII\nDATASET POLYDATA\n";
  VtkPoints(out, mesh);
  out << "LINES " << mesh.getEdges().size() << ' ' << mesh.getEdges().size() * 3
      << '\n';
  for (const auto &edge : mesh.getEdges())
    out << "2 " << edge.Vertex0 << ' ' << edge.Vertex1 << '\n';
  out << "CELL_DATA " << mesh.getEdges().size() << '\n';
  auto scalar = [&](const char *name, auto value) {
    out << "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n";
    for (const auto &edge : mesh.getEdges())
      out << value(edge) << '\n';
  };
  if (s.hasComputedBoundaryScores())
    scalar("boundary_score", [](const MeshEdge &e) { return e.BoundaryScore; });
  scalar("constrained_feature",
         [](const MeshEdge &e) { return e.IsConstrainedFeature ? 1.0 : 0.0; });
  scalar("hard_feature", [](const MeshEdge &e) {
    return e.IsBoundary || e.IsNonManifold || e.IsConstrainedFeature ? 1.0 : 0.0;
  });
  scalar("smooth_surface_transition", [&](const MeshEdge &e) {
    return !e.IsBoundary && !e.IsNonManifold && !e.IsConstrainedFeature &&
                   e.IncidentTriangleIds.size() == 2 &&
                   mesh.getTriangles()[e.IncidentTriangleIds[0]].PatchId !=
                       mesh.getTriangles()[e.IncidentTriangleIds[1]].PatchId
               ? 1.0 : 0.0;
  });
  if (s.hasComputedBoundaryScores()) {
    scalar("normal_discontinuity",
           [](const MeshEdge &e) { return e.Evidence.NormalDiscontinuity; });
    scalar("curvature_discontinuity",
           [](const MeshEdge &e) { return e.Evidence.CurvatureDiscontinuity; });
    scalar("curvature_gradient",
           [](const MeshEdge &e) { return e.Evidence.CurvatureGradient; });
    scalar("surface_fit_discontinuity",
           [](const MeshEdge &e) { return e.Evidence.SurfaceFitDiscontinuity; });
    scalar("tessellation_evidence",
           [](const MeshEdge &e) { return e.Evidence.TessellationEvidence; });
  }
  out.flush();
  return bool(out);
}

bool DebugVisualizer::exportCurvatureVtk(const CadMeshPatchSegmenter &s,
                                         const std::filesystem::path &path,
                                         const std::string &name,
                                         int component) {
  if (!s.hasComputedDifferentialGeometry())
    return false;
  std::ofstream out(path);
  if (!out)
    return false;
  const auto &mesh = s.getMesh();
  VtkMeshHeader(out, mesh);
  out << "POINT_DATA " << mesh.getVertices().size() << "\nSCALARS " << name
      << " double 1\nLOOKUP_TABLE default\n";
  for (const auto &v : mesh.getVertices()) {
    const auto &g = v.Geometry;
    out << (component == 0   ? g.MeanCurvature
            : component == 1 ? g.GaussianCurvature
            : component == 2 ? g.K1
                             : g.K2)
        << '\n';
  }
  out.flush();
  return bool(out);
}

bool DebugVisualizer::exportReportJson(const CadMeshPatchSegmenter &s,
                                       const std::filesystem::path &path) {
  std::ofstream out(path);
  if (!out)
    return false;
  const auto &mesh = s.getMesh();
  const auto &r = mesh.getResolution();
  const auto &c = mesh.getCleanupReport();
  const auto &constraint = s.getRemeshConstraint();
  out << std::setprecision(17) << std::boolalpha;
  out << "{\n  \"schema\": \"cadmesh.remesh_handoff\",\n  \"schema_version\": "
         "1,\n"
         "  \"indexing\": "
         "{\"base\":0,\"vertices\":\"clean_mesh_ply_vertex_order\","
         "\"triangles\":\"clean_mesh_ply_face_order\",\"edges\":\"explicit_"
         "constraint_edge_ids\"},\n"
      << "  "
         "\"boundary_chain_convention\":{\"direction\":\"incident_face_"
         "winding\","
         "\"unresolved_direction\":0,\"closed_vertex_order\":\"repeat_first_at_"
         "end\","
         "\"initial_samples\":\"shared_original_mesh_vertices\"},\n"
      << "  \"diagnostics\":{\"differential_geometry\":{\"computed\":"
      << s.hasComputedDifferentialGeometry()
      << "},\"boundary_scores\":{\"computed\":" << s.hasComputedBoundaryScores()
      << "},\"uncomputed_values\":\"null_or_omitted\"},\n"
      << "  \"partition_valid\":" << s.validatePartition()
      << ",\n  \"mesh\": {\"vertex_count\":" << mesh.getVertices().size()
      << ",\"triangle_count\":" << mesh.getTriangles().size()
      << ",\"edge_count\":" << mesh.getEdges().size() << "},\n";
  out << "  \"resolution\": {\"bbox_diagonal\":";
  Number(out, r.BoundingBoxDiagonal);
  out << ",\"median_edge\":";
  Number(out, r.MedianEdgeLength);
  out << ",\"weld_tolerance\":";
  Number(out, r.WeldTolerance);
  out << ",\"fitting_tolerance\":";
  Number(out, r.FittingTolerance);
  out << ",\"curvature_tolerance\":";
  Number(out, r.CurvatureTolerance);
  out << ",\"angular_tolerance\":";
  Number(out, r.AngularTolerance);
  out << "},\n  \"cleanup\": {\"input_triangles\":" << c.InputTriangles
      << ",\"output_triangles\":" << c.OutputTriangles
      << ",\"degenerate\":" << c.DegenerateTriangles
      << ",\"duplicate\":" << c.DuplicateTriangles
      << ",\"non_manifold_edges\":" << c.NonManifoldEdges
      << ",\"boundary_edges\":" << c.BoundaryEdges << ",\"warnings\":[";
  for (size_t i = 0; i < c.Warnings.size(); ++i) {
    if (i)
      out << ',';
    String(out, c.Warnings[i]);
  }
  out << "]},\n  \"patches\": [\n";
  for (size_t i = 0; i < s.getPatches().size(); ++i) {
    const auto &p = s.getPatches()[i];
    bool analytic =
        p.ProjectionTarget == PatchProjectionTarget::AnalyticSurface;
    out << "    {\"id\":" << p.Id << ",\"type\":\""
        << SurfaceTypeName(p.SurfaceType)
        << "\",\"feature_role\":\"" << RoleName(p.FeatureRole)
        << "\",\"support_patch_ids\":";
    Ids(out, p.SupportPatchIds);
    out << ",\"triangle_count\":" << p.TriangleIds.size() << ",\"rms\":";
    Number(out, p.RmsFittingError);
    out << ",\"max\":";
    Number(out, p.MaxFittingError);
    const bool sampledDeviationComputed = analytic && s.usesModelFirstPartitioning();
    out << ",\"sampled_mesh_deviation\":{\"computed\":" << sampledDeviationComputed
        << ",\"maximum\":";
    if (sampledDeviationComputed)
      Number(out, p.MaxSampledSurfaceDeviation);
    else
      out << "null";
    out << ",\"direction\":\"reference_mesh_to_analytic_surface\","
           "\"sampling\":\"all_triangle_vertices_edge_midpoints_and_centroids\","
           "\"hausdorff_upper_bound\":false}";
    out << ",\"normal_error\":";
    Number(out, p.NormalError);
    out << ",\"confidence\":";
    Number(out, p.Confidence);
    out << ",\"neighbors\":";
    Ids(out, p.NeighborPatchIds);
    out << ",\"triangle_ids\":";
    Ids(out, p.TriangleIds);
    out << ",\"boundary_edge_ids\":";
    Ids(out, p.BoundaryEdgeIds);
    out << ",\"parameters\":";
    Parameters(out, p);
    out << ",\"projection_target\":\""
        << (analytic ? "analytic_surface" : "reference_mesh")
        << "\",\"fitting_error_semantics\":\""
        << (analytic ? (s.usesModelFirstPartitioning()
                             ? "all_patch_vertices_area_weighted_rms_and_vertex_max"
                             : "fitter_vertex_samples_area_weighted_rms_and_vertex_max")
                     : "local_model_diagnostic_not_analytic_accuracy")
        << "\",\"remesh_error_validated\":false,\"consistent_face_"
           "orientation\":"
        << p.HasConsistentFaceOrientation
        << ",\"parameterization\":{\"seam_assessment_required\":"
        << p.NeedsParameterizationSeamAssessment << ",\"seam_generated\":false}"
        << ",\"boundary_chain_refs\":[";
    for (size_t j = 0; j < p.BoundaryChainRefs.size(); ++j) {
      if (j)
        out << ',';
      const auto &ref = p.BoundaryChainRefs[j];
      out << "{\"chain_id\":" << ref.ChainId
          << ",\"direction\":" << ref.Direction << ",\"orientation_status\":\""
          << BoundaryDirectionStatusName(ref.Status) << "\"}";
    }
    out << "]}" << (i + 1 < s.getPatches().size() ? "," : "") << '\n';
  }
  out << "  ],\n  \"adjacency\": [\n";
  for (size_t i = 0; i < s.getAdjacency().size(); ++i) {
    const auto &a = s.getAdjacency()[i];
    out << "    {\"patch0\":" << a.Patch0 << ",\"patch1\":" << a.Patch1
        << ",\"edge_count\":" << a.SharedBoundaryEdges.size()
        << ",\"confidence\":";
    if (s.hasComputedBoundaryScores())
      Number(out, a.BoundaryConfidence);
    else
      out << "null";
    out << ",\"shared_boundary_edge_ids\":";
    Ids(out, a.SharedBoundaryEdges);
    out << '}' << (i + 1 < s.getAdjacency().size() ? "," : "") << '\n';
  }
  out << "  ],\n  \"constraints\": {\"edge_ids\":";
  Ids(out, constraint.ConstraintEdgeIds);
  out << ",\"hard_feature_edge_ids\":";
  Ids(out, constraint.HardFeatureEdgeIds);
  out << ",\"smooth_surface_transition_edge_ids\":";
  Ids(out, constraint.SurfaceTransitionEdgeIds);
  out << ",\"junction_vertex_ids\":";
  Ids(out, constraint.JunctionVertexIds);
  out << ",\"corner_vertex_ids\":";
  Ids(out, constraint.CornerVertexIds);
  out << ",\"corner_angle_radians\":";
  Number(out, constraint.CornerAngleThreshold);
  out << ",\"corners\":[";
  for (size_t i = 0; i < constraint.Corners.size(); ++i) {
    if (i)
      out << ',';
    const auto &corner = constraint.Corners[i];
    out << "{\"vertex_id\":" << corner.VertexId
        << ",\"endpoint\":" << corner.IsEndpoint
        << ",\"junction\":" << corner.IsJunction
        << ",\"sharp_corner\":" << corner.IsSharpCorner
        << ",\"incidence_change\":" << corner.HasIncidenceChange << '}';
  }
  out << "],\"boundary_chains\":[\n";
  for (size_t i = 0; i < constraint.BoundaryChains.size(); ++i) {
    const auto &chain = constraint.BoundaryChains[i];
    out << "    {\"id\":" << chain.Id << ",\"edge_ids\":";
    Ids(out, chain.EdgeIds);
    out << ",\"vertex_ids\":";
    Ids(out, chain.VertexIds);
    out << ",\"incident_patch_ids\":";
    Ids(out, chain.IncidentPatchIds);
    out << ",\"initial_sample_vertex_ids\":";
    Ids(out, chain.InitialSampleVertexIds);
    out << ",\"sampling_target\":\"reference_mesh_polyline\",\"closed\":"
        << chain.IsClosed << ",\"non_manifold\":" << chain.IsNonManifold
        << ",\"hard_feature\":" << chain.IsHardFeature
        << ",\"boundary_kind\":\""
        << (chain.IsHardFeature ? "hard_feature" : "smooth_surface_transition") << '"'
        << ",\"inconsistent_winding\":" << chain.HasInconsistentWinding << '}'
        << (i + 1 < constraint.BoundaryChains.size() ? "," : "") << '\n';
  }
  out << "  ]},\n  \"constraint_edges\": [\n";
  for (size_t i = 0; i < constraint.ConstraintEdgeIds.size(); ++i) {
    int id = constraint.ConstraintEdgeIds[i];
    const auto &edge = mesh.getEdges()[id];
    std::set<int> patches;
    for (int ti : edge.IncidentTriangleIds)
      patches.insert(mesh.getTriangles()[ti].PatchId);
    out << "    {\"id\":" << id << ",\"vertex_ids\":[" << edge.Vertex0 << ','
        << edge.Vertex1 << "],\"incident_triangle_ids\":";
    Ids(out, edge.IncidentTriangleIds);
    out << ",\"incident_patch_ids\":";
    Ids(out, patches);
    out << ",\"open_boundary\":" << edge.IsBoundary
        << ",\"non_manifold\":" << edge.IsNonManifold
        << ",\"constrained_feature\":" << edge.IsConstrainedFeature
        << ",\"hard_feature\":"
        << (edge.IsBoundary || edge.IsNonManifold || edge.IsConstrainedFeature)
        << ",\"boundary_score\":";
    if (s.hasComputedBoundaryScores())
      Number(out, edge.BoundaryScore);
    else
      out << "null";
    out << '}' << (i + 1 < constraint.ConstraintEdgeIds.size() ? "," : "")
        << '\n';
  }
  out << "  ]\n}\n";
  out.flush();
  return bool(out);
}
} // namespace CadMesh
