#pragma once

#include "cad_adaptive/SemanticMesh.h"
#include <algorithm>

namespace cad_adaptive {

// Call after rebuilding topology. Empty accepted sets preserve the active set.
inline void updateDirty2Ring(const SemanticMesh &mesh,
                             const std::vector<uint8_t> &touched,
                             std::vector<uint8_t> &dirty) {
  if (std::none_of(touched.begin(), touched.end(), [](uint8_t x) { return x != 0; }))
    return;
  dirty.assign(mesh.vertexCount(), 0);
  std::vector<int> frontier;
  for (int v = 0; v < mesh.vertexCount() && v < int(touched.size()); ++v)
    if (touched[v]) {
      dirty[v] = 1;
      frontier.push_back(v);
    }
  for (int ring = 0; ring < 2; ++ring) {
    std::vector<int> next;
    for (int v : frontier)
      for (int f : mesh.incidentFaces[v]) {
        if (!mesh.faceAlive[f]) continue;
        for (int n : mesh.face(f))
          if (!dirty[n]) {
            dirty[n] = 1;
            next.push_back(n);
          }
      }
    frontier.swap(next);
  }
}

} // namespace cad_adaptive
