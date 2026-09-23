from pathlib import Path
p=Path(r"F:\\work\\laien_work\\pamo\\experiments\\rxmesh-remesh\\src\\rxmesh\\RawCudaRemesher.cu")
s=p.read_text()
a=s.index("__global__ void collapsePrefilter(")
b=s.index("__global__ void flipCandidates",a)
new=r'''__global__ void collapseCandidates(MeshView m, Candidate *out, float low,
                                   float high, ReferenceSurfaceGpu ref,
                                   bool relaxed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= m.ne)
    return;
  out[id].keep = -1;
  auto e = m.e[id];
  if (e.f0 < 0)
    return;
  int a = e.a, b = e.b;
  if (ref.sizingCount) {
    low *= fminf(sizingAt(ref, mul3(add3(m.v[a].p, m.v[b].p), .5f)),
                 .5f * (m.v[a].target + m.v[b].target)) /
           ref.regularLength;
  }
  const float edgeDist2 = dist2(m.v[a].p, m.v[b].p);
  if (!relaxed && edgeDist2 >= low * low) {
    auto t0 = m.f[e.f0];
    auto n0 = normal3(m.v[t0.v[0]].p, m.v[t0.v[1]].p, m.v[t0.v[2]].p);
    float minArea = .5f * sqrtf(dot3(n0, n0));
    if (e.f1 >= 0) {
      auto t1 = m.f[e.f1];
      auto n1 = normal3(m.v[t1.v[0]].p, m.v[t1.v[1]].p, m.v[t1.v[2]].p);
      minArea = fminf(minArea, .5f * sqrtf(dot3(n1, n1)));
    }
    if (minArea >= low * low / 100.f)
      return;
  }
  int da = m.offsets[a + 1] - m.offsets[a],
      db = m.offsets[b + 1] - m.offsets[b];
  if (relaxed && !((da == 3 || da == 4) && m.v[a].constraint < 2) &&
      !((db == 3 || db == 4) && m.v[b].constraint < 2))
    return;
  bool ma = movable(m, a, id), mb = movable(m, b, id);
  if (!ma && !mb)
    return;