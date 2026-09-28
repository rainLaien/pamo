"""Reuse immutable search trees without changing nearest-triangle tie rules.

The candidate construction and tie handling mirror the installed trimesh
proximity routine. Unlike repeated on_surface calls, the vertex KD tree is
built once per reference mesh. Queries retain bounded 64-point batches.
"""
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def compact_mesh(points,faces):
    """Keep face order while excluding vertices outside this audit region."""
    used=np.unique(faces)
    return trimesh.Trimesh(points[used],np.searchsorted(used,faces),process=False)


class CachedSurfaceQuery:
    def __init__(self, mesh):
        self.tree=mesh.triangles_tree
        self.vertices=cKDTree(mesh.vertices[mesh.referenced_vertices])
        self.triangles=mesh.triangles.view(np.ndarray)
        self.normals=mesh.face_normals

    def on_surface(self, points):
        points=np.asarray(points,dtype=np.float64)
        if points.ndim!=2 or points.shape[1]!=3:raise ValueError('expected (n,3) points')
        if not len(points):return np.empty((0,3)),np.empty(0),np.empty(0,dtype=int)
        radius=self.vertices.query(points)[0][:,None]+trimesh.constants.tol.merge
        bounds=np.column_stack((points-radius,points+radius))
        candidates=[list(self.tree.intersection(b)) for b in bounds]
        counts=np.asarray(list(map(len,candidates)))
        if (counts==0).any():raise ValueError('nearest-triangle candidate set empty')
        ids=np.concatenate(candidates)
        owner=np.repeat(np.arange(len(points)),counts)
        query_points=points[owner]
        close=trimesh.triangles.closest_point(self.triangles[ids],query_points)
        vectors=query_points-close
        squared=trimesh.util.diagonal_dot(vectors,vectors)
        offsets=np.cumsum(counts)[:-1]
        groups=np.array_split(squared,offsets)
        best=np.int32([g.argsort()[:2] if len(g)>1 else [0,0] for g in groups])
        best[1:]+=offsets[:,None]
        two_dist=squared[best]
        two_ids=ids[best]
        result_points=close[best[:,0]].copy()
        result_ids=two_ids[:,0].copy()
        result_dist=two_dist[:,0].copy()
        tol=trimesh.constants.tol.merge
        ambiguous=(np.ptp(two_dist,axis=1)<tol)&np.all(np.abs(two_dist)>tol,axis=1)
        if ambiguous.any():
            normals=self.normals[two_ids[ambiguous]]
            directions=vectors[best[ambiguous]]/two_dist[ambiguous,:,None]**.5
            choice=(normals*directions).sum(axis=2).argmax(axis=1)
            result_ids[ambiguous]=two_ids[ambiguous,choice]
            result_dist[ambiguous]=two_dist[ambiguous,choice]
            result_points[ambiguous]=close[best[ambiguous,choice]]
        result_dist**=.5
        return result_points,result_dist,result_ids
