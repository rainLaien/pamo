"""Conservative planar-strip reconstruction for the partitioned CUDA workflow.

Only convex, almost planar components are accepted. Original vertices and
component creases remain fixed; callers must coordinate neighboring boundaries.
"""
from collections import Counter, defaultdict
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(str(message))


def edges(faces):
    return [(int(a), int(b)) for face in faces
            for a, b in zip(face, np.roll(face, -1))]


def ring_of(faces):
    ee = edges(faces)
    counts = Counter(tuple(sorted(e)) for e in ee)
    boundary = [(a, b) for a, b in ee if counts[tuple(sorted((a, b)))] == 1]
    nxt = dict(boundary)
    require(len(nxt) == len(boundary), 'non-simple boundary')
    require(bool(boundary), "component has no boundary")
    start = boundary[0][0]
    ring = [start]
    while nxt[ring[-1]] != start:
        require(nxt[ring[-1]] not in ring, 'geometry or topology invariant failed: nxt[ring[-1]] not in ring')
        ring.append(nxt[ring[-1]])
    require(len(ring) == len(boundary), 'multiple boundary loops')
    return ring


def quality(p):
    area2 = np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0]))
    denominator = sum(np.dot(p[i]-p[(i+1)%3], p[i]-p[(i+1)%3]) for i in range(3))
    return 2*np.sqrt(3)*area2/denominator if denominator else 0.


def sample_distances(points, p, faces):
    # Plane/edge distance in extended precision. A generic nearest-triangle
    # routine produced nonzero source-to-itself distances on these slivers.
    tri=p[faces].astype(np.longdouble)
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    normal=np.cross(b-a,c-a); norm2=(normal*normal).sum(1)
    require((norm2 > 0).all(), 'geometry or topology invariant failed: (norm2 > 0).all()')
    result=[]
    for start in range(0,len(points),64):
        query=points[start:start+64].astype(np.longdouble)[:,None,:]
        signed=((query-a)*normal).sum(2)
        projected=query-signed[:,:,None]*normal/norm2[None,:,None]
        inside=np.ones(signed.shape,dtype=bool)
        best=np.full(signed.shape,np.inf,dtype=np.longdouble)
        for u,v in ((a,b),(b,c),(c,a)):
            edge=v-u
            side=(np.cross(edge,projected-u)*normal).sum(2)
            inside &= side>=-1e-12*norm2
            t=np.clip(((query-u)*edge).sum(2)/(edge*edge).sum(1),0,1)
            displacement=query-(u+t[:,:,None]*edge)
            best=np.minimum(best,(displacement*displacement).sum(2))
        best=np.where(inside,np.minimum(best,signed*signed/norm2),best)
        result.extend(np.sqrt(best.min(1)).astype(float))
    return np.array(result)


def triangulate(poly, vertices, normal):
    # Maximize the minimum triangle quality over triangulations of a convex
    # polygon, retaining collinear boundary samples without zero-area faces.
    n = len(poly)
    scores = np.full((n, n), -1.)
    choice = {}
    for i in range(n-1):
        scores[i, i+1] = 1.
    for width in range(2, n):
        for i in range(n-width):
            j = i+width
            for k in range(i+1, j):
                p = np.array([vertices[poly[t]] for t in (i, k, j)])
                if np.dot(np.cross(p[1]-p[0], p[2]-p[0]), normal) <= 1e-14:
                    continue
                score = min(scores[i,k], scores[k,j], quality(p))
                if score > scores[i,j]:
                    scores[i,j] = score
                    choice[i,j] = k
    require(scores[0, n - 1] > 0, 'cannot triangulate nondegenerately')
    def visit(i,j):
        if j-i < 2:
            return []
        k = choice[i,j]
        return [(poly[i],poly[k],poly[j])] + visit(i,k) + visit(k,j)
    return visit(0,n-1)


def make_groups(p, faces, merge, tolerance):
    normals = np.cross(p[faces[:,1]]-p[faces[:,0]], p[faces[:,2]]-p[faces[:,0]])
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    ee = defaultdict(list)
    for i, face in enumerate(faces):
        for a,b in edges([face]):
            ee[tuple(sorted((a,b)))].append(i)
    adjacency = defaultdict(list)
    if merge:
        for ff in ee.values():
            if len(ff) == 2 and np.dot(normals[ff[0]],normals[ff[1]]) > 1-1e-10:
                adjacency[ff[0]].append(ff[1]); adjacency[ff[1]].append(ff[0])
    seen = set(); groups=[]
    for i in range(len(faces)):
        if i in seen:
            continue
        seen.add(i); stack=[i]; component=[]
        while stack:
            j=stack.pop(); component.append(j)
            for k in adjacency[j]:
                if k not in seen:
                    stack.append(k); seen.add(k)
        ring=ring_of(faces[component]); normal=normals[i]
        pp=p[ring]
        require(np.max(np.abs((pp - pp[0]) @ normal)) < tolerance, 'nonplanar component')
        # All vertices must lie on the inside of each oriented polygon edge.
        for a,b in zip(pp,np.roll(pp,-1,axis=0)):
            require(np.min(np.cross(b - a, pp - a) @ normal) > -1e-08, 'nonconvex component')
        groups.append((ring,normal))
    return groups


def remesh(p, faces, target_length, shared_grid, tolerance):
    groups=make_groups(p,faces,True,tolerance)
    axis=int(np.argmax(np.ptp(p,axis=0)))
    lo,hi=p[:,axis].min(),p[:,axis].max()
    # A small margin accounts for cross-strip displacement in diagonal edges.
    count=int(np.ceil((hi-lo)/(.95*target_length)))
    require(count <= 4096, "more than 4096 strip stations requested")
    stations=np.linspace(lo,hi,count+1)
    if shared_grid is not None:
        grid=shared_grid[axis]
        stations=np.r_[lo,grid[(grid>lo+1e-10)&(grid<hi-1e-10)],hi]
    vertices=list(p.copy()); cache={}; output=[]
    # Clipping uses coordinate keys at 1e-9, far below source float32 resolution.
    # Original vertices are always returned unchanged when on a clipping plane.
    def crossing(a,b,x):
        if abs(vertices[a][axis]-x)<1e-10: return a
        if abs(vertices[b][axis]-x)<1e-10: return b
        t=(x-vertices[a][axis])/(vertices[b][axis]-vertices[a][axis])
        q=vertices[a]+t*(vertices[b]-vertices[a]); q[axis]=x
        key=tuple(np.round(q,9))
        if key not in cache:
            cache[key]=len(vertices); vertices.append(q)
        return cache[key]
    def clip(poly,x,lower):
        result=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            ia=vertices[a][axis]>=x-1e-10 if lower else vertices[a][axis]<=x+1e-10
            ib=vertices[b][axis]>=x-1e-10 if lower else vertices[b][axis]<=x+1e-10
            if ia: result.append(a)
            if ia!=ib: result.append(crossing(a,b,x))
        result=[v for i,v in enumerate(result) if i==0 or v!=result[i-1]]
        if len(result)>1 and result[0]==result[-1]: result.pop()
        return result
    for ring,normal in groups:
        for x0,x1 in zip(stations[:-1],stations[1:]):
            poly=clip(clip(ring,x0,True),x1,False)
            if len(poly)>=3:
                output.extend(triangulate(poly,vertices,normal))
    output=np.array(output,dtype=np.int64)
    used=np.unique(output)
    require(np.isin(np.arange(len(p)), used).all(), 'lost source vertex')
    remap=np.full(len(vertices),-1,dtype=np.int64); remap[used]=np.arange(len(used))
    return np.array(vertices)[used],remap[output],len(groups)
