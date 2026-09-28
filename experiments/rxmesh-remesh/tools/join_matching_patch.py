"""Join a completed candidate patch only under an exact boundary contract.
Interior vertices keep candidate-local identity. Matching boundary coordinates
are used only when each side has a unique boundary ID and identical edge graph.
Caller must separately audit original geometry, feature lineage and size fields.
"""
import numpy as np
from narrow_strip_geometry import require

def boundary(p,faces):
    edges={};directions={}
    for face in faces:
        for a,b in zip(face,np.roll(face,-1)):
            edge=tuple(sorted((int(a),int(b))));edges[edge]=edges.get(edge,0)+1;directions[edge]=(int(a),int(b))
    require(all(c<=2 for c in edges.values()),'nonmanifold patch')
    edge_set=set();vertices={}
    for edge,count in edges.items():
        if count!=1:continue
        a,b=directions[edge]
        for v in (a,b):
            key=tuple(p[v]);require(key not in vertices or vertices[key]==v,'ambiguous boundary identity')
            vertices[key]=v
        key=(tuple(p[a]),tuple(p[b]))
        require(key not in edge_set,'coincident boundary edges');edge_set.add(key)
    require(edge_set,'closed or empty patch needs a different join contract')
    return vertices,edge_set

def join(bp,bf,bl,cp,cf,cl,patch):
    require(np.isfinite(bp).all() and np.isfinite(cp).all(),'non-finite coordinates')
    old=bf[bl==patch];new=cf[cl==patch];require(len(old)>0 and len(new)>0,'patch missing')
    old_vertices,old_edges=boundary(bp,old);new_vertices,new_edges=boundary(cp,new)
    require(old_edges==new_edges,'boundary graph differs')
    require(set(old_vertices)==set(new_vertices),'boundary vertices differ')
    mapping={new_vertices[k]:old_vertices[k] for k in old_vertices}
    positions=list(bp.copy())
    for v in np.unique(new):
        if int(v) not in mapping:mapping[int(v)]=len(positions);positions.append(cp[v].copy())
    joined=np.asarray([[mapping[int(v)] for v in face] for face in new],dtype=np.int64)
    faces=np.vstack((bf[bl!=patch],joined));labels=np.r_[bl[bl!=patch],np.full(len(new),patch)]
    used=np.unique(faces)
    return np.asarray(positions)[used],np.searchsorted(used,faces),labels
