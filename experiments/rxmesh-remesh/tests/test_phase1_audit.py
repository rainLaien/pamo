import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_phase1 import chain_audit, quality_regions, selected_region_quality, read_ply
from sweep_strip_sizes import evaluate_candidate, interface_scope
from compare_regional_candidate import nonregression, canonical_triangles, candidate_decision

class IndependentAuditTests(unittest.TestCase):
    def test_ply_coordinate_types_match_independent_library(self):
        import tempfile,trimesh
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'types.ply'
            template='ply\nformat ascii 1.0\nelement vertex 3\nproperty {kind} x\nproperty {kind} y\nproperty {kind} z\nelement face 1\nproperty list uchar int vertex_indices\nproperty uint patch_id\nend_header\n300.123456789 0 0\n301 0 0\n300 1 0\n3 0 1 2 0\n'
            for kind in ['float','double']:
                path.write_text(template.format(kind=kind))
                p,faces,labels=read_ply(path)
                ref=trimesh.load(path,force='mesh',process=False)
                np.testing.assert_array_equal(p,ref.vertices)
                np.testing.assert_array_equal(faces,ref.faces)
                self.assertEqual(labels.tolist(),[0])
                if kind=='double':self.assertEqual(p[0,0],300.123456789)
                else:self.assertEqual(p[0,0],float(np.float32(300.123456789)))
    def test_empty_repair_selection_reports_unavailable_quality(self):
        row=selected_region_quality(np.array([.2]),np.array([1.]),np.array([False]),0,.1)
        self.assertEqual(row,dict(patches=0,faces=0,quality_mean=None,quality_p05=None,low_quality_area=0.))

    def test_cached_proximity_matches_library_distances_and_ties(self):
        import trimesh
        from cached_proximity import CachedSurfaceQuery,compact_mesh
        # Coplanar adjacent faces and oppositely oriented parallel layers
        # exercise equal-distance normal selection; unused vertices must not
        # alter the radius used to find candidate triangles.
        p=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0],
                    [0,0,1],[1,0,1],[1,1,1],[0,1,1],[.5,.5,.25]])
        f=np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6]])
        mesh=trimesh.Trimesh(p,f,process=False)
        rng=np.random.default_rng(281)
        queries=np.vstack([rng.uniform(-.5,1.5,(256,3)),
                           [[.5,.5,.5],[.5,.5,0],[.5,.5,1],[0,0,.5]]])
        cached=CachedSurfaceQuery(mesh)
        compact=CachedSurfaceQuery(compact_mesh(p,f))
        for start in range(0,len(queries),64):
            batch=queries[start:start+64]
            old=trimesh.proximity.closest_point(mesh,batch)
            new=cached.on_surface(batch)
            for a,b in zip(old,new):np.testing.assert_array_equal(a,b)
            for a,b in zip(new,compact.on_surface(batch)):np.testing.assert_array_equal(a,b)

    def test_regional_progress_is_distinct_from_endpoint_acceptance(self):
        self.assertEqual(candidate_decision(True,[],True,False,False),(True,False))
        self.assertEqual(candidate_decision(True,['area_weighted_mean'],True,True,True),(False,False))
        self.assertEqual(candidate_decision(True,[],True,True,True),(True,True))

    def test_regional_delta_preserves_orientation(self):
        p=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        f=np.array([[0,1,2]])
        self.assertTrue(np.array_equal(canonical_triangles(p,f),canonical_triangles(p,np.array([[1,2,0]]))))
        self.assertFalse(np.array_equal(canonical_triangles(p,f),canonical_triangles(p,np.array([[0,2,1]]))))

    def test_regional_comparison_rejects_area_weighted_loss(self):
        before=dict(quality_mean=.5,quality_p05=.1,area_weighted_mean=.6,low_quality_area=2.,largest_low_quality_area=1.)
        after=dict(before,quality_mean=.7,quality_p05=.2,area_weighted_mean=.59)
        self.assertEqual(nonregression(before,after),['area_weighted_mean'])
        self.assertEqual(nonregression(before,before),[])

    def test_interface_sweep_detects_unmatched_stations(self):
        p=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0],[2,0,0],[2,1,0]])
        r=np.array([[0,1,2,0],[0,2,3,0],[1,4,5,1],[1,5,2,1]])
        matched=interface_scope(p,r,[0,1],{0:p,1:p},1e-6)
        self.assertEqual(matched['mismatched_selected_interfaces'],0)
        unmatched=interface_scope(p,r,[0,1],{0:np.vstack([p,[1,.5,0]]),1:p},1e-6)
        self.assertEqual(unmatched['mismatched_selected_interfaces'],1)

    def test_strip_size_sweep_retains_boundary_and_improves_quality(self):
        p=np.array([[0.,0,0],[10,0,0],[10,.2,0],[0,.2,0]])
        f=np.array([[0,1,2],[0,2,3]])
        _, _, coarse=evaluate_candidate(p,f,2.,.3,.01)
        _, _, fine=evaluate_candidate(p,f,.2,.3,.01)
        self.assertTrue(fine['checks']['local_checks_pass'])
        self.assertGreater(fine['metrics']['area_weighted_mean'],coarse['metrics']['area_weighted_mean'])
        self.assertLess(fine['metrics']['low_quality_area'],coarse['metrics']['low_quality_area'])
        self.assertGreater(fine['metrics']['faces'],coarse['metrics']['faces'])

    def test_chain_identity_survives_subdivision(self):
        p=np.array([[0.,0,0],[2,0,0]])
        op=np.vstack([p,[1,0,0]])
        edges=np.array([[0,2],[1,2]])
        result=chain_audit(p,np.array([[0,1]]),op,edges,1e-6)
        self.assertEqual(result['missing_chains'],0)
        self.assertEqual(result['missing_anchors'],0)
        result=chain_audit(p,np.array([[0,1]]),op,edges[:1],1e-6)
        self.assertEqual(result['missing_chains'],1)

    def test_bad_region_connects_across_patch_ids(self):
        p=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
        faces=np.array([[0,1,2],[0,2,3]])
        result,*_=quality_regions(p,faces,np.array([0,1]),.99,1.)
        self.assertEqual(result['low_quality_components'],1)
        self.assertAlmostEqual(result['largest_low_quality_area'],1.)

if __name__=='__main__':unittest.main()
