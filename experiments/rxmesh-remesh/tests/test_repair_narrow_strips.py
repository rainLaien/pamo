"""Regression checks for shared-edge repair, source immutability and rejection."""
import struct
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from repair_narrow_strips import audit, read_snapshot, repair, write_snapshot, regional_targets, read_patch_targets
from extract_partition_blocks import extract


class StripRepairTests(unittest.TestCase):
    def test_diagnostic_extraction_retains_support_closure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root)
            h,p,r,patches,features=read_snapshot(source)
            patches[0][0][4]=1
            patches[0]=(patches[0][0],struct.pack('<I',1)+patches[0][1])
            write_snapshot(source,h,p,r,patches,features)
            original=source.read_bytes()
            report=extract(source,root/'subset.cadpart',[0],1)
            self.assertEqual(report['original_patch_ids'],[0,1])
            self.assertEqual(source.read_bytes(),original)
            self.assertFalse(report['complete_pipeline'])
            with self.assertRaisesRegex(ValueError,'positive'):extract(source,root/'bad.cadpart',[0],0)

    def test_graded_regional_neighbor_retains_constraints_and_plane(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root)
            result=repair(source,root/'graded.cadpart',target_length=1.4,patch_targets={0:.5},graded_neighbors=True)
            self.assertTrue(result['shared_edge_incidence_matches'])
            self.assertTrue(result['open_boundary_identity_matches'])
            self.assertTrue(result['graded_neighbors'])
            self.assertEqual(result['after']['float32_zero_faces'],0)
            self.assertLess(result['float32_changed_face_centroid_sample_error'],1e-5)
            self.assertTrue(any('graded_candidate' in x for x in result['neighbor_checks']))
            self.assertTrue(all(x['graded_candidate']['float32_reverse_sample_error']<1e-5 for x in result['neighbor_checks'] if 'graded_candidate' in x))

    def test_pinned_regional_selection_rejects_different_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root)
            config=root/'targets.json';config.write_text(json.dumps({'snapshot_sha256':'wrong','targets':{'0':.5}}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):read_patch_targets(config,source)

    def test_connected_regional_grid_uses_minimum_target(self):
        rows=np.array([[0,1,2,0],[2,1,3,1],[4,5,6,2]],dtype=np.uint32)
        sizes=regional_targets(rows,[0,1,2],{0:.5},2.)
        self.assertEqual(sizes,{0:.5,1:.5,2:2.})

    def test_regional_target_synchronizes_neighbors_and_invalidates_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root);original=source.read_bytes()
            output=root/'regional.cadpart'
            result=repair(source,output,target_length=1.4,patch_targets={0:.5})
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(result['regional_effective_patch_targets'],{0:.5})
            self.assertEqual(result['regional_affected_patch_ids'],[0,1])
            self.assertTrue(result['shared_edge_incidence_matches'])
            state=root/'regional.json';state.write_text(json.dumps(result),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'different regional targets'):
                repair(source,root/'second.cadpart',target_length=1.4,previous_report=state,patch_targets={0:.25})
            with self.assertRaisesRegex(ValueError,'invalid regional'):
                repair(source,root/'bad.cadpart',target_length=1.4,patch_targets={0:2.})

    def test_bounded_rebuild_scope_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root)
            with self.assertRaisesRegex(ValueError,'explicit targets'):
                repair(source,root/'bad.cadpart',target_length=1.4,only_requested_regions=True)
            result=repair(source,root/'bounded.cadpart',target_length=1.4,
                          patch_targets={0:.5},only_requested_regions=True)
            self.assertEqual(result['candidate_patch_ids'],[0])
            self.assertTrue(result['only_requested_regions'])
            self.assertTrue(result['shared_edge_incidence_matches'])
            report=root/'bounded.json';report.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'regional scope'):
                repair(source,root/'reuse.cadpart',target_length=1.4,patch_targets={0:.5},previous_report=report)

    def test_constrained_neighbor_scope_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root)
            with self.assertRaisesRegex(ValueError,'graded neighbors'):
                repair(source,root/'bad.cadpart',target_length=1.4,patch_targets={0:.5},constrained_neighbors=True)
            result=repair(source,root/'constrained.cadpart',target_length=1.4,
                          patch_targets={0:.5},graded_neighbors=True,only_requested_regions=True,constrained_neighbors=True)
            self.assertTrue(result['constrained_neighbors'])
            self.assertTrue(result['shared_edge_incidence_matches'])
            self.assertTrue(result['open_boundary_identity_matches'])
            self.assertEqual(result['after']['float32_zero_faces'],0)
            report=root/'constrained.json';report.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'neighbor triangulation'):
                repair(source,root/'reuse.cadpart',target_length=1.4,patch_targets={0:.5},graded_neighbors=True,
                       only_requested_regions=True,previous_report=report)

    def make_source(self,directory,strip_type=6,internal_feature=False):
        p=np.array([[0,0,0],[20,0,0],[20,.1,0],[0,.1,0],[20,1,0],[0,1,0]],dtype=float)
        records=np.array([[0,1,2,0],[0,2,3,0],[3,2,4,1],[3,4,5,1]],dtype=np.uint32)
        features=np.array([[1,0,1,0],[1,1,2,0],[1,2,3,0],[1,0,3,0]],dtype=np.uint32)
        if internal_feature:
            features=np.vstack([features,[2,0,2,0]]).astype(np.uint32)
        patches=[([strip_type,1,0,2,0],struct.pack('<9d',0,0,0,0,0,1,0,0,0)),
                 ([1,1,1,2,0],struct.pack('<9d',0,0,0,0,0,1,0,0,0))]
        header=b'CADPART1'+struct.pack('<4I6d',len(p),len(records),len(patches),len(features),*([0.]*6))
        source=directory/'source.cadpart'
        write_snapshot(source,header,p,records,patches,features)
        return source

    def test_shared_edge_and_feature_chains_remain_conforming(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root)
            original=source.read_bytes();output=root/'repaired.cadpart'
            result=repair(source,output,target_length=1.4)
            _,p,records,_,features=read_snapshot(output)
            checks,_,_=audit(p,records,features)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(result['candidate_patch_ids'],[0])
            self.assertTrue(result['all_original_vertices_unchanged'])
            self.assertTrue(result['shared_edge_incidence_matches'])
            self.assertGreater(result['neighbor_output_faces'],result['neighbor_source_faces'])
            self.assertEqual(checks['nonmanifold_edges'],0)
            self.assertEqual(checks['float32_zero_faces'],0)
            self.assertEqual(checks['missing_feature_edges'],0)
            self.assertEqual(checks['connected_components'],1)
            self.assertEqual(checks['euler_characteristic'],1)
            tri=p[records[records[:,3]==0,:3]]
            self.assertLessEqual(np.linalg.norm(tri-np.roll(tri,-1,axis=1),axis=2).max(),1.4*4/3)

    def test_no_candidates_is_an_exact_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root,strip_type=1)
            output=root/'copy.cadpart';result=repair(source,output,target_length=1.4)
            self.assertEqual(source.read_bytes(),output.read_bytes())
            self.assertEqual(result['candidate_patch_ids'],[])
            self.assertFalse(result['candidate_set_changed'])

    def test_internal_feature_is_reported_and_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root,internal_feature=True)
            output=root/'copy.cadpart';result=repair(source,output,target_length=1.4)
            self.assertEqual(source.read_bytes(),output.read_bytes())
            self.assertEqual(result['rebuilt_patches'],{})
            self.assertIn('internal protected edge',result['rejected_patches'][0])

    def test_truncated_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'broken.cadpart';path.write_bytes(b'CADPART1')
            with self.assertRaisesRegex(ValueError,'CADPART1'):
                read_snapshot(path)

    def test_cannot_overwrite_source_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root);original=source.read_bytes()
            with self.assertRaisesRegex(ValueError,'immutable'):
                repair(source,source,target_length=1.4)
            self.assertEqual(source.read_bytes(),original)

    def test_unchanged_candidates_reuse_checked_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.make_source(root);first=root/'first.cadpart'
            result=repair(source,first,target_length=1.4)
            state=root/'first.json';state.write_text(json.dumps(result),encoding='utf-8')
            second=root/'second.cadpart'
            reused=repair(source,second,target_length=1.4,previous_report=state)
            self.assertFalse(reused['candidate_set_changed'])
            self.assertEqual(first.read_bytes(),second.read_bytes())
            with self.assertRaisesRegex(ValueError,'different target'):
                repair(source,root/'third.cadpart',target_length=2.,previous_report=state)


if __name__=='__main__':
    unittest.main()
