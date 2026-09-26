"""Regression checks for shared-edge repair, source immutability and rejection."""
import struct
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from repair_narrow_strips import audit, read_snapshot, repair, write_snapshot


class StripRepairTests(unittest.TestCase):
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
