"""Verify that a valid saved candidate cannot silently pass a failed quality gate."""
import json
import subprocess
import struct
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXE=ROOT/'build_rx/Release/cad_raw_partition_cli.exe'

class EndpointCliTests(unittest.TestCase):
    def test_legal_output_and_quality_acceptance_have_distinct_exit_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'square.cadpart'
            source.write_bytes(b'CADPART1'+struct.pack('<4I6d',4,2,1,0,*([0.]*6))+
                struct.pack('<12d',0,0,0,1,0,0,1,1,0,0,1,0)+
                struct.pack('<8I',0,1,2,0,0,2,3,0)+struct.pack('<5I',1,1,0,2,0)+
                struct.pack('<9d',0,0,0,0,0,1,0,0,0))
            rows=[]
            for name,target in [('already_sized',5),('refined',.4)]:
                output=root/(name+'.ply')
                completed=subprocess.run([str(EXE),str(source),str(output),'--target',str(target),
                    '--max-error','.1','--iters','2','--workers','1','--smooth-passes','3'],capture_output=True,text=True)
                data=json.loads(Path(str(output)+'.json').read_text())
                self.assertTrue(output.exists(),completed.stderr);self.assertTrue(data['topology_valid'])
                self.assertTrue(data['boundaries_held']);self.assertEqual(data['fallback'],0)
                self.assertEqual(completed.returncode,0 if data['quality_accepted'] else 4)
                rows.append(data)
            self.assertTrue(rows[0]['quality_accepted']);self.assertEqual(rows[0]['output_faces'],2)
            self.assertFalse(rows[1]['quality_accepted']);self.assertGreater(rows[1]['output_faces'],2)
            self.assertTrue(rows[1]['regressed_patch_ids'])

if __name__=='__main__':unittest.main()
