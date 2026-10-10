import sys,unittest,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from quant import pack_codes,unpack_codes,encode_matrix,decode_matrix,allocate
class QuantTest(unittest.TestCase):
 def test_packing(self):
  for bits in (1,2,3,4,8):
   a=(np.arange(531,dtype=np.uint16)%(2**bits)).astype(np.uint8)
   self.assertTrue(np.array_equal(a,unpack_codes(pack_codes(a,bits),len(a),bits)))
 def test_round_trip(self):
  rng=np.random.default_rng(1);a=rng.normal(size=(19,37)).astype('float32')
  for b in (1,2,3,4,8):
   enc=encode_matrix(a,b,64); dec=decode_matrix(enc,a.shape,b,64)
   self.assertEqual(dec.shape,a.shape)
   self.assertTrue(np.isfinite(dec).all())
   self.assertLess(np.mean((a-dec)**2),1.2)
 def test_residual(self):
  rng=np.random.default_rng(3);a=rng.normal(size=(30,21)).astype('float32');enc=encode_matrix(a,2,64,residual_rank=2)
  b=decode_matrix(enc,a.shape,2,64,residual_rank=2)
  self.assertTrue(np.isfinite(b).all())
 def test_budget(self):
  errors={str(b):1/b for b in (1,2,3,4,8)}
  e={'a':dict(n=65536,kind='matrix_ffn',errors=errors),'b':dict(n=33,kind='fp16')}
  alloc,total=allocate(e,200000)
  self.assertLessEqual(total,200000)
  self.assertEqual(alloc['b'],16)
if __name__=='__main__':unittest.main()

class IntegrationTest(unittest.TestCase):
 def test_checkpoint_pipeline(self):
  import torch
  from quant import inspect_checkpoint,build,restore,verify
  with tempfile.TemporaryDirectory() as d:
   d=Path(d)
   state={'emb.weight':torch.randn(80,40),'blocks.0.ffn.key.weight':torch.randn(128,64),'blocks.0.ln1.weight':torch.randn(40)}
   torch.save(state,d/'original.pth')
   p=inspect_checkpoint(d/'original.pth',d/'plan.json',group_size=32,sample=1024,threshold=256,budget_mb=0.02,residual_rank=2)
   build(d/'original.pth',d/'plan.json',d/'packed.rwq')
   restore(d/'packed.rwq',d/'restored.pth')
   r=verify(d/'packed.rwq',d/'original.pth')
   self.assertEqual(r['weight_elements'],sum(x.numel() for x in state.values()))
   reconstructed=torch.load(d/'restored.pth',weights_only=True)
   self.assertEqual(set(state),set(reconstructed))
   self.assertLess((d/'packed.rwq').stat().st_size,20000)