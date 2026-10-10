"""Native low-bit CUDA matvec via Triton. PyTorch/Triton/CUDA GPU required.

This is a *linear operator backend*, not a complete RWKV-7 generation engine.
Weights remain bitpacked on GPU; a kernel decodes codes into registers and
accumulates the dot product, without materializing the dense FP16 weight matrix.
"""
import argparse, json, zipfile
from dataclasses import dataclass
import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is not None:
    @triton.jit
    def _qmatvec(C, S, X, Y,
                 M:tl.constexpr, N:tl.constexpr, BITS:tl.constexpr,
                 GROUP:tl.constexpr, BLOCK:tl.constexpr):
        row=tl.program_id(0)
        col=tl.arange(0,BLOCK)
        ok=col<N
        idx=row*N+col
        # The bit stream is MSB-first; each individual code can straddle bytes.
        bitidx=idx*BITS
        shift=bitidx%8
        p0=bitidx//8
        first=tl.load(C+p0,mask=ok,other=0).to(tl.uint32)
        second=tl.load(C+p0+1,mask=ok & (shift+BITS>8),other=0).to(tl.uint32)
        pair=(first<<8)|second
        code=(pair>>(16-shift-BITS))&((1<<BITS)-1)
        scale=tl.load(S+idx//GROUP,mask=ok,other=0).to(tl.float32)
        w=(code.to(tl.float32)-(float((1<<BITS)-1)/2.0))*scale
        x=tl.load(X+col,mask=ok,other=0).to(tl.float32)
        val=tl.sum(w*x,0)
        tl.store(Y+row,val)

@dataclass
class LowBitMatrix:
    shape:tuple
    bits:int
    group_size:int
    codes:torch.Tensor
    scales:torch.Tensor
    res_l:torch.Tensor|None=None
    res_r:torch.Tensor|None=None

    @classmethod
    def from_blobs(cls, blobs, shape, bits, group_size, residual_rank=0, device='cuda'):
        if bits not in (1,2,3,4,8):
            raise ValueError('Supported packed bits: 1, 2, 3, 4, 8')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required')
        codes=np.frombuffer(blobs['codes'],dtype=np.uint8).copy()
        scales=np.frombuffer(blobs['scales'],dtype=np.float16).copy()
        expected=(int(np.prod(shape))*bits+7)//8
        if len(codes)!=expected:
            raise ValueError(f'codes length {len(codes)} != {expected}')
        groups=(int(np.prod(shape))+group_size-1)//group_size
        if len(scales)!=groups:
            raise ValueError(f'scales length {len(scales)} != {groups}')
        l=r=None
        if residual_rank:
            l=np.frombuffer(blobs['res_l'],dtype=np.float16).copy().reshape(shape[0],residual_rank)
            r=np.frombuffer(blobs['res_r'],dtype=np.float16).copy().reshape(residual_rank,shape[1])
            l=torch.from_numpy(l).to(device)
            r=torch.from_numpy(r).to(device)
        # Padding extra byte ensures valid address for boundary accesses; reads still masked.
        codes=np.pad(codes,(0,1))
        return cls(tuple(shape),bits,group_size,torch.from_numpy(codes).to(device),
                   torch.from_numpy(scales).to(device),l,r)

    def matvec(self,x):
        if triton is None:
            raise ImportError('Install triton for native CUDA kernels')
        if not x.is_cuda:
            raise ValueError('Input must be on CUDA')
        if x.ndim!=1 or x.numel()!=self.shape[1]:
            raise ValueError(f'Expected vector [{self.shape[1]}]')
        x=x.contiguous()
        m,n=self.shape
        y=torch.empty((m,),device=x.device,dtype=torch.float32)
        # Current kernel uses one CTA per row, intended for per-token generation.
        # Larger matrices need segmented/blocked reductions for high throughput.
        block=triton.next_power_of_2(n)
        if block>65536:
            raise ValueError('N > 65536 not supported by single-CTA kernel')
        _qmatvec[(m,)](self.codes,self.scales,x,y,m,n,self.bits,self.group_size,block,
                       num_warps=4 if block<=2048 else 8)
        if self.res_l is not None:
            y=y+(self.res_l.float() @ (self.res_r.float()@x.float()))
        return y

class LowBitArchive:
    """Open an RWQ archive, materialize selected linear weights as GPU low-bit ops."""
    def __init__(self,path,device='cuda'):
        self.path=path
        self.device=device
        with zipfile.ZipFile(path) as z:
            self.manifest=json.loads(z.read('manifest.json'))
        self.cache={}
    def names(self): return tuple(self.manifest['tensors'])
    def load(self,name):
        if name in self.cache: return self.cache[name]
        meta=self.manifest['tensors'][name]
        if meta['bits']==16:
            raise ValueError('FP16 tensor; load through regular torch path')
        with zipfile.ZipFile(self.path) as z:
            blobs={k:z.read(v) for k,v in meta['files'].items()}
        matrix=LowBitMatrix.from_blobs(blobs,meta['shape'],meta['bits'],
                     self.manifest['group_size'],meta.get('residual_rank',0),self.device)
        self.cache[name]=matrix
        return matrix
    def matvec(self,name,x):return self.load(name).matvec(x)

def selftest():
    from quant import encode_matrix,decode_matrix
    rng=np.random.default_rng(12)
    for bits in (1,2,3,4,8):
        for m,n in ((7,31),(64,257),(5,1024)):
            a=rng.normal(size=(m,n)).astype(np.float32)
            x=rng.normal(size=n).astype(np.float32)
            rank=2 if m==7 else 0
            blobs=encode_matrix(a,bits,group_size=32,residual_rank=rank)
            matrix=LowBitMatrix.from_blobs(blobs,a.shape,bits,32,rank)
            actual=matrix.matvec(torch.from_numpy(x).cuda()).cpu().numpy()
            reference=decode_matrix(blobs,a.shape,bits,32,rank)@x
            np.testing.assert_allclose(actual,reference,atol=5e-3,rtol=3e-3)
            print('PASS',bits,m,n,'max_abs',np.max(np.abs(actual-reference)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    x=p.parse_args()
    if x.selftest:selftest()
    else:p.print_help()