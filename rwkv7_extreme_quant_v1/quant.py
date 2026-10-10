"""RWKV-7 weight-only extreme quantizer. No architecture or tensor-key modifications."""
import argparse, json, math, zipfile
from pathlib import Path
import numpy as np

BITS = (1, 2, 3, 4, 8)

def pack_codes(codes, bits):
    a = np.asarray(codes, dtype=np.uint8).ravel()
    if bits == 8: return a.tobytes()
    # Chunking avoids allocating an O(N*bits) temporary for large matrices.
    out=bytearray()
    chunk=8*131072  # aligned to whole bytes for all bit depths
    shifts=np.arange(bits-1,-1,-1,dtype=np.uint8)
    for start in range(0,len(a),chunk):
        sub=a[start:start+chunk]
        planes=((sub[:,None] >> shifts)&1).reshape(-1)
        out.extend(np.packbits(planes).tobytes())
    return bytes(out)

def unpack_codes(raw, count, bits):
    if bits == 8: return np.frombuffer(raw, np.uint8, count=count).copy()
    byte = np.frombuffer(raw, np.uint8)
    out = np.zeros(count, dtype=np.uint8)
    for i in range(bits):
        pos = np.arange(count, dtype=np.int64) * bits + i
        out |= (((byte[pos // 8] >> (7 - pos % 8)) & 1) << (bits - i - 1)).astype(np.uint8)
    return out

def qgroup(x, bits):
    """Uniform signed midrise codebook; optimize scale over candidate thresholds."""
    levels = (1 << bits)
    x = np.asarray(x, np.float32)
    if x.size == 0: return np.empty(0, np.uint8), 1.0, 0.0
    mx = float(np.max(np.abs(x)))
    if not np.isfinite(mx): raise ValueError('non-finite tensor')
    if mx == 0: return np.zeros(x.size, np.uint8), 0.0, 0.0
    positive_max = (levels-1)/2
    scales = [mx / positive_max * v for v in (0.35,0.50,0.65,0.8,1.0,1.2,1.5)]
    best = None
    for sc in scales:
        code = np.clip(np.rint(x / sc + positive_max), 0, levels-1).astype(np.uint8)
        deq = (code.astype(np.float32)-positive_max)*sc
        err = float(np.mean((x-deq)**2))
        if best is None or err < best[0]: best = (err, code, sc)
    return best[1], best[2], best[0]

def encode_matrix(arr, bits, group_size=256, residual_rank=0):
    a = np.asarray(arr, np.float32)
    shape = a.shape
    if a.ndim != 2: raise ValueError('encode_matrix expects 2d')
    residual = None
    if residual_rank:
        # randomized low-rank pilot approximation, suitable for limited ranks; CPU memory caution
        rng = np.random.default_rng(78)
        r = min(residual_rank, *a.shape)
        omega = rng.normal(size=(a.shape[1],r+4)).astype(np.float32)
        q,_ = np.linalg.qr(a @ omega)
        u,s,vh = np.linalg.svd(q.T @ a, full_matrices=False)
        L = (q @ u[:,:r] * np.sqrt(s[:r])).astype(np.float16)
        R = (np.sqrt(s[:r])[:,None] * vh[:r]).astype(np.float16)
        # quantify residual relative to lowrank reconstruction
        a = a-L.astype(np.float32) @ R.astype(np.float32)
        residual=(L,R)
    flat = a.ravel()
    scales = np.empty(math.ceil(flat.size/group_size),np.float16)
    codes = np.empty(flat.size,np.uint8)
    # Vectorized batches of groups to make multi-billion-weight checkpoints feasible.
    max_groups=max(1, 1048576//group_size)
    levels=(1<<bits); limit=(levels-1)/2
    factors=np.asarray((0.35,0.50,0.65,0.8,1.,1.2,1.5),np.float32)
    for g0 in range(0,len(scales),max_groups):
        g1=min(len(scales),g0+max_groups)
        begin=g0*group_size; finish=min(g1*group_size,len(flat))
        batch=np.zeros((g1-g0,group_size),np.float32)
        batch.ravel()[:finish-begin]=flat[begin:finish]
        mx=np.max(np.abs(batch),axis=1)
        base=mx/limit
        best_error=np.full(len(mx),np.inf,np.float32)
        best_sc=np.zeros(len(mx),np.float32)
        best_code=np.zeros_like(batch,dtype=np.uint8)
        for factor in factors:
            sc=base*factor
            safe=np.where(sc>0,sc,1.)
            c=np.clip(np.rint(batch/safe[:,None]+limit),0,levels-1).astype(np.uint8)
            err=np.sum((batch-(c.astype(np.float32)-limit)*sc[:,None])**2,axis=1)
            take=err<best_error
            best_error[take]=err[take];best_sc[take]=sc[take];best_code[take]=c[take]
        scales[g0:g1]=best_sc.astype(np.float16)
        codes[begin:finish]=best_code.ravel()[:finish-begin]
    blobs={'codes':pack_codes(codes,bits), 'scales':scales.tobytes()}
    if residual:
        blobs['res_l']=residual[0].tobytes()
        blobs['res_r']=residual[1].tobytes()
    return blobs

def decode_matrix(blobs,shape,bits,group_size=256,residual_rank=0):
    n=int(np.prod(shape)); code=unpack_codes(blobs['codes'],n,bits).astype(np.float32)
    sc=np.frombuffer(blobs['scales'],np.float16).astype(np.float32)
    out=(code-(2**bits-1)/2)*np.repeat(sc,group_size)[:n]
    out=out.reshape(shape)
    if residual_rank:
        l=np.frombuffer(blobs['res_l'],np.float16).reshape(shape[0],residual_rank).astype(np.float32)
        r=np.frombuffer(blobs['res_r'],np.float16).reshape(residual_rank,shape[1]).astype(np.float32)
        out += l@r
    return out

def classify(name,arr,threshold):
    if arr.ndim !=2 or arr.size < threshold: return 'fp16'
    if name in ('emb.weight','head.weight'): return 'matrix_sensitive'
    if '.att.' in name: return 'matrix_att'
    if '.ffn.' in name: return 'matrix_ffn'
    return 'matrix_other'

def sensitivity(a,group_size, sample=8192, seed=7):
    """Deterministic sampled relative reconstruction MSE; NOT task-loss sensitivity."""
    v=np.asarray(a,dtype=np.float32).ravel()
    rng=np.random.default_rng(seed)
    if v.size>sample: v=v[rng.choice(v.size,sample,replace=False)]
    denom=float(np.mean(v*v))+1e-12
    metrics={}
    for bits in BITS:
        se=0; total=0
        for i in range(0,len(v),group_size):
            x=v[i:i+group_size]; c,s,_=qgroup(x,bits)
            y=(c.astype(np.float32)-(2**bits-1)/2)*s
            se+=float(np.sum((x-y)**2));total+=len(x)
        metrics[bits] = se/max(total,1)/denom
    return metrics

def allocate(entries,budget_bytes,group_size=256, min_bits=1):
    """Greedy marginal gain per extra byte. Includes estimated scale, fp16 and metadata cost."""
    allocation={k:(min_bits if v['kind']!='fp16' else 16) for k,v in entries.items()}
    def cost(e,b):
        n=e['n']
        if b==16:return n*2
        return math.ceil(n*b/8)+2*math.ceil(n/group_size)+384
    used=sum(cost(e,allocation[k]) for k,e in entries.items())+16384
    if used>budget_bytes: raise RuntimeError(f'Budget impossible even with {min_bits}-bit matrices: {used/1e6:.2f} MB > {budget_bytes/1e6:.2f} MB. Reduce protected tensors or use a larger budget.')
    # target error weighted by either optional observed activity or a conservative structural multiplier
    weights={'matrix_sensitive':2.0,'matrix_att':1.75,'matrix_ffn':1.0,'matrix_other':1.1}
    while True:
        best=None
        for key,e in entries.items():
            b=allocation[key]
            if b not in BITS or b==8: continue
            nxt=BITS[BITS.index(b)+1]
            delta=cost(e,nxt)-cost(e,b)
            if delta > budget_bytes-used: continue
            gain=max(0., e['errors'][str(b)]-e['errors'][str(nxt)])*e['n']*weights.get(e['kind'],1.)
            ratio=gain/max(delta,1)
            if best is None or ratio>best[0]:best=(ratio,key,nxt,delta)
        if best is None or best[0]<=0:break
        _,key,nxt,delta=best;allocation[key]=nxt;used+=delta
    return allocation,used

def extract_state(obj):
    if isinstance(obj,dict):
        for key in ('state_dict','model'):
            if key in obj and isinstance(obj[key],dict):return obj[key]
        return obj
    raise ValueError('Checkpoint must be dict')

def tensor_numpy(t):
    # BF16 cannot be exported directly through torch.numpy on many versions.
    return t.detach().cpu().to(dtype=__import__('torch').float32).numpy()

def load_state(path):
    import torch
    return extract_state(torch.load(path,map_location='cpu',weights_only=True,mmap=True))

def inspect_checkpoint(path,out,group_size=256,sample=8192,threshold=65536,budget_mb=700,residual_rank=4):
    state=load_state(path)
    entries={}
    for i,(key,weight) in enumerate(state.items()):
        if not hasattr(weight,'shape'):continue
        a=tensor_numpy(weight);kind=classify(key,a,threshold)
        if not np.issubdtype(a.dtype,np.number):raise ValueError(f'Non numeric: {key}')
        if kind=='fp16': entries[key]={'kind':kind,'shape':list(a.shape),'n':a.size}
        else:
            metrics=sensitivity(a,group_size,sample,seed=1024+i)
            entries[key]={'kind':kind,'shape':list(a.shape),'n':a.size,'errors':{str(k):v for k,v in metrics.items()}}
        if (i+1)%30==0:print('Inspected',i+1,'tensors',flush=True)
    # Reserve a bounded residual pool; attach factors to the highest-priority
    # sensitive matrices. Optimizer accounts for factors before bit assignment.
    eligible=[(k,v) for k,v in entries.items() if v['kind']=='matrix_sensitive']
    residuals={}
    max_reserve=min(int(budget_mb*1e6*0.012),8_000_000)
    reserved=0
    for k,v in eligible:
        rank=min(residual_rank,*v['shape']) if residual_rank>0 else 0
        extra=2*rank*(v['shape'][0]+v['shape'][1])
        if rank and reserved+extra<=max_reserve:
            residuals[k]=rank;reserved+=extra
    alloc,estimate=allocate(entries,int(budget_mb*1e6)-reserved,group_size)
    estimate+=reserved
    payload={'format':'rwkv7-eq-v1-plan','source':str(path),'group_size':group_size,'threshold':threshold,'budget_bytes':int(budget_mb*1e6),'estimated_bytes':estimate,'entries':entries,'allocation':alloc,'residual_ranks':residuals,'reserved_residual_bytes':reserved,'sensitivity_method':'sampled relative reconstruction mse, not activation/task calibrated'}
    Path(out).write_text(json.dumps(payload,indent=2))
    return payload

def build(checkpoint,plan_path,output,strict=True):
    plan=json.loads(Path(plan_path).read_text());state=load_state(checkpoint)
    metadata={'format':'rwkv7-eq-v1','group_size':plan['group_size'],'tensors':{},'source_name':Path(checkpoint).name,'architecture':'RWKV7 keys preserved, weights only'}
    output=Path(output)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for i,(name,e) in enumerate(plan['entries'].items()):
            weight=state[name].detach().cpu().numpy()
            if list(weight.shape)!=e['shape']:raise ValueError(f'Shape changed {name}')
            bits=plan['allocation'][name];rank=plan.get('residual_ranks',{}).get(name,0) if bits!=16 else 0;prefix=f'tensors/{i:05d}/'
            if bits==16:
                data=np.asarray(weight,dtype='<f2').tobytes()
                archive.writestr(prefix+'fp16',data)
                files={'fp16':prefix+'fp16'}
            else:
                blobs=encode_matrix(weight,bits,plan['group_size'],residual_rank=rank)
                files={}
                for section,blob in blobs.items():
                    files[section]=prefix+section
                    archive.writestr(prefix+section,blob)
            metadata['tensors'][name]={'shape':e['shape'],'bits':bits,'residual_rank':rank,'files':files}
            if (i+1)%30==0:print('Packed',i+1,'/',len(plan['entries']),flush=True)
        archive.writestr('manifest.json',json.dumps(metadata,separators=(',',':')))
    result=output.stat().st_size
    print(f'Wrote {output}: {result/1e6:.3f} MB; budget {plan["budget_bytes"]/1e6:.3f} MB')
    if strict and result>plan['budget_bytes']:
        raise RuntimeError(f'Actual archive exceeds budget by {(result-plan["budget_bytes"])/1e6:.3f} MB. Increase budget or lower allocation.')
    return result

def restore(archive_path,output):
    import torch
    restored={}
    with zipfile.ZipFile(archive_path) as z:
        meta=json.loads(z.read('manifest.json'))
        for name,e in meta['tensors'].items():
            shape=e['shape'];bits=e['bits'];files=e['files']
            if bits==16:
                arr=np.frombuffer(z.read(files['fp16']),dtype='<f2').reshape(shape).copy()
            else:
                blobs={k:z.read(v) for k,v in files.items()}
                arr=decode_matrix(blobs,shape,bits,meta['group_size'],e.get('residual_rank',0)).astype(np.float16)
            restored[name]=torch.from_numpy(arr)
    torch.save(restored,output)
    print('Restored',len(restored),'tensors to',output)

def verify(archive_path,checkpoint,threshold=1.0):
    orig=load_state(checkpoint); se=0.;energy=0.;n=0;details=[]
    with zipfile.ZipFile(archive_path) as z:
        meta=json.loads(z.read('manifest.json'))
        for name,e in meta['tensors'].items():
            a=tensor_numpy(orig[name]);shape=e['shape'];bits=e['bits'];files=e['files']
            if bits==16:b=np.frombuffer(z.read(files['fp16']),'<f2').reshape(shape).astype(np.float32)
            else:b=decode_matrix({k:z.read(v) for k,v in files.items()},shape,bits,meta['group_size'],e.get('residual_rank',0))
            error=float(np.sum((a-b)**2,dtype=np.float64));norm=float(np.sum(a*a,dtype=np.float64))
            se+=error;energy+=norm;n+=a.size
            details.append({'name':name,'bits':bits,'relative_mse':error/max(norm,1e-30)})
    report={'relative_mse_global':se/max(energy,1e-30),'rmse_global':math.sqrt(se/max(n,1)),'weight_elements':n,'weight_only_pass':se/max(energy,1e-30)<=threshold,'per_tensor':details,'warning':'weight reconstruction checks only; NOT perplexity, task accuracy, or inference parity'}
    print(json.dumps({k:v for k,v in report.items() if k!='per_tensor'},indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description='RWKV-7 Extreme Quant V1 - weight-only converter')
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('plan');a.add_argument('--checkpoint',required=True);a.add_argument('--out',required=True);a.add_argument('--budget-mb',type=float,default=700);a.add_argument('--group-size',type=int,default=256);a.add_argument('--sample',type=int,default=8192);a.add_argument('--threshold',type=int,default=65536);a.add_argument('--residual-rank',type=int,default=4)
    b=sub.add_parser('pack');b.add_argument('--checkpoint',required=True);b.add_argument('--plan',required=True);b.add_argument('--out',required=True)
    c=sub.add_parser('restore');c.add_argument('--archive',required=True);c.add_argument('--out',required=True)
    d=sub.add_parser('verify');d.add_argument('--archive',required=True);d.add_argument('--checkpoint',required=True);d.add_argument('--out',required=True)
    x=p.parse_args()
    if x.cmd=='plan':inspect_checkpoint(x.checkpoint,x.out,x.group_size,x.sample,x.threshold,x.budget_mb,x.residual_rank)
    elif x.cmd=='pack':build(x.checkpoint,x.plan,x.out)
    elif x.cmd=='restore':restore(x.archive,x.out)
    else:Path(x.out).write_text(json.dumps(verify(x.archive,x.checkpoint),indent=2))
if __name__=='__main__':main()