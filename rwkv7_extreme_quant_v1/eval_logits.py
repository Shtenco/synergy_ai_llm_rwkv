"""Compute strict teacher/quantized token-level comparisons from exported logits.
NPZ inputs: logits [T,V] or [B,T,V] and labels [T] or [B,T];
label -100 ignored. Both files must use the same ordered tokens and tokenizer.
This evaluates outputs from a real model runner, not weight reconstruction.
"""
import argparse,json,numpy as np

def evaluate(ref_path,quant_path,nll_ratio_limit=1.05,top1_limit=0.90):
    a=np.load(ref_path);b=np.load(quant_path)
    ref=np.asarray(a['logits'],dtype=np.float64);q=np.asarray(b['logits'],dtype=np.float64)
    labels=np.asarray(a['labels'],dtype=np.int64)
    if ref.shape!=q.shape or labels.shape!=ref.shape[:-1]: raise ValueError('Shapes must match')
    if 'labels' in b and not np.array_equal(labels,b['labels']):raise ValueError('Label arrays differ')
    if not (np.isfinite(ref).all() and np.isfinite(q).all()):raise ValueError('Nonfinite logits')
    valid=labels.reshape(-1)!=-100
    r=ref.reshape(-1,ref.shape[-1])[valid];z=q.reshape(-1,q.shape[-1])[valid]
    y=labels.reshape(-1)[valid]
    if len(y)==0 or np.any(y<0) or np.any(y>=ref.shape[-1]):raise ValueError('Invalid labels')
    def nll(x):
        m=np.max(x,axis=1)
        return m+np.log(np.exp(x-m[:,None]).sum(axis=1))-x[np.arange(len(y)),y]
    n_ref=float(np.mean(nll(r)));n_quant=float(np.mean(nll(z)))
    top1=float(np.mean(np.argmax(r,axis=1)==np.argmax(z,axis=1)))
    report={'n_tokens':int(len(y)),'mean_nll_reference':n_ref,'mean_nll_quantized':n_quant,'ppl_reference':float(np.exp(min(n_ref,700))),'ppl_quantized':float(np.exp(min(n_quant,700))),'ppl_ratio':float(np.exp(min(n_quant-n_ref,700))),'top1_agreement':top1}
    report['pass']=report['ppl_ratio']<=nll_ratio_limit and top1>=top1_limit
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ref',required=True);p.add_argument('--quant',required=True);p.add_argument('--out',required=True);p.add_argument('--ppl-limit',type=float,default=1.05);p.add_argument('--top1-limit',type=float,default=0.90)
    x=p.parse_args();report=evaluate(x.ref,x.quant,x.ppl_limit,x.top1_limit)
    with open(x.out,'w') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['pass'] else 2)