"""End-to-end checkpoint assessment via the user's installed compatible RWKV-7 runner.

Runner interface: a Python module path exporting:
  def evaluate(checkpoint_path: str, texts: list[str], output_npz: str) -> None
It MUST export aligned next-token logits and labels to NPZ, using the same
checkpoint architecture, tokenizer and token sequence across both calls.
Output is verified via eval_logits.evaluate.
"""
import argparse,importlib,json,sys
from pathlib import Path
from eval_logits import evaluate

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--original',required=True)
 p.add_argument('--archive',required=True)
 p.add_argument('--runner',required=True,help='module:function, e.g. my_rwkv_runner:evaluate')
 p.add_argument('--texts',required=True,help='UTF-8 JSON array of evaluation texts')
 p.add_argument('--work-dir',default='evaluation')
 p.add_argument('--max-ppl-ratio',type=float,default=1.05)
 p.add_argument('--min-top1',type=float,default=0.90)
 a=p.parse_args()
 folder=Path(a.work_dir);folder.mkdir(parents=True,exist_ok=True)
 texts=json.loads(Path(a.texts).read_text(encoding='utf-8'))
 if not isinstance(texts,list) or not texts or not all(isinstance(x,str) for x in texts):raise ValueError('texts JSON must be nonempty list of strings')
 mod,fn=a.runner.split(':',1)
 func=getattr(importlib.import_module(mod),fn)
 from quant import restore
 recon=folder/'restored_quantized.pth'
 restore(a.archive,recon)
 ref=folder/'fp16_logits.npz';quant=folder/'quant_logits.npz'
 func(str(a.original),texts,str(ref))
 func(str(recon),texts,str(quant))
 report=evaluate(ref,quant,a.max_ppl_ratio,a.min_top1)
 report.update({'original':a.original,'archive':a.archive,'runner':a.runner,
                'disclaimer':'Numerical output parity, not a comprehensive reasoning benchmark.'})
 (folder/'report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
 return 0 if report['pass'] else 2
if __name__=='__main__':sys.exit(main())