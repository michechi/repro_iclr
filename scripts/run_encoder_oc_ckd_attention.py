#!/usr/bin/env python3
"""Detached BERT/RoBERTa OC and CKD fits/audits after the current Qwen queue."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/encoder_oc_ckd_attention_20260923'
UPSTREAM=ROOT/'results/qwen_oc_clinical_attention_20260922_seed9950'
GPU_LOCK=ROOT/'results/.kip_data_efficiency_gpu.lock'
SMOKE_ROOT=Path('runs/encoder_smokes')
SOURCES=(
 'scripts/run_encoder_oc_ckd_attention.py','src/experiments/encoder_oc_ckd.py',
 'src/analysis/encoder_oc_ckd_attention.py','docs/encoder_oc_ckd_attention_20260923.md',
 'src/experiments/kip_encoder_training.py','src/analysis/kip_checkpoint_attention.py',
 'src/analysis/kip_attention_core.py','src/analysis/oc_attention_core.py',
 'src/analysis/qwen_oc_audit.py','src/analysis/ckd_attention_audit.py',
 'src/experiments/kip_efficiency_models.py','src/experiments/kip_llama_training.py',
 'src/experiments/kip_data_efficiency.py','src/mimic/qwen_ckd_training.py',
 'repro/src/analysis/mechanism_id/scripts/common.py',
)


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
 return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name('.'+path.name+'.tmp');temp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');temp.replace(path)


def environment(root):
 env=dict(os.environ)
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
  HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',
  OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
  PYTHONPATH=os.pathsep.join([str(ROOT/'.deps/kip_eff'),str(ROOT)]),
  MPLCONFIGDIR=str(root/'kernel_cache/matplotlib'),TRITON_CACHE_DIR=str(root/'kernel_cache/triton'),
  CUDA_CACHE_PATH=str(root/'kernel_cache/cuda'),TORCHINDUCTOR_CACHE_DIR=str(root/'kernel_cache/inductor'))
 return env


def prepare(root):
 sys.path.insert(0,str(ROOT/'.deps/kip_eff'));sys.path.insert(0,str(ROOT))
 import numpy as np
 import pandas as pd
 from transformers import AutoTokenizer
 from src.experiments.encoder_oc_ckd import EncoderDataset
 from src.analysis.oc_attention_core import label
 if (root/'launch_plan.json').exists():raise FileExistsError('existing plan; do not replace it')
 root.mkdir(parents=True,exist_ok=True)
 upstream=read(UPSTREAM/'launch_plan.json')
 old_ckd=read(ROOT/'results/qwen38_27b_ckd_20260921_seed42/launch_plan.json')
 models={};model_bytes=0
 for name in ('bert','roberta'):
  old=read(ROOT/f'results/kip_encoders_20260916_seed9950/runs/full/{name}/kip_m4/ordered/seed9950/manifest.json')['config']
  models[name]=dict(path=old['model_source'],name=old['model_name'],files_sha256=old['source_file_sha256'])
  for filename,h in models[name]['files_sha256'].items():
   p=Path(models[name]['path'])/filename
   if digest(p)!=h:raise ValueError('base model mismatch: '+str(p))
  model_bytes+=(Path(models[name]['path'])/'model.safetensors').stat().st_size
 datasets={};preflight={};input_hashes={}
 for task,conditions in [('oc',['ordered']),('ckd',['ordered','shuffled'])]:
  datasets[task]={};preflight[task]={}
  folder=Path(upstream['data_dir'] if task=='oc' else upstream['clinical_data_dir'])
  previous={}
  for condition in conditions:
   datasets[task][condition]={};preflight[task][condition]={}
   for split,expected_rows in [('train',400000 if task=='oc' else 8887),('val',50000 if task=='oc' else 2963),('test',50000 if task=='oc' else 2963)]:
    tag='ocdet' if task=='oc' else condition
    xp,yp=folder/f'X_{split}_{tag}.csv',folder/f'y_{split}_{tag}.csv'
    for p in (xp,yp):
     actual=digest(p)
     expected=upstream['input_sha256'][str(p)] if task=='oc' else old_ckd['data_sha256'][p.name]
     if actual!=expected:raise ValueError('input differs from existing Qwen protocol')
     input_hashes[str(p)]=actual
    x=pd.read_csv(xp,dtype=str,keep_default_na=False)['Sequences'].tolist()
    y=pd.read_csv(yp)['Outcome'].to_numpy(dtype=int)
    if len(x)!=expected_rows or len(y)!=len(x) or not np.isin(y,[0,1]).all():raise ValueError('invalid split')
    if task=='oc' and any(label(s)!=target for s,target in zip(x,y)):raise ValueError('OC oracle mismatch')
    if task=='ckd':
     multisets=[sorted(s.split('\x1f')) for s in x]
     if condition=='ordered':previous[split]=(y,multisets)
     elif not np.array_equal(y,previous[split][0]) or multisets!=previous[split][1]:raise ValueError('clinical pairs differ')
    datasets[task][condition][split]=dict(features=str(xp),labels=str(yp),features_sha256=input_hashes[str(xp)],
      labels_sha256=input_hashes[str(yp)],rows=len(y),positive_rows=int(y.sum()))
    preflight[task][condition][split]={}
    for name,record in models.items():
     tokenizer=AutoTokenizer.from_pretrained(record['path'],local_files_only=True,trust_remote_code=False)
     tokenizer.truncation_side='right'
     data=EncoderDataset(x,y,tokenizer,task)
     preflight[task][condition][split][name]=data.audit
     del data,tokenizer
    print(json.dumps(dict(event='preflight',task=task,condition=condition,split=split,tokenization=preflight[task][condition][split])),flush=True)
 jobs=[]
 def add_pair(model,task,condition,smoke=False):
  key=('smoke_' if smoke else '')+f'{model}_{task}_{condition}'
  folder=(SMOKE_ROOT if smoke else root/'fits')/key
  jobs.append(dict(key=key+'_fit',kind='fit',model=model,task=task,condition=condition,smoke=smoke,
                   output_dir=str(folder/'attempt_001'),status='pending'))
  jobs.append(dict(key=key+'_attention',kind='attention',model=model,task=task,condition=condition,smoke=smoke,
                   fit_job=key+'_fit',output_dir=str((SMOKE_ROOT/'attention' if smoke else root/'attention')/key),status='pending'))
 for model in models:
  for task in ('oc','ckd'):add_pair(model,task,'ordered',True)
 for model in models:add_pair(model,'oc','ordered')
 for model in models:
  for condition in ('ordered','shuffled'):add_pair(model,'ckd',condition)
 jobs.append(dict(key='ckd_paired_comparison',kind='summary',smoke=False,output_dir=str(root/'ckd_paired_comparison'),status='pending'))
 plan=dict(schema_version=1,task='encoder_oc_ckd_attention',models=models,datasets=datasets,
  recipes=dict(oc=dict(seed=9950,train_rows=200000,epochs=20,min_epochs=1,patience=3,lr=2e-5,batch_size=128,effective_batch_size=256,eval_batch_size=256),
               ckd=dict(seed=42,epochs=30,min_epochs=5,patience=5,lr=2e-5,batch_size=32,effective_batch_size=32,eval_batch_size=32)),
  tuning='full',samples=1000,sample_seed=9950,perturbations=200,jobs=jobs,
  upstream_status=str(UPSTREAM/'queue_status.json'),upstream_plan_sha256=digest(UPSTREAM/'launch_plan.json'),
  input_sha256=input_hashes,source_sha256={p:digest(ROOT/p) for p in SOURCES},
  checkpoint_policy='retain six selected full models; no persistent optimizer checkpoints; unfinished fits restart',
  estimated_selected_weight_bytes=model_bytes*3,ckd_context_policy='native 512-token right truncation; quantify visible history; not full-context Qwen replication',
  created_at_unix=time.time())
 plan['plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
 write(root/'preflight.json',preflight);write(root/'launch_plan.json',plan)
 write(root/'queue_status.json',dict(status='prepared',jobs=jobs,plan_sha256=plan['plan_sha256'],supervisor_pid=None,child_pid=None,current_job=None,updated_at_unix=time.time()))
 print(json.dumps(dict(status='prepared',jobs=len(jobs),plan_sha256=plan['plan_sha256'],run_root=str(root))),flush=True)


def verify(plan,models=False):
 for relative,h in plan['source_sha256'].items():
  if digest(ROOT/relative)!=h:raise ValueError('frozen source changed: '+relative)
 for filename,h in plan['input_sha256'].items():
  if digest(filename)!=h:raise ValueError('frozen input changed: '+filename)
 if digest(UPSTREAM/'launch_plan.json')!=plan['upstream_plan_sha256']:raise ValueError('upstream plan changed')
 if models:
  for record in plan['models'].values():
   for name,h in record['files_sha256'].items():
    if digest(Path(record['path'])/name)!=h:raise ValueError('base weights changed')


def upstream_ready(state):
 return state.get('status')=='completed' and not state.get('child_pid') and all(j.get('status')=='completed' for j in state.get('jobs',[])) and bool(state.get('jobs'))


def command_for(root,job,batch=None,attempt=1):
 if job['kind']=='fit':
  output=Path(job['output_dir']).parent/f'attempt_{attempt:03d}'
  cmd=[sys.executable,'-B','-u','-m','src.experiments.encoder_oc_ckd','--run-root',str(root),'--job',job['key'],'--output-dir',str(output)]
  if batch:cmd+=['--batch-size',str(batch)]
 elif job['kind']=='attention':
  output=Path(job['output_dir']);cmd=[sys.executable,'-B','-u','-m','src.analysis.encoder_oc_ckd_attention','--run-root',str(root),'--job',job['key']]
 else:
  output=Path(job['output_dir']);cmd=[sys.executable,'-B','-u','-m','src.analysis.encoder_oc_ckd_attention','--run-root',str(root),'--paired-summary']
 return cmd,output


def supervise(root):
 with (root/'queue.lock').open('a+') as local:
  fcntl.flock(local.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=read(root/'launch_plan.json');verify(plan)
  state=read(root/'queue_status.json')
  if state.get('status')=='completed':raise RuntimeError('queue completed already')
  state.update(supervisor_pid=os.getpid(),child_pid=None,current_job=None,error=None)
  def progress(status=None,**detail):
   if status:state['status']=status
   state.update(detail,updated_at_unix=time.time());write(root/'queue_status.json',state)
  try:
   progress('waiting_for_qwen')
   while not upstream_ready(read(Path(plan['upstream_status']))):
    progress('waiting_for_qwen',upstream_status=read(Path(plan['upstream_status'])).get('status'));time.sleep(15)
   with GPU_LOCK.open('a+') as gpu:
    while True:
     try:fcntl.flock(gpu.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
     except BlockingIOError:progress('waiting_for_gpu');time.sleep(15)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('unexpected GPU process')
    cancellation=read(ROOT/'results/kip_m4_llama70b_qlora_20260921_seed9950/user_cancellation_20260923.json')
    if cancellation['status'] not in ('cancellation_armed','cancelled'):raise RuntimeError('Llama70 cancellation not confirmed')
    verify(plan,models=True)
    remaining_bytes=sum((Path(plan['models'][j['model']]['path'])/'model.safetensors').stat().st_size
      for j in state['jobs'] if j['kind']=='fit' and not j['smoke'] and j.get('status')!='completed')
    if shutil.disk_usage(root).free<remaining_bytes+768*1024**2:raise OSError('insufficient space to retain selected encoder checkpoints')
    for job in state['jobs']:
     if job.get('status')=='completed':
      if read(Path(job['output_dir'])/'metrics.json').get('status')!='completed':raise ValueError('completed job lacks metrics')
      continue
     verify(plan)
     original=next(j for j in plan['jobs'] if j['key']==job['key'])
     batch=plan['recipes'][job['task']]['batch_size'] if job['kind']=='fit' else None
     attempt=len(job.get('attempts',[]))+1
     while True:
      cmd,out=command_for(root,original,batch,attempt)
      if job['kind']=='fit' and out.exists() and any(out.iterdir()):raise FileExistsError('fresh attempt output already exists')
      if shutil.disk_usage(root).free<640*1024**2:raise OSError('low output headroom')
      log=root/'logs'/f"{job['key']}_attempt_{attempt:03d}.log";log.parent.mkdir(exist_ok=True)
      job.update(status='running',output_dir=str(out))
      job.setdefault('attempts',[]).append(dict(command=cmd,output_dir=str(out),log=str(log),batch_size=batch,started_at_unix=time.time()))
      with log.open('a') as stream:
       child=subprocess.Popen(cmd,cwd=ROOT,env=environment(root),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(local.fileno(),gpu.fileno()))
       progress('running',current_job=job['key'],child_pid=child.pid)
       while child.poll() is None:time.sleep(10);progress()
      job['attempts'][-1].update(exit_code=child.returncode,finished_at_unix=time.time());progress(child_pid=None)
      if child.returncode==0 and (out/'metrics.json').exists() and read(out/'metrics.json').get('status')=='completed':
       job.update(status='completed',metrics_sha256=digest(out/'metrics.json'))
       if job.get('smoke'):
        dest=root/'smoke_records'/job['key'];dest.mkdir(parents=True,exist_ok=True)
        for p in out.iterdir():
         if p.is_file() and p.stat().st_size<20*1024**2:shutil.copy2(p,dest/p.name)
       progress();break
      if job['kind']=='fit' and batch>1 and 'out of memory' in log.read_text(errors='replace')[-16000:].lower():
       batch//=2;attempt+=1;progress('retrying_oom');continue
      job['status']='failed';raise RuntimeError('job failed: '+job['key']+'; see '+str(log))
   progress('completed',current_job=None,child_pid=None,finished_at_unix=time.time())
  except BaseException as error:
   progress('failed',error=f'{type(error).__name__}: {error}');raise


def background(root):
 verify(read(root/'launch_plan.json'))
 with (root/'queue.lock').open('a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 with (root/'supervisor.log').open('a') as stream:
  child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--run-root',str(root)],cwd=ROOT,
   env=environment(root),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 (root/'supervisor.pid').write_text(str(child.pid)+'\n')
 for _ in range(150):
  time.sleep(.1)
  if child.poll() is not None:raise RuntimeError('supervisor exited; inspect log')
  state=read(root/'queue_status.json')
  if state.get('supervisor_pid')==child.pid and state.get('status') in ('waiting_for_qwen','waiting_for_gpu','running'):
   print(json.dumps(dict(status=state['status'],supervisor_pid=child.pid,run_root=str(root))),flush=True);return
 raise RuntimeError('startup not confirmed; inspect before retrying')


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,default=RUN)
 group=parser.add_mutually_exclusive_group();group.add_argument('--prepare',action='store_true');group.add_argument('--background',action='store_true')
 args=parser.parse_args();args.run_root=args.run_root.resolve()
 prepare(args.run_root) if args.prepare else background(args.run_root) if args.background else supervise(args.run_root)
