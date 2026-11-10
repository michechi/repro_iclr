"""Matched-batch OC and CLS-readout CKD attention for full BERT/RoBERTa fits."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.analysis.kip_attention_core import event_matrix, fixed_sample, letter_token_groups
from src.analysis.kip_checkpoint_attention import capture_native_attention, digest, infer, read_json, write_json
from src.analysis.oc_attention_core import METRICS, label, oc_statistics, perturbations
from src.analysis.qwen_oc_audit import reference_batch_groups, replace_batch_row
from src.analysis.ckd_attention_audit import summarize_readout
from src.experiments.encoder_oc_ckd import EncoderDataset, classification_metrics, prompt, read_data
from src.experiments.kip_encoder_training import DynamicPadding


def encode_visible(tokenizer, sequence, task):
    text = prompt(sequence,task)
    args = dict(return_offsets_mapping=True,return_token_type_ids=False)
    if task == "ckd":
        args.update(truncation=True,max_length=512)
    else:
        args.update(truncation=False)
    encoded = tokenizer(text,**args)
    events = sequence.split("\x1f")
    if task == "oc":
        groups = letter_token_groups(encoded["offset_mapping"],events)
        return encoded,events,groups,0
    groups,visible,partial = [],[],0
    start = len("Sequential events: ")
    for event in events:
        end = start+len(event)
        tokens = [i for i,(a,b) in enumerate(encoded["offset_mapping"]) if b>a and b>start and a<end]
        if tokens:
            a = min(encoded["offset_mapping"][i][0] for i in tokens)
            b = max(encoded["offset_mapping"][i][1] for i in tokens)
            if a<=start and b>=end:
                groups.append(tokens);visible.append(event)
            else:
                partial += 1
        start=end+1
    flattened=[t for group in groups for t in group]
    if not groups or len(set(flattened))!=len(flattened):
        raise ValueError("missing or ambiguously merged complete clinical events")
    return encoded,visible,groups,partial


@contextmanager
def capture_cls_attention(captured, row_lengths):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.bert import modeling_bert
    from transformers.models.roberta import modeling_roberta
    from unittest.mock import patch
    native=ALL_ATTENTION_FUNCTIONS["sdpa"]

    def record(module,query,key,attention_mask,dropout=0.,scaling=None,**kwargs):
        if module.training or dropout or getattr(module,"is_causal",False):
            raise ValueError("requires bidirectional encoder evaluation")
        layer=getattr(module,"layer_idx",None)
        if layer is None or layer in captured:
            raise ValueError("invalid physical attention layer")
        rows={}
        for row,length in row_lengths.items():
            if not 0<=row<query.shape[0] or not 1<=length<=query.shape[-2]:
                raise ValueError("invalid selected row/length")
            q=query[row:row+1,:,:1]
            k=key[row:row+1]
            if q.shape[1]!=k.shape[1]:
                raise ValueError("unexpected encoder head layout")
            scores=torch.matmul(q,k.transpose(-1,-2))*(scaling if scaling is not None else query.shape[-1]**-.5)
            if attention_mask is not None:
                if attention_mask.ndim!=4 or attention_mask.shape[0] not in (1,query.shape[0]):
                    raise ValueError("unsupported padding mask")
                mask=attention_mask
                offset=0 if mask.shape[0]==1 else row
                mask=mask[offset:offset+1,:,:1,:]
                scores=scores.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else scores+mask
            scores[...,length:]=float('-inf')
            rows[row]=torch.softmax(scores,dim=-1,dtype=torch.float32).to(query.dtype)[0,:,0,:length].float().cpu().numpy()
        captured[layer]=rows

    def capture(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        record(module,query,key,attention_mask,dropout,scaling,**kwargs)
        return native(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)

    def wrap_encoder_forward(forward):
        def wrapped(module,*args,**kwargs):
            native_sdpa=torch.nn.functional.scaled_dot_product_attention

            def encoder_sdpa(query,key,value,attn_mask=None,dropout_p=0.,
                             is_causal=False,*,scale=None,enable_gqa=False):
                if is_causal:
                    raise ValueError("CLS capture requires bidirectional attention")
                record(module,query,key,attn_mask,dropout_p,scale)
                options={}
                if scale is not None:options["scale"]=scale
                if enable_gqa:options["enable_gqa"]=True
                return native_sdpa(query,key,value,attn_mask,dropout_p,is_causal,**options)

            with patch.object(torch.nn.functional,"scaled_dot_product_attention",encoder_sdpa):
                return forward(module,*args,**kwargs)
        return wrapped

    with ExitStack() as stack:
        for namespace,name in ((modeling_bert,"BertSdpaSelfAttention"),
                               (modeling_roberta,"RobertaSdpaSelfAttention")):
            cls=getattr(namespace,name,None)
            if cls is not None:
                stack.enter_context(patch.object(cls,"forward",wrap_encoder_forward(cls.forward)))
        ALL_ATTENTION_FUNCTIONS.register("sdpa",capture)
        try:
            yield
        finally:
            ALL_ATTENTION_FUNCTIONS.register("sdpa",native)


def audit(args):
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    plan=read_json(args.run_root/"launch_plan.json")
    job=next(j for j in plan["jobs"] if j["key"]==args.job)
    fit=next(j for j in plan["jobs"] if j["key"]==job["fit_job"])
    if (args.run_root/"queue_status.json").exists():
        active=next(j for j in read_json(args.run_root/"queue_status.json")["jobs"] if j["key"]==job["fit_job"])
        if active.get("status")!="completed":raise ValueError("fit has not completed")
        fit={**fit,"output_dir":active.get("output_dir",fit["output_dir"])}
    run=Path(fit["output_dir"]);output=Path(job["output_dir"])
    if (output/"metrics.json").exists():raise FileExistsError("completed audit exists")
    output.mkdir(parents=True,exist_ok=True)
    manifest=read_json(run/"manifest.json");fit_metrics=read_json(run/"metrics.json")
    if fit_metrics.get("status")!="completed" or fit_metrics["is_smoke"]!=job["smoke"]:
        raise ValueError("source fit not complete or wrong smoke status")
    checkpoint=run/"best_model/model.safetensors";checkpoint_hash=digest(checkpoint)
    if checkpoint_hash!=fit_metrics["selected_checkpoint_sha256"]:raise ValueError("selected weights changed")
    for split in ("train","val","test"):
        if digest(run/f"{split}_indices.npy")!=manifest["index_sha256"][split]:raise ValueError("split indices changed")
    started=time.monotonic()

    def progress(status,**detail):
        write_json(output/"status.json",dict(status=status,job=args.job,pid=os.getpid(),updated_at_unix=time.time(),
                                           wall_seconds=time.monotonic()-started,**detail))

    progress("loading_checkpoint")
    torch.set_num_threads(4);torch.set_float32_matmul_precision("high")
    device=torch.device(args.device)
    tokenizer=AutoTokenizer.from_pretrained(run/"best_model",local_files_only=True,trust_remote_code=False)
    tokenizer.truncation_side="right"
    model,loading=AutoModelForSequenceClassification.from_pretrained(run/"best_model",local_files_only=True,
        trust_remote_code=False,dtype=torch.float32,attn_implementation="sdpa",output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
        raise ValueError("selected classifier was not restored exactly")
    model.to(device).eval()
    sequences,labels=read_data(plan,job["task"],job["condition"],"test")
    order=np.load(run/"test_indices.npy",allow_pickle=False)
    count=16 if job["smoke"] else plan["samples"]
    indices=order[fixed_sample(labels[order],count,plan["sample_seed"])]
    reference=pd.read_csv(run/"test_predictions.csv").set_index("row_index").loc[indices]
    if not np.array_equal(reference["label"].to_numpy(),labels[indices]):raise ValueError("reference labels differ")
    dataset=EncoderDataset(sequences,labels,tokenizer,job["task"])
    batches=reference_batch_groups(indices,order,manifest["recipe"]["eval_batch_size"])
    maximum=dataset.max_length
    collate=DynamicPadding(tokenizer.pad_token_id,maximum)
    layers=list(range(model.config.num_hidden_layers))
    probabilities=np.full(len(indices),np.nan);examples=[None]*3
    masses,baselines=[None]*len(indices),[None]*len(indices)
    records,code_stats,patient_mass,coverage=[],{}, {},[]
    counterfactual=[];done=0;native_delta=None
    cf_positions=set(np.random.default_rng(plan["sample_seed"]+1).choice(count,min(count,plan["perturbations"]),replace=False))
    for batch_rows,selected in batches:
        rows=[dataset[int(i)] for i in batch_rows]
        inputs={k:v.to(device) for k,v in collate(rows).items() if k!="labels"}
        lengths={offset:len(rows[offset][0]) for _,offset in selected}
        if native_delta is None:before,_=infer(model,inputs,False)
        captured={}
        manager=capture_native_attention if job["task"]=="oc" else capture_cls_attention
        with manager(captured,lengths):p,_=infer(model,inputs,False)
        if sorted(captured)!=layers:raise ValueError("missing captured attention layers")
        if native_delta is None:
            native_delta=float(np.max(np.abs(p-before)))
            if native_delta>1e-6:raise ValueError("attention capture changed native predictions")
        for n,offset in selected:
            row=int(indices[n]);probabilities[n]=float(p[offset])
            error=abs(probabilities[n]-float(reference.iloc[n]["probability"]))
            if error>.02:
                write_json(output/"reproduction_failure.json",dict(row_index=row,error=error,tolerance=.02))
                raise ValueError("original batched test predictions do not reproduce")
            encoded,events,groups,partial=encode_visible(tokenizer,sequences[row],job["task"])
            if encoded["input_ids"]!=dataset.input_ids[row]:raise ValueError("attention tokenization differs from training")
            if job["task"]=="oc":
                if label(sequences[row])!=labels[row]:raise ValueError("OC oracle mismatch")
                ml,bl,em=[],[],[]
                for layer in layers:
                    a=captured[layer][offset]
                    mass,baseline=oc_statistics(a,groups,events,causal=False,readout=0)
                    ml.append(mass);bl.append(baseline)
                    if n<3:em.append(event_matrix(a,groups))
                masses[n],baselines[n]=np.stack(ml),np.stack(bl)
                if n<3:examples[n]=(events,np.stack(em))
                if n in cf_positions:
                    for variant,(changed,target) in perturbations(sequences[row],plan["sample_seed"]+row).items():
                        enc,_,_,_=encode_visible(tokenizer,changed,"oc")
                        changed_inputs=replace_batch_row(inputs,offset,torch.tensor(enc["input_ids"]))
                        p2,_=infer(model,changed_inputs,False)
                        counterfactual.append(dict(row_index=row,variant=variant,original_label=int(labels[row]),
                            expected_label=int(target),original_probability=probabilities[n],probability=float(p2[offset]),
                            original_sequence=sequences[row],changed_sequence=changed))
            else:
                full_count=len(sequences[row].split("\x1f"));distribution=defaultdict(float);example=[]
                coverage.append(dict(row_index=row,full_events=full_count,visible_complete_events=len(events),
                    visible_fraction=len(events)/full_count,partial_events=partial,
                    full_tokens=dataset.full_lengths[row],input_tokens=len(encoded["input_ids"])))
                for layer in layers:
                    weights=captured[layer][offset]
                    mass,baseline,entropy=summarize_readout(weights,groups)
                    example.append(mass.mean(0))
                    quartiles=np.minimum(np.arange(len(events))*4//len(events),3)
                    for h in range(len(mass)):
                        records.append(dict(row_index=row,label=int(labels[row]),layer=layer,head=h,
                            visible_events=len(events),full_events=full_count,event_mass=float(mass[h].sum()),
                            other_token_mass=float(1-mass[h].sum()),normalized_token_entropy=float(entropy[h]),
                            **{f"visible_quartile_{q+1}_mass":float(mass[h,quartiles==q].sum()) for q in range(4)}))
                    for code in sorted(set(events)):
                        pos=[i for i,c in enumerate(events) if c==code]
                        value=mass[:,pos].sum(-1);expected=float(baseline[pos].sum());key=(layer,code)
                        if key not in code_stats:code_stats[key]=[0,np.zeros(len(value)),np.zeros(len(value))]
                        code_stats[key][0]+=1;code_stats[key][1]+=value;code_stats[key][2]+=value/expected
                        distribution[code]+=float(value.mean())/len(layers)
                patient_mass[str(row)]=dict(distribution)
                if n<3:examples[n]=(events,np.stack(example))
            done+=1
        progress("extracting",samples_completed=done,samples=count)
    errors=np.abs(probabilities-reference["probability"].to_numpy())
    metrics=dict(status="completed",task=job["task"],model=job["model"],condition=job["condition"],
        is_smoke=job["smoke"],samples=count,plan_sha256=plan["plan_sha256"],
        max_reference_probability_error=float(errors.max()),mean_reference_probability_error=float(errors.mean()),
        capture_probability_delta=native_delta,full_test_reference=fit_metrics["test"],
        sample_metrics=classification_metrics(labels[indices],probabilities,.5),
        source_checkpoint_sha256=checkpoint_hash,attention_readout="CLS token 0",causal=False)
    pd.DataFrame(dict(row_index=indices,label=labels[indices],probability=probabilities,
        reference_probability=reference["probability"].to_numpy())).to_csv(output/"predictions.csv",index=False)
    title=f"{job['model']} full | {job['task']} {job['condition']}"
    progress("saving_analysis",samples_completed=done,samples=count)
    if job["task"]=="oc":
        mass,base=np.stack(masses).astype(np.float32),np.stack(baselines).astype(np.float32)
        np.savez_compressed(output/"per_example_attention.npz",masses=mass,uniform_baseline=base,
            row_indices=indices,labels=labels[indices],probabilities=probabilities,
            physical_layer_ids=np.array(layers),metric_names=np.array(METRICS))
        correct=(probabilities>=.5)==labels[indices]
        conditions=dict(all=np.ones(count,bool),positive=labels[indices]==1,negative=labels[indices]==0,
                        correct=correct,incorrect=~correct)
        summaries=[]
        for condition,selection in conditions.items():
            for l,layer in enumerate(layers):
                for m,name in enumerate(METRICS):
                    for h in range(mass.shape[-1]):
                        values,b=mass[selection,l,m,h],base[selection,l,m]
                        valid=np.isfinite(values)&np.isfinite(b)&(b>0)
                        if valid.any():summaries.append(dict(condition=condition,layer=layer,head=h,metric=name,
                            examples=int(valid.sum()),mean_mass=float(values[valid].mean()),
                            mean_uniform_baseline=float(b[valid].mean()),mean_enrichment=float((values[valid]/b[valid]).mean())))
        pd.DataFrame(summaries).to_csv(output/"attention_summary.csv",index=False)
        pd.DataFrame(counterfactual).to_csv(output/"counterfactuals.csv",index=False)
        metrics["counterfactuals"]={}
        for variant in sorted({r["variant"] for r in counterfactual}):
            rr=[r for r in counterfactual if r["variant"]==variant]
            pp=np.array([r["probability"] for r in rr]);old=np.array([r["original_probability"] for r in rr])
            yy=np.array([r["expected_label"] for r in rr])
            metrics["counterfactuals"][variant]=dict(samples=len(rr),accuracy_at_0_5=float(np.mean((pp>=.5)==yy)),
                prediction_flip_rate=float(np.mean((pp>=.5)!=(old>=.5))),mean_absolute_probability_change=float(np.abs(pp-old).mean()))
        oc_plots(output,pd.DataFrame(summaries),examples,layers,title)
    else:
        frame=pd.DataFrame(records);frame.to_csv(output/"per_patient_head_summary.csv.gz",index=False)
        pd.DataFrame(coverage).to_csv(output/"visible_history_coverage.csv",index=False)
        code_rows=[]
        for (layer,code),(support,total,enrichment) in code_stats.items():
            for h in range(len(total)):code_rows.append(dict(layer=layer,head=h,code=code,patients_with_code=support,
                mean_mass_present=float(total[h]/support),mean_mass_all_patients=float(total[h]/count),
                mean_enrichment_present=float(enrichment[h]/support)))
        codes=pd.DataFrame(code_rows);codes.to_csv(output/"code_attention_summary.csv",index=False)
        write_json(output/"patient_visible_code_mass.json",patient_mass)
        metrics.update(context_policy="first 512 tokens; complete visible events only; partial event tokens grouped with other tokens",
            truncated_patients=sum(r["full_tokens"]>512 for r in coverage),
            mean_visible_event_fraction=float(np.mean([r["visible_fraction"] for r in coverage])),
            interpretation="descriptive CLS attention; truncation changes visible code sets across ordered/shuffled inputs")
        ckd_plots(output,frame,codes,examples,layers,title)
    write_json(output/"manifest.json",dict(plan_sha256=plan["plan_sha256"],fit_manifest_sha256=digest(run/"manifest.json"),
        source_checkpoint_sha256=checkpoint_hash,sample_indices=indices.tolist(),eval_batch_size=manifest["recipe"]["eval_batch_size"],
        inference="original evaluation batches and padding; native SDPA; diagnostic attention leaves predictions unchanged",
        readout_query=0,causal=False,source_sha256=plan["source_sha256"]))
    if digest(checkpoint)!=checkpoint_hash:raise ValueError("selected weights changed during analysis")
    metrics.update(wall_seconds=time.monotonic()-started,artifact_sha256={p.name:digest(p) for p in output.iterdir()
        if p.is_file() and p.name not in {"metrics.json","status.json"}})
    write_json(output/"metrics.json",metrics);progress("completed",samples_completed=count)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig,path):
    for suffix in ("png","pdf","svg"):fig.savefig(path.with_suffix('.'+suffix),dpi=150,bbox_inches="tight")


def oc_plots(output,table,examples,layers,title):
    plt=pyplot();fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ("key_to_lag_key","key_to_other_key","key_to_distractor"):
        groups=table[(table.condition=="all")&(table.metric==name)].groupby("layer")
        for ax,column in zip(axes,("mean_mass","mean_enrichment")):
            values=groups[column].mean();ax.plot(values.index,values.values,marker='.',label=name)
    axes[0].set_ylabel("Mean attention mass");axes[1].set_ylabel("Enrichment over uniform visible tokens")
    axes[1].axhline(1,color='gray',linestyle='--')
    for ax in axes:ax.set_xlabel("Physical layer (zero based)");ax.legend(fontsize=7)
    fig.suptitle(title+" | bidirectional attention");fig.tight_layout();save_figure(fig,output/"layer_profiles");plt.close(fig)
    shown=sorted(set([0,len(layers)//2,len(layers)-1]))
    for n,(events,mass) in enumerate(examples):
        fig,axes=plt.subplots(1,len(shown),figsize=(5*len(shown),4),squeeze=False)
        vmax=max(float(mass[i].max()) for i in shown)
        for ax,l in zip(axes[0],shown):
            im=ax.imshow(mass[l],vmin=0,vmax=vmax,cmap='viridis')
            ax.set_xticks(range(20),events,fontsize=7);ax.set_yticks(range(20),events,fontsize=7)
            ax.set_title(f"Layer {layers[l]}");ax.set_xlabel("Target event");ax.set_ylabel("Query event")
        fig.colorbar(im,ax=list(axes[0]),shrink=.7);fig.suptitle(f"{title}, fixed sample {n}")
        fig.savefig(output/f"example_{n}.png",dpi=140,bbox_inches='tight');plt.close(fig)


def ckd_plots(output,frame,codes,examples,layers,title):
    plt=pyplot();fig,ax=plt.subplots(figsize=(8,4))
    for q in range(1,5):
        values=frame.groupby('layer')[f'visible_quartile_{q}_mass'].mean()
        ax.plot(values.index,values.values,label=f'Visible-history quartile {q}')
    ax.set_xlabel('Physical layer (zero based)');ax.set_ylabel('Mean CLS attention mass');ax.legend()
    ax.set_title(title+' | visible 512-token window');fig.tight_layout();save_figure(fig,output/'input_position_attention');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,7))
    values=codes.groupby('code').mean_mass_all_patients.mean().nlargest(20).sort_values();values.plot.barh(ax=ax)
    ax.set_xlabel('Mean CLS attention mass');ax.set_title(title+' | complete visible CCS codes')
    fig.tight_layout();save_figure(fig,output/'ccs_attention');plt.close(fig)
    for n,(events,mass) in enumerate(examples):
        bins=np.array_split(np.arange(len(events)),min(80,len(events)))
        values=np.stack([mass[:,b].sum(-1) for b in bins],-1)
        fig,ax=plt.subplots(figsize=(12,4));im=ax.imshow(values,aspect='auto',cmap='viridis')
        ax.set_yticks(range(len(layers)),layers);ax.set_ylabel('Physical layer');ax.set_xlabel('Consecutive visible-event bins')
        ax.set_title(f'{title}, fixed sample {n}; CLS attention, visible history only')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(output/f'sample_{n}.png',dpi=140);plt.close(fig)


def paired_summary(run_root):
    plan=read_json(run_root/'launch_plan.json');output=run_root/'ckd_paired_comparison';output.mkdir(exist_ok=True)
    summary={}
    for model in ('bert','roberta'):
        dirs={c:Path(next(j['output_dir'] for j in plan['jobs'] if j['key']==f'{model}_ckd_{c}_attention')) for c in ('ordered','shuffled')}
        distributions={c:read_json(p/'patient_visible_code_mass.json') for c,p in dirs.items()}
        if set(distributions['ordered'])!=set(distributions['shuffled']):raise ValueError('different sampled patients')
        rows=[]
        for row in sorted(distributions['ordered'],key=int):
            left,right=distributions['ordered'][row],distributions['shuffled'][row]
            codes=sorted(set(left)|set(right));a=np.array([left.get(c,0) for c in codes]);b=np.array([right.get(c,0) for c in codes])
            a/=a.sum();b/=b.sum()
            rows.append(dict(row_index=int(row),visible_code_distribution_overlap=float(1-.5*np.abs(a-b).sum()),
                visible_code_set_jaccard=len(set(left)&set(right))/len(set(left)|set(right))))
        pd.DataFrame(rows).to_csv(output/f'{model}_paired_overlap.csv',index=False)
        summary[model]=dict(patients=len(rows),mean_visible_code_distribution_overlap=float(np.mean([r['visible_code_distribution_overlap'] for r in rows])),
                            interpretation='condition-trained weights and truncated visible code sets can both differ')
    write_json(output/'metrics.json',dict(status='completed',models=summary,plan_sha256=plan['plan_sha256']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--job')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda');parser.add_argument('--paired-summary',action='store_true')
    args=parser.parse_args()
    paired_summary(args.run_root) if args.paired_summary else audit(args)
