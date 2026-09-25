import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.analysis.encoder_oc_ckd_attention import capture_cls_attention, encode_visible, audit
from src.analysis.kip_checkpoint_attention import capture_native_attention, digest, infer, write_json
from src.analysis.oc_attention_core import label, oc_statistics
from src.experiments.encoder_oc_ckd import EncoderDataset, train, split_indices
from scripts.run_encoder_oc_ckd_attention import command_for, upstream_ready

ROOT=Path(__file__).resolve().parents[1]


def cached_tokenizer(model):
    from transformers import AutoTokenizer
    # These integration checks require a user-provided local tokenizer. The
    # anonymous package does not bundle historical cache paths or checkpoints.
    source = os.environ.get(f"ICLR_TEST_{model.upper()}_TOKENIZER")
    if not source:
        raise unittest.SkipTest(f"Set ICLR_TEST_{model.upper()}_TOKENIZER to a local tokenizer directory")
    return AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)


class AttentionContracts(unittest.TestCase):
    def test_200k_oc_subset_is_shared_stratified_and_keeps_evaluation_complete(self):
        labels = np.concatenate([np.ones(116685, dtype=int), np.zeros(283315, dtype=int)])
        recipe = dict(seed=9950, train_rows=200000)
        bert = split_indices(labels, "train", recipe)
        roberta = split_indices(labels, "train", recipe)
        np.testing.assert_array_equal(bert, roberta)
        self.assertEqual(len(bert), 200000)
        self.assertEqual(len(np.unique(bert)), 200000)
        self.assertGreaterEqual(bert.min(), 0)
        self.assertLess(bert.max(), len(labels))
        self.assertLessEqual(abs(labels[bert].sum() - labels.sum() / 2), .5)
        for split in ("val", "test"):
            np.testing.assert_array_equal(split_indices(labels[:50000], split, recipe), np.arange(50000))
        np.testing.assert_array_equal(split_indices(labels[:8887], "train", dict(seed=42)), np.arange(8887))
        self.assertEqual(len(split_indices(labels, "train", recipe, smoke=True)), 64)
        with self.assertRaises(ValueError):
            split_indices(labels[:1000], "train", recipe)

    def test_cls_capture_matches_native_attention_with_padding(self):
        from transformers import BertConfig,BertForSequenceClassification,RobertaConfig,RobertaForSequenceClassification
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        for cfg,cls in [(BertConfig,BertForSequenceClassification),(RobertaConfig,RobertaForSequenceClassification)]:
            config=cfg(vocab_size=40,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32,
                       pad_token_id=0,attn_implementation='sdpa')
            model=cls(config).eval()
            inputs=dict(input_ids=torch.tensor([[2,4,5,3,0,0,0,0],[2,6,7,8,3,0,0,0]]),
                        attention_mask=torch.tensor([[1,1,1,1,0,0,0,0],[1,1,1,1,1,0,0,0]]))
            before,_=infer(model,inputs,False);full={};captured={};native=ALL_ATTENTION_FUNCTIONS['sdpa']
            with capture_native_attention(full,{0:4,1:5}):infer(model,inputs,False)
            with capture_cls_attention(captured,{0:4,1:5}):after,_=infer(model,inputs,False)
            np.testing.assert_array_equal(before,after)
            self.assertIs(native,ALL_ATTENTION_FUNCTIONS['sdpa'])
            for layer in (0,1):
                for row in (0,1):
                    np.testing.assert_allclose(captured[layer][row],full[layer][row][:,0,:],atol=.003)
                    np.testing.assert_allclose(captured[layer][row].sum(-1),1,atol=.01)
            with self.assertRaises(RuntimeError):
                with capture_cls_attention({},{}):raise RuntimeError('restore interface')
            self.assertIs(native,ALL_ATTENTION_FUNCTIONS['sdpa'])

    def test_truncation_and_event_mapping_use_training_token_ids(self):
        for model in ('bert','roberta'):
            tokenizer=cached_tokenizer(model)
            sequence='\x1f'.join(['CCS_123','CCS_17']*300)
            data=EncoderDataset([sequence],np.array([1]),tokenizer,'ckd')
            enc,visible,groups,partial=encode_visible(tokenizer,sequence,'ckd')
            self.assertEqual(data.input_ids[0],enc['input_ids'])
            self.assertEqual(len(enc['input_ids']),512)
            self.assertEqual(data.audit['truncated_rows'],1)
            self.assertLess(len(visible),600)
            self.assertTrue(all(groups));self.assertLessEqual(partial,1)
            self.assertEqual(len(set(t for g in groups for t in g)),sum(map(len,groups)))

    def test_encoder_uniform_baseline_is_bidirectional(self):
        letters=['B']*20;letters[0]='W';letters[7]='D';letters[14]='Q'
        attention=np.ones((2,22,22))/22
        mass,base=oc_statistics(attention,[[i+1] for i in range(20)],letters,causal=False,readout=0)
        valid=np.isfinite(base)&(base>0)
        np.testing.assert_allclose(mass[valid]/base[valid,None],1)

    def test_waits_for_successful_upstream_and_keeps_fit_attempts_separate(self):
        self.assertTrue(upstream_ready(dict(status='completed',child_pid=None,jobs=[dict(status='completed')])))
        for change in [dict(status='running'),dict(child_pid=1),dict(jobs=[]),dict(jobs=[dict(status='failed')])]:
            self.assertFalse(upstream_ready({**dict(status='completed',child_pid=None,jobs=[dict(status='completed')]),**change}))
        job=dict(kind='fit',key='bert_ckd_ordered_fit',output_dir='/tmp/run/fits/bert_ckd_ordered/attempt_001')
        cmd,out=command_for(Path('/tmp/run'),job,batch=8,attempt=2)
        self.assertEqual(out.name,'attempt_002');self.assertIn('8',cmd)


class SmallEndToEnd(unittest.TestCase):
    def test_tiny_real_encoders_fit_restore_and_analyze_both_tasks(self):
        from transformers import BertConfig,BertForSequenceClassification,RobertaConfig,RobertaForSequenceClassification
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix='encoder_oc_ckd_test_') as temporary:
            root=Path(temporary);datasets={}
            for task in ('oc','ckd'):
                datasets[task]={'ordered':{}}
                for split,n in [('train',64),('val',32),('test',32)]:
                    seq=[];y=np.arange(n)%2
                    for target in y:
                        if task=='oc':
                            events=['B']*20;events[0],events[7],events[14]='W','D','Q' if target else 'W'
                            s='\x1f'.join(events);self.assertEqual(label(s),target)
                        else:s='\x1f'.join((['CCS_1','CCS_123'] if target else ['CCS_2','CCS_17'])*100)
                        seq.append(s)
                    xp=root/f'{task}_{split}_x.csv';yp=root/f'{task}_{split}_y.csv'
                    pd.DataFrame({'Sequences':seq}).to_csv(xp,index=False);pd.DataFrame({'Outcome':y}).to_csv(yp,index=False)
                    datasets[task]['ordered'][split]=dict(features=str(xp),labels=str(yp),features_sha256=digest(xp),labels_sha256=digest(yp),rows=n)
            models={};jobs=[]
            for name,config_type,model_type in [('bert',BertConfig,BertForSequenceClassification),('roberta',RobertaConfig,RobertaForSequenceClassification)]:
                tokenizer=cached_tokenizer(name);source=root/name
                config=config_type(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32,
                                   pad_token_id=tokenizer.pad_token_id,max_position_embeddings=514 if name=='roberta' else 512)
                model_type(config).save_pretrained(source);tokenizer.save_pretrained(source)
                models[name]=dict(path=str(source),name=name,files_sha256={p.name:digest(p) for p in source.iterdir() if p.is_file()})
                for task in ('oc','ckd'):
                    key=f'{name}_{task}_ordered'
                    jobs.append(dict(key=key+'_fit',kind='fit',model=name,task=task,condition='ordered',smoke=True,output_dir=str(root/'fits'/key)))
                    jobs.append(dict(key=key+'_attention',kind='attention',model=name,task=task,condition='ordered',smoke=True,fit_job=key+'_fit',output_dir=str(root/'attention'/key)))
            recipe=dict(seed=42,epochs=1,min_epochs=1,patience=2,lr=2e-5,batch_size=4,effective_batch_size=8,eval_batch_size=4)
            plan=dict(plan_sha256='test-only',models=models,datasets=datasets,jobs=jobs,recipes=dict(oc=recipe,ckd=recipe),
                      samples=16,sample_seed=9950,perturbations=4,source_sha256={})
            write_json(root/'launch_plan.json',plan)
            for job in jobs:
                args=argparse.Namespace(run_root=root,job=job['key'],device='cpu',batch_size=None,output_dir=None)
                train(args) if job['kind']=='fit' else audit(args)
                metrics=json.loads((Path(job['output_dir'])/'metrics.json').read_text())
                self.assertEqual(metrics['status'],'completed')
                if job['kind']=='attention':
                    self.assertLessEqual(metrics['max_reference_probability_error'],.02)
                    self.assertEqual(metrics['capture_probability_delta'],0)
                    self.assertEqual(metrics['samples'],16)
                else:
                    self.assertFalse(metrics['test_informed_stopping']);self.assertTrue(metrics['is_smoke'])


if __name__=='__main__':unittest.main()
