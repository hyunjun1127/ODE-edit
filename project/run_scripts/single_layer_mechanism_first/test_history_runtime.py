"""Tiny CPU model/cache fixtures only, not actual Llama validation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import history_runtime as module
from .history_runtime import HistoryFactory, canonical_paths
from .history import receive_all, StreamingHistoryOracle
from .decision import EndpointBinding, FactorArchive, DecisionError
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import WEIGHT,model_guard


class Tokenizer:
    name_or_path="cpu-fixture"
    padding_side="right"
    pad_token_id=2;bos_token_id=1;eos_token_id=2;unk_token_id=0
    add_bos_token=True;add_eos_token=False
    def encode(self,text,add_special_tokens=False):
        values=[3+ord(c)%12 for c in text.strip()]
        return ([1] if add_special_tokens else [])+values
    def __call__(self,text,add_special_tokens=True):
        return {"input_ids":self.encode(text,add_special_tokens=add_special_tokens)}


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm=torch.nn.Identity()
        self.mlp=torch.nn.Module()
        self.mlp.up_proj=torch.nn.Linear(3,4,bias=False)
        self.mlp.down_proj=torch.nn.Linear(4,3,bias=False)
        self.calls=0
    def forward(self,hidden,**kwargs):
        self.calls+=1
        x=self.post_attention_layernorm(hidden)
        return (hidden+self.mlp.down_proj(torch.tanh(self.mlp.up_proj(x))),)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens=torch.nn.Embedding(16,3)
        self.layers=torch.nn.ModuleList([Layer() for _ in range(6)])
        self.norm=torch.nn.Identity()
    def _update_causal_mask(self,*args):return None
    def rotary_emb(self,hidden,position_ids):return (torch.ones_like(hidden),torch.zeros_like(hidden))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Decoder()
        self.lm_head=torch.nn.Linear(3,16,bias=False)
        self.config=SimpleNamespace(model_type="llama",pretraining_tp=1,_attn_implementation="eager",vocab_size=16)
        self.eval()
        for p in self.parameters():p.requires_grad_(False)


def record(index=0,new="ab",old="c"):
    r=dict(case_id=index,requested_rewrite=dict(subject=f"s{index}",relation_id="r",prompt="{} p",
            target_new={"str":new}))
    if old is not None:r["requested_rewrite"]["target_true"]={"str":old}
    return r


class HistoryRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.root_patch=patch.object(module,"ROOT",self.root);self.root_patch.start()
        self.transformers_patch=patch.dict(sys.modules,{"transformers":SimpleNamespace(__version__="4.44.2")})
        self.transformers_patch.start()
        self.before_tf32=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(123);torch.set_num_threads(2)
        model=Model()
        self.rt=SimpleNamespace(model=model,etok=Tokenizer(),oracles=[],
            W=dict(model.named_parameters())[WEIGHT],identity={"W0":"fixture"},
            lock={"model_revision":"fixture","model_config_sha256":"a"*64,
                  "model_weights_identity_sha256":"b"*64,"tokenizer_identity_sha256":"c"*64,
                  "torch":str(torch.__version__),"transformers":"4.44.2","transformers_import":"fixture",
                  "execution":{"commit":"d"*40},"lock_identity":"e"*64})
        original=module._nonselected(model_guard(model))
        def guard():
            if module._nonselected(model_guard(model))!=original:raise RuntimeError("fixture nonselected mutation")
        self.rt.guard=guard
        self.records=receive_all([],[record(0),record(1)])

    def tearDown(self):
        self.transformers_patch.stop();self.root_patch.stop()
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=self.before_tf32
        self.temp.cleanup()

    def factory(self,records=None,directory="prefix"):
        return HistoryFactory(self.rt,self.records if records is None else records,self.root/directory)

    def test_tokenization_and_only_two_canonical_paths(self):
        packs,rows=canonical_paths(self.rt,self.records[0])
        self.assertEqual(len(packs),2)
        self.assertEqual({r["kind"] for r in rows},{"canonical"})
        self.assertEqual([r["branch"] for r in rows],["new","old"])
        for p,r in zip(packs,rows):
            self.assertEqual(p["input_ids"][0,-1].item(),rows[0]["labels"][0] if r["branch"]=="new" else self.rt.etok.encode("s0 p")[-1])
            self.assertEqual(r["positions"][-1],p["input_ids"].numel()-1)

    def test_same_input_dedup_and_missing_old(self):
        packs,rows=canonical_paths(self.rt,record(new="a",old="b"))
        self.assertEqual(len(packs),1);self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["cache"],rows[1]["cache"])
        packs,rows=canonical_paths(self.rt,record(old=None))
        self.assertEqual(len(packs),1);self.assertEqual([r["branch"] for r in rows],["new"])

    def test_create_once_then_zero_prefix_forward_and_no_leak(self):
        factory=self.factory()
        with factory(self.records[0]) as (oracle,rows):
            self.assertEqual(oracle.work["prefix_forwards"],2)
            keys=[c.keys.clone() for c in oracle.caches]
        self.assertEqual(oracle.caches,[])
        first=self.rt.model.model.layers[0].calls
        with factory(self.records[0]) as (other,_):
            self.assertEqual(other.work["prefix_forwards"],0)
            for a,b in zip(keys,[c.keys for c in other.caches]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(self.rt.model.model.layers[0].calls,first)
        self.assertEqual(self.rt.oracles,[])
        self.assertEqual(factory.work["peak_resident_requests"],1)
        self.assertEqual(factory.work["peak_resident_paths"],2)
        self.assertEqual(factory.work["live_paths"],0)
        self.assertEqual(len(list((self.root/"prefix"/"requests").glob("*.pt"))),1)

    def test_l4_write_rebind_keeps_upstream_cache(self):
        factory=self.factory()
        with factory(self.records[0]) as (oracle,_):
            original=oracle.history_cache_binding["current_physical_selected"]["weight_sha256"]
        with torch.no_grad():self.rt.W.add_(.1)
        before=self.rt.model.model.layers[0].calls
        with factory(self.records[0]) as (oracle,_):
            changed=oracle.history_cache_binding["current_physical_selected"]["weight_sha256"]
            self.assertNotEqual(original,changed)
            self.assertFalse(oracle.history_cache_binding["selected_weight_equality_to_creator_required"])
        self.assertEqual(self.rt.model.model.layers[0].calls,before)

    def test_existing_cache_first_load_fullhash_only_once(self):
        factory=self.factory()
        with factory(self.records[0]):pass
        second=self.factory()
        with second(self.records[0]):pass
        with second(self.records[0]):pass
        self.assertEqual(second.work["complete_file_hash_checks"],1)
        self.assertEqual(second.work["new_prefix_forwards"],0)

    def test_streaming_scores_gradient_and_J_reuse_disk_without_prefix(self):
        factory=self.factory()
        history=StreamingHistoryOracle(self.records,factory)
        entry=EndpointBinding(self.rt.W,"cpu",endpoint_id="entry",source_identity={"test":True})
        observed=history.evaluate(entry)
        prefix_count=self.rt.model.model.layers[0].calls
        native=EndpointBinding(self.rt.W+.02,"cpu",endpoint_id="native",source_identity={"test":True})
        archive=FactorArchive(self.root/"factors",{"test":True})
        scored=history.evaluate(native,entry_anchor=observed.anchor(),gradient=True,factor_sink=archive)
        j=history.jacobian(scored,[torch.ones_like(self.rt.W).double()])
        self.assertEqual(tuple(j.shape),(2,1))
        self.assertTrue(torch.isfinite(j).all())
        self.assertEqual(self.rt.model.model.layers[0].calls,prefix_count)
        self.assertEqual(factory.work["live_requests"],0)
        self.assertEqual(self.rt.oracles,[])
        entry.close();native.close()

    def test_nested_context_rejected_outer_stays_live(self):
        factory=self.factory()
        with factory(self.records[0]):
            with self.assertRaises(DecisionError):
                with factory(self.records[1]):pass
            self.assertEqual(factory.work["live_requests"],1)
        self.assertEqual(factory.work["live_requests"],0)

    def test_exception_releases_prefix_and_preserves_artifact(self):
        factory=self.factory()
        with self.assertRaisesRegex(ValueError,"fixture abort"):
            with factory(self.records[0]) as (oracle,_):raise ValueError("fixture abort")
        self.assertEqual(oracle.caches,[])
        self.assertEqual(factory.work["live_requests"],0)
        self.assertEqual(factory.work["failed_contexts"],1)
        with factory(self.records[0]) as (oracle,_):self.assertEqual(oracle.work["prefix_forwards"],0)

    def test_cache_mutation_is_error_not_silent_reload(self):
        factory=self.factory()
        with self.assertRaises(RuntimeError):
            with factory(self.records[0]) as (oracle,_):oracle.caches[0].keys.add_(1)
        self.assertEqual(factory.work["live_requests"],0)
        self.assertGreater(factory.work["cleanup_guard_failures"],0)

    def test_wrong_record_and_nonsel_mutation(self):
        factory=self.factory()
        altered=deepcopy(self.records[0]);altered["requested_rewrite"]["target_new"]["str"]="different"
        with self.assertRaises(DecisionError):
            with factory(altered):pass
        with torch.no_grad():self.rt.model.model.layers[0].mlp.down_proj.weight.add_(1)
        with self.assertRaises(DecisionError):
            with factory(self.records[0]):pass

    def test_corruption_or_partial_not_recomputed(self):
        factory=self.factory()
        with factory(self.records[0]):pass
        path=Path(factory.receipt["cache_members"][0]["path"])
        with path.open("r+b") as handle:handle.seek(-1,2);handle.write(b"X")
        before=self.rt.model.model.layers[0].calls
        with self.assertRaises(DecisionError):
            with factory(self.records[0]):pass
        self.assertEqual(self.rt.model.model.layers[0].calls,before)
        packs,rows=canonical_paths(self.rt,self.records[1])
        key=factory._request_identity(self.records[1],packs,rows)
        partial=factory.requests_directory/f"{key}.pt"
        with partial.open("xb") as handle:torch.save({"partial":True},handle)
        with self.assertRaisesRegex(DecisionError,"PARTIAL_HISTORY"):
            with factory(self.records[1]):pass
        self.assertTrue(partial.exists())

    def test_lock_tokenizer_and_scope_changes_fail(self):
        factory=self.factory()
        self.rt.etok.padding_side="left"
        with self.assertRaises(DecisionError):
            with factory(self.records[0]):pass
        self.rt.etok.padding_side="right";self.rt.lock["model_revision"]="wrong"
        with self.assertRaises(DecisionError):
            with factory(self.records[0]):pass
        with self.assertRaises(DecisionError):HistoryFactory(self.rt,self.records,self.root.parent/"outside")

    def test_external_cache_change_during_context_fails(self):
        factory=self.factory()
        with self.assertRaisesRegex(DecisionError,"CHANGED_DURING_CONTEXT"):
            with factory(self.records[0]):
                path=Path(factory.receipt["cache_members"][0]["path"])
                with path.open("r+b") as handle:handle.seek(-1,2);handle.write(b"X")
        self.assertEqual(factory.work["live_requests"],0)

    def test_context_error_not_masked_by_oracle_registration_leak(self):
        factory=self.factory()
        with self.assertRaisesRegex(ValueError,"original") as raised:
            with factory(self.records[0]):
                self.rt.oracles.append(object())
                raise ValueError("original")
        self.assertIn("LIFETIME_LEAK"," ".join(raised.exception.__notes__))
        self.assertEqual(factory.work["live_requests"],0)

    def test_payload_is_cpu_weights_only_and_compact_receipt_raw_free(self):
        factory=self.factory()
        with factory(self.records[0]):pass
        row=factory.receipt["cache_members"][0]
        payload=torch.load(row["path"],weights_only=True,mmap=True,map_location="cpu")
        self.assertEqual(payload["prefix_source_identity"],factory.source_identity)
        self.assertTrue(all(c["keys"].device.type=="cpu" for c in payload["caches"]))
        compact=json.dumps(factory.receipt)
        self.assertNotIn('"labels"',compact)
        self.assertNotIn('"input_ids"',compact)


if __name__=="__main__":unittest.main()
