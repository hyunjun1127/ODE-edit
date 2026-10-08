"""Metadata/CPU fixture tests only; no real W0/native/GPU qualification claim."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext

from official.evaluation import factual
from official.evaluation import w0_reference as w
from official.evaluation.generation.common import digest as gd
from official.evaluation.generation.metrics import reduce_cases
from official.evaluation.generation.native_profile import PROFILE, ROUTE, SOURCE, runtime_identity
from official.evaluation.reduce import counterfact, zsre


def sha(char="a"):
    return char * 64


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return w.member(path)


def content():
    file = dict(bytes=4, sha256=sha())
    return dict(model=dict(model_id="TEST_ONLY/model", revision="b"*40, config_sha256=sha(), weights={"model.safetensors":file}),
        tokenizer=dict(files={"tokenizer.json":file}, settings=dict(prompt_add_special_tokens=True,
            target_add_special_tokens=False, padding="right", target_prefix=" ", lookup=factual.TOKENIZATION)),
        datasets={name:dict(stream_sha256=sha("c" if name=="cf" else "d"),
            ordered_occurrences_sha256=w._digest(list(range(1,2001))), ordered_queries_sha256=sha(), requests=2000)
            for name in ("cf","zsre")},
        sources=dict(factual=sha(), reduce=sha(), w0_reference=sha(), generation={name:sha() for name in w.GENERATION_SOURCES}),
        runtime=dict(versions={name:"TEST_ONLY_VERSION" for name in w.VERSIONS}, numeric=dict(weights_dtype="float32",
            attention="eager", matmul_tf32=False, cudnn_tf32=False, autocast=False, factual_use_cache=False, generation_use_cache=True)),
        generation=dict(profile=PROFILE, seed=20261007, case_batching="ALL_GENERATION_PROMPTS", sampling_scope="ENDPOINT_GLOBAL_BATCH_STREAM",
            top_k=5, max_total_tokens=100, n_gen_per_prompt=1, EOS_stop=False, decode=SOURCE["decode"]),
        references=dict(identity_sha256=sha("e"), files={name:file for name in w.REFERENCE_FILES},
            nltk=dict(resources={"TEST_ONLY/resource":file}, sources={"TEST_ONLY/tokenizer.py":file}), vectorizer="TEST_ONLY_VECTOR"))


def external(fp, dataset="zsre", *, consumer=False):
    return dict(model="llama3", model_revision=fp["content"]["model"]["revision"], tokenizer_sha256=w._tokenizer_sha(fp),
        stream_sha256=fp["content"]["datasets"][dataset]["stream_sha256"],
        runtime={key:fp["content"]["runtime"]["versions"][key] for key in ("torch","transformers","numpy")},
        precision="FP32_EAGER_TF32_OFF_NO_AUTOCAST", code_commit="f"*40 if consumer else "a"*40,
        official_tree="e"*40 if consumer else "b"*40, assets_sha256=sha(),raw_local_only=True)


def candidate(ordinal, kind, target_kind):
    return dict(case_index=ordinal-1, occurrence_index=ordinal, kind=kind, prompt_index=0,
        prompt="TEST_ONLY_PROMPT", target_kind=target_kind, target="TEST_ONLY_TARGET", input_token_ids=[1,2], target_start=1,
        target_token_ids=[2], predicted_token_ids=[2], token_count=1, token_correct=[True], token_correct_count=1,
        strict_correct=True, nll_by_token=[1.0 if target_kind=="new" else 2.0], mean_nll=1.0 if target_kind=="new" else 2.0)


def endpoint(dataset, fp):
    cases=[]
    for ordinal in range(1,2001):
        row=dict(case_id=ordinal+5000, occurrence_index=ordinal)
        for kind in ("rewrite","paraphrase","neighborhood"):
            if dataset=="cf":
                new,true=candidate(ordinal,kind,"new"),candidate(ordinal,kind,"true")
                row[kind+"_observations"]=[dict(prompt_index=0,prompt="TEST_ONLY_PROMPT", target_new=new,target_true=true)]
                row[kind+"_prompts_probs"]=[dict(target_new=new["mean_nll"],target_true=true["mean_nll"])]
            else:
                row[kind+"_observations"]=[candidate(ordinal,kind,"loc_ans" if kind=="neighborhood" else "new")]
            row[kind+"_prompts_correct"]=[True]
        if dataset=="zsre":
            row["neighborhood_W0_agreement"]=[True]
        cases.append(row)
    fp["content"]["datasets"][dataset]["ordered_queries_sha256"]=w._digest(w._signatures(cases,dataset))
    fp["sha256"]=w._digest(fp["content"])
    identity=dict(schema=factual.SCHEMA,dataset=dataset,tokenization=factual.TOKENIZATION,
        cohort_sha256=fp["content"]["datasets"][dataset]["ordered_queries_sha256"], ordered_occurrences=list(range(1,2001)),
        external_identity=external(fp,dataset),padding="RIGHT_EXPLICIT_ATTENTION_MASK",use_cache=False,W0_reference_sha256=None)
    accuracy={kind:factual._accuracy([desired for row in cases for desired in
        ([observation["target_true" if kind=="neighborhood" else "target_new"] for observation in row[kind+"_observations"]]
         if dataset=="cf" else row[kind+"_observations"])]) for kind in ("rewrite","paraphrase","neighborhood")}
    return dict(cases=cases,accuracy=accuracy,summary=counterfact(cases) if dataset=="cf" else zsre(cases),identity=identity,
        identity_sha256=w._digest(identity),model_no_mutation=True,RNG_restored=True,raw_local_only=True)


def reference(ep):
    identity=dict(schema=factual.W0_SCHEMA,tokenization=factual.TOKENIZATION,external_identity=ep["identity"]["external_identity"],
        cohort_sha256=ep["identity"]["cohort_sha256"],ordered_occurrences=list(range(1,2001)),state="W0_COLD_BASE_MODEL")
    value=dict(schema=factual.W0_SCHEMA,identity=identity,identity_sha256=w._digest(identity),evaluation=ep,raw_local_only=True,
        cases=[dict(case_id=row["case_id"],occurrence_index=row["occurrence_index"],
            queries=[query for query in sig["queries"] if query["kind"]=="neighborhood"],predictions=[[2]])
            for row,sig in zip(ep["cases"],w._signatures(ep["cases"],"zsre"))])
    value["payload_sha256"]=w._digest(value)
    return value


def generation(root, fp, execution, cf_records):
    runtime=runtime_identity(dict(model_identity=dict(model="TEST_ONLY",revision=fp["content"]["model"]["revision"],
        tokenizer_sha256=w._tokenizer_sha(fp)),generation_source_sha=dict(code_commit=execution["source"]["main_commit"],
            official_tree=execution["source"]["official_tree"])), fp["content"]["references"]["identity_sha256"])
    runtime_sha=gd(runtime)
    runtime_member=put(root/"observer-identity.json",dict(identity=runtime,identity_sha256=runtime_sha))
    from official.evaluation.generation.observer import _record_identity
    records=[_record_identity(row,index) for index,row in enumerate(cf_records,1)]
    stream=gd(dict(runtime=runtime_sha,eval_seed=20261007,ordered_record_identities=records))
    state=dict(base_model=fp["content"]["model"],actual_model_edits=0)
    rows=[]
    for record in records:
        ordinal=record["ordered_occurrence"]
        identity=dict(runtime=runtime_sha,state_identity=state,record_identity=record,sampling_stream_sha256=stream)
        tokens=list(range(100))
        observation=dict(profile=PROFILE,route=ROUTE,seed=20261007,sampling_scope="ENDPOINT_GLOBAL_BATCH_STREAM",
            prompt="TEST_ONLY_PROMPT",occurrence=ordinal,prompt_index=0,endpoint_RNG_restore_guard_required=True,EOS_stop=False,
            case_batch_prompt_count=1,initial_batch_width=100,model_forwards=0,physical_forward_calls=0,
            sampling=dict(top_k=5,temperature=1,top_p=1,max_total_tokens=100,n_gen_per_prompt=1),
            input_token_ids=tokens,continuation_token_ids=[],full_token_ids=tokens,padded_input_token_ids=tokens,
            padded_decode_token_ids=tokens,input_token_count=100,continuation_token_count=0,stop_reason="length_cap_no_continuation",
            text="TEST_ONLY_TEXT",prefill_query_tokens=0,decode_query_tokens=0,full_prefix_token_work=0)
        metrics=dict(ngram_entropy=0.0,reference_score=0.5,fluency_valid=True,consistency_valid=True,
            reasons=["length_cap_no_continuation"],generation_prompt_count=1,generated_token_count=0,length_cap_no_continuation_count=1)
        raw=dict(identity=identity,identity_sha256=gd(identity),raw_local_only=True,checkpoint_saved=False,
            occurrence=ordinal,case_id=record["case_id"],observations=[observation],metrics=metrics)
        raw["payload_sha256"]=gd(raw)
        raw_member=put(root/"raw"/(str(ordinal)+".json"),raw)
        rows.append(dict(occurrence=ordinal,case_id=record["case_id"],identity_sha256=raw["identity_sha256"],
            payload_sha256=raw["payload_sha256"],observation_path=raw_member["path"],metrics=metrics,
            provenance=dict(raw_member=raw_member,runtime_sha256=runtime_sha,generation_source_sha=runtime["generation_source_sha"],route=ROUTE)))
    identity=dict(runtime=runtime_sha,state_sha256=gd(state),endpoint="W0",cohort="first2000",ordered_occurrences=list(range(1,2001)),
        observation_identities=[row["identity_sha256"] for row in rows],sampling_stream_sha256=stream)
    native=dict(identity=identity,identity_sha256=gd(identity),profile=PROFILE,route=ROUTE,native_execution_complete=True,
        qualification_performed=False,no_fallback=True,RNG_restored=True,observer_no_mutation=True,
        physical_forward_calls=0,prefill_query_tokens=0,decode_query_tokens=0)
    value=dict(identity=identity,identity_sha256=gd(identity),summary=reduce_cases(rows),rows=rows,RNG_restored=True,
        observer_no_mutation=True,raw_local_only=True,native_execution_member=put(root/"native-execution.json",native))
    return value,runtime_member


class W0ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(); cls.root=Path(cls.tmp.name)
        data=content()
        cls.records={name:[dict(case_id=n+5000,occurrence_index=n,generation_prompts=["TEST_ONLY_PROMPT"]) for n in range(1,2001)]
                     for name in ("cf","zsre")}
        cls.dataset_members={name:put(cls.root/"datasets"/(name+".json"),rows) for name,rows in cls.records.items()}
        for name,row in cls.dataset_members.items():data["datasets"][name]["stream_sha256"]=row["sha256"]
        cls.source_members={}
        for name in ("factual.py","reduce.py","w0_reference.py", *("generation/"+n for n in w.GENERATION_SOURCES)):
            path=cls.root/"source"/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# TEST_ONLY_SOURCE_METADATA_NOT_EXECUTED\n"+name)
            cls.source_members[name]=w.member(path)
            if name.startswith("generation/"):data["sources"]["generation"][name.split("/",1)[1]]=cls.source_members[name]["sha256"]
            else:data["sources"][name.removesuffix(".py")]=cls.source_members[name]["sha256"]
        cls.fp=w.computational_fingerprint(data)
        cls.cf=endpoint("cf",cls.fp); cls.zsre=endpoint("zsre",cls.fp); cls.ref=reference(cls.zsre)
        cls.execution=dict(source=dict(main_commit="a"*40,official_tree="b"*40),
            path="/producer",hardware="TEST_ONLY_GPU_METADATA_NOT_ACTUAL_GPU")
        cls.consumer=dict(source=dict(main_commit="f"*40,official_tree="e"*40),path="/consumer",hardware="OTHER_GPU")
        cls.gen,cls.runtime_member=generation(cls.root/"generation",cls.fp,cls.execution,cls.records["cf"])
        proof={}
        for method in ("FT","MEMIT","MEMIT_FE"):
            identity=dict(model_revision=cls.fp["content"]["model"]["revision"],tokenizer_sha256=w._tokenizer_sha(cls.fp),
                code_commit=cls.execution["source"]["main_commit"],official_tree_sha256=cls.execution["source"]["official_tree"])
            ready=dict(schema="official-server1-native-resume-READY-v1",method=method,identity=identity,passed=True,
                actual_native_B3_and_B2_resume=True,actual_fit_calls=6,generation_calls=0,
                checks=dict(selected_weights="EXACT_SHA256",contexts="EXACT",RNG="EXACT",factual="EXACT_RAW_VALUES_AND_TOKEN_IDENTITY",no_tolerance_relaxation=True),
                native_parity=None)
            measured=copy.deepcopy(cls.cf);measured["cases"]=measured["cases"][:300]
            measured["summary"]=counterfact(measured["cases"])
            selected_state=dict(successful_calls=3,method=method,TEST_ONLY_NO_REAL_NATIVE_CALLS=True)
            factual_member=put(cls.root/"qualification"/method/"B3-factual.json",measured)
            live_measured=copy.deepcopy(measured)
            # Complete structural metadata fixture for stored original300 proof.
            # The reviewed task validator independently accepts the same fake
            # receipt below; neither test actually observes a model or GPU.
            from official.runners.server1.native_parity import PLAN, validate_report
            sigs=w._signatures(measured["cases"],"cf")
            live_measured["identity"]["ordered_occurrences"]=list(range(1,301))
            live_measured["identity"]["cohort_sha256"]=w._digest(sigs)
            live_measured["identity_sha256"]=w._digest(live_measured["identity"])
            measured=copy.deepcopy(live_measured)
            live_measured["work"]=dict(candidate_sequences=1800,forward_calls=113,physical_input_tokens=3600,
                padded_input_tokens=3600,target_tokens=1800,seconds=0.0,TEST_ONLY_METADATA_NOT_ACTUAL_FORWARD=True)
            factual_member=put(cls.root/"qualification"/method/"B3-factual.json",measured)
            bind=dict(model_identity=dict(revision=cls.fp["content"]["model"]["revision"]),
                tokenizer_identity=dict(sha256=w._tokenizer_sha(cls.fp)),state_identity=selected_state)
            ni=dict(schema="official-cf-original-native-reference-v1",external_identity=live_measured["identity"]["external_identity"],
                reference_binding=bind,source_sha256=PLAN["native_reference_upstream_sha256"],source_commit=PLAN["native_reference_upstream_commit"],
                source_bytes=7941,original_function="test_batch_prediction",evidence_scope="ACTUAL_MATCHED_SUBSET",ordered_occurrences=list(range(1,301)),
                cohort_sha256=w._digest(sigs),native_device="cuda",padding="NATIVE_RIGHT",model_dtype="FP32",use_cache=False,
                locked_cohort_sha256=w._digest([dict(case_id=row["case_id"],occurrence_index=row["occurrence_index"]) for row in measured["cases"]]))
            native=dict(schema="official-cf-original-native-reference-v1",status="OBSERVED_UNCOMPARED",identity=ni,
                identity_sha256=w._digest(ni),canonical_payload_sha256=w._digest(live_measured),raw_local_only=True,model_no_mutation=True,
                RNG_restored=True,checkpoint_saved=False,cases=copy.deepcopy(measured["cases"]),summary=measured["summary"],case_signatures=sigs,
                work=dict(live_measured["work"],forward_calls=300))
            comparison=dict(schema="official-cf-original-native-reference-v1",status="PASS",evidence_scope="ACTUAL_MATCHED_SUBSET",
                evidence={"TEST_ONLY_CPU_FIXTURE":"NOT_OBSERVED","ACTUAL_GPU_SMOKE":"NOT_OBSERVED","ACTUAL_MATCHED_SUBSET":"PASS","ACTUAL_FULL_2K":"NOT_OBSERVED"},
                mismatches=[],display_mismatches=[],raw_local_only=True,checkpoint_saved=False,scientific_performance_promotion=False,
                tolerances=dict(nll_abs_nats=1e-4,nll_relative_to_native=1e-5,strict_booleans="EXACT",aggregate_abs=1e-10),
                canonical_identity_sha256=live_measured["identity_sha256"],native=native,work=native["work"])
            parity=dict(schema="official-server1-native-CF-parity-report-v1",plan_sha256=w._digest(PLAN),
                external_identity=live_measured["identity"]["external_identity"],reference_binding=bind,canonical=live_measured,
                canonical_payload_sha256=w._digest(live_measured),canonical_payload_unchanged=True,canonical_observation="EXISTING_B3_FIRST300",
                additional_canonical_forward_calls=0,state_before=selected_state,state_after=selected_state,comparison=comparison,
                RNG_restored=True,input_records_unchanged=True,raw_local_only=True,performance_gate=False,additional_fit_calls=0,generation_calls=0,
                TEST_ONLY_METADATA_FIXTURE_NOT_ACTUAL_GPU=True)
            ready["native_parity"]=validate_report(parity,live_measured["identity"]["external_identity"],PLAN)
            assert ready["native_parity"]==w._native_parity_proof(parity,measured,selected_state)
            parity_member=put(cls.root/"qualification"/method/"B3-native-parity.json",parity)
            for key,name,batches,calls in (("continuous","continuous",3,3),("stopped","stop",2,2),("resumed","resume",3,1)):
                ready[key]=put(cls.root/"qualification"/method/(key+".json"),dict(schema="official-server1-native-qualification-stage-v1",
                    stage=name,completed_batch=batches,actual_native_fit_calls=calls,GPU_actual=True,actual_model_loaded=True,
                    identity=identity,method=method,selected_state=selected_state,contexts_sha256=sha(),RNG_sha256=sha(),checkpoint_RNG_sha256=sha(),
                    factual_member=factual_member if key in ("continuous","resumed") else None,
                    native_parity_member=parity_member if key=="continuous" else None,TEST_ONLY_METADATA_FIXTURE_NOT_ACTUAL_GPU=True))
            proof[method]=put(cls.root/"qualification"/method/"READY.json",ready)
        cls.validation={name:dict(actual_complete=True,actual_model_edits=0,state="W0_COLD_BASE_MODEL",requests=2000,
            ordered_queries_sha256=cls.fp["content"]["datasets"]["zsre" if name=="zsre_reference" else "cf"]["ordered_queries_sha256"],
            native_qualification_state=dict(status="VALIDATED",evidence_members=proof,evidence_sha256=w._digest(proof))) for name in w.COMPONENTS}
        cls.members={name:put(cls.root/(name+".json"),value) for name,value in
            (("cf_factual",cls.cf),("cf_generation",cls.gen),("zsre_reference",cls.ref))}
        cls.ready=w.make_ready(producer_execution_identity=cls.execution,fingerprint=cls.fp,members=cls.members,
            component_validation=cls.validation,generation_runtime_member=cls.runtime_member,
            source_members=cls.source_members,dataset_members=cls.dataset_members)
        cls.ready_path=cls.root/"READY.json";put(cls.ready_path,cls.ready)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def bind(self, reference=None, **kw):
        options=dict(producer_external_identity=external(self.fp),consumer_external_identity=external(self.fp,consumer=True),
            producer_fingerprint=self.fp,consumer_fingerprint=self.fp,producer_execution_identity=self.execution,consumer_execution_identity=self.consumer)
        options.update(kw)
        return w.bind_zsre_reference(copy.deepcopy(self.ref if reference is None else reference),**options)

    def test_equal_content_distinct_execution_preserves_original_zsre(self):
        original=copy.deepcopy(self.ref); view=self.bind()
        ref,binding=w.validate_zsre_reference(view,consumer_external_identity=external(self.fp,consumer=True),
            signatures=w._signatures(self.zsre["cases"],"zsre"))
        self.assertEqual(ref,original);self.assertEqual(self.ref,original)
        self.assertNotEqual(binding["producer_execution_identity"],binding["consumer_execution_identity"])
        self.assertFalse(binding["producer_raw_relabelled"])

    def test_factual_portable_contract_and_classic_identity_fail_before_fake_infer(self):
        # Fake planning/inference only; exercises the real API branch, no LM.
        base=dict(case_id=5001,occurrence_index=1)
        signature=w._signatures(self.zsre["cases"],"zsre")[0]
        results=[copy.deepcopy(self.zsre["cases"][0][kind+"_observations"][0])
                 for kind in ("rewrite","paraphrase","neighborhood")]
        consumer=external(self.fp,consumer=True)
        with patch.object(factual,"_plan",return_value=([base],results,[signature])), \
             patch.object(factual,"_observation",return_value=nullcontext()), \
             patch.object(factual,"_infer",return_value=(results,dict(TEST_ONLY_FAKE_INFERENCE=True))) as infer:
            with self.assertRaisesRegex(factual.FactualError,"EXTERNAL_IDENTITY"):
                factual.evaluate_zsre(None,None,[],w0_reference=self.ref,identity=consumer)
            self.assertEqual(infer.call_count,0)
            result=factual.evaluate_zsre(None,None,[],w0_reference=self.bind(),identity=consumer)
            self.assertEqual(infer.call_count,1)
            self.assertEqual(result["identity"]["external_identity"],consumer)
            self.assertEqual(result["W0_reference_binding"]["producer_external_identity"],external(self.fp))
            self.assertEqual(result["identity"]["W0_reference_sha256"],self.ref["identity_sha256"])
            changed=dict(consumer,code_commit="FALSE_CONSUMER")
            with self.assertRaises(w.ReferenceInputError):
                factual.evaluate_zsre(None,None,[],w0_reference=self.bind(),identity=changed)
            self.assertEqual(infer.call_count,1)

    def test_ready_all_raw_sha_proofs_and_actual_content_reader(self):
        view=w.read_ready(self.ready_path,consumer_execution_identity=self.consumer,consumer_fingerprint=self.fp)
        self.assertEqual(view.values["cf_factual"],self.cf)
        self.assertFalse(view.binding["execution_identities_equal"])
        self.assertEqual(len([key for key in view.members if key.startswith("cf_generation.raw.")]),2000)
        self.assertEqual(view.zsre(consumer_external_identity=external(self.fp,consumer=True)).reference,self.ref)

    def test_explicit_copied_member_path_is_content_only_and_original_unchanged(self):
        path=self.root/"consumer-copy.json";path.write_bytes(Path(self.members["cf_factual"]["path"]).read_bytes())
        view=w.read_ready(self.ready_path,consumer_execution_identity=self.consumer,consumer_fingerprint=self.fp,
            member_paths={"cf_factual":str(path)})
        self.assertTrue(view.members["cf_factual"]["path_differs"])
        self.assertEqual(view.ready,self.ready)

    def test_missing_ready_typed_no_auto_creation(self):
        path=self.root/"absent.json"
        with self.assertRaises(w.ReferenceInputError) as error:
            w.read_ready(path,consumer_execution_identity=self.consumer,consumer_fingerprint=self.fp)
        self.assertEqual(error.exception.code,"REFERENCE_INPUT_MISSING");self.assertFalse(path.exists())

    def test_unknown_fields_paths_hardware_missing_versions_and_numeric_false_type(self):
        for mutate in (lambda c:c.update(tree="ignored"),lambda c:c["model"].update(path="/model"),
            lambda c:c["runtime"]["versions"].pop("torch"),lambda c:c["runtime"]["numeric"].update(matmul_tf32=0),
            lambda c:c["generation"].update(EOS_stop=0),lambda c:c["sources"].pop("w0_reference")):
            value=copy.deepcopy(self.fp["content"]);mutate(value)
            with self.assertRaises(w.ReferenceInputError):w.computational_fingerprint(value)

    def test_every_computational_section_mismatch_rejected(self):
        for section, mutate in (("model",lambda c:c["model"].update(config_sha256=sha("f"))),
            ("tokenizer",lambda c:c["tokenizer"]["files"]["tokenizer.json"].update(sha256=sha("f"))),
            ("datasets",lambda c:c["datasets"]["cf"].update(stream_sha256=sha("f"))),
            ("sources",lambda c:c["sources"].update(factual=sha("f"))),
            ("runtime",lambda c:c["runtime"]["versions"].update(torch="OTHER")),
            ("references",lambda c:c["references"].update(identity_sha256=sha("f")))):
            value=copy.deepcopy(self.fp["content"]);mutate(value)
            with self.assertRaisesRegex(w.ReferenceInputError,section):self.bind(consumer_fingerprint=w.computational_fingerprint(value))

    def test_producer_actual_other_model_runtime_stream_or_tokenizer_spoof_rejected(self):
        for key,value in (("model_revision","c"*40),("tokenizer_sha256",sha("f")),("stream_sha256",sha("f")),
                           ("runtime",dict(torch="OTHER",transformers="TEST_ONLY_VERSION",numpy="TEST_ONLY_VERSION"))):
            ref=copy.deepcopy(self.ref);ref["identity"]["external_identity"][key]=value
            ref["identity_sha256"]=w._digest(ref["identity"]);ref["payload_sha256"]=w._digest({k:v for k,v in ref.items() if k!="payload_sha256"})
            with self.assertRaises(w.ReferenceInputError):self.bind(ref,producer_external_identity=ref["identity"]["external_identity"])

    def test_raw_mutation_resigned_token_denominator_summary_or_source_rows_rejected(self):
        for mutate in (lambda r:r["evaluation"]["cases"][0]["rewrite_observations"][0].update(token_count=4),
            lambda r:r["evaluation"]["cases"][0]["rewrite_observations"][0]["input_token_ids"].append(9),
            lambda r:r["evaluation"]["summary"].update(Efficacy=0),lambda r:r["cases"][0].update(predictions=[])):
            ref=copy.deepcopy(self.ref);mutate(ref);ref["payload_sha256"]=w._digest({k:v for k,v in ref.items() if k!="payload_sha256"})
            with self.assertRaises(w.ReferenceInputError):self.bind(ref)

    def test_consumer_identity_or_query_change_rejected_before_use(self):
        view=self.bind();sigs=w._signatures(self.zsre["cases"],"zsre");sigs[0]["queries"][-1]["target_token_ids"]=[3]
        with self.assertRaises(w.ReferenceInputError):w.validate_zsre_reference(view,consumer_external_identity=external(self.fp,consumer=True),signatures=sigs)
        changed=external(self.fp,consumer=True);changed["code_commit"]="RELABELED"
        with self.assertRaises(w.ReferenceInputError):w.validate_zsre_reference(view,consumer_external_identity=changed,signatures=[])

    def test_binding_or_original_payload_mutation_rejected(self):
        view=self.bind();view.binding["producer_execution_identity"]["path"]="/spoof"
        with self.assertRaises(w.ReferenceInputError):w.validate_zsre_reference(view,consumer_external_identity=external(self.fp,consumer=True),signatures=[])
        view=self.bind();view.reference["cases"][0]["predictions"][0][0]=3
        with self.assertRaises(w.ReferenceInputError):w.validate_zsre_reference(view,consumer_external_identity=external(self.fp,consumer=True),signatures=[])

    def test_incomplete_cold_or_unknown_metadata_ready_rejected(self):
        for mutate in (lambda r:r.update(actual_complete=False),lambda r:r.update(actual_model_edits=1),
                       lambda r:r.pop("computational_fingerprint"),lambda r:r["component_validation"]["cf_generation"].update(requests=1999)):
            ready=copy.deepcopy(self.ready);mutate(ready);ready["ready_sha256"]=w._digest({k:v for k,v in ready.items() if k!="ready_sha256"})
            path=self.root/"tampered-READY.json";put(path,ready)
            with self.assertRaises(w.ReferenceInputError):w.read_ready(path,consumer_execution_identity=self.consumer,consumer_fingerprint=self.fp)

    def test_mutated_source_bytes_with_same_json_and_consumer_file_sha_rejected(self):
        path=self.root/"byte-mutated.json";path.write_bytes(Path(self.members["cf_factual"]["path"]).read_bytes()+b" ")
        with self.assertRaisesRegex(w.ReferenceInputError,"BYTES_SHA"):
            w.read_ready(self.ready_path,consumer_execution_identity=self.consumer,consumer_fingerprint=self.fp,member_paths={"cf_factual":str(path)})

    def test_generation_actual_other_model_reference_runtime_and_cold_state_rejected(self):
        identity=self.gen["identity"]
        runtime=json.loads(Path(self.runtime_member["path"]).read_text())
        for mutate in (lambda r:r["identity"]["model_identity"].update(revision="c"*40),
            lambda r:r["identity"]["model_identity"].update(tokenizer_sha256=sha("f")),
            lambda r:r["identity"].update(reference_assets_sha256=sha("f")),lambda r:r["identity"].update(eval_seed=0)):
            value=copy.deepcopy(runtime);mutate(value);value["identity_sha256"]=gd(value["identity"])
            with self.assertRaises(w.ReferenceInputError):w._generation_runtime(value,self.fp,dict(identity,runtime=value["identity_sha256"]))

    def test_qualification_status_string_without_actual_hashed_proofs_rejected(self):
        value=copy.deepcopy(self.validation);value["cf_factual"]["native_qualification_state"].pop("evidence_members")
        with self.assertRaises(w.ReferenceInputError):w._component_validation(value,self.fp)

    def test_resigned_cf_probability_and_zsre_prediction_not_measured_rows_rejected(self):
        value=copy.deepcopy(self.cf);value["cases"][0]["rewrite_prompts_probs"][0]["target_new"]=10
        value["summary"]=counterfact(value["cases"])
        with self.assertRaisesRegex(w.ReferenceInputError,"OBSERVATION_BINDING"):
            w._cold_component(value,"cf",self.fp)
        ref=copy.deepcopy(self.ref);ref["cases"][0]["predictions"][0][0]=3
        ref["payload_sha256"]=w._digest({k:v for k,v in ref.items() if k!="payload_sha256"})
        with self.assertRaisesRegex(w.ReferenceInputError,"ACTUAL_MEASUREMENT_BINDING"):self.bind(ref)

    def test_original_source_or_producer_execution_relabel_rejected(self):
        sources=copy.deepcopy(self.source_members);sources["factual.py"]["sha256"]=sha("f")
        with self.assertRaisesRegex(w.ReferenceInputError,"CONSUMED_SOURCE"):
            w._sources(sources,self.fp)
        execution=copy.deepcopy(self.execution);execution["source"]["main_commit"]="RELABELED"
        with self.assertRaisesRegex(w.ReferenceInputError,"PRODUCER_EXECUTION"):
            w._cold_component(self.cf,"cf",self.fp,execution)
        runtime=json.loads(Path(self.runtime_member["path"]).read_text())
        with self.assertRaisesRegex(w.ReferenceInputError,"PRODUCER_EXECUTION"):
            w._generation_runtime(runtime,self.fp,self.gen["identity"],execution)

    def test_generation_other_stream_case_cohort_is_not_source_compatible(self):
        records=copy.deepcopy(self.records["cf"]);records[0]["case_id"]=-1
        with self.assertRaisesRegex(w.ReferenceInputError,"ORDER_STREAM"):
            w._generation(self.gen,self.fp,{},self.runtime_member,records,collect=True)

    def test_generation_nonzero_cold_state_not_allowed_even_with_resigned_hashes(self):
        value=copy.deepcopy(self.gen);row=value["rows"][0]
        raw=json.loads(Path(row["observation_path"]).read_text());raw["identity"]["state_identity"]["actual_model_edits"]=1
        raw["identity_sha256"]=gd(raw["identity"]);raw["payload_sha256"]=gd({k:v for k,v in raw.items() if k!="payload_sha256"})
        new=put(self.root/"noncold-row.json",raw)
        row.update(observation_path=new["path"],identity_sha256=raw["identity_sha256"],payload_sha256=raw["payload_sha256"])
        row["provenance"]["raw_member"]=new
        with self.assertRaisesRegex(w.ReferenceInputError,"ACTUAL_COLD_MODEL_STATE"):
            w._generation(value,self.fp,{},self.runtime_member,self.records["cf"],collect=True)

    def test_actual_stage_without_measured_B3_or_native_proof_cannot_unlock_ready(self):
        validation=copy.deepcopy(self.validation)
        old=json.loads(Path(validation["cf_factual"]["native_qualification_state"]["evidence_members"]["FT"]["path"]).read_text())
        stage=json.loads(Path(old["continuous"]["path"]).read_text());stage["factual_member"]=None
        old["continuous"]=put(self.root/"unproven-stage.json",stage)
        proof=put(self.root/"unproven-READY.json",old)
        for row in validation.values():
            q=row["native_qualification_state"];q["evidence_members"]["FT"]=proof;q["evidence_sha256"]=w._digest(q["evidence_members"])
        with self.assertRaisesRegex(w.ReferenceInputError,"FACTUAL_PROOF_REQUIRED"):
            w._qualification(validation,self.fp,self.execution,{},collect=True)
        value=copy.deepcopy(self.validation);value["cf_factual"]["native_qualification_state"]["evidence_sha256"]=sha("f")
        with self.assertRaises(w.ReferenceInputError):w._component_validation(value,self.fp)

    def test_native_stored_receipt_replays_exact_source_raw_tolerance_work_contract(self):
        ready=json.loads(Path(self.validation["cf_factual"]["native_qualification_state"]["evidence_members"]["FT"]["path"]).read_text())
        stage=json.loads(Path(ready["continuous"]["path"]).read_text())
        measured=json.loads(Path(stage["factual_member"]["path"]).read_text())
        proof=json.loads(Path(stage["native_parity_member"]["path"]).read_text())
        self.assertEqual(w._native_parity_proof(proof,measured,stage["selected_state"]),ready["native_parity"])
        for mutate in (lambda r:r["comparison"].pop("native"),
            lambda r:r["comparison"]["tolerances"].update(nll_abs_nats=0.1),
            lambda r:r["comparison"]["native"]["identity"].update(source_sha256=sha("f")),
            lambda r:r["comparison"]["native"]["cases"][0]["rewrite_prompts_probs"][0].update(target_new=10.0),
            lambda r:r["comparison"]["native"]["cases"][0]["rewrite_prompts_correct"].__setitem__(0,False),
            lambda r:r["comparison"]["native"]["cases"][0]["neighborhood_prompts_correct"].__setitem__(0,1),
            lambda r:r["comparison"]["native"]["cases"][0].update(neighborhood_prompts_correct=[0]),
            lambda r:r["comparison"]["native"]["cases"][0].update(neighborhood_prompts_correct=(True,)),
            lambda r:r["comparison"]["native"]["work"].update(forward_calls=0),
            lambda r:r["comparison"]["evidence"].update(ACTUAL_FULL_2K="PASS")):
            value=copy.deepcopy(proof);mutate(value)
            with self.assertRaises(w.ReferenceInputError):w._native_parity_proof(value,measured,stage["selected_state"])


if __name__=="__main__":unittest.main()
