"""Metadata/CPU fixture tests only; no real W0/native/GPU qualification claim."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext, ExitStack

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
                row[kind+"_observations"]=[dict(prompt_index=0,prompt="TEST_ONLY_PROMPT", target_new=new,target_true=true,
                    desired_target="true" if kind=="neighborhood" else "new",margin_true_minus_new=true["mean_nll"]-new["mean_nll"])]
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
        identity_sha256=w._digest(identity),model_no_mutation=True,RNG_restored=True,raw_local_only=True,
        work=dict(factual.retained_query_work(w._signatures(cases,dataset),dataset),seconds=.001))


def execution_fixture(fp, *, consumer=False, role=None):
    runtime=dict(python="/fixture/python",python_version=fp["content"]["runtime"]["versions"]["python"],
                 dependency_versions={k:fp["content"]["runtime"]["versions"][k] for k in ("torch","transformers","numpy")})
    return dict(server="server4" if consumer else "server1", role=role or ("GPU_CONSUMER" if consumer else "GPU_PRODUCER"),
        source=dict(main_commit=("f" if consumer else "a")*40,official_tree=("e" if consumer else "b")*40),
        config_sha256=sha("a"),assets_manifest=dict(path="/fixture/assets.json",bytes=8,sha256=sha()),
        runtime=runtime,input_stream_bundle=dict(path="/fixture/bundle.json",bytes=8,sha256=sha()),
        output="/consumer" if consumer else "/producer",base_W0_input="/producer",
        hardware=dict(device="CPU" if role=="CPU_REDUCER" else "synthetic-unit-GPU",capability=[] if role=="CPU_REDUCER" else [8,6],
                      cross_hardware_bitwise_claim=False),slurm=dict(SLURM_JOB_ID="700" if consumer else "600"))


def state_fixture(method, calls=3):
    module=dict(FT="official.baselines.easyedit.models.ft.ft_main",MEMIT="official.baselines.sphere.memit.memit_main",
                MEMIT_FE="official.baselines.easyedit.models.memit_FE.memit_FE_main")[method]
    context=dict(schema="official-server1-native-context-v1",method=method,module=module,successful_calls=calls,
                 CONTEXT_TEMPLATES_CACHE=None if method=="FT" else [["{}"],["synthetic context {}"]])
    if method=="MEMIT":context["GLOBAL_EDIT_COUNT"]=calls
    context["identity_sha256"]=w._digest(context)
    state=dict(method=method,successful_calls=calls,cache_c={},contexts_sha256=context["identity_sha256"],
               selected_weights={"model.layers."+str(layer)+".mlp.down_proj.weight":dict(sha256=sha(),shape=[4096,14336],dtype="torch.float32")
                                 for layer in ((21,) if method=="FT" else (4,5,6,7,8))})
    state["identity_sha256"]=w._digest(state)
    tensor=lambda dtype,shape:dict(dtype=dtype,shape=shape,sha256=sha())
    rng=dict(python=[3,[0]*624+[624],None],numpy=["MT19937",tensor("uint32",[624]),624,0,0.0],
             torch_cpu=tensor("torch.uint8",[5056]),torch_cuda=[tensor("torch.uint8",[5056])])
    return state,context,rng


def sliced_endpoint(ep,n):
    value=copy.deepcopy(ep);value["cases"]=value["cases"][:n]
    signatures=w._signatures(value["cases"],"cf")
    value["identity"]["ordered_occurrences"]=list(range(1,n+1))
    value["identity"]["cohort_sha256"]=w._digest(signatures)
    value["identity_sha256"]=w._digest(value["identity"])
    value["summary"]=counterfact(value["cases"])
    value["accuracy"]={kind:factual._accuracy([row["target_true" if kind=="neighborhood" else "target_new"]
            for case in value["cases"] for row in case[kind+"_observations"]]) for kind in ("rewrite","paraphrase","neighborhood")}
    value["work"]=dict(factual.retained_query_work(signatures,"cf"),seconds=.001)
    return value


def reference(ep):
    identity=dict(schema=factual.W0_SCHEMA,tokenization=factual.TOKENIZATION,external_identity=ep["identity"]["external_identity"],
        cohort_sha256=ep["identity"]["cohort_sha256"],ordered_occurrences=list(range(1,2001)),state="W0_COLD_BASE_MODEL")
    value=dict(schema=factual.W0_SCHEMA,identity=identity,identity_sha256=w._digest(identity),evaluation=ep,raw_local_only=True,
        cases=[dict(case_id=row["case_id"],occurrence_index=row["occurrence_index"],
            queries=[query for query in sig["queries"] if query["kind"]=="neighborhood"],predictions=[[2]])
            for row,sig in zip(ep["cases"],w._signatures(ep["cases"],"zsre"))])
    value["payload_sha256"]=w._digest(value)
    return value


def generation(root, fp, execution, cf_records, *, assets=None):
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
        if assets is not None:
            from official.evaluation.generation.metrics import score_case
            observation["text"]="one two three"
            metrics=score_case([observation],["one two three"],assets.vectorizer,str.split)
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
        # Real tiny reference bytes and CPU metric code, with explicit fake
        # NLTK/runtime boundaries. No model/oracle/GPU observations occur.
        cls.scoring_fixture=type("LocalScoringFixture",(W0GenerationScoringTests,),{})
        cls.scoring_fixture.setUpClass()
        cls.asset_stack=cls.scoring_fixture.asset_fixture();cls.asset_stack.__enter__()
        data["references"]=copy.deepcopy(cls.scoring_fixture.fp["content"]["references"])
        data["runtime"]["versions"].update(cls.scoring_fixture.versions)
        cls.generation_assets=cls.scoring_fixture.descriptor
        cls.records={name:[dict(case_id=n+5000,occurrence_index=n,generation_prompts=["TEST_ONLY_PROMPT"],
            requested_rewrite=dict(prompt="{}",subject="TEST_ONLY_PROMPT",relation_id="R",
                target_new=dict(str="TEST_ONLY_TARGET",id="T"),target_true=dict(str="TEST_ONLY_TARGET",id="U")),
            paraphrase_prompts=["TEST_ONLY_PROMPT"],neighborhood_prompts=["TEST_ONLY_PROMPT"] if name=="cf"
                else [dict(prompt="TEST_ONLY_PROMPT",target="TEST_ONLY_TARGET")]) for n in range(1,2001)]
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
        cls.execution=execution_fixture(cls.fp)
        cls.consumer=execution_fixture(cls.fp,consumer=True)
        cls.gen,cls.runtime_member=generation(cls.root/"generation",cls.fp,cls.execution,cls.records["cf"],assets=cls.scoring_fixture.assets)
        proof={}
        for method in ("FT","MEMIT","MEMIT_FE"):
            identity=dict(model_revision=cls.fp["content"]["model"]["revision"],tokenizer_sha256=w._tokenizer_sha(cls.fp),
                code_commit=cls.execution["source"]["main_commit"],official_tree_sha256=cls.execution["source"]["official_tree"],
                stream_sha256=cls.fp["content"]["datasets"]["cf"]["stream_sha256"],config_sha256=sha(),assets_sha256=sha())
            ready=dict(schema="official-server1-native-resume-READY-v1",method=method,identity=identity,passed=True,
                actual_native_B3_and_B2_resume=True,actual_fit_calls=6,generation_calls=0,
                checks=dict(selected_weights="EXACT_SHA256",contexts="EXACT",RNG="EXACT",factual="EXACT_RAW_VALUES_AND_TOKEN_IDENTITY",no_tolerance_relaxation=True),
                native_parity=None)
            measured=sliced_endpoint(cls.cf,300)
            selected_state,context,rng=state_fixture(method)
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
                padded_input_tokens=3600,target_tokens=1800,seconds=.001)
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
                work=dict(live_measured["work"],forward_calls=300),
                evidence={"TEST_ONLY_CPU_FIXTURE":"NOT_OBSERVED","ACTUAL_GPU_SMOKE":"NOT_OBSERVED",
                          "ACTUAL_MATCHED_SUBSET":"NOT_OBSERVED","ACTUAL_FULL_2K":"NOT_OBSERVED"})
            comparison=dict(schema="official-cf-original-native-reference-v1",status="PASS",evidence_scope="ACTUAL_MATCHED_SUBSET",
                evidence={"TEST_ONLY_CPU_FIXTURE":"NOT_OBSERVED","ACTUAL_GPU_SMOKE":"NOT_OBSERVED","ACTUAL_MATCHED_SUBSET":"PASS","ACTUAL_FULL_2K":"NOT_OBSERVED"},
                mismatches=[],display_mismatches=[],raw_local_only=True,checkpoint_saved=False,scientific_performance_promotion=False,
                tolerances=dict(nll_abs_nats=1e-4,nll_relative_to_native=1e-5,strict_booleans="EXACT",aggregate_abs=1e-10),
                canonical_identity_sha256=live_measured["identity_sha256"],native=native,work=native["work"])
            parity=dict(schema="official-server1-native-CF-parity-report-v1",plan_sha256=w._digest(PLAN),
                external_identity=live_measured["identity"]["external_identity"],reference_binding=bind,canonical=live_measured,
                canonical_payload_sha256=w._digest(live_measured),canonical_payload_unchanged=True,canonical_observation="EXISTING_B3_FIRST300",
                additional_canonical_forward_calls=0,state_before=selected_state,state_after=selected_state,comparison=comparison,
                RNG_restored=True,input_records_unchanged=True,raw_local_only=True,performance_gate=False,additional_fit_calls=0,generation_calls=0)
            ready["native_parity"]=validate_report(parity,live_measured["identity"]["external_identity"],PLAN)
            assert ready["native_parity"]==w._native_parity_proof(parity,measured,selected_state)
            parity_member=put(cls.root/"qualification"/method/"B3-native-parity.json",parity)
            for key,name,batches,calls in (("continuous","continuous",3,3),("stopped","stop",2,2),("resumed","resume",3,1)):
                selected_state,context,rng=state_fixture(method,batches)
                ready[key]=put(cls.root/"qualification"/method/(key+".json"),dict(schema="official-server1-native-qualification-stage-v1",
                    stage=name,completed_batch=batches,actual_native_fit_calls=calls,GPU_actual=True,actual_model_loaded=True,
                    identity=identity,method=method,selected_state=selected_state,contexts_sha256=w._digest(context),
                    RNG_sha256=w._digest(rng),checkpoint_RNG_sha256=w._digest(rng),context_content=context,
                    RNG_content=rng,checkpoint_RNG_content=copy.deepcopy(rng),
                    execution_identity=execution_fixture(cls.fp,role="QUALIFICATION"),
                    factual_member=factual_member if key in ("continuous","resumed") else None,
                    native_parity_member=parity_member if key=="continuous" else None))
            proof[method]=put(cls.root/"qualification"/method/"READY.json",ready)
        cls.validation={name:dict(actual_complete=True,actual_model_edits=0,state="W0_COLD_BASE_MODEL",requests=2000,
            ordered_queries_sha256=cls.fp["content"]["datasets"]["zsre" if name=="zsre_reference" else "cf"]["ordered_queries_sha256"],
            native_qualification_state=dict(status="VALIDATED",evidence_members=proof,evidence_sha256=w._digest(proof))) for name in w.COMPONENTS}
        cls.members={name:put(cls.root/(name+".json"),value) for name,value in
            (("cf_factual",cls.cf),("cf_generation",cls.gen),("zsre_reference",cls.ref))}
        cls.ready=w.make_ready(producer_execution_identity=cls.execution,fingerprint=cls.fp,members=cls.members,
            component_validation=cls.validation,generation_runtime_member=cls.runtime_member,
            source_members=cls.source_members,dataset_members=cls.dataset_members,generation_assets=cls.generation_assets)
        cls.ready_path=cls.root/"READY.json";put(cls.ready_path,cls.ready)

    @classmethod
    def tearDownClass(cls):
        cls.asset_stack.__exit__(None,None,None)
        cls.scoring_fixture.tearDownClass()
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
        view=w.read_ready(self.ready_path,consumer_execution_identity=self.consumer,consumer_fingerprint=self.fp,generation_assets=self.generation_assets)
        self.assertEqual(view.values["cf_factual"],self.cf)
        self.assertFalse(view.binding["execution_identities_equal"])
        self.assertEqual(len([key for key in view.members if key.startswith("cf_generation.raw.")]),2000)
        self.assertEqual(view.zsre(consumer_external_identity=external(self.fp,consumer=True)).reference,self.ref)

    def test_explicit_copied_member_path_is_content_only_and_original_unchanged(self):
        path=self.root/"consumer-copy.json";path.write_bytes(Path(self.members["cf_factual"]["path"]).read_bytes())
        view=w.read_ready(self.ready_path,consumer_execution_identity=self.consumer,consumer_fingerprint=self.fp,
            member_paths={"cf_factual":str(path)},generation_assets=self.generation_assets)
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
            with self.assertRaises(w.ReferenceInputError):w.read_ready(path,consumer_execution_identity=self.consumer,consumer_fingerprint=self.fp,generation_assets=self.generation_assets)

    def test_mutated_source_bytes_with_same_json_and_consumer_file_sha_rejected(self):
        path=self.root/"byte-mutated.json";path.write_bytes(Path(self.members["cf_factual"]["path"]).read_bytes()+b" ")
        with self.assertRaisesRegex(w.ReferenceInputError,"BYTES_SHA"):
            w.read_ready(self.ready_path,consumer_execution_identity=self.consumer,consumer_fingerprint=self.fp,
                         member_paths={"cf_factual":str(path)},generation_assets=self.generation_assets)

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
            w._generation(self.gen,self.fp,{},self.runtime_member,records,collect=True,generation_assets=self.generation_assets)

    def test_generation_nonzero_cold_state_not_allowed_even_with_resigned_hashes(self):
        value=copy.deepcopy(self.gen);row=value["rows"][0]
        raw=json.loads(Path(row["observation_path"]).read_text());raw["identity"]["state_identity"]["actual_model_edits"]=1
        raw["identity_sha256"]=gd(raw["identity"]);raw["payload_sha256"]=gd({k:v for k,v in raw.items() if k!="payload_sha256"})
        new=put(self.root/"noncold-row.json",raw)
        row.update(observation_path=new["path"],identity_sha256=raw["identity_sha256"],payload_sha256=raw["payload_sha256"])
        row["provenance"]["raw_member"]=new
        with self.assertRaisesRegex(w.ReferenceInputError,"ACTUAL_COLD_MODEL_STATE"):
            w._generation(value,self.fp,{},self.runtime_member,self.records["cf"],collect=True,generation_assets=self.generation_assets)

    def test_actual_stage_without_measured_B3_or_native_proof_cannot_unlock_ready(self):
        validation=copy.deepcopy(self.validation)
        old=json.loads(Path(validation["cf_factual"]["native_qualification_state"]["evidence_members"]["FT"]["path"]).read_text())
        stage=json.loads(Path(old["continuous"]["path"]).read_text());stage["factual_member"]=None
        old["continuous"]=put(self.root/"unproven-stage.json",stage)
        proof=put(self.root/"unproven-READY.json",old)
        for row in validation.values():
            q=row["native_qualification_state"];q["evidence_members"]["FT"]=proof;q["evidence_sha256"]=w._digest(q["evidence_members"])
        with self.assertRaisesRegex(w.ReferenceInputError,"FACTUAL_PROOF_REQUIRED"):
            w._qualification(validation,self.fp,self.execution,{},collect=True,
                             cf_records=self.records["cf"],cf_signatures=w._signatures(self.cf["cases"],"cf"),expected_assets_sha256=sha())
        value=copy.deepcopy(self.validation);value["cf_factual"]["native_qualification_state"]["evidence_sha256"]=sha("f")
        with self.assertRaises(w.ReferenceInputError):w._component_validation(value,self.fp)

    def test_locked_cold_stream_cannot_be_replaced_by_mutually_resigned_other_stream_proofs(self):
        validation=copy.deepcopy(self.validation)
        original=validation["cf_factual"]["native_qualification_state"]["evidence_members"]["FT"]
        proof=json.loads(Path(original["path"]).read_text());proof["identity"]["stream_sha256"]=sha("f")
        # Re-signing the receipt does not make its OTHER stream the verified CF.
        changed=put(self.root/"other-stream-qualified.json",proof)
        for component in validation.values():
            component["native_qualification_state"]["evidence_members"]["FT"]=changed
            component["native_qualification_state"]["evidence_sha256"]=w._digest(component["native_qualification_state"]["evidence_members"])
        with self.assertRaisesRegex(w.ReferenceInputError,"ACTUAL_NATIVE_READY_IDENTITY"):
            w._qualification(validation,self.fp,self.execution,{},collect=True,cf_records=self.records["cf"],
                             cf_signatures=w._signatures(self.cf["cases"],"cf"),expected_assets_sha256=sha())

    def test_actual_dataset_query_target_order_not_just_case_ids(self):
        for key in ("subject","target_new","target_true"):
            records=copy.deepcopy(self.records["cf"])
            if key=="subject":records[0]["requested_rewrite"][key]="OTHER_QUERY"
            else:records[0]["requested_rewrite"][key]["str"]="OTHER_TARGET"
            with self.subTest(key=key),self.assertRaisesRegex(w.ReferenceInputError,"QUERY_OR_TARGET"):
                w._case_order(self.cf,records,"cf")

    def test_missing_or_zero_cold_work_cannot_unlock_complete_ready(self):
        for edit in (lambda e:e.pop("work"),lambda e:e["work"].update(forward_calls=0),
                     lambda e:e["work"].update(physical_input_tokens=0,target_tokens=0,candidate_sequences=0),
                     lambda e:e["work"].update(seconds=0)):
            value=copy.deepcopy(self.cf);edit(value)
            with self.assertRaises(w.ReferenceInputError):
                w._cold_component(value,"cf",self.fp,self.execution)

    def test_execution_provenance_missing_role_and_slurm_fake_claims_are_rejected(self):
        for field in ("source","config_sha256","assets_manifest","runtime","output","hardware","slurm"):
            value=copy.deepcopy(self.execution);value.pop(field)
            with self.subTest(field=field),self.assertRaises(w.ReferenceInputError):w._execution(value,"PRODUCER")
        for edit in (lambda e:e.update(role="CPU_REDUCER"),lambda e:e["hardware"].update(device="CPU"),
                     lambda e:e["slurm"].update(SLURM_JOB_ID="UNKNOWN")):
            value=copy.deepcopy(self.execution);edit(value)
            with self.assertRaises(w.ReferenceInputError):w._execution(value,"PRODUCER")

    def test_labels_or_malformed_weight_context_rng_content_cannot_replace_actual_state_receipt(self):
        selected,context,rng=state_fixture("FT")
        base=dict(selected_state=selected,context_content=context,contexts_sha256=w._digest(context),
                  RNG_content=rng,checkpoint_RNG_content=copy.deepcopy(rng),
                  RNG_sha256=w._digest(rng),checkpoint_RNG_sha256=w._digest(rng))
        w._state_content(base,method="FT",calls=3)
        for edit in (lambda e:e["selected_state"].update(selected_weights="EXACT_SHA256"),
                     lambda e:e["selected_state"].update(TEST_ONLY_NO_REAL_NATIVE_CALLS=True),
                     lambda e:e["context_content"].update(module="official.baselines.WRONG"),
                     lambda e:e["RNG_content"]["python"][1].__setitem__(-1,625),
                     lambda e:e["RNG_content"]["numpy"].__setitem__(3,True)):
            value=copy.deepcopy(base);edit(value)
            value["RNG_sha256"]=w._digest(value["RNG_content"])
            with self.assertRaises((w.ReferenceInputError,KeyError,TypeError)):
                w._state_content(value,method="FT",calls=3)

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


class W0GenerationScoringTests(unittest.TestCase):
    """Synthetic CPU reference/scoring fixtures; no native/model observation."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from official.evaluation.generation import assets as a
        from official.evaluation.generation.metrics import score_case
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        refs = cls.root / "references"
        snippets = put(refs / "attribute_snippets.json", [dict(relation_id="R", target_id="T", samples=[dict(text="one two three")])])
        vocab = put(refs / "tfidf_vocab.json", {"one": 0, "two": 1, "three": 2})
        np.save(refs / "idf.npy", np.ones(3, dtype=np.float64), allow_pickle=False)
        idf = w.member(refs / "idf.npy")
        resource = put(cls.root / "nltk_data/tokenizers/punkt_tab/english/collocations.tab", {"CPU_FIXTURE_ONLY": True})
        source = dict(put(cls.root / "nltk_package/tokenize/punkt.py", {"CPU_FIXTURE_ONLY": True}), relative="tokenize/punkt.py")
        versions = {key: "TEST_ONLY_SCORING_VERSION" for key in ("numpy", "scipy", "sklearn", "nltk")}
        cls.nltk = dict(nltk_version=versions["nltk"], language="english", required_family="punkt_tab",
                        required_resources=[resource], source_members=[source])
        manifest = dict(schema=a.SCHEMA, status="READY", files={"attribute_snippets.json": snippets, "idf.npy": idf, "tfidf_vocab.json": vocab},
            versions=versions, tokenizer=copy.deepcopy(cls.nltk),
            snippet_schema=dict(entries=1, relation_target_pairs=1, samples=1,
                                selection="all samples per exact relation_id/target_new.id, original order"),
            vocabulary_size=3, idf_shape=[3], idf_dtype="float64",
            fixed_vectorizer=dict(fit_calls=0, refit=False, class_name="sklearn.feature_extraction.text.TfidfVectorizer"))
        manifest["identity_sha256"] = a.digest(a._identity(manifest))
        cls.descriptor = dict(manifest=put(cls.root / "asset-manifest.json", manifest))
        cls.versions = versions
        data = content()
        data["runtime"]["versions"].update(versions)
        data["references"] = dict(identity_sha256=manifest["identity_sha256"],
            files={name: {key: row[key] for key in ("bytes", "sha256")} for name, row in manifest["files"].items()},
            nltk=dict(resources={"tokenizers/punkt_tab/english/collocations.tab": {key: resource[key] for key in ("bytes", "sha256")}},
                      sources={"tokenize/punkt.py": {key: source[key] for key in ("bytes", "sha256")}}),
            vectorizer=manifest["fixed_vectorizer"]["class_name"] + ":PUBLIC_IDF_SETTER_NO_FIT")
        cls.fp = w.computational_fingerprint(data)
        cls.records = [dict(case_id=5000 + index, occurrence_index=index, generation_prompts=["TEST_ONLY_PROMPT"],
                            requested_rewrite=dict(relation_id="R", target_new=dict(id="T"))) for index in range(1, 2001)]
        execution = dict(source=dict(main_commit="a" * 40, official_tree="b" * 40))
        cls.observed, cls.runtime_member = generation(cls.root / "generation", cls.fp, execution, cls.records)
        with cls.asset_fixture():
            cls.assets = a.load_assets(cls.descriptor)
        for row in cls.observed["rows"]:
            path = Path(row["observation_path"])
            raw = json.loads(path.read_text())
            raw["observations"][0]["text"] = "one two three"
            raw["metrics"] = score_case(raw["observations"], ["one two three"], cls.assets.vectorizer, str.split)
            raw["payload_sha256"] = gd({key: value for key, value in raw.items() if key != "payload_sha256"})
            original = put(path, raw)
            row.update(metrics=copy.deepcopy(raw["metrics"]), payload_sha256=raw["payload_sha256"])
            row["provenance"]["raw_member"] = original
        cls.observed["summary"] = reduce_cases(cls.observed["rows"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def asset_fixture(cls, nltk=None):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("official.evaluation.generation.assets.dependency_versions", return_value=cls.versions))
        stack.enter_context(patch("official.evaluation.generation.assets.nltk_binding", return_value=copy.deepcopy(cls.nltk if nltk is None else nltk)))
        stack.enter_context(patch("official.evaluation.generation.assets._nltk_callable", return_value=str.split))
        return stack

    def verify(self, value=None, *, descriptor=None, fingerprint=None, overrides=None):
        with self.asset_fixture():
            return w._generation(self.observed if value is None else value, self.fp if fingerprint is None else fingerprint,
                {}, self.runtime_member, self.records, collect=True, overrides=overrides,
                generation_assets=self.descriptor if descriptor is None else descriptor)

    def altered_raw(self, mutate, label):
        endpoint = copy.deepcopy(self.observed)
        row = endpoint["rows"][0]
        raw = json.loads(Path(row["observation_path"]).read_text())
        mutate(raw)
        raw["payload_sha256"] = gd({key: value for key, value in raw.items() if key != "payload_sha256"})
        receipt = put(self.root / "adversarial" / (self._testMethodName + "-" + label + ".json"), raw)
        row.update(observation_path=receipt["path"], payload_sha256=raw["payload_sha256"], metrics=copy.deepcopy(raw["metrics"]))
        row["provenance"]["raw_member"] = receipt
        endpoint["summary"] = reduce_cases(endpoint["rows"])
        return endpoint

    def test_actual_cpu_assets_score_and_observation_counts_are_consumed(self):
        receipts = self.verify()
        self.assertIn("reference.manifest", receipts)
        self.assertIn("reference.files.idf.npy", receipts)
        self.assertIn("reference.nltk.sources.tokenize/punkt.py", receipts)
        self.assertEqual(self.observed["summary"]["generation_prompt_count"], 2000)
        self.assertEqual(self.observed["summary"]["generated_token_count"], 0)
        self.assertEqual(self.observed["summary"]["length_cap_no_continuation_prompt_count"], 2000)
        self.assertEqual(self.observed["summary"]["fluency_count"], 2000)
        self.assertEqual(self.observed["summary"]["consistency_count"], 2000)
        # Keep native cosine rounding; do not clip a measured 1+1 ULP.
        self.assertEqual(self.observed["summary"]["reference_score"],
                         self.observed["rows"][0]["metrics"]["reference_score"])

    def test_missing_assets_and_untrusted_preloaded_objects_fail_closed(self):
        for descriptor in (None, self.assets, {}, dict(manifest=self.descriptor["manifest"], allow_missing=True)):
            with self.asset_fixture(), self.assertRaises(w.ReferenceInputError):
                w._generation(self.observed, self.fp, {}, self.runtime_member, self.records, collect=True,
                              generation_assets=descriptor)

    def test_resigned_fabricated_score_counts_cap_and_missingness_rejected(self):
        mutations = {
            "prompts": lambda raw: raw["metrics"].update(generation_prompt_count=17),
            "tokens": lambda raw: raw["metrics"].update(generated_token_count=42),
            "cap": lambda raw: raw["metrics"].update(length_cap_no_continuation_count=0),
            "consistency": lambda raw: raw["metrics"].update(reference_score=999.0),
            "fluency": lambda raw: raw["metrics"].update(ngram_entropy=999.0),
            "missing": lambda raw: raw["metrics"].update(reasons=["missing_reference"]),
            "validity": lambda raw: raw["metrics"].update(fluency_valid=False),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), self.assertRaises(w.ReferenceInputError):
                self.verify(self.altered_raw(mutate, label))

    def test_equal_bool_float_count_spoofs_rejected_without_numeric_relaxation(self):
        mutations = {
            "metric_bool_count": lambda raw: raw["metrics"].update(generation_prompt_count=True),
            "metric_integer_flag": lambda raw: raw["metrics"].update(fluency_valid=1),
            "observation_float": lambda raw: raw["observations"][0].update(continuation_token_count=0.0),
            "observation_bool": lambda raw: raw["observations"][0].update(prompt_index=False),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), self.assertRaises(w.ReferenceInputError):
                self.verify(self.altered_raw(mutate, label))

    def test_consumed_reference_nltk_source_and_resource_sha_mismatch_rejected(self):
        for key in ("files", "nltk_resource", "nltk_source", "versions", "vectorizer"):
            fp = copy.deepcopy(self.fp)
            if key == "files":
                fp["content"]["references"]["files"]["idf.npy"]["sha256"] = sha("f")
            elif key == "nltk_resource":
                fp["content"]["references"]["nltk"]["resources"]["tokenizers/punkt_tab/english/collocations.tab"]["sha256"] = sha("f")
            elif key == "nltk_source":
                fp["content"]["references"]["nltk"]["sources"]["tokenize/punkt.py"]["sha256"] = sha("f")
            elif key == "versions":
                fp["content"]["runtime"]["versions"]["nltk"] = "OTHER_CPU_FIXTURE_VERSION"
            else:
                fp["content"]["references"]["vectorizer"] = "NOT_THE_CONSUMED_VECTORIZER"
            fp["sha256"] = w._digest(fp["content"])
            with self.subTest(key=key), self.assertRaises(w.ReferenceInputError):
                self.verify(fingerprint=fp)

    def test_wrong_reference_override_bytes_and_extra_override_keys_rejected(self):
        changed = put(self.root / "wrong-snippets.json", [])
        for overrides in ({"attribute_snippets.json": changed["path"]}, {"NLTK": changed["path"]}):
            descriptor = dict(self.descriptor, asset_paths=overrides)
            with self.assertRaises(w.ReferenceInputError):
                self.verify(descriptor=descriptor)

    def test_local_manifest_copy_preserves_original_descriptor_and_producer_receipt(self):
        original = copy.deepcopy(self.descriptor)
        target = self.root / "consumer-local-manifest.json"
        target.write_bytes(Path(original["manifest"]["path"]).read_bytes())
        receipts = self.verify(overrides={"reference.manifest": str(target)})
        receipt = receipts["reference.manifest"]
        self.assertEqual(receipt["producer_member"], original["manifest"])
        self.assertEqual(receipt["consumed_member"], w.member(target))
        self.assertTrue(receipt["path_differs"])
        self.assertEqual(self.descriptor, original)
        target.write_text('{"CPU_FIXTURE_CHANGED_MANIFEST":true}')
        with self.assertRaises(w.ReferenceInputError):
            self.verify(overrides={"reference.manifest": str(target)})

    def test_actual_nltk_source_and_resource_bytes_independently_rehashed(self):
        for kind, key, relative in (("source", "source_members", "tokenize/punkt.py"),
                                   ("resource", "required_resources", "tokenizers/punkt_tab/english/collocations.tab")):
            actual = copy.deepcopy(self.nltk)
            target = self.root / ("bad-local-nltk-" + kind) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("CPU_FIXTURE_BYTES_CHANGED_BUT_STALE_METADATA")
            # Stale original SHA labels alone must not pass the consumed proof.
            actual[key][0]["path"] = str(target)
            with self.asset_fixture(nltk=actual), self.assertRaises(w.ReferenceInputError):
                w._generation(self.observed, self.fp, {}, self.runtime_member, self.records,
                              collect=True, generation_assets=self.descriptor)


if __name__=="__main__":unittest.main()
