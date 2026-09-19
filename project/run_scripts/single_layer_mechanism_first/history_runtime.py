"""Task-local canonical-history prefix cache, bounded to one request.

First use captures only the provided canonical new/old paths. Subsequent
Current/Past candidate checks and factor contractions load frozen CPU prefix
artifacts with zero new prefix/model forward. The L4 output weight may change;
all upstream/nonselected state, model/token/source/input bindings must not.
No object is registered in Runtime.oracles and no all-Past tensor bank remains
resident. Shared native/alltoken/tokenizer modules are imported read-only.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import torch

from .config import ROOT
from .decision import DecisionError, _json_sha, _require
from .history import rewrite, valid_target
from project.run_scripts.single_layer_edit_preserving_correction import binding as inherited_binding
from project.run_scripts.single_layer_edit_preserving_correction import alltoken
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import (
    FullTokenCache, FullWeightLlamaOracle, PACK_FIELDS, WEIGHT, model_guard, tensor_sha256)


def _file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""):h.update(block)
    return h.hexdigest()


def _stat(path):
    p=Path(path);s=p.lstat()
    _require(stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_uid==os.getuid(),
             "HISTORY_CACHE_OWNED_REGULAR_FILE_REQUIRED")
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,
                ctime_ns=s.st_ctime_ns,owner=s.st_uid)


def _sync_directory(directory):
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _publish(path,payload,*,tensor=False):
    """Create once; incomplete temp artifacts survive any failed publication."""
    path=Path(path)
    _require(not path.exists() and not path.is_symlink(),"HISTORY_CACHE_OVERWRITE_FORBIDDEN")
    temporary=path.with_name(path.name+f".partial-{os.getpid()}-{time.time_ns()}")
    with temporary.open("xb") as handle:
        if tensor:torch.save(payload,handle)
        else:handle.write((json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())
        handle.flush();os.fsync(handle.fileno())
    os.link(temporary,path)
    os.unlink(temporary)  # Only this invocation's newly published temp hardlink.
    _sync_directory(path.parent)
    return dict(path=str(path),sha256=_file_sha(path),**_stat(path))


def _nonselected(guard):
    return tuple(x for x in guard[0] if x[1]!=WEIGHT),guard[1]


def _selected_fingerprint(weight):
    return (weight.data_ptr(),weight._version,tuple(weight.shape),str(weight.dtype),str(weight.device))


def canonical_paths(rt,record):
    """The exact inherited canonical token contract, without native contexts."""
    request=deepcopy(rewrite(record))
    request["case_id"]=record["case_id"]
    _require(valid_target(record),"HISTORY_NEW_TARGET_REQUIRED")
    contracts=inherited_binding._token_contracts()
    prompt=request["prompt"].format(request["subject"])
    prefix=contracts.prompt_token_ids(rt.etok,prompt)
    packs=[];rows=[];lookup={}
    for branch,key in (("new","target_new"),("old","target_true")):
        text=request.get(key,{}).get("str")
        if not text:
            _require(branch=="old","HISTORY_NEW_TARGET_REQUIRED")
            continue
        labels=contracts.target_token_ids(rt.etok,text)
        ids=tuple(prefix+labels[:-1])
        if ids not in lookup:
            lookup[ids]=len(packs);packs.append(inherited_binding.pack(ids))
        rows.append(dict(case_id=record["case_id"],kind="canonical",branch=branch,
            context=0,cache=lookup[ids],positions=list(range(len(prefix)-1,len(ids))),
            labels=list(labels),sequence_id=f'{record["case_id"]}:canonical:{branch}'))
    _require(1<=len(packs)<=2 and 1<=len(rows)<=2,"HISTORY_MAX_TWO_CANONICAL_PATHS")
    return packs,rows


class FrozenHistoryOracle(FullWeightLlamaOracle):
    """Inherited physical suffix/head with an optional no-forward prefix load."""
    def __init__(self,model,packs,*,retained=None):
        self._closed=False
        self._retained=retained
        self._load_index=0
        super().__init__(model,packs,teacher_loader=None,score_slice=(0,1),
                         require_reference_length=None,head_chunk_positions=16)
        if retained is not None:
            _require(self._load_index==len(retained),"HISTORY_RETAINED_CACHE_CARDINALITY")
        self._retained=None

    def _guard(self):
        _require(not self._closed,"HISTORY_ORACLE_CLOSED")
        super()._guard()

    def _prepare(self,source):
        if self._retained is None:
            return super()._prepare(source)
        _require(self._load_index<len(self._retained),"HISTORY_RETAINED_PATH_MISSING")
        stored=self._retained[self._load_index];self._load_index+=1
        packed=self._pack(source)
        _require(all(torch.equal(packed[k],stored.packed[k]) for k in PACK_FIELDS),
                 "HISTORY_RETAINED_INPUT_BYTES_MISMATCH")
        _require(stored.keys.device.type=="cpu" and stored.residual.device.type=="cpu" and
                 stored.keys.dtype==torch.float32 and stored.residual.dtype==torch.float32 and
                 stored.keys.shape==(*packed["input_ids"].shape,self.shape[1]) and
                 stored.residual.shape==(*packed["input_ids"].shape,self.shape[0]),
                 "HISTORY_RETAINED_CACHE_SCHEMA")
        self.work.setdefault("retained_prefix_loads",0)
        self.work["retained_prefix_loads"]+=1
        # Complete file SHA + finite/tensor hashes are validated once per file
        # per process by HistoryFactory. Do not rehash each candidate/row.
        return FullTokenCache(packed,stored.keys,stored.residual,deepcopy(stored.input_identity))

    def close(self):
        try:self._guard()
        finally:
            self.caches.clear()
            self.cache_guard=()
            self._retained=None
            self._closed=True


class HistoryFactory:
    """Callable request context for StreamingHistoryOracle.

    Keep ``directory`` stable for this same frozen source/upstream across
    sequential batches if prefixes should be reused. Membership is local to
    the supplied active records; the persistent store's source binding does
    not depend on the current subset or current edited L4 bytes.
    """
    def __init__(self,rt,records,directory):
        self.rt=rt
        self.directory=Path(directory)
        _require(self.directory.resolve().is_relative_to(ROOT.resolve()) and
                 not self.directory.is_symlink(),"HISTORY_CACHE_TASK_LOCAL_DIRECTORY")
        self.directory.mkdir(parents=True,exist_ok=True)
        self.requests_directory=self.directory/"requests"
        self.requests_directory.mkdir(exist_ok=True)
        _require(not self.requests_directory.is_symlink(),"HISTORY_REQUEST_DIR_SYMLINK")
        self.records={str(r["case_id"]):deepcopy(r) for r in records}
        _require(len(self.records)==len(records) and len(records)<=900 and
                 all(valid_target(r) for r in records),"HISTORY_ACTIVE_RECORDS_REQUIRED")
        self.record_identities={k:_json_sha(v) for k,v in self.records.items()}
        rt.guard()  # One binding boundary; do not rescan dense M for every Past.
        self._nonselected_guard=_nonselected(model_guard(rt.model))
        self._selected_bindings={}
        self._known={}
        self._active=None
        self.work=dict(request_contexts=0,new_request_prefix_artifacts=0,
            request_prefix_cache_hits=0,new_prefix_forwards=0,new_prefix_valid_tokens=0,
            new_prefix_input_tokens=0,prefix_CPU_payload_bytes_written=0,
            cache_bytes_loaded=0,complete_file_hash_checks=0,complete_file_hash_bytes=0,
            selected_weight_hash_calls=0,selected_weight_hash_D2H_bytes=0,
            peak_resident_requests=0,peak_resident_paths=0,peak_resident_prefix_bytes=0,
            live_requests=0,live_paths=0,official_P_N_reads=0,native_history_appends=0,
            runtime_oracles_registrations=0,failed_contexts=0,
            cleanup_guard_failures=0,
            cache_load_seconds=0.,cache_store_seconds=0.,new_prefix_seconds=0.,
            request_context_seconds=0.)
        self._oracle_work={}
        self.source_binding=self._source_binding()
        self._lock_model_binding=deepcopy(self.source_binding["model"])
        self.source_identity=_json_sha(self.source_binding)
        self.creation_selected=self._selected_binding()
        self._tokenizer_guard=self._tokenizer_settings()
        identity_file=self.directory/"prefix-source-binding.json"
        if identity_file.exists():
            _stat(identity_file)
            loaded=json.loads(identity_file.read_text())
            _require(loaded==dict(schema_version=1,identity=self.source_identity,binding=self.source_binding),
                     "HISTORY_PREFIX_SOURCE_BINDING_MISMATCH")
        else:
            _publish(identity_file,dict(schema_version=1,identity=self.source_identity,binding=self.source_binding))
        self._binding_file_stat=_stat(identity_file)
        self._guard()

    def _tokenizer_settings(self):
        return {key:getattr(self.rt.etok,key,None) for key in
                ("name_or_path","padding_side","pad_token_id","bos_token_id","eos_token_id",
                 "unk_token_id","add_bos_token","add_eos_token")}

    def _source_binding(self):
        lock=self.rt.lock
        required=("model_revision","model_config_sha256","model_weights_identity_sha256",
                  "tokenizer_identity_sha256","torch","transformers","transformers_import")
        _require(all(lock.get(k) for k in required),"HISTORY_REQUIRED_MODEL_SOURCE_LOCK")
        _require(self.rt.identity.get("W0") is not None,"HISTORY_W0_SOURCE_BINDING_REQUIRED")
        contracts=inherited_binding._token_contracts()
        paths={"alltoken":Path(alltoken.__file__),"binding":Path(inherited_binding.__file__),
               "token_contract":Path(contracts.__file__),"history_runtime":Path(__file__)}
        return dict(model={key:lock[key] for key in required},physical_module=WEIGHT,
            fixed_dtype="torch.float32",attention="eager",matmul_tf32=False,cudnn_tf32=False,
            tokenizer_settings=self._tokenizer_settings(),
            source_sha256={key:_file_sha(path) for key,path in paths.items()},
            runtime_W0_binding=deepcopy(self.rt.identity.get("W0")),
            upstream_contract="ALL_NONSEL_PARAMS_FIXED;L4_DOWN_PROJ_INPUT_AND_PRE_MLP_RESIDUAL_INDEPENDENT_OF_EDITED_L4",
            actual_stationarity_validation="CALLER_T0_EVIDENCE_REQUIRED;NOT_ASSIGNED_BY_CACHE")

    def _selected_binding(self):
        fingerprint=_selected_fingerprint(self.rt.W)
        if fingerprint not in self._selected_bindings:
            _require(self.rt.W.dtype==torch.float32 and bool(torch.isfinite(self.rt.W).all()),
                     "HISTORY_PHYSICAL_SELECTED_FINITE_FP32")
            self._selected_bindings[fingerprint]=dict(weight_sha256=tensor_sha256(self.rt.W),
                shape=list(self.rt.W.shape),dtype=str(self.rt.W.dtype),
                physical_parameter_version=int(self.rt.W._version))
            self.work["selected_weight_hash_calls"]+=1
            if self.rt.W.device.type=="cuda":
                self.work["selected_weight_hash_D2H_bytes"]+=self.rt.W.numel()*self.rt.W.element_size()
        return deepcopy(self._selected_bindings[fingerprint])

    def _guard(self):
        _require(_nonselected(model_guard(self.rt.model))==self._nonselected_guard,
                 "HISTORY_UPSTREAM_OR_NONSEL_MODEL_CHANGED")
        _require({key:self.rt.lock.get(key) for key in self._lock_model_binding}==self._lock_model_binding,
                 "HISTORY_RUNTIME_MODEL_LOCK_CHANGED")
        _require(not self.rt.W.requires_grad and all(p.grad is None for p in self.rt.model.parameters()),
                 "HISTORY_MODEL_GRAD_STATE_CHANGED")
        _require(self._tokenizer_settings()==self._tokenizer_guard,"HISTORY_TOKENIZER_SETTINGS_CHANGED")
        _require(_stat(self.directory/"prefix-source-binding.json")==self._binding_file_stat,
                 "HISTORY_SOURCE_BINDING_FILE_CHANGED")

    def _request_identity(self,record,packs,rows):
        return _json_sha(dict(prefix_source_identity=self.source_identity,case_id=record["case_id"],
            record_identity=self.record_identities[str(record["case_id"])],rows=rows,
            packed=[{key:pack[key].tolist() for key in PACK_FIELDS} for pack in packs]))

    def _load(self,key,packs,rows):
        path=self.requests_directory/f"{key}.pt"
        meta_path=self.requests_directory/f"{key}.json"
        _require(path.exists() and meta_path.exists(),"PARTIAL_HISTORY_PREFIX_CACHE_PRESERVED")
        stat_now,meta_stat=_stat(path),_stat(meta_path)
        first=key not in self._known
        if first:
            metadata=json.loads(meta_path.read_text())
            _require(metadata["request_identity"]==key and metadata["prefix_source_identity"]==self.source_identity
                     and metadata["complete"] is True and metadata["artifact"]["bytes"]==stat_now["bytes"],
                     "HISTORY_PREFIX_MANIFEST_IDENTITY")
            started=time.perf_counter()
            _require(_file_sha(path)==metadata["artifact"]["sha256"],"HISTORY_PREFIX_FILE_SHA")
            self.work["complete_file_hash_checks"]+=1
            self.work["complete_file_hash_bytes"]+=stat_now["bytes"]
        else:
            known=self._known[key]
            _require(stat_now==known["stat"] and meta_stat==known["meta_stat"],"HISTORY_PREFIX_FILE_CHANGED")
            metadata=known["metadata"]
        started=time.perf_counter()
        payload=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        _require(payload["request_identity"]==key and payload["prefix_source_identity"]==self.source_identity
                 and payload["rows"]==rows and len(payload["caches"])==len(packs),"HISTORY_PREFIX_PAYLOAD_IDENTITY")
        restored=[]
        for index,item in enumerate(payload["caches"]):
            packed=item["packed"];k,r=item["keys"],item["residual"]
            _require(set(packed)==set(PACK_FIELDS) and all(torch.equal(packed[name],packs[index][name]) for name in PACK_FIELDS),
                     "HISTORY_PREFIX_PACK_MISMATCH")
            _require(k.device.type=="cpu" and r.device.type=="cpu" and k.dtype==torch.float32 and r.dtype==torch.float32
                     and k.shape==(*packed["input_ids"].shape,self.rt.W.shape[1])
                     and r.shape==(*packed["input_ids"].shape,self.rt.W.shape[0]),"HISTORY_PREFIX_TENSOR_SCHEMA")
            if first:
                _require(bool(torch.isfinite(k).all()) and bool(torch.isfinite(r).all()),"NONFINITE_HISTORY_PREFIX_CACHE")
                _require(tensor_sha256(k)==item["input_identity"]["keys_sha256"] and
                         tensor_sha256(r)==item["input_identity"]["residual_sha256"],"HISTORY_PREFIX_TENSOR_SHA")
            restored.append(FullTokenCache(packed,k,r,deepcopy(item["input_identity"])))
        _require(_stat(path)==stat_now and _stat(meta_path)==meta_stat,
                 "HISTORY_PREFIX_FILE_CHANGED_DURING_LOAD")
        if first:self._known[key]=dict(metadata=metadata,stat=stat_now,meta_stat=meta_stat)
        self.work["cache_bytes_loaded"]+=stat_now["bytes"]
        self.work["cache_load_seconds"]+=time.perf_counter()-started
        return restored,metadata

    def _store(self,key,rows,oracle):
        started=time.perf_counter()
        caches=[dict(packed={k:v.detach().cpu() for k,v in c.packed.items()},keys=c.keys.detach().cpu(),
                     residual=c.residual.detach().cpu(),input_identity=deepcopy(c.input_identity)) for c in oracle.caches]
        creator=dict(execution_commit=self.rt.lock.get("execution",{}).get("commit"),
                     lock_identity=self.rt.lock.get("lock_identity"),physical_selected=self._selected_binding(),
                     factory_creation_selected=deepcopy(self.creation_selected))
        artifact=_publish(self.requests_directory/f"{key}.pt",dict(schema_version=1,request_identity=key,
            prefix_source_identity=self.source_identity,rows=deepcopy(rows),caches=caches,creator=creator),tensor=True)
        payload_bytes=sum(t.numel()*t.element_size() for c in oracle.caches for t in (*c.packed.values(),c.keys,c.residual))
        metadata=dict(schema_version=1,request_identity=key,prefix_source_identity=self.source_identity,
            complete=True,artifact=artifact,creator=creator,canonical_paths=len(caches),canonical_rows=len(rows),
            prefix_tensor_bytes=payload_bytes,input_identities=[deepcopy(c.input_identity) for c in oracle.caches],
            prefix_work=deepcopy(oracle.work),official_P_N_or_native_augmentation=False)
        meta_path=self.requests_directory/f"{key}.json"
        _publish(meta_path,metadata)
        self._known[key]=dict(metadata=metadata,stat=_stat(artifact["path"]),meta_stat=_stat(meta_path))
        self.work["new_request_prefix_artifacts"]+=1
        self.work["new_prefix_forwards"]+=oracle.work["prefix_forwards"]
        self.work["new_prefix_valid_tokens"]+=oracle.work["prefix_valid_tokens"]
        self.work["new_prefix_input_tokens"]+=oracle.work["prefix_input_tokens"]
        self.work["new_prefix_seconds"]+=oracle.work["prefix_seconds"]
        self.work["prefix_CPU_payload_bytes_written"]+=payload_bytes
        self.work["cache_store_seconds"]+=time.perf_counter()-started
        return metadata

    def _guard_file(self,key):
        known=self._known[key]
        _require(_stat(self.requests_directory/f"{key}.pt")==known["stat"] and
                 _stat(self.requests_directory/f"{key}.json")==known["meta_stat"],
                 "HISTORY_PREFIX_FILE_CHANGED_DURING_CONTEXT")

    @contextmanager
    def __call__(self,record):
        _require(self._active is None,"HISTORY_CONCURRENT_REQUEST_CONTEXT_FORBIDDEN")
        case=str(record["case_id"])
        _require(case in self.records and _json_sha(record)==self.record_identities[case],
                 "HISTORY_RECORD_NOT_IN_BOUND_ACTIVE_SET")
        self._guard()
        packs,rows=canonical_paths(self.rt,record)
        key=self._request_identity(record,packs,rows)
        path=self.requests_directory/f"{key}.pt";meta_path=self.requests_directory/f"{key}.json"
        started=time.perf_counter();oracle=None;body_error=None
        list_before=tuple(id(x) for x in self.rt.oracles)
        self._active=key;self.work["live_requests"]=1
        try:
            if path.exists() or meta_path.exists():
                retained,metadata=self._load(key,packs,rows)
                oracle=FrozenHistoryOracle(self.rt.model,packs,retained=retained)
                del retained
                _require(oracle.work["prefix_forwards"]==0,"HISTORY_RELOAD_RERAN_PREFIX")
                self.work["request_prefix_cache_hits"]+=1
            else:
                oracle=FrozenHistoryOracle(self.rt.model,packs)
                metadata=self._store(key,rows,oracle)
            self.work["request_contexts"]+=1
            self.work["live_paths"]=len(oracle.caches)
            self.work["peak_resident_requests"]=max(self.work["peak_resident_requests"],1)
            self.work["peak_resident_paths"]=max(self.work["peak_resident_paths"],len(oracle.caches))
            self.work["peak_resident_prefix_bytes"]=max(self.work["peak_resident_prefix_bytes"],metadata["prefix_tensor_bytes"])
            oracle.history_cache_binding=dict(request_identity=key,prefix_source_identity=self.source_identity,
                creator=metadata["creator"],current_physical_selected=self._selected_binding(),
                selected_weight_equality_to_creator_required=False,
                guard_ack="FRESH_CURRENT_PARAMETER_GUARD_AND_FIXED_NONSEL_STATE;NO_AUTOMATIC_FAILURE_REBIND")
            yield oracle,deepcopy(rows)
            oracle._guard()
            self._guard_file(key)
        except BaseException as exc:
            body_error=exc
            self.work["failed_contexts"]+=1
            raise
        finally:
            try:
                if oracle is not None:
                    for name,value in oracle.work.items():
                        if isinstance(value,(int,float)):
                            self._oracle_work[name]=self._oracle_work.get(name,0)+value
                    try:oracle.close()
                    except BaseException as close_error:
                        self.work["cleanup_guard_failures"]+=1
                        if body_error is None:raise
                        body_error.add_note(f"History close guard also failed: {close_error!r}")
            finally:
                self._active=None;self.work["live_requests"]=0;self.work["live_paths"]=0
                self.work["request_context_seconds"]+=time.perf_counter()-started
                if tuple(id(x) for x in self.rt.oracles)!=list_before:
                    leak=DecisionError("HISTORY_RT_ORACLES_LIFETIME_LEAK")
                    if body_error is None:raise leak
                    body_error.add_note(str(leak))

    @property
    def receipt(self):
        return dict(schema_version=1,prefix_source_identity=self.source_identity,
            source_binding=deepcopy(self.source_binding),active_request_count=len(self.records),
            active_record_order_sha256=_json_sha(list(self.record_identities.values())),
            work=deepcopy(self.work),oracle_work=deepcopy(self._oracle_work),
            cache_members=[dict(request_identity=key,path=value["metadata"]["artifact"]["path"],
                sha256=value["metadata"]["artifact"]["sha256"],bytes=value["metadata"]["artifact"]["bytes"],
                canonical_paths=value["metadata"]["canonical_paths"]) for key,value in self._known.items()],
            payload_verification="FULL_FILE_AND_TENSOR_SHA_ON_FIRST_EXISTING_FILE_LOAD;SUBSEQUENT_OWNED_INODE_SIZE_TIME_BINDING",
            source_or_model_stationarity_actual_PASS_assigned=False,
            teacher_reads=0,official_observer_reads=0,native_history_appends=0,
            timer_nesting="request_context inclusive; do not add store/prefix/load subphases to it")
