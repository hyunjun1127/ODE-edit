"""Read-only full-SHA inventory of approved completed GPT-J zsRE checkpoints."""
import hashlib
import json
from pathlib import Path

ROOT = Path("/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/registration-no-gpu-qual-r1")
JOBS = dict(FT=61726, MEMIT=61728, ALPHAEDIT=61730, ALPHAEDIT_BLUE=61732, MEMIT_FE=61734, SPHERE=61735)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()

def member(path):
    path=Path(path)
    before=path.stat()
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(block)
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError("SOURCE_CHANGED")
    return dict(path=str(path),bytes=after.st_size,sha256=h.hexdigest(),device=after.st_dev,inode=after.st_ino,mtime_ns=str(after.st_mtime_ns))

def inventory():
    manifest=json.loads((ROOT/"manifest.json").read_text())
    rows=[]
    objects=set()
    for method,job in JOBS.items():
        folder=ROOT/("ZSRE_"+method)
        row=dict(method=method,original_job_id=job,model="gptj",dataset="zsre")
        try:
            result=json.loads((folder/"result.json").read_text())
            if result["status"]!="SCIENTIFIC_COMPLETE" or result["batches"]!=20 or len(result["commits"])!=20:
                raise ValueError("NOT_COMPLETE_W20")
            cp=Path(result["checkpoint_folder"])
            ref=json.loads((cp/"latest.json").read_text())
            identity=result["checkpoint_identity"]
            if ref["batch"]!=20 or ref["final_W20"] is not True or ref["identity_sha256"]!=digest(identity):
                raise ValueError("FINAL_IDENTITY_MISMATCH")
            if Path(ref["file"]).name!=ref["file"] or (cp/ref["file"]).is_symlink():
                raise ValueError("CHECKPOINT_MEMBER_SCOPE")
            payload=member(cp/ref["file"])
            if payload["sha256"]!=ref["sha256"]:
                raise ValueError("CHECKPOINT_SHA_MISMATCH")
            obj=(payload["device"],payload["inode"])
            if obj in objects:
                raise ValueError("DUPLICATE_CHECKPOINT_OBJECT")
            objects.add(obj)
            row.update(status="FINAL_W20_FULLSHA_VERIFIED",checkpoint=payload,latest=member(cp/"latest.json"),
                result=member(folder/"result.json"),identity=identity,original_source=result["code_commit"],
                checkpoint_folder=str(cp),original_manifest=member(ROOT/"manifest.json"))
        except Exception as error:
            row.update(status="MISSING_OR_INCOMPLETE",error_type=type(error).__name__,error=str(error))
        rows.append(row)
    return dict(schema="server2-zsre-saved-weights-inventory-v1",
        instruction_id="USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1",rows=rows,
        source_input_status="SOURCE_INPUT_PENDING",model_loads=0,GPU=0,checkpoint_deserializations=0,
        original_artifacts_unchanged=True,original_manifest_path=str(ROOT/"manifest.json"),
        model_revision=manifest.get("model_revision"),model_snapshot=manifest.get("model_snapshot"),
        tokenizer_sha256=manifest.get("tokenizer_sha256"),streams={key:manifest["streams"]["zsre"][key]
            for key in ("path","member","source","lock_member")},
        stat_encoding="mtime_ns decimal string avoids JSON consumer IEEE754 precision loss")

if __name__=="__main__":
    print(json.dumps(inventory(),indent=2))
