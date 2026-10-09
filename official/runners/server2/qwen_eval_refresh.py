"""Future Qwen caller binding to GH public zsRE queries; prior W0 stays historical."""
from pathlib import Path
from official.experiments.prepare import digest,file_sha,read

def require(x,c):
    if not x:raise ValueError(c)

def planned(order,kept_ids):
    lanes=[kept_ids['cf'],kept_ids['zsre'],None,None]
    result=[]
    for i,row in enumerate(order,2):
        ds=row['config']['dataset'];lane=i%4
        deps=sorted({x for x in (lanes[lane],kept_ids[ds]) if x})
        result.append((row,lane,deps));lanes[lane]=row['logical_main_row']
    return result

def public_zsre(model,tokenizer,records,identity):
    from official.evaluation.zsre_paper import evaluate
    observed=evaluate(model,tokenizer,records,model_family="qwen25",batch_size=16,identity=identity)
    # The new query API returns official request macro only. Do not invent
    # old multi-token TF diagnostic observations or W0 agreement.
    observed["accuracy"]={}
    observed["legacy_accuracy_status"]="NOT_PROVIDED_BY_PUBLIC_QUERY_API"
    observed["work"]=dict(observed["work"], evaluation_profile="zsre-public-query-v1",
        query_sha256=observed["query_sha256"], token_denominators=observed["token_denominators"])
    return observed

def w0_folder(root,dataset):
    root=Path(root)
    plan=root/"w0-parent.json"
    if not plan.exists():return root/"shared-w0"/("qwen25-"+dataset)
    binding=read(plan)
    require(binding["producer_source"]=="69bfbb2cdffe24072733950c671e47597ff9fbd5",
            "W0_PARENT_SOURCE")
    return Path(binding["root"])/"shared-w0"/("qwen25-"+dataset)

def verify_parent(root,dataset,stream_lock,consumer_source,consumer_assets):
    binding=read(Path(root)/"w0-parent.json")
    producer=Path(binding["root"])
    for item in (binding["source_lock"],binding["asset_preflight"]):
        require(file_sha(item["path"])==item["sha256"],"W0_PARENT_MEMBER_CHANGED")
    source=read(binding["source_lock"]["path"])
    assets=read(binding["asset_preflight"]["path"])
    require(source["code_commit"]==binding["producer_source"],"W0_PARENT_SOURCE")
    require(assets["asset_identity"]==consumer_assets["asset_identity"]
        and assets["assets_sha256"]==consumer_assets["assets_sha256"]
        and assets["runtime"]["packages"]==consumer_assets["runtime"]["packages"],
        "W0_PARENT_MODEL_TOKEN_ASSET_RUNTIME")
    folder=w0_folder(root,dataset);r=read(folder/"w0-receipt.json")
    expected=dict(model="qwen25",dataset=dataset,endpoint="W0",observed_requests=2000,
        stream_sha256=stream_lock["stream_sha256"],code_commit=source["code_commit"],
        official_tree_sha256=source["official_tree_sha256"],assets_sha256=assets["assets_sha256"])
    require(all(r.get(k)==v for k,v in expected.items()),"W0_PARENT_RECEIPT_BINDING")
    require(r["receipt_sha256"]==digest({k:v for k,v in r.items() if k!="receipt_sha256"}),
            "W0_PARENT_DIGEST")
    require(file_sha(folder/"w0-cases.json")==r["factual"]["cases_sha256"],"W0_PARENT_RAW_HASH")
    require(len(read(folder/"w0-cases.json"))==2000,"W0_PARENT_FULL_COHORT")
    # Original raw/source/metric meaning remains untouched. The new zsRE
    # evaluator has no W0 dependency; this is physical cold-state provenance only.
    return r,dict(producer_receipt_path=str(folder/"w0-receipt.json"),
        producer_receipt_sha256=file_sha(folder/"w0-receipt.json"),
        producer_source=source["code_commit"],consumer_source=consumer_source,
        historical_W0_not_relabelled=True,new_public_query_W0_observation=False,
        numerical_cross_source_parity="NOT_CLAIMED",role="COLD_MODEL_PROVENANCE_NOT_NEW_METRICS")
