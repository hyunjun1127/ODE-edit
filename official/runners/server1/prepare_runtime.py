"""Six exact official configs plus future READY paths; CPU/source preparation only."""
import argparse
import copy
from pathlib import Path

from official.experiments.prepare import build_matrix, digest, load_plan, write_new
from .common import LOCAL_ROOT, METHODS, member, read, require, validate_config, DEFERRED_W20, CF_CHECKPOINT_AUTHORITY
from .native_parity import PLAN as NATIVE_PARITY_PLAN


QUALIFICATION_PLAN = dict(schema="official-native-resume-plan-v1", batches=[1, 2, 3],
    external_batch_size=100, split_after_batch=2, selected_weight_comparison="EXACT_SHA256",
    RNG_comparison="EXACT", context_comparison="EXACT",
    factual_comparison="EXACT_VALUES_WITHOUT_ELAPSED_WORK", extra_generation=False, extra_science_chain=False,
    native_reference_matched_B3=NATIVE_PARITY_PLAN)


def prepare(assets_path, streams_path, output, *, tracking_env_file, attempt="official-r1", cf_checkpoint_only=False):
    output = Path(output).absolute()
    require(output.is_relative_to(LOCAL_ROOT) and not output.exists(), "FRESH_OWN_RUNTIME_NAMESPACE_REQUIRED")
    assets, streams = member(assets_path), member(streams_path)
    asset_value = read(assets_path)
    require(read(streams_path)["assets_sha256"] == asset_value["assets_sha256"], "STREAM_BUNDLE_ASSETS_IDENTITY")
    contract, profiles = load_plan()
    runtime = output / "runs"
    configs = {}
    for row in build_matrix(contract, profiles):
        if row["model"] != "llama3" or row["method"] not in METHODS:
            continue
        if cf_checkpoint_only and row["dataset"] != "cf":
            continue
        value = dict(copy.deepcopy(row), schema="official-server1-runtime-v1", assets_member=assets,
            stream_bundle_member=streams, qualification_plan=QUALIFICATION_PLAN,
            qualification_outputs={method: str(runtime / ("qualification-" + method.lower())) for method in METHODS},
            base_W0_output=str(runtime / "base-w0"), zsre_smoke_output=str(runtime / "zsre-smoke"),
            tracking=dict(module="official.tracking.client", env_file=str(Path(tracking_env_file).absolute()),
                          entity="wkdguswns2256", project="layer allocation", attempt=attempt),
            actual_GPU_qualification=False, actual_online_validation=False,
            automatic_retry=False, protected_old_jobs_keep=True)
        if cf_checkpoint_only:
            value.update(cf_W20_generation=DEFERRED_W20, scope_override=CF_CHECKPOINT_AUTHORITY)
            value["evaluation"]["generation"]["edited_endpoints"] = []
            value["evaluation"]["generation"]["deferred_to_checkpoint"] = 20
        value.pop("config_sha256", None)
        value["config_sha256"] = digest(value)
        validate_config(value)
        path = output / "configs" / (row["run_id"] + ".json")
        write_new(path, value)
        configs[row["method"], row["dataset"]] = path
    require(len(configs) == (3 if cf_checkpoint_only else 6), "SERVER1_EXACT_ROW_CONFIGS")
    result = dict(schema="official-server1-runtime-preparation-v1", server="server1", model="llama3",
        configs=[dict(method=method, dataset=dataset, member=member(path))
                 for (method, dataset), path in sorted(configs.items())], output_root=str(runtime),
        assets_member=assets, stream_bundle_member=streams, qualification_plan_sha256=digest(QUALIFICATION_PLAN),
        actual_GPU_qualification=False, submitted_jobs=[], W0_measured=False, online_verified=False,
        source_main_freeze_required=True, old_jobs_mutated=False)
    write_new(output / "preparation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-env-file", type=Path, required=True)
    parser.add_argument("--cf-checkpoint-only", action="store_true")
    args = parser.parse_args()
    value = prepare(args.assets, args.streams, args.output, tracking_env_file=args.tracking_env_file,
                    cf_checkpoint_only=args.cf_checkpoint_only)
    print({key: value[key] for key in ("server", "model", "output_root", "actual_GPU_qualification", "submitted_jobs")})


if __name__ == "__main__":
    main()
