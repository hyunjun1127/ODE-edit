"""GH independent analysis of sealed publications; never loads a model/checkpoint.

Run with the existing EasyEdit .venv Python. Inputs stay read-only; output must
not exist. Recomputes small controller identities, physical ratios and roots.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def read_rows(root, name):
    with (root / name).open(newline="") as f:
        return list(csv.DictReader(f))


def verify(root, manifest_name):
    manifest = json.loads((root / manifest_name).read_text())
    members = []
    for m in manifest["members"]:
        path = root / m["path"]
        assert path.parent == root and not path.is_symlink() and path.is_file()
        assert sha(path) == m["sha256"] and path.stat().st_size == m["bytes"], path
        n = len(read_rows(root, m["path"])) if path.suffix == ".csv" else None
        if m.get("rows") is not None:
            assert n == m["rows"], (path, n)
        members.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, rows=n))
    if manifest_name == "analysis-manifest.json":
        receipt = json.loads((root / "rooted-receipt.json").read_text())
        assert hashlib.sha256(canonical(manifest["members"])).hexdigest() == receipt["members_root"]
        body = dict(receipt)
        identity = body.pop("receipt_identity")
        assert hashlib.sha256(canonical(body)).hexdigest() == identity
    else:
        receipt = json.loads((root / "rooted-package-receipt.json").read_text())
        # This older publisher uses json.dumps' default ensure_ascii=True.
        assert hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == receipt["root_sha256"]
    assert sha(root / manifest_name) == receipt["manifest_sha256"]
    assert sha(root / "factual-report-ko.md") == receipt["report_sha256"]
    return members


def save_rows(root, name, rows):
    with (root / name).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sh1", type=Path, required=True)
    parser.add_argument("--sh2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    checks = verify(a.sh1, "analysis-manifest.json") + verify(a.sh2, "package-manifest.json")
    a.output.mkdir(parents=True, exist_ok=False)
    node_rows, batch_rows = [], []
    allocations = read_rows(a.sh1, "layer_allocation_nodes.csv")
    assert len(allocations) == 80
    cumulative = {}
    qrefs = {}
    for row in allocations:
        model, batch, node = row["alias"], int(row["batch"]), int(row["node"])
        c, g, hess, metric = [np.asarray(json.loads(row[k]), dtype=np.float64)
                              for k in ("c", "g", "full_H", "G")]
        assert c.shape == (5,) and metric.shape == hess.shape == (5, 5)
        v, after = float(row["V_before"]), float(row["V_after"])
        gain, response, action = float(g @ c), float(c @ hess @ c), float(c @ metric @ c)
        dual = (hess + .1 * metric) @ c - g
        kkt_residual = max(float(np.max(np.abs(dual[c > 0]))) if np.any(c > 0) else 0.,
                           max(0., -float(dual[c == 0].min())) if np.any(c == 0) else 0.)
        assert np.min(c) >= 0
        assert abs(gain - response - .1 * action) < 1e-10
        assert action <= v / .2 + 1e-10
        db = v - after - .05 * action
        predicted_db_affine = .375 * response
        defect = predicted_db_affine - db
        assert abs(defect - float(row["finite_step_dissipation_defect"])) < 1e-12
        key = (model, batch)
        cumulative[key] = cumulative.get(key, 0.) + db
        if key in qrefs:
            assert qrefs[key] == float(row["qN_ref"])
        qrefs[key] = float(row["qN_ref"])
        cosine = json.loads(row["response_residual_cosine"])
        node_rows.append(dict(model=model, batch=batch, node=node,
            kkt_residual=kkt_residual, dissipation_residual=gain-response-.1*action,
            normalized_native_velocity_action=action, native_speed_bound=v/.2,
            gain=gain, predicted_response_squared=response,
            actual_delta_barrier=db, affine_predicted_delta_barrier=predicted_db_affine,
            cumulative_barrier=cumulative[key], finite_defect=defect,
            defect_to_predicted_increment=abs(defect)/predicted_db_affine if predicted_db_affine else None,
            V_before=v, V_after=after, V_ratio=float(row["V_ratio"]),
            qN_ref=qrefs[key], min_response_Gram_eig=float(np.linalg.eigvalsh((hess+hess.T)/2).min()),
            max_response_Gram_eig=float(np.linalg.eigvalsh((hess+hess.T)/2).max()),
            **{f"L{l}_response_cosine":cosine[l-4] for l in range(4,9)}))
    assert len({(x['model'], x['batch'], x['node']) for x in node_rows}) == 80
    batches = read_rows(a.sh1, "batches.csv")
    for row in batches:
        if row["arm"] != "JV_NATIVE":
            continue
        model, batch = row["alias"], int(row["batch"])
        official = next(x for x in batches if x["alias"] == model and x["arm"] == "O_NATIVE" and int(x["batch"]) == batch)
        raw, norm = float(row["raw_native_work"]), float(row["normalized_native_work"])
        assert abs(raw / qrefs[(model,batch)] - norm) < 1e-12
        assert abs(float(row["history_work"]) + float(row["L2_work"]) - raw) < 1e-8
        bnodes = [x for x in node_rows if x["model"] == model and x["batch"] == batch]
        batch_rows.append(dict(model=model, batch=batch,
            JV_total_batch_energy=float(row["total_batch_net_energy"]),
            Official_total_batch_energy=float(official["total_batch_net_energy"]),
            JV_over_Official_total_energy=float(row["total_batch_net_energy"])/float(official["total_batch_net_energy"]),
            JV_over_Official_L4_7_energy=float(row["L4_7_batch_net_energy"])/float(official["L4_7_batch_net_energy"]),
            JV_L8_share=float(row["L8_energy_share"]), Official_L8_share=float(official["L8_energy_share"]),
            history_fraction=float(row["history_raw_work_ratio"]), qN_ref=qrefs[(model,batch)],
            raw_native_work=raw, normalized_native_work=norm,
            final_V_ratio=float(row["final_V_ratio"]),
            min_actual_delta_barrier=min(x["actual_delta_barrier"] for x in bnodes),
            final_barrier=cumulative[(model,batch)],
            barrier_decreasing_nodes=sum(x["actual_delta_barrier"] < 0 for x in bnodes)))
    model_summaries = {}
    for model in sorted({x['model'] for x in node_rows}):
        nodes = [x for x in node_rows if x['model'] == model]
        normal = [x for x in nodes if not(model.startswith('llama') and x['batch']==10)]
        model_summaries[model] = dict(
            max_kkt_residual=max(x['kkt_residual'] for x in nodes),
            max_abs_dissipation_residual=max(abs(x['dissipation_residual']) for x in nodes),
            barrier_decreasing_nodes=[dict(batch=x['batch'],node=x['node'],db=x['actual_delta_barrier']) for x in nodes if x['actual_delta_barrier'] < 0],
            cosine_medians_excluding_llama_B10={f'L{l}':float(np.median([x[f'L{l}_response_cosine'] for x in normal])) for l in range(4,9)},
            max_abs_finite_defect=max(abs(x['finite_defect']) for x in nodes),
            max_finite_defect_to_affine_prediction=max(x['defect_to_predicted_increment'] for x in nodes))
    save_rows(a.output, 'controller_barrier_audit.csv', node_rows)
    save_rows(a.output, 'batch_mechanism_comparison.csv', batch_rows)
    output = dict(status="PUBLISHED_TABLE_REHASH_AND_CPU_IDENTITY_CHECKS_PASS",
                  source_sha256=sha(Path(__file__)), input_members=checks, members_checked=len(checks),
                  coverage="SH1_34_AND_SH2_48_PUBLICATION_MEMBERS; NOT_EXTERNAL_SERVER2_RAW",
                  new_GPU_model_evaluation=0, models=model_summaries)
    (a.output/'verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'figure.dpi':130, 'savefig.dpi':160})
    fig, axes = plt.subplots(2,2,figsize=(11,7), constrained_layout=True)
    for model, color, label in [('llama3-8b-inst','#c34a36','Llama'),('qwen2.5-7b-inst','#267baf','Qwen')]:
        rr = sorted([x for x in batch_rows if x['model']==model], key=lambda x:x['batch'])
        xx = [x['batch'] for x in rr]
        axes[0,0].plot(xx,[x['JV_over_Official_total_energy'] for x in rr],'-o',color=color,label=label)
        axes[0,1].plot(xx,[x['JV_over_Official_L4_7_energy'] for x in rr],'-o',color=color,label=label)
        axes[1,0].plot(xx,[100*x['history_fraction'] for x in rr],'-o',color=color,label=label)
        axes[1,1].plot(xx,[x['min_actual_delta_barrier'] for x in rr],'-o',color=color,label=label)
    axes[0,0].set(title='Total batch-net weight energy: JV / Official',yscale='log',ylabel='Ratio (Frobenius squared)')
    axes[0,1].set(title='L4-L7 batch-net weight energy: JV / Official',yscale='log',ylabel='Ratio (Frobenius squared)')
    axes[1,0].set(title='History fraction of JV raw native work',ylabel='Percent')
    axes[1,1].set(title='Minimum observed barrier increment per batch',yscale='symlog',ylabel='Delta b (native dissipation, not locality)')
    axes[1,1].axhline(0,color='gray',ls=':'); axes[1,1].set_yscale('symlog',linthresh=1e-5)
    for ax in axes.flat:
        ax.set_xlabel('Sequential B100 batch');ax.set_xticks(range(1,11));ax.grid(alpha=.18);ax.legend()
    fig.suptitle('Mechanism evidence: fixed sample, descriptive comparison')
    fig.savefig(a.output/'mechanism-evidence.png',metadata={'Software':'GH reproducible analysis'})
    plt.close(fig)
    print(json.dumps(dict(members_checked=len(checks),models=model_summaries),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
