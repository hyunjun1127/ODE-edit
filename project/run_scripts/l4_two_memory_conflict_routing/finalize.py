"""One-shot CPU publication CLI over already frozen complete inputs.

No scheduler submission, model loading, polling, or experiment execution.
Use a new output/reproduction directory for any analysis-only technical retry.
"""
import argparse
import json
from pathlib import Path
import torch
from . import analysis,diagnostics,input_preservation,snapshot_check,final_checks,accounting,publication
from .identity import save,member

def main(args):
    torch.set_num_threads(8)
    ns=argparse.Namespace
    analysis.main(ns(runs=args.runs,observations=args.observations,geometry=args.geometry,output=args.output,partial=False))
    diagnostics.main(ns(runs=args.runs,observations=args.observations,output=args.output))
    input_preservation.main(ns(output=args.output))
    snapshot_check.main(ns(runs=args.runs,output=str(Path(args.output)/'snapshot-check.json')))
    final_checks.main(ns(runs=args.runs,output=args.output))
    accounting.main(ns(allocations=args.allocations,output=args.output))
    spec=json.loads(Path(args.allocations).read_text())
    save(Path(args.output)/'reproduction-command.json',dict(
        command=['python','-m','project.run_scripts.l4_two_memory_conflict_routing.finalize',
            '--runs',*args.runs,'--observations',args.observations,'--geometry',args.geometry,
            '--allocations',args.allocations,'--output',args.output,'--reproduction',args.reproduction],
        allocation_spec=member(args.allocations),model_loads=0,GPU_actions=0))
    publication.main(ns(report=args.output,reproduction=args.reproduction,jobs=[int(r['job']) for r in spec['allocations']]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    for name in ('observations','geometry','allocations','output','reproduction'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
