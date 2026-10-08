"""Create only the authorized tuning saved view; never creates or mutates runs.

Execute with the existing isolated wandb-tracking environment, not science Python.
Only view metadata and scalar metric names are sent. Existing views are preserved.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import quote

ENTITY = 'wkdguswns2256'
PROJECT = 'layer allocation'
NAME = 'Qwen Tuning — Held-out 500'
TASK = 'qwen-price-hparam-tier2-20261009'


def sections(ws, wr):
    def panel(prefix, field, title, percent=False):
        return wr.LinePlot(title=title, x='edits', y=[f'{prefix}/{field}'],
            title_x='Actual held-out edit applications',
            title_y='Percent' if percent else 'Measured value',
            smoothing_type='none', smoothing_factor=0, aggregate=False,
            ignore_outliers=False, max_runs_to_show=20,
            legend_template='${run:displayName}')

    result = [ws.Section(name='Scope and availability', is_open=True, panels=[
        wr.MarkdownPanel(markdown='''# Qwen PRICE tuning — held-out 500
CF ordered slice [2000:2500], separate from eval first2000. Q0–Q7 Tier1 use
the same held-out B1; selected Tier2 arms use B1–B5. No historical first2K
or baseline results are imported. Qwen model / MEMIT writer only.

Each line is a distinct measured run/arm/attempt with actual job number.
Current remains incoming 100; all-seen is measured W5/500 only. W0_first500
is the exact matched cold cohort, not first2000 or full10k. R/P desired=new,
N desired=true; success/ACC are percent and NLL/margins are unscaled nats.
Missing measurements are omitted, never zero-filled. Fit uses its separate
monotonic candidate axis. Tier1 and Tier2 must be distinguished by run config.

This saved view does not launch experiments or repair the shared logger schema.
An empty chart means no matching uploaded measurement, not a zero score.
Smoke/validation runs are excluded by cohort_role=heldout_tuning.
''')])]
    for prefix, label in [('current/pre', 'Current100 pre'),
                          ('current/post', 'Current100 post'),
                          ('all_seen/post', 'W5 all-seen500'),
                          ('W0_first500', 'Matched cold W0 first500')]:
        result.append(ws.Section(name=label, is_open=True, panels=[
            panel(prefix, f'{kind}/success_pct', f'{label}: {kind} success', True)
            for kind in 'RPN'] + [
            panel(prefix, 'success_harmonic_pct', f'{label}: harmonic', True)]))
    for field in ('strict_acc_pct', 'token_acc_pct', 'prompt_acc_pct',
                  'true_nll', 'new_nll', 'margin_true_minus_new'):
        result.append(ws.Section(name='Current100 post — '+field, panels=[
            panel('current/post', f'{kind}/{field}', f'{kind}: {field}', field.endswith('_pct'))
            for kind in 'RPN']))
    result.append(ws.Section(name='Fit — independent candidate axis', panels=[
        wr.LinePlot(title=metric, x='fit/global_candidate', y=[metric],
                    aggregate=False, smoothing_type='none', smoothing_factor=0)
        for metric in ('fit/loss', 'fit/nll', 'fit/kl', 'fit/norm')]))
    return result


def main():
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    from wandb_workspaces._graphql import execute_graphql
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    api = wandb.Api(timeout=30)
    query = '''query Views($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        allViews(viewType: "project-view") { edges { node { name displayName } } }
      }
    }'''
    data = execute_graphql(api, query, {'entity': ENTITY, 'project': PROJECT})
    if data.get('project') is None:
        raise RuntimeError('PROJECT_UNAVAILABLE')
    matches = [e['node'] for e in data['project']['allViews']['edges']
               if e['node']['displayName'] == NAME]
    if len(matches) > 1:
        raise RuntimeError('DUPLICATE_VIEW_NAME_NO_MUTATION')
    if matches:
        name = matches[0]['name'].removeprefix('nw-').removesuffix('-v')
        view = ws.Workspace.from_url(
            f'https://wandb.ai/{ENTITY}/{quote(PROJECT)}?nw={name}')
        action = 'REUSED_NO_MUTATION'
    else:
        view = ws.Workspace(entity=ENTITY, project=PROJECT, name=NAME,
            sections=sections(ws, wr), auto_generate_panels=False,
            runset_settings=ws.RunsetSettings(filters=[
                expr.Config('task_id') == TASK,
                expr.Config('cohort_role') == 'heldout_tuning']),
            settings=ws.WorkspaceSettings(max_runs=20))
        view.save()
        action = 'CREATED'
    loaded = ws.Workspace.from_url(view.url)
    assert loaded.name == NAME
    assert len(loaded.sections) == 12
    filters = str(loaded.runset_settings.filters)
    assert TASK in filters and 'heldout_tuning' in filters
    receipt = dict(name=NAME, entity=ENTITY, project=PROJECT, url=view.url,
        action=action, remote_view_readback='VERIFIED', sections=len(loaded.sections),
        run_filter_task_id=TASK, run_filter_cohort_role='heldout_tuning',
        new_runs=0, metric_points_uploaded=0, existing_runs_modified=0,
        existing_other_views_modified=0,
        logger_schema='HELDOUT_EXTENSION_STILL_REQUIRED',
        scientific_execution='NOT_SUBMITTED_BY_THIS_VIEW_OPERATION')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
