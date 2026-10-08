"""Server4 orchestration of SH1's published factual API; no new evaluator."""
import copy
import inspect
from official.evaluation import factual, reduce
from official.experiments.prepare import digest, file_sha, read, write_new
from official.tracking.method import OFFICIAL_FIELDS


def verify_api():
    expected=('model','tokenizer','records','dataset','w0_reference','batch_size',
              'device','progress','identity')
    signature=inspect.signature(factual.evaluate)
    if factual.SCHEMA!='official-factual-causal-v1' or tuple(signature.parameters)!=expected:
        raise ValueError('SH1_FACTUAL_API_NOT_BOUND')
    for name in expected[4:]:
        if signature.parameters[name].kind is not inspect.Parameter.KEYWORD_ONLY:
            raise ValueError('SH1_FACTUAL_KEYWORD_API')
    reference=inspect.signature(factual.build_zsre_w0_reference)
    if tuple(reference.parameters)!=('model','tokenizer','records','identity',
                                     'batch_size','device','progress'):
        raise ValueError('SH1_ZSRE_REFERENCE_API_NOT_BOUND')
    return dict(schema=factual.SCHEMA,source_sha256=file_sha(factual.__file__),
                evaluate_signature=str(signature),reference_signature=str(reference))


def external_identity(assets):
    return dict(model='llama3', revision=assets['model_revision'],
                tokenizer_sha256=assets['tokenizer_sha256'],
                runtime_sha256=assets['runtime_sha256'], precision='FP32_EAGER_TF32_OFF',
                evaluator_sha256=file_sha(factual.__file__))


def subset(observation, records, dataset):
    lookup={row['occurrence_index']:row for row in observation['cases']}
    if len(lookup)!=len(observation['cases']):raise ValueError('FACTUAL_DUPLICATE_OCCURRENCE')
    if len({r['occurrence_index'] for r in records})!=len(records):
        raise ValueError('FACTUAL_SUBSET_DUPLICATE_OCCURRENCE')
    selected=[]
    for record in records:
        row=lookup[record['occurrence_index']]
        if row['case_id']!=record['case_id']:raise ValueError('FACTUAL_SUBSET_IDENTITY')
        selected.append(copy.deepcopy(row))
    summary=(reduce.counterfact if dataset=='cf' else reduce.zsre)(selected)
    identity=dict(observation['identity'],subset=True,
                  parent_cohort_sha256=observation['identity']['cohort_sha256'],
                  ordered_occurrences=[r['occurrence_index'] for r in records])
    # No new tokenization/forward. The parent token cohort remains provenance,
    # not a false claim that the selected 100 have the parent's cohort hash.
    identity.pop('cohort_sha256')
    return dict(cases=selected,summary=summary,identity=identity,
                identity_sha256=digest(identity),
                parent_identity_sha256=observation['identity_sha256'],subset=True)


def scores(observation, prefix, batch):
    values={'edits':batch*100,'post_state_edits':batch*100,'batch':batch}
    if prefix=='current/pre':values['pre_state_edits']=(batch-1)*100
    values.update({'official/'+prefix+'/'+key:value for key,value in observation['summary'].items()
                   if key in OFFICIAL_FIELDS})
    return values


class FactualObserver:
    def __init__(self, model, tokenizer, dataset, assets, output, qualification=False):
        verify_api()
        self.model,self.tokenizer,self.dataset=model,tokenizer,dataset
        self.identity=external_identity(assets)
        self.out=output
        self.reference=None;self.w0=None
        member=assets.get('w0',{}).get(dataset)
        if member:
            if file_sha(member['path'])!=member['sha256']:raise ValueError('W0_MEMBER_SHA')
            payload=read(member['path'])
            if dataset=='zsre':
                self.reference=payload;self.w0=payload['evaluation']
                if payload['identity']['external_identity']!=self.identity:raise ValueError('W0_RUNTIME_IDENTITY')
            else:self.w0=payload
            if (self.w0['identity']['external_identity']!=self.identity
                    or self.w0['identity']['dataset']!=dataset
                    or self.w0['identity']['schema']!=factual.SCHEMA
                    or self.w0['identity_sha256']!=digest(self.w0['identity'])
                    or self.w0['summary']['requests']!=2000
                    or len({r['occurrence_index'] for r in self.w0['cases']})!=2000
                    or len(self.w0['cases'])!=2000):
                raise ValueError('W0_EXTERNAL_IDENTITY')
        elif not qualification:
            raise ValueError('SHARED_W0_INPUT_REQUIRED')
        if dataset=='zsre' and self.reference is None:
            raise ValueError('ZSRE_W0_PREDICTIONS_REQUIRED')

    def observe(self, records, endpoint):
        result=factual.evaluate(self.model,self.tokenizer,records,self.dataset,
            w0_reference=self.reference,batch_size=8,identity=self.identity)
        # Wall time is observational, not part of reproducible numeric payload.
        write_new(self.out/(endpoint+'.json'),result)
        return result

    def verify_w0(self,records):
        if self.w0 is None:return None
        expected=[(r['occurrence_index'],r['case_id']) for r in records]
        if [(r['occurrence_index'],r['case_id']) for r in self.w0['cases']]!=expected:
            raise ValueError('W0_COHORT_ROW_ORDER')
        _,_,signatures=factual._plan(records,self.dataset,self.tokenizer)
        if self.w0['identity']['cohort_sha256']!=digest(signatures):
            raise ValueError('W0_COHORT_TOKEN_IDENTITY')
        return self.w0


def numeric_receipt(value):
    return dict(cases=value['cases'], summary=value['summary'],identity=value['identity'])
