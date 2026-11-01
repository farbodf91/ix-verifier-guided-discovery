"""Check teaching decisions against saved experiments, including failure branches."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tutorial_lib import workflow
results=[]
for mu,expected_id,expected_nmse in [(0.5,118,0.0006985121154),(1.5,100,0.00326709827)]:
    case=workflow.make_case(mu)
    pool=workflow.load_pool(mu).sort_values('rollout_rank')
    shortlist=pool.head(10)
    checked=workflow.verify_shortlist(shortlist,case,suite='teaching')
    saved=workflow.load_shortlist(mu).sort_values('rollout_rank')
    assert checked.candidate_id.tolist()==saved.candidate_id.tolist()
    assert checked.full_suite_admissible.tolist()==saved.verifier_admissible.tolist()
    assert checked.verifier_admissible.tolist()==checked.full_suite_admissible.tolist()
    selected=workflow.select_admissible(checked)
    assert int(selected.candidate_id)==expected_id
    final,checks,decision=workflow.accept_refinement(selected.equation,workflow.saved_models(mu)['vg_refined'],case,selected.local_dynamics_limit,suite='teaching')
    assert checks['verifier_admissible'] and checks['full_suite_admissible']
    nmse=float(np.median(workflow.score_bank(final,case['test'])))
    assert np.isclose(nmse,expected_nmse,rtol=1e-4,atol=1e-9),(mu,nmse)
    rejected=checked.copy();rejected['verifier_admissible']=False
    assert workflow.select_admissible(rejected) is None
    fallback,_,fallback_decision=workflow.accept_refinement(selected.equation,'0*x_0 | 0*x_1',case,selected.local_dynamics_limit,suite='teaching')
    assert fallback==selected.equation
    results.append({'mu':mu,'selected':expected_id,'median_test_nmse':nmse,'full_suite_agreement':True,'fallback':fallback_decision})
flow=workflow.load_cylinder()
model=flow['model']
assert model['candidate_id']==20
pred=workflow.rollout(model['optimized_equation'],flow['test']['t'],flow['test']['x0'])
assert pred is not None and pred.shape==flow['test']['x'].shape
results.append({'cylinder_candidate':20,'full_test_nmse':workflow.nmse(flow['test']['x'],pred),'test_states':len(pred)})
print(json.dumps(results,indent=2))
