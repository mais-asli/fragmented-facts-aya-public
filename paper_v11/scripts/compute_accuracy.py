from pathlib import Path
HERE = Path(__file__).resolve().parent            # nlp final/paper_refined/scripts
ROOT = HERE.parent.parent                         # nlp final/
R = str(ROOT / "results") + "/"
W = str(HERE.parent / "_work") + "/"
OUT = str(HERE.parent / "figures") + "/"
import json, collections, math, sys
import numpy as np
sys.path.insert(0,'.')
import importlib.util
src=open(str(ROOT/'src'/'fragmented_facts'/'analysis.py')).read()
# extract only the pure functions
ns={}
exec("import collections, numpy as np\n"+src[src.index('def relation_macro'):src.index('def effect_for_rows')], ns)
cluster_effect=ns['cluster_effect']; sign_flip_p=ns['sign_flip_p']

def load(f): return [json.loads(l) for l in open(W+f, encoding='utf-8')]
e1=load('e1.jsonl'); e2=load('e2.jsonl')
out={}
def acc(rows):
    v=[float(r['entity_correct']) for r in rows]; s=[r['subject_qid'] for r in rows]; rel=[r['relation'] for r in rows]
    return cluster_effect(v,s,rel)
for lang in ['en','he','ar']:
    rows=[r for r in e1 if r['key']['language']==lang]
    out[f'E1_{lang}_U']=acc(rows)
for lang in ['he','ar']:
    for v in ['U','D']:
        rows=[r for r in e2 if r['key']['language']==lang and r['key']['variant']==v]
        rows.sort(key=lambda r:(r['key']['fact_id'],r['key']['template']))
        out[f'E2_{lang}_{v}']=acc(rows)
    U={(r['key']['fact_id'],r['key']['template']):r for r in e2 if r['key']['language']==lang and r['key']['variant']=='U'}
    D={(r['key']['fact_id'],r['key']['template']):r for r in e2 if r['key']['language']==lang and r['key']['variant']=='D'}
    ks=sorted(U)
    vals=[float(D[k]['entity_correct'])-float(U[k]['entity_correct']) for k in ks]
    s=[U[k]['subject_qid'] for k in ks]; rel=[U[k]['relation'] for k in ks]
    out[f'E2_{lang}_delta_acc']=cluster_effect(vals,s,rel); out[f'E2_{lang}_delta_acc']['p']=sign_flip_p(vals,s,rel)
    vals=[D[k]['gold_sum']-U[k]['gold_sum'] for k in ks]
    out[f'E2_{lang}_delta_ll']=cluster_effect(vals,s,rel); out[f'E2_{lang}_delta_ll']['p']=sign_flip_p(vals,s,rel)
    # micro
    out[f'E2_{lang}_micro']={'U':sum(U[k]['entity_correct'] for k in ks),'D':sum(D[k]['entity_correct'] for k in ks),'n':len(ks)}
for k,v in out.items():
    print(k, {kk:(round(vv,4) if isinstance(vv,float) else ([round(x,4) for x in vv] if isinstance(vv,list) else vv)) for kk,vv in v.items() if kk in ('estimate','ci95','p','U','D','n','n_subjects')})
json.dump(out,open(str(HERE.parent/'_work'/'computed.json'),'w'),indent=1)
