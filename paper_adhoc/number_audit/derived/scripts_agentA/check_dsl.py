import yaml, json, re, os, time, numpy as np, pandas as pd
PIPE="/media/harshin/data_drive/route based UTCI"; PAPER="/home/harshin/files/fastUTEC paper"
def rp(p):
    if p.startswith("RUN/"): return PIPE+"/run_output/"+p[4:]
    if p.startswith("PIPE/"): return PIPE+"/"+p[5:]
    if p.startswith("/"): return p
    return PAPER+"/"+p
def J(p,key=""):
    d=json.load(open(rp(p)))
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", key):
        d = d[int(part[1:-1])] if part.startswith("[") else d[part]
    return d
CSV=lambda p: pd.read_csv(rp(p))
def TXT(p,rx): return float(re.search(rx,open(rp(p)).read()).group(1))
def TXTALL(p,rx): return [float(x) for x in re.findall(rx,open(rp(p)).read())]
CODE=TXT
items=yaml.safe_load(open(os.path.join(os.path.dirname(__file__),"result.yaml")))
bad=0
for it in items:
    t=time.time()
    try: v=eval(it["expr"])
    except Exception as e: print("ERR",it["loc"],it["printed"],e); bad+=1; continue
    dt=time.time()-t
    exp=it["value"]
    try:
        ok=np.allclose(np.asarray(v,float),np.asarray(exp,float),rtol=1e-3,atol=6e-4) if not isinstance(exp,str) else True
    except Exception: ok=False
    if not ok or dt>2: print("CHK",it["loc"],it["printed"],"->",v,"vs",exp,"%.1fs"%dt); bad+=1
print(len(items),"items,",bad,"flagged")
