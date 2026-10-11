"""Generate candidate routes between the campus endpoints on the domain-clipped
network and screen pairs where a longer route is predicted to strain less."""
import pickle, itertools, numpy as np, pandas as pd, networkx as nx, osmnx as ox
P="/media/harshin/data_drive/route based UTCI"; OUT="/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/rsearch"
E=pickle.load(open(f"{OUT}/edge_sun.pkl","rb")); D=pd.read_csv(f"{OUT}/calib.csv")
U=1.3; fp=lambda el: 0.308*np.cos(np.deg2rad(el*(0.998-el**2/50000)))
# walking cost: neutral rise per minute; radiant cost per kJ m-2 of beam on the body (from the sweep)
sw=pd.read_csv(f"{P}/run_output/MMC/postprocessing/departure_sweep/departure_sweep_fixedclothing.csv")
D=D.merge(sw[["route_id","departure_hour","neutral_tcore_rise_c"]].rename(columns={"route_id":"route","departure_hour":"h0"}),on=["route","h0"])
A_walk=np.polyfit(D.dur,D.neutral_tcore_rise_c,1)[0]; B_rad=np.polyfit(D.dose_model,D.rad,1)[0]
print(f"walking {A_walk:.4f} C/min, radiation {B_rad*1000:.3f} C per MJ m-2 of beam")
G=ox.load_graphml(f"{P}/run_output/MMC/osm_paths/pedestrian_network_domain.graphml")
S,T=5520494042,7028624065
# per directed edge: sample offsets and tau
edges=E["edges"]; eidx={(u,v,k):i for i,(u,v,k,L) in enumerate(edges)}
owner=E["owner"]; order=np.argsort(owner,kind="stable"); starts=np.searchsorted(owner[order],np.arange(len(edges)))
def edge_pts(i):
    a=starts[i]; b=starts[i+1] if i+1<len(edges) else len(owner); return order[a:b]
Gs=nx.Graph()
for i,(u,v,k,L) in enumerate(edges):
    if not Gs.has_edge(u,v) or Gs[u][v]["length"]>L: Gs.add_edge(u,v,length=L,eid=i)
hours,elev,dni,tau=E["hours"],E["elev"],E["dni"],E["tau"]
def path_points(path):
    idx=[]
    for u,v in zip(path[:-1],path[1:]):
        i=Gs[u][v]["eid"]; uu,vv,_,_=edges[i]; p=edge_pts(i)
        idx.append(p if uu==u else p[::-1])
    return np.concatenate(idx)
def score(path,h0s):
    p=path_points(path); L=sum(Gs[u][v]["length"] for u,v in zip(path[:-1],path[1:]))
    cum=np.linspace(0,L,len(p)); out=[]
    for h0 in h0s:
        ta=h0+cum/U/3600; j=np.clip(np.searchsorted(hours,ta)-1,0,len(hours)-2); w=(ta-hours[j])/(hours[j+1]-hours[j])
        tv=(1-w)*tau[j,p]+w*tau[j+1,p]; load=tv*np.maximum(0,(1-w)*fp(elev[j])*dni[j]+w*fp(elev[j+1])*dni[j+1])
        out.append(float((load*np.gradient(cum)/U).sum()/1000))
    sun=float(np.mean(tau[np.argmin(abs(hours-13)),p]>0.5))
    return L,np.array(out),sun
H0=np.arange(7,19.01,0.5)
cands={}
def add(path,tag):
    key=tuple(path)
    if key not in cands: cands[key]=tag
# (a) k shortest by length
for n,pth in enumerate(nx.shortest_simple_paths(Gs,S,T,weight="length")):
    add(pth,f"k{n}")
    if n>=400: break
# (b) shade- and sun-weighted shortest paths at several departures
for h in (9,11,13,15,17):
    j=np.argmin(abs(hours-h))
    for lam in (0.25,0.5,1,2,4,8):
        for sign in (+1,-1):
            for u,v,dd in Gs.edges(data=True):
                s=float(tau[j,edge_pts(dd["eid"])].mean())
                dd["w"]=dd["length"]*max(0.05,1+sign*lam*(s-0.5))
            add(nx.shortest_path(Gs,S,T,weight="w"),f"{'shade' if sign>0 else 'sun'}{lam}@{h}")
print(len(cands),"candidate routes")
rows=[]
for pth,tag in cands.items():
    L,dz,sun=score(list(pth),H0); rows.append(dict(path=pth,tag=tag,L=L,dur=L/U/60,sun13=sun,**{f"d{h:g}":z for h,z in zip(H0,dz)}))
C=pd.DataFrame(rows).sort_values("L").reset_index(drop=True)
pickle.dump(dict(C=C,A=A_walk,B=B_rad),open(f"{OUT}/cands.pkl","wb"))
print(C[["tag","L","sun13"]].describe().round(2).to_string())
# pairs: B longer than A by >= 40 m, predicted rise of B lower than A at some departure by >= 0.02 C
best=[]
for i in range(len(C)):
    for k in range(i+1,len(C)):
        a,b=C.iloc[i],C.iloc[k]
        if b.L-a.L<40: continue
        ea=set(zip(a.path[:-1],a.path[1:]))|set(zip(a.path[1:],a.path[:-1])); eb=set(zip(b.path[:-1],b.path[1:]))
        share=len(ea&eb)/max(1,len(eb))
        for h in H0:
            adv=A_walk*(b.dur-a.dur)+B_rad*(b[f"d{h:g}"]-a[f"d{h:g}"])
            if adv< -0.015: best.append(dict(i=i,k=k,h0=h,dL=b.L-a.L,LA=a.L,LB=b.L,pred_adv=adv,share=share,tagA=a.tag,tagB=b.tag))
B=pd.DataFrame(best)
print(len(B),"pair-departures where the longer route is predicted >=0.015 C better")
if len(B):
    g=B.groupby(["i","k"]).agg(n_dep=("h0","size"),best_adv=("pred_adv","min"),dL=("dL","first"),LA=("LA","first"),LB=("LB","first"),share=("share","first"),h_best=("h0",lambda s: list(s)[int(np.argmin(B.loc[s.index,'pred_adv']))]))
    print(g.sort_values(["n_dep","best_adv"],ascending=[False,True]).head(15).round(3).to_string())
    B.to_csv(f"{OUT}/pairs.csv",index=False)
