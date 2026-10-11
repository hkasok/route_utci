import pickle, numpy as np, pandas as pd
OUT="/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/rsearch"
X=pickle.load(open(f"{OUT}/cands.pkl","rb")); C,A,B=X["C"],X["A"],X["B"]; D=pd.read_csv(f"{OUT}/calib.csv")
E=pickle.load(open(f"{OUT}/edge_sun.pkl","rb"))
x0,x1,y0,y1=562027.7,563202.1,2848417.0,2849228.8
import osmnx as ox; G=ox.load_graphml("/media/harshin/data_drive/route based UTCI/run_output/MMC/osm_paths/pedestrian_network_domain.graphml")
def edge_margin(path):
    xy=np.array([[G.nodes[n]["x"],G.nodes[n]["y"]] for n in path]); return float(np.min(np.minimum.reduce([xy[:,0]-x0,x1-xy[:,0],xy[:,1]-y0,y1-xy[:,1]])))
H0=np.arange(7,19.01,0.5); rows=[]
for r in range(1,5):
    Rr=D[D.route==r].set_index("h0")
    for i,c in C.iterrows():
        if c.L-Rr.dur.iloc[0]*1.3*60 < 50: continue
        adv=np.array([A*(c.dur-Rr.loc[h,"dur"])+B*(c[f"d{h:g}"]-Rr.loc[h,"dose_model"]) for h in H0])
        rows.append(dict(cand=i,vs=r,L=c.L,Lr=Rr.dur.iloc[0]*78,dL=c.L-Rr.dur.iloc[0]*78,adv13=adv[H0==13][0],n_better=int((adv<-0.025).sum()),worst=adv.max(),sun13=c.sun13,tag=c.tag,margin=edge_margin(list(c.path))))
R=pd.DataFrame(rows)
print(R.sort_values("adv13").head(12).round(3).to_string()); print(R.sort_values(["n_better","adv13"],ascending=[False,True]).head(12).round(3).to_string())
