"""Strict vs sign-tolerant success curves for masked second-order CPA, and trace multiplier needed to resolve the sign."""
import csv, numpy as np
import lbsca as L, engine as E
Q=L.Q; rng=np.random.default_rng(8008)
Ms=[L.second_order_table(j)[0] for j in (1,2,3)]
rows=[]
for sg,grid in [(1,[20,80,300,1200,5000,20000]),(2,[40,150,600,2500,10000,40000]),(4,[100,400,1500,6000,24000])]:
    Nmax=grid[-1]; res=[];tr=[]
    for k in range(30):
        s=int(L.center(rng.integers(0,Q)))
        u=L.center(rng.integers(0,Q,Nmax)); ui=E.residx(u)
        s0=L.center(rng.integers(0,Q,Nmax)); s1=L.center(s-s0)
        l0=L.hw_points(s0,u)[...,1:4]+rng.normal(0,sg,(Nmax,3)).astype(np.float32)
        l1=L.hw_points(s1,u)[...,1:4]+rng.normal(0,sg,(Nmax,3)).astype(np.float32)
        C=((l0-l0.mean(0))*(l1-l1.mean(0))).astype(np.float32)
        res.append(E.corr_checkpoints([C[:,j] for j in range(3)],[E.rowtable_getter(Ms[j],ui) for j in range(3)],grid,s%Q)); tr.append(s%Q)
    for gi,N in enumerate(grid):
        st=np.mean([r[gi][0]==t for r,t in zip(res,tr)]); tl=np.mean([(r[gi][0]==t) or (r[gi][0]==(Q-t)%Q) for r,t in zip(res,tr)])
        rows.append([sg,N,st,tl]); print(rows[-1],flush=True)
with open('../data/e8_sign.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['sigma','N','SR_strict','SR_signtol']); w.writerows(rows)
