"""Strict (exact key) versus sign-tolerant success for e1 (first order) and e3 part A (second order). Same seeds as e1/e3."""
import sys, csv, numpy as np
import lbsca as L, engine as E
Q=L.Q
def both(res,tr):
    G=len(res[0]); st=np.zeros(G); tl=np.zeros(G)
    for r,t in zip(res,tr):
        for g,(b,_) in enumerate(r):
            st[g]+=(b==t); tl[g]+=(b==t) or (b==(Q-t)%Q)
    return st/len(res),tl/len(res)
which=sys.argv[1]
rows=[]
if which=='e1':
    rng=np.random.default_rng(1001); sigmas=[1,2,4,8,16]
    grid=[1,2,3,5,8,12,20,30,50,80,120,200,300,500,800,1200,2000,3000,5000]; trials=60; T=L.hw_tables()
    for sg in sigmas:
        res=[];tr=[]
        for k in range(trials):
            s=int(L.center(rng.integers(0,Q))); Nmax=grid[-1]
            u=L.center(rng.integers(0,Q,Nmax)); uidx=E.residx(u)
            lk=L.hw_points(np.full(Nmax,s),u)+rng.normal(0,sg,(Nmax,4)).astype(np.float32)
            res.append(E.corr_checkpoints([lk[:,j] for j in (1,2,3)],[E.table_getter(T[j],uidx) for j in (1,2,3)],grid,s%Q)); tr.append(s%Q)
        st,tl=both(res,tr)
        rows.append([sg,E.mtd_from_sr(grid,st),E.mtd_from_sr(grid,tl)]); print(rows[-1],flush=True)
else:
    rng=np.random.default_rng(3003); trials=30
    Ms=[L.second_order_table(j)[0] for j in (1,2,3)]
    sigmas=[1,2,4,8,16]
    gridA={1:[5,10,20,40,80,150,300,600,1200],2:[5,10,20,40,80,150,300,600,1200,2500],
       4:[10,20,50,100,200,400,800,1500,3000,6000],8:[50,100,200,400,800,1500,3000,6000,12000,20000],
       16:[200,500,1000,2000,4000,8000,16000,32000]}
    for sg in sigmas:
        grid=gridA[sg]; Nmax=grid[-1]; res=[];tr=[]
        for k in range(trials):
            s=int(L.center(rng.integers(0,Q)))
            u=L.center(rng.integers(0,Q,Nmax)); uidx=E.residx(u)
            s0=L.center(rng.integers(0,Q,Nmax)); s1=L.center(s-s0)
            l0=L.hw_points(s0,u)[...,1:4]+rng.normal(0,sg,(Nmax,3)).astype(np.float32)
            l1=L.hw_points(s1,u)[...,1:4]+rng.normal(0,sg,(Nmax,3)).astype(np.float32)
            Cc=((l0-l0.mean(0))*(l1-l1.mean(0))).astype(np.float32)
            res.append(E.corr_checkpoints([Cc[:,j] for j in range(3)],[E.rowtable_getter(Ms[j],uidx) for j in range(3)],grid,s%Q)); tr.append(s%Q)
        st,tl=both(res,tr)
        rows.append([sg,E.mtd_from_sr(grid,st),E.mtd_from_sr(grid,tl)]); print(rows[-1],flush=True)
with open('../data/e7_strict_%s.csv'%which,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['sigma','MTD_strict','MTD_signtol']); w.writerows(rows)
