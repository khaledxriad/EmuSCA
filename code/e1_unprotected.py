import sys, time, numpy as np, csv
import lbsca as L, engine as E
Q=L.Q
rng = np.random.default_rng(1001)
sigmas=[1,2,4,8,16]
grid=[1,2,3,5,8,12,20,30,50,80,120,200,300,500,800,1200,2000,3000,5000]
trials=int(sys.argv[1]) if len(sys.argv)>1 else 60
T=L.hw_tables()
rows=[]; mtds=[]
for sg in sigmas:
    res=[]; tr=[]
    t0=time.time()
    for k in range(trials):
        s=int(L.center(rng.integers(0,Q)))
        Nmax=grid[-1]
        u=L.center(rng.integers(0,Q,Nmax)); uidx=E.residx(u)
        lk=L.hw_points(np.full(Nmax,s),u)+rng.normal(0,sg,(Nmax,4)).astype(np.float32)
        cols=[lk[:,j] for j in (1,2,3)]
        getters=[E.table_getter(T[j],uidx) for j in (1,2,3)]
        r=E.corr_checkpoints(cols,getters,grid,s%Q)
        res.append(r); tr.append(s%Q)
    sr=E.success_curve(res,tr)
    mtd=E.mtd_from_sr(grid,sr)
    mtds.append(mtd)
    print('sigma',sg,'MTD90',mtd,'time',round(time.time()-t0,1),flush=True)
    for g,N in enumerate(grid): rows.append([sg,N,sr[g]])
with open('../data/e1_unprotected_sr.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['sigma','N','SR']); w.writerows(rows)
with open('../data/e1_unprotected_mtd.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['sigma','MTD90']); [w.writerow([s,m]) for s,m in zip(sigmas,mtds)]
