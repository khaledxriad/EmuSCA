import sys, time, csv, numpy as np
import lbsca as L, engine as E
Q=L.Q; T=L.hw_tables()
rng=np.random.default_rng(2002)
sg=2.0
trials=int(sys.argv[1]) if len(sys.argv)>1 else 30
cfg={1:[1,2,3,5,8,12,20,30,50,80,120,200,300,500],
     2:[2,5,10,20,40,80,150,250,400,700,1000],
     4:[5,10,20,40,80,150,300,500,800,1200,2000,3000],
     8:[10,20,40,80,150,300,600,1000,2000,3500,6000],
     16:[20,50,100,200,400,800,1500,3000,6000,10000,20000]}
rows=[]; mt=[]
for w,grid in cfg.items():
    resF=[];resI=[];tr=[]; t0=time.time()
    Nmax=grid[-1]
    for k in range(trials):
        s=int(L.center(rng.integers(0,Q))); 
        u=L.center(rng.integers(0,Q,(Nmax,w))); s_arr=L.center(rng.integers(0,Q,(Nmax,w)))
        pi=rng.integers(0,w,Nmax)
        s_arr[np.arange(Nmax),pi]=s
        lk=L.hw_points(s_arr,u)+rng.normal(0,sg,(Nmax,w,4)).astype(np.float32)
        ut=u[np.arange(Nmax),pi]; uidx=E.residx(ut)
        colsF=[lk[:,0,j] for j in (1,2,3)]
        colsI=[lk[:,:,j].sum(1) for j in (1,2,3)]
        g=[E.table_getter(T[j],uidx) for j in (1,2,3)]
        resF.append(E.corr_checkpoints(colsF,g,grid,s%Q))
        resI.append(E.corr_checkpoints(colsI,g,grid,s%Q))
        tr.append(s%Q)
    srF=E.success_curve(resF,tr); srI=E.success_curve(resI,tr)
    mF=E.mtd_from_sr(grid,srF); mI=E.mtd_from_sr(grid,srI)
    mt.append((w,mF,mI))
    print('w',w,'MTD fixed',mF,'integrated',mI,'time',round(time.time()-t0),flush=True)
    for gi,N in enumerate(grid): rows.append([w,N,srF[gi],srI[gi]])
with open('../data/e2_shuffle_sr.csv','w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['w','N','SR_fixed','SR_integrated']); wr.writerows(rows)
with open('../data/e2_shuffle_mtd.csv','w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['w','MTD_fixed','MTD_integrated']); wr.writerows(mt)
