import sys, time, csv, numpy as np
import lbsca as L, engine as E
Q=L.Q
rng=np.random.default_rng(3003)
trials=int(sys.argv[1]) if len(sys.argv)>1 else 30
Ms=[L.second_order_table(j)[0] for j in (1,2,3)]
# part A: masked, second-order CPA vs noise (w=1)
sigmas=[1,2,4,8,16]
gridA={1:[5,10,20,40,80,150,300,600,1200],2:[5,10,20,40,80,150,300,600,1200,2500],
       4:[10,20,50,100,200,400,800,1500,3000,6000],8:[50,100,200,400,800,1500,3000,6000,12000,20000],
       16:[200,500,1000,2000,4000,8000,16000,32000]}
rowsA=[]; mA=[]
def leak3(s,u,beta=0.25):
    return L.hw_points(s,u,beta)[...,1:4]
for sg in sigmas:
    grid=gridA[sg]; Nmax=grid[-1]; res=[];tr=[]; t0=time.time()
    for k in range(trials):
        s=int(L.center(rng.integers(0,Q)))
        u=L.center(rng.integers(0,Q,Nmax)); uidx=E.residx(u)
        s0=L.center(rng.integers(0,Q,Nmax)); s1=L.center(s-s0)
        l0=leak3(s0,u)+rng.normal(0,sg,(Nmax,3)).astype(np.float32)
        l1=leak3(s1,u)+rng.normal(0,sg,(Nmax,3)).astype(np.float32)
        Cc=((l0-l0.mean(0))*(l1-l1.mean(0))).astype(np.float32)
        res.append(E.corr_checkpoints([Cc[:,j] for j in range(3)],[E.rowtable_getter(Ms[j],uidx) for j in range(3)],grid,s%Q)); tr.append(s%Q)
    sr=E.success_curve(res,tr); m=E.mtd_from_sr(grid,sr); mA.append(m)
    print('A sigma',sg,'MTD90',m,round(time.time()-t0),flush=True)
    for gi,N in enumerate(grid): rowsA.append([sg,N,sr[gi]])
with open('../data/e3a_masked_sr.csv','w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['sigma','N','SR']); wr.writerows(rowsA)
with open('../data/e3a_masked_mtd.csv','w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['sigma','MTD90']); [wr.writerow([a,b]) for a,b in zip(sigmas,mA)]
# part B: masked + shuffled, sigma=4
sg=4.0
cfgB={2:[50,100,200,400,800,1500,3000,6000],4:[100,200,500,1000,2000,4000,8000,16000],
      8:[200,500,1000,2000,4000,8000,16000,32000,48000]}
rowsB=[]; mB=[]
for w,grid in cfgB.items():
    Nmax=grid[-1]; resF=[];resI=[];tr=[]; t0=time.time()
    for k in range(max(10,trials//2)):
        s=int(L.center(rng.integers(0,Q)))
        pi=rng.integers(0,w,Nmax)
        u=L.center(rng.integers(0,Q,(Nmax,w)))
        s0=L.center(rng.integers(0,Q,(Nmax,w)))
        sa=L.center(rng.integers(0,Q,(Nmax,w)))                 # other coefficients
        s1=sa.copy(); s1[np.arange(Nmax),pi]=L.center(s-s0[np.arange(Nmax),pi])
        s1o=L.center(rng.integers(0,Q,(Nmax,w))); mask=np.ones((Nmax,w),bool); mask[np.arange(Nmax),pi]=False
        s1=np.where(mask,s1o,s1)
        l0=leak3(s0,u)+rng.normal(0,sg,(Nmax,w,3)).astype(np.float32)
        l1=leak3(s1,u)+rng.normal(0,sg,(Nmax,w,3)).astype(np.float32)
        a0=l0-l0.mean((0,1)); a1=l1-l1.mean((0,1))
        P3=a0*a1
        ut=u[np.arange(Nmax),pi]; uidx=E.residx(ut); g=[E.rowtable_getter(Ms[j],uidx) for j in range(3)]
        resF.append(E.corr_checkpoints([P3[:,0,j] for j in range(3)],g,grid,s%Q)); resI.append(E.corr_checkpoints([P3[:,:,j].sum(1) for j in range(3)],g,grid,s%Q)); tr.append(s%Q)
    srF=E.success_curve(resF,tr); srI=E.success_curve(resI,tr)
    mF=E.mtd_from_sr(grid,srF); mI=E.mtd_from_sr(grid,srI); mB.append((w,mF,mI))
    print('B w',w,'fixed',mF,'integrated',mI,round(time.time()-t0),flush=True)
    for gi,N in enumerate(grid): rowsB.append([w,N,srF[gi],srI[gi]])
with open('../data/e3b_masked_shuffled_sr.csv','w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['w','N','SR_fixed','SR_integrated']); wr.writerows(rowsB)
with open('../data/e3b_masked_shuffled_mtd.csv','w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['w','MTD_fixed','MTD_integrated']); wr.writerows(mB)
