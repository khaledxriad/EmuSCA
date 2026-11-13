import csv, numpy as np
import lbsca as L, engine as E
Q=L.Q; rng=np.random.default_rng(5005)
N=400000
s=int(L.center(rng.integers(0,Q)))
T=L.hw_tables()
u=L.center(rng.integers(0,Q,N)); uidx=E.residx(u)
def pear(a,b):
    a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
Hm=T[3][s%Q,uidx].astype(np.float64)            # model, true key, point 3 (result)
clean=L.hw_points(np.full(N,s),u,0.25)[:,3].astype(np.float64)
V=clean.var(); rho0=pear(clean,Hm)
rows=[]
for sg in [0,1,2,4,8,16,32]:
    l=clean+rng.normal(0,sg,N)
    emp=pear(l,Hm); th=rho0/np.sqrt(1+sg**2/V)
    rows.append(['lemma_corr',sg,emp,th])
# shuffling dilution lemma: slot-0 correlation vs w, and integrated
sg=2.0
base=None
for w in [1,2,4,8,16]:
    Nn=200000
    uu=L.center(rng.integers(0,Q,(Nn,w))); ss=L.center(rng.integers(0,Q,(Nn,w))); pi=rng.integers(0,w,Nn)
    ss[np.arange(Nn),pi]=s
    lk=L.hw_points(ss,uu,0.25)[...,3]+rng.normal(0,sg,(Nn,w))
    ut=uu[np.arange(Nn),pi]; H=T[3][s%Q,E.residx(ut)].astype(np.float64)
    rf=pear(lk[:,0].astype(np.float64),H); ri=pear(lk.sum(1).astype(np.float64),H)
    if base is None: base=rf
    rows.append(['shuffle_fixed',w,rf,base/w]); rows.append(['shuffle_integrated',w,ri,base/np.sqrt(w)])
# second-order lemma: rho2 vs sigma
M,mu=L.second_order_table(3)
Nn=400000
u=L.center(rng.integers(0,Q,Nn)); uidx=E.residx(u)
s0=L.center(rng.integers(0,Q,Nn)); s1=L.center(s-s0)
c0=L.hw_points(s0,u,0.25)[:,3].astype(np.float64); c1=L.hw_points(s1,u,0.25)[:,3].astype(np.float64)
Hm=M[uidx,s%Q].astype(np.float64)
for sg in [1,2,4,8,16,32]:
    l0=c0+rng.normal(0,sg,Nn); l1=c1+rng.normal(0,sg,Nn)
    C=(l0-l0.mean())*(l1-l1.mean())
    emp=pear(C,Hm)
    # theory: rho2 = sd(m)/sqrt(Var(C)) with Var(C) estimated from the model constants
    a=c0-c0.mean(); b=c1-c1.mean()
    varC=sg**4+sg**2*(a.var()+b.var())+ (a*a*b*b).mean()-((a*b).mean())**2
    sdm=Hm.std()
    rows.append(['second_order',sg,emp,sdm/np.sqrt(varC)])
with open('../data/e5_validation.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['test','param','empirical','theory']); w.writerows(rows)
print('V=',V,'rho0=',rho0)
for r in rows: print(r)
