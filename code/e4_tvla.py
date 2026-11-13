import sys, time, csv, numpy as np
import lbsca as L, ntt_emul as NT
Q=L.Q
rng=NT.rng=np.random.default_rng(4004)
sg=2.0; bsz=1000
grid=[250,500,1000,2000,5000,10000,20000]
fixed=NT.cbd2(rng,(256,))
gamma_list={'unprotected':None,'shuffled8':None,'masked':0.0,'masked_trans':1.0,'masked_trans_shuf8':1.0}
def gen(cfg,cls,n):
    s=np.tile(fixed,(n,1)) if cls==0 else NT.cbd2(rng,(n,256))
    if cfg=='unprotected': return NT.ntt_leak(s,sg,rng)
    if cfg=='shuffled8': return NT.ntt_leak(s,sg,rng,shuffle_w=8)
    w=8 if cfg.endswith('shuf8') else 1
    s0=L.center(rng.integers(0,Q,(n,256))); s1=L.center(s-s0)
    la=NT.ntt_leak(s0,sg,rng,w); lb=NT.ntt_leak(s1,sg,rng,w)
    g=gamma_list[cfg]
    parts=[la,lb]
    if g:   # distance (transition) leakage between consecutive loads of the two shares of coefficient i
        tl=g*L.hw16((s0^s1)).astype(np.float32)+rng.normal(0,sg,(n,256)).astype(np.float32)
        parts=[tl]+parts
    return np.concatenate(parts,axis=1)
rows=[]; layer_rows=[]
bsz=500; grid=[500,1000,2000,5000,10000,20000]
for cfg in gamma_list:
    t0=time.time()
    acc=[dict(n=0,s=None,s2=None),dict(n=0,s=None,s2=None)]
    for N in grid:
        while acc[0]['n']<N:
            for c in (0,1):
                x=gen(cfg,c,bsz).astype(np.float64); a=acc[c]
                if a['s'] is None: a['s']=x.sum(0); a['s2']=(x*x).sum(0)
                else: a['s']+=x.sum(0); a['s2']+=(x*x).sum(0)
                a['n']+=bsz
        n0=acc[0]['n']; n1=acc[1]['n']
        m0=acc[0]['s']/n0; m1=acc[1]['s']/n1
        v0=(acc[0]['s2']/n0-m0**2)*n0/(n0-1); v1=(acc[1]['s2']/n1-m1**2)*n1/(n1-1)
        tt=(m0-m1)/np.sqrt(v0/n0+v1/n1+1e-12)
        rows.append([cfg,n0,float(np.abs(tt).max())])
    off=256 if gamma_list[cfg] else 0
    layer_rows.append([cfg,0,float(np.abs(tt[:off]).max()) if off else 0.0])   # stage 0 = share split/load
    sizes=[ln_ja[1].size*5 for ln_ja in NT.LAYERS]; pos=off
    for li,sz in enumerate(sizes):
        layer_rows.append([cfg,li+1,float(np.abs(tt[pos:pos+sz]).max())]); pos+=sz
    print(cfg,'maxT final',float(np.abs(tt).max()),'time',round(time.time()-t0),flush=True)
with open('../data/e4_tvla_vs_N.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['config','N_per_class','max_abs_t']); w.writerows(rows)
with open('../data/e4_tvla_layers.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['config','layer','max_abs_t']); w.writerows(layer_rows)
