import numpy as np
import lbsca as L
Q=L.Q

def brv7(k): return int('{:07b}'.format(k)[::-1],2)
def zetas_mont():
    z=[int(L.center((pow(17,brv7(k),Q)*65536)%Q)) for k in range(128)]
    return np.array(z,dtype=np.int64)
ZET=zetas_mont()

def layer_orders():
    """Butterfly index lists (top index ja, zeta index) for the 7 layers, in reference-code order."""
    layers=[]; k=1
    ln=128
    while ln>=2:
        ja=[];zk=[]
        for start in range(0,256,2*ln):
            zeta_idx=k; k+=1
            for j in range(start,start+ln):
                ja.append(j); zk.append(zeta_idx)
        layers.append((ln,np.array(ja),np.array(zk)))
        ln>>=1
    return layers
LAYERS=layer_orders()

def ntt_leak(r, noise_sigma, rng, shuffle_w=1):
    """r: (N,256) signed coefficients. Returns leakage array (N, 896*5) float32 in execution order.
    5 points per butterfly: HW16(b), HW32(zeta*b), HW16(t), HW16(a-t), HW16(a+t)."""
    r=r.astype(np.int64).copy(); N=r.shape[0]; chunks=[]; per_layer=[]
    for ln,ja,zk in LAYERS:
        a=r[:,ja]; b=r[:,ja+ln]
        z=ZET[zk][None,:]
        p=z*b; t,_=L.mont_reduce(p)
        lo=a-t; hi=a+t
        r[:,ja+ln]=lo; r[:,ja]=hi
        lk=np.stack([L.hw16(b),L.hw32(p),L.hw16(t),L.hw16(lo),L.hw16(hi)],axis=-1).astype(np.float32) # (N,128,5)
        if shuffle_w>1:
            g=128//shuffle_w
            perm=np.argsort(rng.random((N,g,shuffle_w)),axis=2)
            idx=(np.arange(g)[None,:,None]*shuffle_w+perm).reshape(N,128)
            lk=np.take_along_axis(lk,idx[:,:,None],axis=1)
        chunks.append(lk.reshape(N,-1))
    out=np.concatenate(chunks,axis=1)
    return out+rng.normal(0,noise_sigma,out.shape).astype(np.float32)

def cbd2(rng,shape):
    a=rng.integers(0,2,shape+(4,)).astype(np.int64)
    return a[...,0]+a[...,1]-a[...,2]-a[...,3]
