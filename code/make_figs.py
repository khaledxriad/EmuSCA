import csv, math, os, collections
import numpy as np
D='../data/'; OUT_PDF='../figures/'
os.makedirs(OUT_PDF,exist_ok=True)
def rd(name):
    with open(D+name) as f: return list(csv.DictReader(f))
# ---------- load data ----------
e1sr=rd('e1_unprotected_sr.csv'); e1m=rd('e1_unprotected_mtd.csv')
e2m=rd('e2_shuffle_mtd.csv'); e3a=rd('e3a_masked_mtd.csv'); e3c=rd('e3c_masked_shuffled_s2_mtd.csv')
e4n=rd('e4_tvla_vs_N.csv'); e4l=rd('e4_tvla_layers.csv'); e5=rd('e5_validation.csv'); e6=rd('e6_cost.csv')
f=lambda x: float(x)
# ---------- chart specs ----------
SPEC={}
cfg_names={'unprotected':'Unprotected','shuffled8':'Shuffled ($w{=}8$)','masked':'Masked (ideal)','masked_trans':'Masked + transition leak','masked_trans_shuf8':'Masked + trans. + shuffled'}
cfg_names_plain={'unprotected':'Unprotected','shuffled8':'Shuffled (w=8)','masked':'Masked (ideal)','masked_trans':'Masked + transition leak','masked_trans_shuf8':'Masked + trans. + shuffled'}
# 1 TVLA vs N
ser=[]
for c in cfg_names:
    pts=[(f(r['N_per_class']),f(r['max_abs_t'])) for r in e4n if r['config']==c]
    ser.append((cfg_names[c],cfg_names_plain[c],[p[0] for p in pts],[p[1] for p in pts]))
SPEC['fig_tvla_n']=dict(kind='line',xlabel='Traces per class $N$',ylabel='Max $|t|$ over all samples',logx=True,logy=True,series=ser,hline=(4.5,'$|t|=4.5$'),ylim=(1,3000))
# 2 TVLA layers
layers=['S']+[str(i) for i in range(1,8)]
ser=[]
for c in cfg_names:
    d={int(r['layer']):f(r['max_abs_t']) for r in e4l if r['config']==c}
    ser.append((cfg_names[c],cfg_names_plain[c],layers,[max(d[i],1.0) for i in range(0,8)]))
SPEC['fig_tvla_layers']=dict(kind='bar',xlabel='Stage (S: share split/load; 1--7: NTT layers)',ylabel='Max $|t|$ (log scale)',logy=True,series=ser,hline=(4.5,'$|t|=4.5$'),ylim=(1,8000))
# 3 SR vs N
sig=sorted(set(int(r['sigma']) for r in e1sr)); ser=[]
for s in sig:
    pts=[(f(r['N']),f(r['SR'])) for r in e1sr if int(r['sigma'])==s]
    ser.append(('$\\sigma{=}%d$'%s,'sigma=%d'%s,[p[0] for p in pts],[p[1] for p in pts]))
SPEC['fig_sr_unprot']=dict(kind='line',xlabel='Number of traces $N$',ylabel='Success rate',logx=True,logy=False,series=ser,hline=(0.9,'SR $=0.9$'),ylim=(0,1.02))
# 4 MTD vs sigma (+model)
s1=[f(r['sigma']) for r in e1m]; m1=[f(r['MTD90']) for r in e1m]
a=[(f(r['sigma']),f(r['MTD90'])) for r in e3a if math.isfinite(f(r['MTD90']))]
s2=[p[0] for p in a]; m2=[p[1] for p in a]
th1={f(r['param']):f(r['theory']) for r in e5 if r['test']=='lemma_corr'}
th2={f(r['param']):f(r['theory']) for r in e5 if r['test']=='second_order'}
def fitc(sx,my,th):
    return math.exp(np.mean([math.log(m)+2*math.log(th[s]) for s,m in zip(sx,my)]))
c1=fitc(s1,m1,th1); c2=fitc(s2,m2,th2)
sg_fine=[1,2,4,8,16,32]
model1=[c1/th1[s]**2 for s in sg_fine]; model2=[c2/th2[s]**2 for s in sg_fine]
SPEC['fig_mtd_sigma']=dict(kind='line',xlabel='Noise standard deviation $\\sigma$',ylabel='Traces for 90\\% success',logx=True,logy=True,
  series=[('Unprotected (1st order)','Unprotected (1st order)',s1,m1,'marker'),('Masked (2nd order)','Masked (2nd order)',s2,m2,'marker'),
          ('Model, 1st order','Model, 1st order',sg_fine,model1,'dashed'),('Model, 2nd order','Model, 2nd order',sg_fine,model2,'dashed')],ylim=(3,1e6))
MODEL_CONST=(c1,c2)
# 5 MTD vs w
w_s=[int(f(r['w'])) for r in e2m]
base_masked=[f(r['MTD90']) for r in e3a if f(r['sigma'])==2][0]
wc=[1]+[int(f(r['w'])) for r in e3c]
mF=[base_masked]+[f(r['MTD_fixed']) for r in e3c]; mI=[base_masked]+[f(r['MTD_integrated']) for r in e3c]
ref=[1,2,4,8,16]
SPEC['fig_mtd_shuffle']=dict(kind='line',xlabel='Shuffling window $w$',ylabel='Traces for 90\\% success',logx=True,logy=True,
  series=[('Shuffled, fixed slot','Shuffled, fixed slot',w_s,[f(r['MTD_fixed']) for r in e2m],'marker'),
          ('Shuffled, integrated','Shuffled, integrated',w_s,[f(r['MTD_integrated']) for r in e2m],'marker'),
          ('Masked+shuffled, fixed slot','Masked+shuffled, fixed slot',wc,mF,'marker'),
          ('Masked+shuffled, integrated','Masked+shuffled, integrated',wc,mI,'marker'),
          ('slope 2 ($w^2$)','slope 2 (w^2)',ref,[14.7*x*x for x in ref],'dashed'),
          ('slope 1 ($w$)','slope 1 (w)',ref,[14.7*x for x in ref],'dashed')],ylim=(5,1e5))
# 6 frontier
cost={(r['config'],int(r['w'])):f(r['overhead']) for r in e6}
sh=[(cost[('shuffled',w)],min(f(r['MTD_fixed']),f(r['MTD_integrated']))) for w,r in zip(w_s[1:],e2m[1:])]
ms=[(cost[('masked+shuffled',w)],min(f(r['MTD_fixed']),f(r['MTD_integrated']))) for w,r in zip(wc[1:],e3c)]
unp=(1.0,f(e1m[1]['MTD90'])); mk=(cost[('masked',1)],base_masked)
SPEC['fig_frontier']=dict(kind='line',xlabel='Estimated cost overhead (x unprotected)',ylabel='Traces for 90\\% success ($\\sigma{=}2$)',logx=False,logy=True,
  series=[('Shuffling only','Shuffling only',[unp[0]]+[p[0] for p in sh],[unp[1]]+[p[1] for p in sh],'marker'),
          ('Masking + shuffling','Masking + shuffling',[mk[0]]+[p[0] for p in ms],[mk[1]]+[p[1] for p in ms],'marker')],ylim=(5,1e5))
# 7 overhead vs literature (bar horizontal -> treat as vertical bars w/ rotated labels)
lit_over=[('Ours: shuffled $w{=}8$ (model)','Ours: shuffled w=8 (model)',cost[('shuffled',8)]),('Ours: masked (model)','Ours: masked (model)',cost[('masked',1)]),
 ('Ours: masked+shuffled $w{=}8$ (model)','Ours: masked+shuffled w=8 (model)',cost[('masked+shuffled',8)]),
 ('Bos et al. masked decaps [bos21]','Bos et al. masked decaps',3.5),('Ait Manssour et al. decaps [aitmanssour26]','Ait Manssour et al. decaps',1.25),
 ('Bitwise fine shuffling, decaps [aitmanssour26]','Bitwise fine shuffling, decaps',1.51),('Nagpal et al. RNR, NTT only [nagpal25]','Nagpal et al. RNR (NTT only)',1.628)]
SPEC['fig_overhead_lit']=dict(kind='hbar',xlabel='Overhead factor (x unprotected)',items=lit_over,xlim=(0,4.2))
# 8 traces vs literature
tr_items=[('Kennaway et al. [kennaway25], unprot.','Kennaway et al., unprotected (HW)',179),('Tosun et al. [tosun25], 2nd-order masked','Tosun et al., 2nd-order masked',2200),('Tosun et al. [tosun25], 3rd-order masked','Tosun et al., 3rd-order masked',14500),
 ('Ours: unprotected, $\\sigma{=}8$ (emul.)','Ours: unprotected, sigma=8 (emul.)',f(e1m[3]['MTD90'])),('Ours: masked, $\\sigma{=}4$ (emul.)','Ours: masked, sigma=4 (emul.)',f(e3a[2]['MTD90'])),('Ours: masked, $\\sigma{=}8$ (emul.)','Ours: masked, sigma=8 (emul.)',f(e3a[3]['MTD90']))]
SPEC['fig_traces_lit']=dict(kind='hbar',xlabel='Traces to recover the key',items=tr_items,logx=True,xlim=(50,40000))
# ---------- matplotlib ----------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':6.5,'axes.labelsize':7,'legend.fontsize':5.8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':0.5,
  'lines.linewidth':0.9,'lines.markersize':2.6,'font.family':'serif','mathtext.fontset':'cm','pdf.fonttype':42})
MK=['o','s','^','D','v','P']; LS=['-','-','-','-','-']
for name,sp in SPEC.items():
    fig,ax=plt.subplots(figsize=(3.45,2.45) if sp['kind']!='hbar' else (3.45,2.2),constrained_layout=True)
    if sp['kind']=='line':
        k=0
        for s in sp['series']:
            lab,_,x,y=s[:4]; st=s[4] if len(s)>4 else 'marker'
            if st=='dashed': ax.plot(x,y,'--',color='gray',lw=0.7,label=lab)
            else: ax.plot(x,y,marker=MK[k%6],label=lab); k+=1
        if sp.get('hline'): ax.axhline(sp['hline'][0],color='k',ls=':',lw=0.7,label=sp['hline'][1])
        if sp.get('logx'): ax.set_xscale('log')
        if sp.get('logy'): ax.set_yscale('log')
        if sp.get('ylim'): ax.set_ylim(*sp['ylim'])
        ax.set_xlabel(sp['xlabel']); ax.set_ylabel(sp['ylabel']); ax.grid(True,which='major',lw=0.3,alpha=0.6)
        ax.legend(frameon=True,framealpha=0.9,borderpad=0.3,labelspacing=0.25,handlelength=1.6,loc=('upper left' if name in ('fig_tvla_n','fig_mtd_shuffle','fig_frontier') else 'lower right' if name in ('fig_sr_unprot','fig_mtd_sigma') else 'best'))
    elif sp['kind']=='bar':
        n=len(sp['series']); wd=0.8/n; xs=np.arange(8)
        for i,s in enumerate(sp['series']): ax.bar(xs+(i-(n-1)/2)*wd,s[3],wd,label=s[0],bottom=0)
        ax.set_xticks(xs); ax.set_xticklabels(sp['series'][0][2]); ax.set_yscale('log'); ax.set_ylim(*sp['ylim'])
        ax.axhline(sp['hline'][0],color='k',ls=':',lw=0.7,label=sp['hline'][1])
        ax.set_xlabel(sp['xlabel']); ax.set_ylabel(sp['ylabel']); ax.legend(ncol=2,frameon=True,framealpha=0.9,columnspacing=0.8,handlelength=1.2,loc='upper right')
        ax.grid(True,axis='y',which='major',lw=0.3,alpha=0.6)
    else:
        items=sp['items']; yy=np.arange(len(items))[::-1]
        cols=['#4c72b0' if it[1].startswith('Ours') else '#c44e52' for it in items]
        ax.barh(yy,[it[2] for it in items],color=cols,height=0.62)
        ax.set_yticks(yy); ax.set_yticklabels([it[1] for it in items])
        if sp.get('logx'): ax.set_xscale('log')
        ax.set_xlim(*sp['xlim']); ax.set_xlabel(sp['xlabel']); ax.grid(True,axis='x',which='major',lw=0.3,alpha=0.6)
        for yi,it in zip(yy,items): ax.text(it[2]*(1.06 if sp.get('logx') else 1)+(0 if sp.get('logx') else 0.04),yi,('%.4g'%it[2]),va='center',fontsize=5.8)
    fig.savefig(OUT_PDF+name+'.pdf'); plt.close(fig)
print('model constants',MODEL_CONST)
import json; json.dump({k:{kk:vv for kk,vv in v.items() if kk in ('series','items')} for k,v in SPEC.items()},open(D+'chart_data.json','w'),default=str)
