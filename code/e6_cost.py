import csv
# Assumed Cortex-M4 instruction costs (cycles). RNG word = 32 random bits from an on-chip TRNG.
LD=2; ST=2; ALU=1; MUL=1
R=40                       # cycles to obtain one 32-bit random word (assumption)
FQMUL=MUL+MUL+ALU+MUL+ALU  # smulbb, mul, sxth, mls/smlabb, asr
BFLY=3*LD+FQMUL+2*ALU+2*ST # loads a,b,zeta; fqmul; add,sub; two stores
NBFLY=896
NTT=NBFLY*BFLY
BM=4*LD+5*FQMUL+3*ALU+2*ST # one pair: 4 loads, 5 fqmul, 3 add/sub, 2 stores
NBM=128
BASEMUL=NBM*BM
BASE=NTT+BASEMUL
SPLIT=256*(R/2+LD+ALU*3+2*ST)   # draw s0 (two 16-bit values per word), reduce, s1=s-s0, store both
RECOMB=256*(2*LD+2*ALU+ST)
def shuf_cost(n_items, w):
    if w<=1: return 0.0
    groups=n_items/w
    swap=(R/4+2*LD+2*ST)           # one random byte per Fisher-Yates swap
    perm=groups*(w-1)*swap
    indirection=n_items*3           # index load + address arithmetic per item
    return perm+indirection
rows=[]
def add(name,w,cyc): rows.append([name,w,round(cyc),round(cyc/BASE,3)])
add('unprotected',1,BASE)
for w in (2,4,8,16):
    add('shuffled',w,BASE+shuf_cost(NBFLY,w)+shuf_cost(NBM,w))
add('masked',1,2*NTT+2*BASEMUL+SPLIT+RECOMB)
for w in (2,4,8):
    # both shares use the same permutation (generated once)
    add('masked+shuffled',w,2*NTT+2*BASEMUL+SPLIT+RECOMB+shuf_cost(NBFLY,w)+shuf_cost(NBM,w)+(2*NBFLY+2*NBM)*3)
with open('../data/e6_cost.csv','w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['config','w','cycles','overhead']); wr.writerows(rows)
print('BFLY',BFLY,'NTT',NTT,'BM',BM,'BASEMUL',BASEMUL,'BASE',BASE,'SPLIT',SPLIT,'RECOMB',RECOMB)
for r in rows: print(r)
