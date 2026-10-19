"""Article 3 - grouped sensor-fault stress test.
Usage: python eval3.py
Requires irrigation_clean.parquet from prep3.py.
Outputs article3_results.csv and article3_sensor_criticality.csv.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (f1_score,matthews_corrcoef,recall_score,precision_score,
    balanced_accuracy_score,average_precision_score,confusion_matrix)
from scipy.stats import wilcoxon
import lightgbm as lgb

SEED=42
FEATURES=['Soil Percent','Soil Raw','Suhu Tanah','Suhu Udara','Humidity']
d=pd.read_parquet('irrigation_clean.parquet')
X=d[FEATURES].astype(float); y=d.Label.astype(int).to_numpy(); groups=d.DateGroup.to_numpy()
cv=StratifiedGroupKFold(n_splits=5)  # no shuffle: deterministic, as used for the reported results
splits=list(cv.split(X,y,groups))

def build_model():
    return lgb.LGBMClassifier(n_estimators=150,num_leaves=31,learning_rate=.05,
        class_weight='balanced',random_state=SEED,n_jobs=-1,verbose=-1)

def augment(Xt,yt,std,rng):
    m=int(.5*len(Xt)); idx=rng.choice(len(Xt),size=m,replace=True)
    A=Xt.iloc[idx].copy().reset_index(drop=True); ya=yt[idx].copy(); z=A.to_numpy()
    blocks=np.array_split(np.arange(m),3)
    rr=blocks[0]; cc=rng.integers(0,A.shape[1],size=len(rr)); z[rr,cc]=np.nan
    rr=blocks[1]; cc=rng.integers(0,A.shape[1],size=len(rr)); z[rr,cc]+=rng.normal(0,.5,size=len(rr))*std[cc]
    rr=blocks[2]; cc=rng.integers(0,A.shape[1],size=len(rr)); z[rr,cc]+=rng.choice([-1,1],size=len(rr))*std[cc]
    A.iloc[:,:]=z
    return pd.concat([Xt.reset_index(drop=True),A],ignore_index=True), np.concatenate([yt,ya])

def corrupt(Xv,kind,std,med,rng):
    Z=Xv.copy(); a=Z.to_numpy(copy=True); n,p=a.shape
    if kind=='missing':
        mask=rng.random((n,p))<.08; a[mask]=np.nan
    elif kind=='noise':
        mask=rng.random((n,p))<.15; noise=rng.normal(size=a.shape)*std[None,:]*.5; a[mask]+=noise[mask]
    elif kind=='bias':
        j=int(rng.integers(0,p)); rows=rng.random(n)<.40; a[rows,j]+=rng.choice([-1,1])*std[j]
    elif kind=='stuck':
        j=int(rng.integers(0,p)); L=max(1,int(.25*n)); start=int(rng.integers(0,max(1,n-L))); a[start:start+L,j]=med[j]
    elif kind=='drift':
        j=int(rng.integers(0,p)); a[:,j]+=np.linspace(0,std[j],n)
    return pd.DataFrame(a,columns=Xv.columns,index=Xv.index)

def score(yt,pr,pb):
    tn,fp,fn,tp=confusion_matrix(yt,pr,labels=[0,1]).ravel()
    return dict(F1=f1_score(yt,pr),MCC=matthews_corrcoef(yt,pr),Recall=recall_score(yt,pr),
        Precision=precision_score(yt,pr,zero_division=0),BAcc=balanced_accuracy_score(yt,pr),
        PRAUC=average_precision_score(yt,pb),TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn))

models={'standard':[],'aware':[]}; stats=[]
for fi,(tr,va) in enumerate(splits):
    Xt=X.iloc[tr]; yt=y[tr]; std=Xt.std().replace(0,1).to_numpy(); med=Xt.median().to_numpy(); stats.append((std,med))
    models['standard'].append(build_model().fit(Xt,yt))
    Xa,ya=augment(Xt,yt,std,np.random.default_rng(500+fi))
    models['aware'].append(build_model().fit(Xa,ya))

FAULTS=['clean','missing','noise','bias','stuck','drift']; rows=[]
for seed in range(5):
    for training in ['standard','aware']:
        for fault in FAULTS:
            yy=[]; pp=[]; bb=[]
            for fi,(tr,va) in enumerate(splits):
                Xv=X.iloc[va]; yv=y[va]; std,med=stats[fi]
                Zv=Xv if fault=='clean' else corrupt(Xv,fault,std,med,np.random.default_rng(10000+seed*100+fi*10+FAULTS.index(fault)))
                pb=models[training][fi].predict_proba(Zv)[:,1]; pr=(pb>=.5).astype(int)
                yy.append(yv); pp.append(pr); bb.append(pb)
            rows.append(dict(seed=seed,training=training,fault=fault,**score(np.concatenate(yy),np.concatenate(pp),np.concatenate(bb))))

res=pd.DataFrame(rows); res.to_csv('article3_results.csv',index=False)

# Sensor-criticality experiment: complete single-channel faults.
crit=[]
for training in ['standard','aware']:
    for j,name in enumerate(FEATURES):
        for fault in ['missing','noise','bias','drift']:
            yy=[];pp=[];bb=[]
            for fi,(tr,va) in enumerate(splits):
                Xv=X.iloc[va].copy(); yv=y[va]; std,med=stats[fi]; a=Xv.to_numpy(copy=True); n=len(a)
                rng=np.random.default_rng(9000+j*100+fi*10+['missing','noise','bias','drift'].index(fault))
                if fault=='missing': a[:,j]=np.nan
                elif fault=='noise': a[:,j]+=rng.normal(0,std[j],size=n)
                elif fault=='bias': a[:,j]+=std[j]
                elif fault=='drift': a[:,j]+=np.linspace(0,std[j],n)
                Z=pd.DataFrame(a,columns=FEATURES,index=Xv.index); pb=models[training][fi].predict_proba(Z)[:,1]; pr=(pb>=.5).astype(int)
                yy.append(yv);pp.append(pr);bb.append(pb)
            crit.append(dict(training=training,sensor=name,fault=fault,**score(np.concatenate(yy),np.concatenate(pp),np.concatenate(bb))))
pd.DataFrame(crit).to_csv('article3_sensor_criticality.csv',index=False)

# Paired statistical tests over 25 fault realizations.
s=res[res.fault!='clean'].sort_values(['fault','seed'])
a=s[s.training=='aware']; b=s[s.training=='standard']
for metric in ['F1','MCC','PRAUC']:
    stat,p=wilcoxon(a[metric].to_numpy(),b[metric].to_numpy(),alternative='greater')
    print(metric,'mean gain',float((a[metric].to_numpy()-b[metric].to_numpy()).mean()),'p',p)
print(res.groupby(['training','fault'])[['F1','MCC','Recall','Precision','PRAUC']].agg(['mean','std']))
