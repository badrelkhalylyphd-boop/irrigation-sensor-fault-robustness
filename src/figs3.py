"""Generate the three article figures and compact table images."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path('figures'); OUT.mkdir(exist_ok=True)
r=pd.read_csv('article3_results.csv'); c=pd.read_csv('article3_sensor_criticality.csv')
faults=['clean','missing','noise','bias','stuck','drift']
fig,ax=plt.subplots(figsize=(6.4,3.3),dpi=220); x=np.arange(len(faults)); w=.36
for i,tr in enumerate(['standard','aware']):
    g=r[r.training==tr].groupby('fault').F1.agg(['mean','std']).reindex(faults)
    ax.bar(x+(i-.5)*w,g['mean']*100,w,yerr=g['std']*100,capsize=2,label='Standard' if tr=='standard' else 'Fault-aware')
ax.set_xticks(x); ax.set_xticklabels(['Clean','Missing','Noise','Bias','Stuck','Drift']); ax.set_ylabel('F1-score (%)'); ax.set_ylim(0,60); ax.legend(frameon=False,ncol=2)
ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'fig1_fault_f1.png',bbox_inches='tight'); plt.close(fig)
p=c[c.training=='standard'].pivot(index='sensor',columns='fault',values='F1').reindex(index=['Soil Percent','Soil Raw','Suhu Tanah','Suhu Udara','Humidity'],columns=['missing','noise','bias','drift'])
fig,ax=plt.subplots(figsize=(6.4,3.5),dpi=220); im=ax.imshow(p.values*100,aspect='auto',vmin=0,vmax=55); ax.set_xticks(range(4)); ax.set_xticklabels(['Missing','Noise','Bias','Drift'])
ax.set_yticks(range(5)); ax.set_yticklabels(['Soil moisture %','Soil raw','Soil temp.','Air temp.','Humidity'])
for i in range(5):
    for j in range(4): ax.text(j,i,f'{p.values[i,j]*100:.1f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=ax).set_label('F1-score (%)'); fig.tight_layout(); fig.savefig(OUT/'fig2_sensor_criticality.png',bbox_inches='tight'); plt.close(fig)
clean=r[r.fault=='clean'].groupby('training').F1.mean(); stress=r[r.fault!='clean'].groupby('training').F1.mean()
vals=[clean['standard']*100,stress['standard']*100,clean['aware']*100,stress['aware']*100]
fig,ax=plt.subplots(figsize=(5.8,3.1),dpi=220); ax.bar(['Std clean','Std faults','Aware clean','Aware faults'],vals)
for i,v in enumerate(vals): ax.text(i,v+1,f'{v:.1f}',ha='center',fontsize=8)
ax.set_ylabel('F1-score (%)'); ax.set_ylim(0,60); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'fig3_retention.png',bbox_inches='tight'); plt.close(fig)
