"""Article 3 - preprocessing.
Usage: python prep3.py /path/to/greenhouse_dataset.csv
Output: irrigation_clean.parquet
"""
import sys
from datetime import datetime
import pandas as pd

FEATURES=['Soil Percent','Soil Raw','Suhu Tanah','Suhu Udara','Humidity']
path=sys.argv[1] if len(sys.argv)>1 else 'greenhouse_dataset.csv'
d=pd.read_csv(path,sep=';')
d=d[d['Label'].notna()].copy()
for c in FEATURES+['Label']:
    d[c]=pd.to_numeric(d[c],errors='coerce')

def parse_timestamp(s):
    for fmt in ('%Y-%m-%d_%H-%M-%S','%Y-%m-%d %H:%M:%S'):
        try: return datetime.strptime(str(s),fmt)
        except ValueError: pass
    return pd.NaT

d['ts']=d['Timestamp'].map(parse_timestamp)
d=d.dropna(subset=FEATURES+['Label','ts']).copy()
d['DateGroup']=d['ts'].dt.date.astype(str)
d=d.drop_duplicates(subset=FEATURES+['Label']).sort_values('ts').reset_index(drop=True)
d.to_parquet('irrigation_clean.parquet',index=False)
print('rows',len(d),'positives',int(d.Label.sum()),'days',d.DateGroup.nunique())
