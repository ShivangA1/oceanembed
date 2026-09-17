from __future__ import annotations
import json, math, os, sys
from datetime import date
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parent
DEPTHS=[0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000]
LAT_MIN,LAT_MAX,LON_MIN,LON_MAX=5.0,30.0,45.0,105.0
CHECKPOINT=ROOT/'models'/'checkpoints'/'best_model.pt'
PROCESSED=ROOT/'data'/'processed'
app=FastAPI(title='OceanEmbed AI-ML Service',version='1.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])

class PredictRequest(BaseModel):
    lat: float=Field(...,ge=LAT_MIN,le=LAT_MAX)
    lon: float=Field(...,ge=LON_MIN,le=LON_MAX)
    date: str

def demo(lat,lon,day):
    doy=day.timetuple().tm_yday
    seasonal=math.sin(2*math.pi*(doy-30)/365.25)
    surface=28.0-0.10*((lat-17.5)/12.5)+0.35*((lon-75)/30)+0.65*seasonal
    temps=[round(surface-(0.015*d+5*(1-math.exp(-d/120))),4) for d in DEPTHS]
    emb=[round(math.sin((i+1)*.17+lat*.03+lon*.01+doy*.005),6) for i in range(64)]
    return temps,emb

def real_prediction(lat,lon,day):
    if not CHECKPOINT.exists(): return None
    try:
        import torch
        from models.oceanembed_model import OceanEmbedModel
    except Exception: return None
    best=None; best_dist=float('inf')
    for split in ('test','val','train'):
        for path in sorted((PROCESSED/split).glob('shard_*.npz')):
            try:
                with np.load(path,allow_pickle=False) as data:
                    meta=json.loads(str(data['metadata'].item()))
                    for idx,s in enumerate(meta.get('samples',[])):
                        if str(s.get('date',''))[:10]!=day.isoformat(): continue
                        dist=(float(s['latitude'])-lat)**2+(float(s['longitude'])-lon)**2
                        if dist<best_dist: best_dist=dist; best=(path,idx,s)
            except Exception: continue
    if best is None: return None
    path,idx,meta=best
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    channels=int(checkpoint.get('in_channels',5)); emb_dim=int(checkpoint.get('embedding_dim',64)); out=int(checkpoint.get('output_depths',15))
    with np.load(path,allow_pickle=False) as data: x=data['X'][idx].astype(np.float32)
    if x.ndim!=3 or x.shape[-1]!=channels: raise RuntimeError(f'Checkpoint expects {channels} channels; sample has {x.shape[-1]}.')
    model=OceanEmbedModel(in_channels=channels,embedding_dim=emb_dim,output_dim=out)
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    with torch.no_grad(): e,p=model(torch.from_numpy(np.transpose(x,(2,0,1))[None,...]))
    return p[0].numpy().round(4).tolist(),e[0].numpy().round(6).tolist(),checkpoint.get('depths_m',DEPTHS),meta

@app.get('/health')
def health(): return {'status':'ok','mode':'trained-model' if CHECKPOINT.exists() else 'demo'}
@app.get('/region-bounds')
def bounds(): return {'lat_min':LAT_MIN,'lat_max':LAT_MAX,'lon_min':LON_MIN,'lon_max':LON_MAX,'depths':DEPTHS}
@app.post('/predict')
def predict(req:PredictRequest):
    try: day=date.fromisoformat(req.date)
    except ValueError: raise HTTPException(400,'Date must be YYYY-MM-DD')
    real=real_prediction(req.lat,req.lon,day)
    if real:
        temps,emb,depths,meta=real
        return {'lat':req.lat,'lon':req.lon,'date':req.date,'depths':depths,'temperatures':temps,'embedding':emb,'mode':'trained-model','matched_lat':meta.get('latitude'),'matched_lon':meta.get('longitude')}
    temps,emb=demo(req.lat,req.lon,day)
    return {'lat':req.lat,'lon':req.lon,'date':req.date,'depths':DEPTHS,'temperatures':temps,'embedding':emb,'mode':'demo','message':'Demo mode until best_model.pt and processed shards are available.'}
