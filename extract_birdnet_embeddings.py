#!/usr/bin/env python
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# Must be set before importing birdnet.
os.environ.setdefault('BIRDNET_APP_DATA', r'C:\BirdNETData')
import soundfile as sf
import birdnet


def read_metadata(path):
    p=Path(path)
    return pd.read_csv(p, sep='\t' if p.suffix.lower() in {'.tsv','.tab'} else None,
                       engine='python' if p.suffix.lower() not in {'.tsv','.tab'} else 'c')


def resolve(root, rel):
    return Path(root) / str(rel).strip().replace('\\', os.sep).replace('/', os.sep)


def extract_one(model, path):
    x, sr = sf.read(str(path), dtype='float32')
    if x.ndim == 2: x = x.mean(axis=1)
    if x.ndim != 1 or x.size == 0: raise ValueError(f'Unexpected audio shape: {x.shape}')
    r = model.encode_arrays([(x, sr)], n_producers=1, n_workers=1, batch_size=1)
    
    emb = np.asarray(r.embeddings)
    if emb.ndim != 3 or emb.shape[0] != 1 or emb.shape[1] != 1:
        raise RuntimeError(
            f"Unexpected BirdNET embedding shape: {emb.shape}; "
            "expected (1, 1, 1024)"
        )

    emb = emb[0, 0, :].astype(np.float32, copy=False)

    if emb.shape != (1024,):
        raise RuntimeError(
          f"Unexpected embedding dimension: {emb.shape}; expected (1024,)"
     )

    if not np.all(np.isfinite(emb)):
        raise RuntimeError("Embedding contains NaN or infinite values.")

    return emb


def save(out, emb, status, failures, meta):
    np.save(out/'embeddings.npy', emb)
    status.to_csv(out/'extraction_status.csv', index=False)
    failures.to_csv(out/'failed_segments.csv', index=False)
    m=meta.copy(); m['embedding_status']=status.status.values; m['embedding_error']=status.error.values
    m.to_csv(out/'metadata_with_embedding_status.csv', index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--audio-root', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--checkpoint-every', type=int, default=50)
    a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    meta=read_metadata(a.metadata)
    if not {'segment_id','segment_file'}.issubset(meta.columns): raise ValueError('Metadata must contain segment_id and segment_file')
    n=len(meta)
    model=birdnet.load('acoustic','2.4','tf',library='litert')
    dim=int(model.get_embeddings_dim())
    if dim != 1024: raise RuntimeError(f'Expected 1024-D embeddings, got {dim}')
    emb=np.zeros((n,dim),np.float32)
    status=pd.DataFrame({'row_index':np.arange(n),'segment_id':meta.segment_id.astype(str),'status':['pending']*n,'error':['']*n})
    failures=[]
    if a.resume and (out/'embeddings.npy').exists() and (out/'extraction_status.csv').exists():
        emb=np.load(out/'embeddings.npy'); status=pd.read_csv(out/'extraction_status.csv')
        if emb.shape != (n,dim) or len(status)!=n: raise RuntimeError('Existing checkpoint shape/row count does not match metadata')
        if (out/'failed_segments.csv').exists(): failures=pd.read_csv(out/'failed_segments.csv').to_dict('records')
    print(f'BirdNET 2.4 LiteRT | dim={dim} | cache={os.environ.get("BIRDNET_APP_DATA")}')
    print(f'Rows: {n}')
    for i,row in meta.iterrows():
        if status.at[i,'status']=='success': continue
        p=resolve(a.audio_root,row.segment_file)
        try:
            if not p.exists(): raise FileNotFoundError(str(p))
            emb[i]=extract_one(model,p); status.at[i,'status']='success'; status.at[i,'error']=''
            print(f'[{i+1}/{n}] OK {row.segment_id}')
        except Exception as ex:
            err=f'{type(ex).__name__}: {ex}'; status.at[i,'status']='failed'; status.at[i,'error']=err
            failures.append({'row_index':i,'segment_id':row.segment_id,'segment_file':row.segment_file,'audio_path':str(p),'error':err})
            print(f'[{i+1}/{n}] FAIL {row.segment_id}: {err}', file=sys.stderr)
        processed=int((status.status!='pending').sum())
        if processed and processed % a.checkpoint_every==0:
            save(out,emb,status,pd.DataFrame(failures),meta); print('checkpoint saved')
    fail=pd.DataFrame(failures, columns=['row_index','segment_id','segment_file','audio_path','error'])
    save(out,emb,status,fail,meta)
    summary={'n_metadata_rows':n,'n_success':int((status.status=='success').sum()),'n_failed':int((status.status=='failed').sum()),'n_pending':int((status.status=='pending').sum()),'embeddings_shape':list(emb.shape),'dtype':str(emb.dtype),'birdnet_app_data':os.environ.get('BIRDNET_APP_DATA'),'birdnet_version':'2.4','backend':'litert','precision':'fp32','embedding_dimension':dim,'model_sample_rate_hz':48000,'segment_duration_s':3.0,'embedding_row_order':'identical_to_input_metadata'}
    (out/'extraction_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
