#!/usr/bin/env python3
"""Research-grade audit for the RANA BirdNET embedding dataset."""
from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
EXPECTED_DIM = 1024

def log(x=""): print(x, flush=True)
def norm(s): return s.astype("string").str.strip().replace({"":pd.NA,"nan":pd.NA,"None":pd.NA})
def col(df, names):
    for n in names:
        if n in df.columns: return n
    return None

def check(ok, p, f, msg):
    log(("[PASS] " if ok else "[FAIL] ")+msg)
    return (p+1,f) if ok else (p,f+1)

def main():
    ap=argparse.ArgumentParser(description="Audit RANA BirdNET embeddings and metadata alignment")
    ap.add_argument("--dataset-dir",default=r"D:\Acoustics\AnuraSet_3sec_all")
    ap.add_argument("--embedding-dir")
    ap.add_argument("--metadata")
    ap.add_argument("--targets")
    ap.add_argument("--report-dir")
    ap.add_argument("--expected-count",type=int,default=25330)
    ap.add_argument("--chunk-rows",type=int,default=2048)
    a=ap.parse_args()
    root=Path(a.dataset_dir); ed=Path(a.embedding_dir) if a.embedding_dir else root/"birdnet_embeddings"
    ep=ed/"embeddings.npy"; emp=ed/"embedding_metadata.csv"; fp=ed/"embedding_failures.csv"; sp=ed/"extraction_summary.json"
    mp=Path(a.metadata) if a.metadata else root/"metadata.csv"; tp=Path(a.targets) if a.targets else root/"segment_targets.csv"
    rd=Path(a.report_dir) if a.report_dir else ed/"audit"; rd.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); jp=rd/f"embedding_audit_{stamp}.json"; xp=rd/f"embedding_audit_{stamp}.txt"
    P=F=0; warnings=[]; errors=[]; out={"files":{}}
    log("="*76); log("RANA / BirdNET embedding dataset audit"); log("="*76)
    log(f"Dataset root: {root}"); log(f"Embeddings: {ep}"); log(f"Embedding metadata: {emp}"); log(f"Metadata: {mp}"); log(f"Targets: {tp}")
    required={"embeddings":ep,"embedding_metadata":emp,"metadata":mp}
    optional={"targets":tp,"embedding_failures":fp,"extraction_summary":sp}
    for k,p in {**required,**optional}.items():
        ok=p.exists(); out["files"][k]={"path":str(p),"exists":ok}
        if ok: log(f"[PASS] {k}: {p}"); P+=1
        elif k in optional: log(f"[WARN] {k} not found: {p}"); warnings.append(f"Optional file not found: {p}")
        else: log(f"[FAIL] Required file not found: {p}"); F+=1; errors.append(f"Required file not found: {p}")
    if not ep.exists() or not emp.exists():
        out["status"]="FAIL"; out["summary"]={"passed_checks":P,"failed_checks":F,"warnings":len(warnings)}
        jp.write_text(json.dumps(out,indent=2),encoding="utf-8"); xp.write_text("RANA embedding audit\n\n"+"\n".join(errors)+"\n",encoding="utf-8")
        return 2
    try: emb=np.load(ep,mmap_mode="r")
    except Exception as e:
        errors.append(str(e)); F+=1; out["status"]="FAIL"; jp.write_text(json.dumps(out,indent=2),encoding="utf-8"); xp.write_text(str(e),encoding="utf-8"); return 2
    out["embeddings"]={"shape":list(emb.shape),"dtype":str(emb.dtype),"ndim":emb.ndim}; n=emb.shape[0] if emb.ndim>=1 else 0
    P,F=check(emb.ndim==2,P,F,f"embeddings.npy is 2-D (ndim={emb.ndim})")
    if emb.ndim==2: P,F=check(emb.shape[1]==EXPECTED_DIM,P,F,f"Embedding dimension is {EXPECTED_DIM} (found {emb.shape[1]})")
    if a.expected_count is not None: P,F=check(n==a.expected_count,P,F,f"Embedding row count is {a.expected_count} (found {n})")
    nan=inf=0; mn=math.inf; mx=-math.inf
    if emb.ndim==2:
        for i in range(0,n,a.chunk_rows):
            b=np.asarray(emb[i:min(i+a.chunk_rows,n)]); nan+=int(np.isnan(b).sum()); inf+=int(np.isinf(b).sum()); v=b[np.isfinite(b)]
            if v.size: mn=min(mn,float(v.min())); mx=max(mx,float(v.max()))
    out["embeddings"].update({"nan_values":nan,"inf_values":inf,"min_finite_value":None if mn==math.inf else mn,"max_finite_value":None if mx==-math.inf else mx})
    P,F=check(nan==0,P,F,f"No NaN values (found {nan})"); P,F=check(inf==0,P,F,f"No infinite values (found {inf})")
    em=pd.read_csv(emp,encoding="utf-8-sig",low_memory=False); out["embedding_metadata"]={"rows":len(em),"columns":list(em.columns)}
    P,F=check(len(em)==n,P,F,f"embedding_metadata rows match embeddings ({len(em)} == {n})")
    ec=col(em,["segment_id","segment","id"]); P,F=check(ec is not None,P,F,f"Embedding metadata contains segment_id (found {ec})")
    eids=norm(em[ec]) if ec else pd.Series(dtype="string")
    if ec:
        mi=int(eids.isna().sum()); du=int(eids.dropna().duplicated().sum()); out["embedding_metadata"].update({"segment_id_column":ec,"missing_segment_ids":mi,"duplicate_segment_ids":du})
        P,F=check(mi==0,P,F,f"No missing embedding segment_id values (found {mi})"); P,F=check(du==0,P,F,f"Embedding segment_id values are unique (duplicates={du})")
    meta=pd.read_csv(mp,encoding="utf-8-sig",low_memory=False); mc=col(meta,["segment_id","segment","id"]); out["metadata"]={"rows":len(meta),"columns":list(meta.columns),"segment_id_column":mc}
    P,F=check(mc is not None,P,F,f"Main metadata contains segment_id (found {mc})")
    if mc and ec:
        mids=norm(meta[mc]); mi=int(mids.isna().sum()); du=int(mids.dropna().duplicated().sum()); P,F=check(mi==0,P,F,f"Main metadata has no missing segment_id values (found {mi})"); P,F=check(du==0,P,F,f"Main metadata segment_id values are unique (duplicates={du})")
        P,F=check(len(mids)==len(eids),P,F,f"metadata.csv row count matches embedding metadata ({len(mids)} == {len(eids)})")
        if len(mids)==len(eids):
            exact=mids.fillna("<NA>").reset_index(drop=True).equals(eids.fillna("<NA>").reset_index(drop=True)); P,F=check(exact,P,F,"embedding_metadata segment_id order exactly matches metadata.csv")
            if not exact:
                es=set(eids.dropna()); ms=set(mids.dropna()); warnings.append(f"ID set differences: embeddings-only={len(es-ms)}, metadata-only={len(ms-es)}")
    if "site_id" in meta.columns:
        vc=meta["site_id"].astype("string").fillna("<MISSING>").value_counts(); out["site_distribution"]={str(k):int(v) for k,v in vc.items()}; log("\nSite distribution:"); [log(f"  {k}: {v}") for k,v in vc.items()]
    if "source_type" in meta.columns:
        vc=meta["source_type"].astype("string").fillna("<MISSING>").value_counts(); out["source_type_distribution"]={str(k):int(v) for k,v in vc.items()}; log("\nSource type distribution:"); [log(f"  {k}: {v}") for k,v in vc.items()]
    if "recording_id" in meta.columns:
        r=norm(meta["recording_id"]); mi=int(r.isna().sum()); vc=r.value_counts(); out["recordings"]={"unique":int(r.nunique(dropna=True)),"missing":mi,"min_segments":int(vc.min()) if len(vc) else None,"median_segments":float(vc.median()) if len(vc) else None,"max_segments":int(vc.max()) if len(vc) else None}; P,F=check(mi==0,P,F,f"No missing recording_id values (found {mi})")
    if tp.exists() and ec:
        try:
            t=pd.read_csv(tp,encoding="utf-8-sig",low_memory=False); tc=col(t,["segment_id","segment","id"])
            if tc:
                tids=set(norm(t[tc]).dropna()); eis=set(eids.dropna()); miss=eis-tids; extra=tids-eis; out["targets"]={"rows":len(t),"missing_target_for_embedding":len(miss),"extra_target_ids":len(extra)}; P,F=check(not miss,P,F,f"Every embedding segment_id has a segment_targets row (missing={len(miss)})")
                if extra: warnings.append(f"segment_targets contains {len(extra)} IDs not present in embeddings")
            else: warnings.append("segment_targets.csv has no segment_id column")
        except Exception as e: warnings.append(f"Could not audit targets: {e}")
    if fp.exists():
        try:
            fdf=pd.read_csv(fp,encoding="utf-8-sig",low_memory=False); P,F=check(len(fdf)==0,P,F,f"Embedding failure audit contains zero failures (rows={len(fdf)})")
        except Exception as e: warnings.append(f"Could not read failure file: {e}")
    if sp.exists():
        try:
            s=json.loads(sp.read_text(encoding="utf-8")); out["extraction_summary"]=s
            for keys,expected,msg in [(["successful","successes","successful_count"],n,"successful count matches embeddings"),(["failures","failure_count","failed"],0,"extraction summary reports zero failures")]:
                val=next((s[k] for k in keys if k in s),None)
                if val is not None: P,F=check(int(val)==expected,P,F,f"{msg} ({val})")
        except Exception as e: warnings.append(f"Could not parse extraction summary: {e}")
    status="PASS" if F==0 else "FAIL"; out.update({"status":status,"warnings":warnings,"errors":errors,"summary":{"passed_checks":P,"failed_checks":F,"warnings":len(warnings),"errors":len(errors)},"audit_timestamp_utc":datetime.now(timezone.utc).isoformat()})
    lines=["RANA / BIRDNET EMBEDDING DATASET AUDIT","="*76,f"Status: {status}",f"Embedding shape: {list(emb.shape)}",f"NaN values: {nan}",f"Inf values: {inf}",f"Passed checks: {P}",f"Failed checks: {F}",f"Warnings: {len(warnings)}",""]
    if warnings: lines += ["WARNINGS","-"*76]+[f"- {x}" for x in warnings]+[""]
    if errors: lines += ["ERRORS","-"*76]+[f"- {x}" for x in errors]+[""]
    if "site_distribution" in out: lines += ["SITE DISTRIBUTION","-"*76]+[f"{k}: {v}" for k,v in out["site_distribution"].items()]+[""]
    if "source_type_distribution" in out: lines += ["SOURCE TYPE DISTRIBUTION","-"*76]+[f"{k}: {v}" for k,v in out["source_type_distribution"].items()]+[""]
    jp.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8"); xp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    log("\n"+"="*76); log(f"AUDIT STATUS: {status}"); log(f"Passed checks: {P}"); log(f"Failed checks: {F}"); log(f"Warnings: {len(warnings)}"); log(f"JSON report: {jp}"); log(f"TXT report: {xp}"); log("="*76)
    return 0 if F==0 else 1
if __name__=="__main__": raise SystemExit(main())
