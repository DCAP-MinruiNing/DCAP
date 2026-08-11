#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZTF Source Classifier v4 — No JPL, avro ssnamenr priority
Layers:
  L1: ALeRCE avro interface fetching raw alert, checking candidate.ssnamenr
  L2: Heidelberg Gaia DR3
  L3: VizieR (Gaia DR3 / UCAC5 / URAT1 / TICv8.2)
  L4: Simbad stellar verification
  L5: RANSAC self-trajectory fitting
"""

import sys, warnings, json, traceback
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from alerce.core import Alerce
from sklearn.linear_model import RANSACRegressor, LinearRegression
from astroquery.utils.tap.core import TapPlus
from astroquery.gaia import conf as gaia_conf
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import requests

warnings.filterwarnings("ignore")

# ============================================================
# Global Configuration
# ============================================================
gaia_conf.timeout = 30
GAIA_JOB_TIMEOUT = 60
GAIA_RETRIES = 2
HEIDELBERG_TAP = "https://gaia.ari.uni-heidelberg.de/tap"
AVRO_API = "https://avro.alerce.online/get_avro_info"

GAIA_EPOCH_MJD   = 57388.5
TIC_EPOCH_MJD    = 58119.0
UCAC5_EPOCH_MJD  = 55200.0
URAT1_EPOCH_MJD  = 56700.0
SIMBAD_EPOCH_MJD = 51544.5

VR_CATALOGS = [
    {'name':'Gaia DR3','id':'I/355/gaiadr3','ra_cols':['RA_ICRS','RA'],'dec_cols':['DE_ICRS','DE'],
     'pmra_col':'pmRA','pmdec_col':'pmDE','id_col':'Source','epoch_mjd':GAIA_EPOCH_MJD,
     'columns':['RA_ICRS','DE_ICRS','pmRA','pmDE','Source','Gmag']},
    {'name':'UCAC5','id':'I/340/ucac5','ra_cols':['_RAJ2000','RAJ2000'],'dec_cols':['_DEJ2000','DEJ2000'],
     'pmra_col':'pmRA','pmdec_col':'pmDE','id_col':None,'epoch_mjd':UCAC5_EPOCH_MJD,
     'columns':['_RAJ2000','_DEJ2000','pmRA','pmDE','Gmag','Rmag']},
    {'name':'URAT1','id':'I/329/urat1','ra_cols':['RAICRS','RA'],'dec_cols':['DEICRS','DE'],
     'pmra_col':'pmRA','pmdec_col':'pmDE','id_col':None,'epoch_mjd':URAT1_EPOCH_MJD,
     'columns':['RAICRS','DEICRS','pmRA','pmDE','Gmag','e_pm']},
    {'name':'TICv8.2','id':'IV/39/tic82','ra_cols':['RAJ2000','RA_ICRS'],'dec_cols':['DEJ2000','DE_ICRS'],
     'pmra_col':'pmRA','pmdec_col':'pmDE','id_col':'TIC','epoch_mjd':TIC_EPOCH_MJD,
     'columns':['RAJ2000','DEJ2000','pmRA','pmDE','TIC','Tmag']},
]

# ============================================================
# Utility Functions
# ============================================================
def _sf(v):
    if v is None: return None
    try:
        s=str(v).strip()
        if s in ('','--','nan','NaN'): return None
        return float(s)
    except: return None

def _sint(v):
    if v is None: return None
    try: return int(v)
    except: return None

def _async_q(url,adql,timeout=GAIA_JOB_TIMEOUT):
    return TapPlus(url=url).launch_job_async(adql,verbose=False).get_results()

def _safe_tap(url,adql,label,retries=GAIA_RETRIES):
    for i in range(retries):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                f=ex.submit(_async_q,url,adql,GAIA_JOB_TIMEOUT)
                return f.result(timeout=GAIA_JOB_TIMEOUT)
        except FuturesTimeout:
            print(f"  [timeout] {label}"); return None
        except Exception as e:
            print(f"  [warn] {label} #{i+1}: {e}"); time.sleep(2)
    return None

def _quick(url,label):
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(_async_q,url,"SELECT TOP 1 * FROM gaiadr3.gaia_source",15).result(timeout=10)
        return True
    except Exception as e:
        print(f"  [{label}] unreachable: {e}"); return False

def _find_col(table,names):
    for n in names:
        if n in table.colnames: return n
    return None

def _mjd_to_utc(mjd):
    t = Time(mjd, format='mjd')
    return t.datetime.strftime('%Y-%m-%d %H:%M:%S')

def _decra_to_hms(ra_deg):
    ra_hours = ra_deg / 15.0
    h = int(ra_hours)
    m = int((ra_hours - h) * 60)
    s = (ra_hours - h - m/60.0) * 3600
    return f"{h:02d}-{m:02d}-{s:05.2f}"

def _decdec_to_dms(dec_deg):
    sign = '+' if dec_deg >= 0 else '-'
    abs_dec = abs(dec_deg)
    d = int(abs_dec)
    m = int((abs_dec - d) * 60)
    s = (abs_dec - d - m/60.0) * 3600
    return f"{sign}{d:02d}-{m:02d}-{s:05.2f}"

# ============================================================
# L1: ALeRCE avro interface fetching raw alert, checking ssnamenr
# ============================================================
def L1_avro_ssnamenr(oid, det, rep):
    print("\n========== L1: ALeRCE avro interface checking ssnamenr ==========")
    rep['L1'] = {'status':'skip','reason':'avro request failed or no candid'}
    if det is None or det.empty:
        print("  [error] No detections data")
        return False
    candid = None
    if 'candid' in det.columns:
        valid_candids = det['candid'].dropna()
        if not valid_candids.empty:
            candid = int(valid_candids.iloc[0])
    if candid is None:
        print("  [error] Unable to obtain candid")
        return False
    print(f"  Requesting avro: oid={oid}, candid={candid}")
    try:
        params = {'oid': oid, 'candid': candid}
        r = requests.get(AVRO_API, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  [HTTP error] {r.status_code}: {r.text[:200]}")
            return False
        avro_data = r.json()
    except Exception as e:
        print(f"  [error] avro request failed: {e}")
        return False
    candidate = avro_data.get('candidate', {})
    ssname = candidate.get('ssnamenr')
    if ssname is not None and str(ssname).strip() not in ('', 'null', 'None'):
        ssdist = candidate.get('ssdistnr')
        ssmag = candidate.get('ssmagnr')
        print(f"  ✅ Found known asteroid: ssnamenr={ssname}")
        if ssdist is not None: print(f"     Distance: {ssdist}″")
        if ssmag is not None: print(f"     Predicted magnitude: {ssmag}")
        rep['L1'] = {
            'status': 'ok',
            'source': 'ZTF_avro_ssnamenr',
            'best': {
                'source_name': f"MPC {ssname}",
                'name': str(ssname),
                'sep': _sf(ssdist),
                'vmag': _sf(ssmag),
                'rejected': False,
            }
        }
        return True
    print("  [info] ssnamenr empty or missing, proceeding to next layers")
    return False

# ============================================================
# L2: Heidelberg Gaia DR3
# ============================================================
def L2_heidelberg(ra,dec,mjd,rep):
    print("\n========== L2: Gaia DR3 @ Heidelberg ==========")
    rep['L2']={'status':'skip','reason':'unreachable'}
    if not _quick(HEIDELBERG_TAP,"Heidelberg"):
        print("  → skip to L3"); return
    for rad in [120,300,600,1200]:
        for pm in [200,150,100,50]:
            r=rad/3600.0
            adql=(f"SELECT source_id,ra,dec,pmra,pmdec,pm,parallax,ruwe "
                  f"FROM gaiadr3.gaia_source WHERE "
                  f"1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{r})) "
                  f"AND pm>{pm} AND ruwe<1.4 AND parallax IS NOT NULL ORDER BY pm DESC")
            print(f"  [Heidelberg] r={rad:>5.0f}″ pm>{pm:>5.0f} ...",end=" ")
            t=_safe_tap(HEIDELBERG_TAP,adql,"L2")
            if t is not None and len(t)>0:
                print(f"✅ {len(t)} cand")
                rep['L2']={'status':'found','n_cand':len(t),'best':_L2_pick(t,ra,dec,mjd)}
                return
            print("❌")
    print("  → skip to L3")

def _L2_pick(cands,ra0,dec0,mjd):
    best=None; bs=-1e9
    for r in cands:
        try:
            pm=float(r['pm']); ra=float(r['ra']); dc=float(r['dec'])
            pmra=float(r['pmra']); pmdec=float(r['pmdec'])
        except: continue
        sid=_sint(r['source_id'])
        dt=(mjd-GAIA_EPOCH_MJD)/365.25
        rp=ra+(pmra*dt)/(1e3 * 3600*np.cos(np.radians(dc)))
        dp=dc+(pmdec*dt)/(1e3 * 3600)
        disp=pm*abs(dt)/1e3
        sep=np.sqrt(((rp-ra0)*np.cos(np.radians(dec0)))**2+(dp-dec0)**2)*3600
        pen=(sep/max(disp,1))**2; bon=pm/1e3
        vz=np.array([(ra0-ra)*np.cos(np.radians(dec0)),dec0-dc])
        vp=np.array([pmra,pmdec])
        nz,np_=np.linalg.norm(vz),np.linalg.norm(vp)
        ca=np.dot(vz,vp)/(nz*np_) if nz>0 and np_>0 else 0
        s=bon+ca-pen*0.5
        if s>bs:
            bs=s
            best={'source_name':f"Gaia DR3 {sid}",'source_id':sid,
                  'ra':ra,'dec':dc,'pmra':pmra,'pmdec':pmdec,'pm':pm,
                  'sep':sep,'disp':disp,'pred_ra':rp,'pred_dec':dp,
                  'rejected':sep>max(disp*5,10)}
    return best

# ============================================================
# L3: VizieR
# ============================================================
def L3_vizier(ra,dec,mjd,rep):
    print("\n========== L3: VizieR (Gaia DR3/UCAC5/URAT1/TICv8.2) ==========")
    rep['L3']={'status':'fail','reason':'no match'}
    coord=SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs')
    for cfg in VR_CATALOGS:
        catname=cfg['name']
        for rad_arcsec in [120,300,600]:
            radius=rad_arcsec*u.arcsec
            print(f"  [VizieR:{catname}] r={rad_arcsec}″ ...",end=" ",flush=True)
            try:
                v=Vizier(catalog=cfg['id'],columns=cfg['columns'],row_limit=20)
                res=v.query_region(coord,radius=radius)
                if res is None or len(res)==0:
                    print("❌ none"); continue
                t=res[0]; print(f"✅ {len(t)} row(s)")
                ra_col=_find_col(t,cfg['ra_cols'])
                dec_col=_find_col(t,cfg['dec_cols'])
                if ra_col is None or dec_col is None:
                    print(f"    ⚠️ Coordinate columns missing, skip"); break
                pmra_col=cfg['pmra_col']; pmdec_col=cfg['pmdec_col']
                if pmra_col not in t.colnames or pmdec_col not in t.colnames:
                    print(f"    ⚠️ Proper motion columns missing, skip"); break
                last_rejected=None
                for row in t:
                    rc=_sf(row[ra_col]); dc2=_sf(row[dec_col])
                    if rc is None or dc2 is None: continue
                    pmra=_sf(row[pmra_col]) or 0.0; pmdec=_sf(row[pmdec_col]) or 0.0
                    pm=np.sqrt(pmra**2+pmdec**2)
                    if pm<10: continue
                    sep=np.sqrt(((rc-ra)*np.cos(np.radians(dec)))**2+(dc2-dec)**2)*3600
                    if sep>rad_arcsec*1.5: continue
                    dt=(mjd-cfg['epoch_mjd'])/365.25
                    rp=rc+(pmra*dt)/(1e3 * 3600*np.cos(np.radians(dc2)))
                    dp=dc2+(pmdec*dt)/(1e3 * 3600)
                    disp=pm*abs(dt)/1e3
                    final_sep=np.sqrt(((rp-ra)*np.cos(np.radians(dec)))**2+(dp-dec)**2)*3600
                    rejected=final_sep>max(disp*5,10)
                    src_name=catname
                    if cfg['id_col'] is not None and cfg['id_col'] in row.colnames:
                        iv=row[cfg['id_col']]
                        if iv is not None:
                            if catname=='Gaia DR3': src_name=f"Gaia DR3 {int(iv)}"
                            elif catname=='TICv8.2': src_name=f"TIC {int(iv)}"
                            else: src_name=f"{catname} {iv}"
                    entry={'source_name':src_name,'catalog':catname,'ra':rc,'dec':dc2,
                           'pmra':pmra,'pmdec':pmdec,'pm':pm,'sep':final_sep,
                           'disp':disp,'pred_ra':rp,'pred_dec':dp,'rejected':rejected,
                           'epoch_mjd':cfg['epoch_mjd']}
                    if not rejected:
                        if 'Gaia' in catname: rep['L3']={'status':'ok','best':entry,'reason':''}
                        else: rep['L4']={'status':'ok','best':entry,'reason':''}
                        print(f"    ✅ Accepted: {src_name} pm={pm:.1f} mas/yr sep={final_sep:.1f}″")
                        return
                    last_rejected=entry
                    print(f"    ❌ Rejected: {src_name} pm={pm:.1f} mas/yr sep={final_sep:.1f}″ > disp={disp:.1f}″")
                if last_rejected is not None:
                    if 'Gaia' in catname:
                        rep['L3']={'status':'rejected','best':last_rejected,'reason':f'offset {last_rejected["sep"]:.0f}″'}
                    else:
                        rep['L4']={'status':'rejected','best':last_rejected,'reason':f'offset {last_rejected["sep"]:.0f}″'}
            except Exception as e:
                print(f"❌ error: {e}"); continue
    print("  → skip to L4")

# ============================================================
# L4: Simbad
# ============================================================
def L4_simbad(ra,dec,mjd,rep):
    print("\n========== L4: Simbad (stellar verification) ==========")
    rep['L4']={'status':'fail','reason':'no match'}
    for rad in [120,300,600]:
        print(f"  [simbad] r={rad}″ ...")
        try:
            s=Simbad(); s.reset_votable_fields()
            s.add_votable_fields('pmra','pmdec','plx_value','otype','ra','dec','main_id')
            s.add_votable_fields('mesPM')
            tab=s.query_region(SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs'),radius=rad*u.arcsec)
        except Exception as e:
            print(f"  [warn] {e}"); continue
        if tab is None or len(tab)==0:
            print("  ❌ none"); continue
        print(f"  [simbad] {len(tab)} raw")
        best=None; bs=-1e9
        for row in tab:
            rc=_sf(row['ra']); dc=_sf(row['dec'])
            if rc is None or dc is None: continue
            sep=np.sqrt(((rc-ra)*np.cos(np.radians(dec)))**2+(dc-dec)**2)*3600
            if sep>rad: continue
            pmra=_sf(row['pmra']); pmdec=_sf(row['pmdec'])
            if pmra is None and pmdec is None:
                pmra=_sf(row['mespm.pmra']) if 'mespm.pmra' in row.colnames else None
                pmdec=_sf(row['mespm.pmde']) if 'mespm.pmde' in row.colnames else None
            pmra=pmra or 0.0; pmdec=pmdec or 0.0
            pm=np.sqrt(pmra**2+pmdec**2)
            if pm<10: continue
            dt=(mjd-SIMBAD_EPOCH_MJD)/365.25
            rp=rc+(pmra*dt)/(1e3 * 3600*np.cos(np.radians(dc)))
            dp=dc+(pmdec*dt)/(1e3 * 3600)
            disp=pm*abs(dt)/1e3
            fsep=np.sqrt(((rp-ra)*np.cos(np.radians(dec)))**2+(dp-dec)**2)*3600
            sc=pm/200.0-sep/1000.0
            if sc>bs:
                bs=sc
                best={'source_name':str(row.get('main_id','?')).strip(),
                      'name':str(row.get('main_id','?')).strip(),
                      'ra':rc,'dec':dc,'pmra':pmra,'pmdec':pmdec,'pm':pm,
                      'sep':fsep,'disp':disp,'pred_ra':rp,'pred_dec':dp,
                      'rejected':fsep>max(disp*5,10)}
        if best is not None:
            rep['L4']={'status':'ok' if not best['rejected'] else 'rejected','best':best}
            if best['rejected']:
                print(f"  ❌ Best candidate rejected: {best['source_name']} sep={best['sep']:.0f}″ > disp={best['disp']:.1f}″")
            else:
                print(f"  ✅ Accepted: {best['source_name']} pm={best['pm']:.1f} mas/yr")
            return
        print("  [simbad] No high proper-motion source")
    print("  → skip to L5")

# ============================================================
# L5: RANSAC
# ============================================================
def L5_ransac(det,ra0,dec0,rep):
    print("\n========== L5: RANSAC (self-detection points only) ==========")
    rep['L5']={'status':'fail','reason':'insufficient data'}
    if len(det)<3:
        print("  Detection points < 3, skipping"); return
    d=det.copy()
    if 'objectId' in d.columns and 'oid' not in d.columns:
        d.rename(columns={'objectId':'oid'},inplace=True)
    d=d.sort_values('mjd').reset_index(drop=True)
    dt_hr=(d['mjd']-d['mjd'].iloc[0])*24.0
    ra_off=(d['ra']-ra0)*np.cos(np.radians(dec0))*3600.0
    dec_off=(d['dec']-dec0)*3600.0
    X=dt_hr.values.reshape(-1,1)
    base=LinearRegression(fit_intercept=False)
    rr=RANSACRegressor(estimator=base,min_samples=3,residual_threshold=0.3)
    dr=RANSACRegressor(estimator=base,min_samples=3,residual_threshold=0.3)
    try:
        rr.fit(X,ra_off.values)
        dr.fit(X,dec_off.values)
    except Exception as e:
        print(f"  RANSAC fit failed: {e}"); return
    mask=rr.inlier_mask_ & dr.inlier_mask_
    if mask.sum()<3:
        print(f"  Insufficient inliers ({mask.sum()}<3)"); return
    sra=rr.estimator_.coef_[0]
    sdc=dr.estimator_.coef_[0]
    v=float(np.sqrt(sra**2+sdc**2))
    pa=float(np.degrees(np.arctan2(sra,sdc))%360)
    resid=np.sqrt((ra_off.values-(sra*X.flatten()))**2+(dec_off.values-(sdc*X.flatten()))**2)[mask]
    rms=float(np.sqrt(np.mean(resid**2)))
    ssr=float(np.sum(resid**2))
    sst=float(np.sum(ra_off.values[mask]**2+dec_off.values[mask]**2))
    r2=1-ssr/sst if sst>0 else 0.0
    span=float((d['mjd'].max()-d['mjd'].min())*24.0)
    pm_est=v*8760 * 1000

    reliable=(r2>=0.5) and (mask.sum()/len(d)>=0.5)
    if not reliable:
        label='unreliable'
    elif v<0.5 and rms<0.5 and r2>0.80 and span>1:
        label='high_proper_motion_star'
    elif v>2 and rms<0.5 and r2>0.90:
        label='sso_candidate'
    elif 0.5<=v<=2 and rms<0.5 and r2>0.70:
        label='gray_zone'
    else:
        label='unknown'

    rep['L5']={'status':'ok' if reliable else 'unreliable','label':label,
               'v_asec_per_hr':v,'pa_deg':pa,'rms':rms,'r2':r2,'span_hr':span,
               'n_inliers':int(mask.sum()),'n_total':len(d),'pm_est_mas_yr':pm_est,
               'mjd_range':(float(d['mjd'].min()),float(d['mjd'].max())),'reliable':reliable}
    print(f"  {'✅' if reliable else '⚠️'} {label}: v={v:.6f}″/hr PM≈{pm_est:.0f} mas/yr "
          f"RMS={rms:.3f}″ R²={r2:.4f} span={span:.0f}h inliers={mask.sum()}/{len(d)} "
          f"{'(reliable)' if reliable else '(unreliable)'}")

# ============================================================
# Report Printing
# ============================================================
def print_report(oid,ra0,dec0,mjd,rep):
    print("\n"+"="*60)
    print(f"📋 Full Report — {oid}")
    print(f"    ZTF: RA={ra0:.6f}° Dec={dec0:.6f}°  Mean MJD={mjd:.1f}")
    print("="*60)

    layer_order = ['L1','L2','L3','L4','L5']
    layer_names = {
        'L1':'avro ssnamenr',
        'L2':'Heidelberg Gaia DR3',
        'L3':'VizieR',
        'L4':'Simbad',
        'L5':'RANSAC'
    }
    for key in layer_order:
        r=rep.get(key,{})
        st=r.get('status','N/A')
        name=layer_names.get(key,key)

        if key=='L1':
            if st=='ok':
                b=r.get('best',{})
                sn=b.get('source_name','?')
                sep=b.get('sep')
                vmag=b.get('vmag')
                sep_str=f"{sep:.1f}″" if sep is not None else "?"
                vmag_str=f"{vmag:.1f}" if vmag is not None else "?"
                print(f"  {key} ✅ {name}: {sn}  Δ={sep_str}  V={vmag_str}")
            else:
                print(f"  {key} ❌ {name}: {r.get('reason','No match')}")
            continue

        if key=='L5':
            if st=='ok':
                label=r.get('label','unknown')
                pm_est=r.get('pm_est_mas_yr',0)
                v=r.get('v_asec_per_hr',0)
                print(f"  {key} {'✅' if r.get('reliable') else '⚠️'} {name}: {label} "
                      f"v={v:.6f}″/hr PM≈{pm_est:.0f} mas/yr "
                      f"R²={r.get('r2',0):.4f} inliers={r.get('n_inliers',0)}/{r.get('n_total',0)}")
            elif st=='unreliable':
                print(f"  {key} ⚠️  {name}: {r.get('label','')} (unreliable, R²={r.get('r2',0):.3f})")
            elif st=='skip':
                print(f"  {key} ⏭️  {name}: skipped ({r.get('reason','')})")
            else:
                print(f"  {key} ❌ {name}: {r.get('reason','No match')}")
            continue

        b=r.get('best')
        if st in ('ok','found'):
            if b is None:
                print(f"  {key} ✅ {name}: No detailed data")
            elif b.get('rejected'):
                sn=b.get('source_name','?')
                print(f"  {key} ⚠️  {name}: Candidate hard-rejected ({sn}, sep={b['sep']:.0f}″ >> disp)")
            else:
                sn=b.get('source_name','?')
                print(f"  {key} ✅ {name}: {sn} pm={b.get('pm',0):.1f} mas/yr "
                      f"pred=({b.get('pred_ra',0):.5f},{b.get('pred_dec',0):.5f}) "
                      f"sep={b.get('sep',0):.1f}″")
        elif st=='rejected':
            if b:
                sn=b.get('source_name','?')
                print(f"  {key} ⚠️  {name}: Candidate hard-rejected ({sn}, sep={b['sep']:.0f}″ >> disp)")
            else:
                print(f"  {key} ⚠️  {name}: Rejected")
        elif st=='skip':
            print(f"  {key} ⏭️  {name}: skipped ({r.get('reason','')})")
        else:
            print(f"  {key} ❌ {name}: {r.get('reason','No match')}")

    # Final conclusion
    print("\n"+"-"*60)
    if rep.get('L1',{}).get('status')=='ok':
        b=rep['L1']['best']
        print(f"✅ Conclusion: Known asteroid (avro ssnamenr={b['name']})")
        if b.get('sep') is not None: print(f"   Offset: {b['sep']:.2f}″")
        if b.get('vmag') is not None: print(f"   V magnitude: {b['vmag']:.1f}")
        return

    for key in ['L2','L3','L4']:
        r=rep.get(key,{})
        if r.get('status') in ('ok','found'):
            b=r.get('best')
            if b and not b.get('rejected'):
                sn=b.get('source_name','Unknown')
                print(f"✅ Conclusion: High proper-motion star confirmed ({key})")
                print(f"   Source: {sn}")
                print(f"   Proper motion: {b.get('pm',0):.1f} mas/yr")
                print(f"   Predicted position: RA={b.get('pred_ra',0):.6f}° Dec={b.get('pred_dec',0):.6f}°")
                print(f"   Offset: {b.get('sep',0):.2f}″")
                return

    if rep.get('L5',{}).get('status')=='ok' and rep['L5'].get('reliable'):
        r=rep['L5']
        print(f"✅ Conclusion: {r.get('label','unknown')} (RANSAC, reliable)")
        print(f"   Estimated proper motion: {r.get('pm_est_mas_yr',0):.0f} mas/yr")
        print(f"   Angular velocity: {r.get('v_asec_per_hr',0):.6f}″/hr  PA={r.get('pa_deg',0):.1f}°")
        print(f"   RMS={r.get('rms',0):.3f}″  R²={r.get('r2',0):.4f}")
        print(f"   Data span: {r.get('span_hr',0):.0f} hours ({r.get('n_inliers',0)}/{r.get('n_total',0)} inliers)")
        return

    print("⚠️  Conclusion: Cannot confirm")
    print("   All layers found no reliable match.")
    print("   Possible causes: low proper-motion star / galaxy / transient / insufficient data.")
    print("   Suggestion: Check Alerce light curve, or manually query Simbad/Gaia DR3.")
    print("-"*60)

# ============================================================
# Main Flow
# ============================================================
def main():
    print("=" * 40)
    print("ZTF Source Classifier v4 — No JPL")
    print("=" * 40)

    try:
        oid = input("Enter ZTF object ID: ").strip()
    except EOFError:
        print("Input interrupted, exiting.")
        sys.exit(1)
    if not oid:
        print("No valid OID entered, exiting.")
        sys.exit(1)

    print(f"\nQuerying {oid} ...")
    client = Alerce()
    try:
        det = client.query_detections(oid=oid, survey="ztf", format="pandas")
    except Exception as e:
        print(f"Query failed: {e}")
        sys.exit(1)

    if det.empty:
        print("No detection records for this OID.")
        sys.exit(1)

    ra0 = float(det['ra'].iloc[0])
    dec0 = float(det['dec'].iloc[0])
    mjd = det['mjd'].mean()
    print(f"Target: {oid}")
    print(f"ZTF coords: RA={ra0:.6f}  Dec={dec0:.6f}  Mean MJD={mjd:.1f}")

    rep = {}

    # L1: avro ssnamenr
    if L1_avro_ssnamenr(oid, det, rep):
        print("\n⏭️  L1 confirmed as known asteroid, skipping subsequent layers.")
    else:
        # L2: Heidelberg Gaia
        L2_heidelberg(ra0, dec0, mjd, rep)
        # L3: VizieR
        L3_vizier(ra0, dec0, mjd, rep)
        # L4: Simbad
        L4_simbad(ra0, dec0, mjd, rep)
        # L5: RANSAC
        L5_ransac(det, ra0, dec0, rep)

    print_report(oid, ra0, dec0, mjd, rep)

if __name__ == '__main__':
    main()