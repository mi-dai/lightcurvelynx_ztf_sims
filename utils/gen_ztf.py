import pandas as pd
import sqlite3
from nested_pandas import read_parquet
import numpy as np

from regions import RectangleSkyRegion
from astropy.coordinates import SkyCoord
import astropy.units as u

from lightcurvelynx.obstable.ztf_obstable import ZTFObsTable, _ztfcam_ccd_gain, _ztfcam_readout_noise
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.models.sncosmo_models import SncosmoWrapperModel
from lightcurvelynx.models.snia_host import SNIaHost
from lightcurvelynx.astro_utils.dustmap import DustmapWrapper,SFDMap
from lightcurvelynx.effects.extinction import ExtinctionEffect
from lightcurvelynx.astro_utils.mag_flux import mag2flux,flux2mag
from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint
from lightcurvelynx.utils.extrapolate import LinearDecayOnMag,ZeroPadding

from utils.analysis_utils import compute_sky_ztfsn_maglimit, compute_sky_ztfmeta_maglim
from ztf_snia_sim_params import SIM_PARAMS

def load_combined_obs_log():
    obs_log = pd.read_parquet('data/ztf_observing_log_combined_w_metadata.parquet')
    return obs_log

def load_ccd_obs_log():
    obs_log_allccd = pd.read_parquet('ztfsniadr2/tables/observing_logs.parquet')
    return obs_log_allccd
    
def load_metadata_db():
    con = sqlite3.connect("data/ztf_metadata_latest.db")
    sql_query = "SELECT * FROM exposures"
    metadata_table = pd.read_sql_query(sql_query, con)
    metadata_table = metadata_table.replace("", np.nan)
    metadata_table = metadata_table.dropna(subset=["fwhm"])
    return metadata_table

def load_sndata():
    globalhostdata = pd.read_csv('ztfsniadr2/tables/globalhost_data.csv')
    localhostdata = pd.read_csv('ztfsniadr2/tables/localhost_data.csv')
    sndata = pd.read_csv('ztfsniadr2/tables/snia_data.csv')
    data = pd.merge(sndata,globalhostdata,on='ztfname')
    return data

def load_lcdata():
    lcdata = read_parquet('data/ztfsniadr2.parquet')
    return lcdata

def get_matched_obs_log(ztfname, sndata=None, lcdata=None, combined_obs_log=None, obs_log_allccd=None, metadata_table=None):

    sn = sndata.loc[sndata.ztfname == ztfname]
    lc = lcdata.loc[lcdata["ztfname"] == ztfname]
    
    colmap = {"ra":"ra",
              "dec":"dec",
              "time":"mjd",
              "zp":"zp_nJy",
              "filter":"filter",
              "sky":"sky_adu",
             }
    
    #ztf ccd size 6144 × 6160 pixel * 16
    pixel_scale = 1.01 #arcsec/pixel
    center = SkyCoord(ra=0.0, dec=0.0, unit="deg", frame="icrs")
    rect_region = RectangleSkyRegion(center=center, width=7.323 * u.deg, 
                                     height=7.504 * u.deg, angle=0.0 * u.deg) # Dekany 2020 Table 3
    ztf_fp = DetectorFootprint(rect_region, pixel_scale=pixel_scale)
    
    ztf_obstable = ZTFObsTable(combined_obs_log,colmap=colmap,detector_footprint=ztf_fp)
    
    ra, dec = sn.ra.values[0], sn.dec.values[0]
    idx = ztf_obstable.range_search(ra,dec)
    table = ztf_obstable._table.iloc[idx]
    
    obs_log_allccd_sn = obs_log_allccd[obs_log_allccd["expid"].isin(table.expid)]

    df2 = lc['lc'].iloc[0]
    df3 = obs_log_allccd_sn
    if df2 is None:
        print(f"WARNING: lc is None for {ztfname}")
        return
    else:
        # match the rcid for that sn
        dflist = []
        for i,row in df2.drop_duplicates(['field_id','rcid']).iterrows():
            df = df3.loc[(df3.fieldid == row['field_id']) & (df3.rcid == row['rcid'])]
            dflist.append(df)
        obs_log = pd.concat(dflist)

        if len(obs_log) == 0:
            print(f"WARNING: matched obs_log is empty for {ztfname}")
            return

        else:
            obs_log["filter"] = obs_log.apply(lambda row: row["band"][-1],axis=1)
            obs_log = pd.merge(obs_log, metadata_table[["expid","filter","exptime","fwhm","obsdate","scibckgnd","ra","dec","maglim"]],on=["filter","expid"])
            gain = _ztfcam_ccd_gain
            obs_log["zp_nJy"] = mag2flux(obs_log["zp"].values + 2.5*np.log10(gain))
            obs_log = obs_log.rename(columns={"zp":"zp_abmag"})  
        
            obs_log["sky_adu_ztfsn"] = obs_log.apply(compute_sky_ztfsn_maglimit,axis=1)
            obs_log["sky_adu_ztfmeta"] = obs_log.apply(compute_sky_ztfmeta_maglim,axis=1)
            
            return obs_log

def gen_single_ztf_sn_lc(ztfname, sky_adu_col=None,  
                         sndata=None, lcdata=None, combined_obs_log=None, 
                         obs_log_allccd=None, metadata_table=None,
                         nsntotal = 30,
                         rng=None):
    
    H0 = SIM_PARAMS["H0"]
    OMEGA_M = SIM_PARAMS["Omega_m"]
    ZP_ERR_MAG = SIM_PARAMS["zp_mag_err"]
    
    obs_log = get_matched_obs_log(ztfname, 
                                  sndata=sndata, lcdata=lcdata, 
                                  combined_obs_log=combined_obs_log, 
                                  obs_log_allccd=obs_log_allccd, 
                                  metadata_table=metadata_table)
    if obs_log is None:
        return

    obs_log["sky_adu"] = obs_log[sky_adu_col]

    sn = sndata.loc[sndata.ztfname == ztfname]
    
    colmap = {"ra":"ra",
          "dec":"dec",
          "time":"mjd",
          "zp":"zp_nJy",
          "filter":"filter",
          "sky":"sky_adu",
         }

    #ztf ccd size 6144 × 6160 pixel * 16
    pixel_scale = 1.01 #arcsec/pixel
    center = SkyCoord(ra=0.0, dec=0.0, unit="deg", frame="icrs")
    rect_region = RectangleSkyRegion(center=center, width=7.323 * u.deg, 
                                     height=7.504 * u.deg, angle=0.0 * u.deg) # Dekany 2020 Table 3
    ztf_fp = DetectorFootprint(rect_region, pixel_scale=pixel_scale)
    
    ztf_obstable = ZTFObsTable(obs_log,colmap=colmap,detector_footprint=ztf_fp)
    ztf_obstable.survey_values["zp_err_mag"] = ZP_ERR_MAG
    
    t_min, t_max = ztf_obstable.time_bounds()
    print(f"Loaded OpSim with {len(ztf_obstable)} rows and times [{t_min}, {t_max}]")
    
    # sky_coverage = ztf_obstable.estimate_coverage(use_footprint=True)
    # print(f"The total sky coverage is {sky_coverage} square degrees")
    
    passband_group = PassbandGroup.from_preset(preset="ZTF", filters=["g", "r", "i"])
    print(f"Loaded Passbands: {passband_group}")
    
    host = SNIaHost(
        ra = sn.ra_host,
        dec = sn.dec,
        hostmass= sn.mass,
        redshift=sn.redshift,
        node_label="host",
    )

    sncosmo_modelname = "salt3"
    time_extrap_before = ZeroPadding()
    time_extrap_after = LinearDecayOnMag(decay_rate=0.02, mag_thres=30.)
    wave_extrap_before = ZeroPadding()
    wave_extrap_after = ZeroPadding()

    source = SncosmoWrapperModel(
        sncosmo_modelname,
        t0=sn.t0.values[0],
        x0=sn.x0.values[0],
        x1=sn.x1.values[0],
        c=sn.c.values[0],
        ra=sn.ra.values[0],
        dec=sn.dec.values[0],
        redshift=sn.redshift.values[0],
        node_label="source",
        time_extrapolation=(time_extrap_before,time_extrap_after),
        wave_extrapolation=(wave_extrap_before,wave_extrap_after),   
    )
    
    mwextinction = SFDMap(
        ra=source.ra,
        dec=source.dec,
        node_label="mwext",
    )
    
    # Create an extinction effect using the EBVs from that dust map.
    ext_effect = ExtinctionEffect(extinction_model="F99", ebv=mwextinction, 
                                  r_v=3.1,frame='observer',backend="dust_extinction")
    source.add_effect(ext_effect)

    lightcurves = simulate_lightcurves(source, int(nsntotal), ztf_obstable, passband_group, 
                                       obstable_save_cols=["expid","zp_nJy","scibckgnd","skynoise",
                                                           "fwhm","maglimit","maglim","sky_adu"],
                                       rng=rng)
    return lightcurves