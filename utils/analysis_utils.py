import numpy as np
from lightcurvelynx.astro_utils.mag_flux import flux2mag
from astropy.coordinates import angular_separation, SkyCoord
import astropy.units as u
from lightcurvelynx.obstable.ztf_obstable import _ztfcam_ccd_gain, _ztfcam_readout_noise
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ

def lc_quality_cuts(flux,mjd,filter,z,n_phases=7, n_before_peak=2, n_after_peak=2, n_bands=2):
    peak_idx = np.argmax(flux)
    phases = np.floor((mjd - mjd[peak_idx])/(1. + z))
    unique_phases,unique_idx = np.unique(phases,return_index=True)
    good_idx = (unique_phases >= -10) & (unique_phases<=40)
    pass_cut = len(unique_phases[good_idx]) >= n_phases
    flux_new = flux[unique_idx][good_idx]
    peak_idx_new = np.argmax(flux_new)
    pass_cut &= (peak_idx_new >= n_before_peak - 1) & (len(flux_new) - peak_idx_new >= n_after_peak - 1)
    pass_cut &= len(np.unique(filter[unique_idx][good_idx])) >= n_bands
    return {"pass_quality_cuts": pass_cut}

def spec_selection_func(flux,p0=None,m0=18.8,s=4.5):
    if p0 is None:
        p0 = np.random.uniform(0,1)
    m = flux2mag(np.max(flux))
    p = np.power(1. + np.exp((m - m0)*s), -1)
    return {"pass_spec_selection": p0 < p}

def get_sn_host_sep(host_ra, host_dec, sn_ra, sn_dec):
    host_ra = np.array(host_ra)
    host_dec = np.array(host_dec)
    sn_ra = np.array(sn_ra)
    sn_dec = np.array(sn_dec)
    c_host = SkyCoord(host_ra, host_dec, unit="deg")
    c_sn = SkyCoord(sn_ra, sn_dec, unit="deg")
    sep = angular_separation(c_host.ra, c_host.dec, c_sn.ra, c_sn.dec)
    return sep.to(u.arcsec).value

# let's try to derive skynoise using maglim and zp
# snr = flux/fluxerr
# fluxerr = sqrt(flux + sky_adu*npix*gain + readnoise**2*nexposure*npix + darkcurrent*npix*exptime*nexposure)
# 5 = flux/fluxerr
# 25 = flux**2/(flux + sky_adu*npix*gain + readnoise**2*nexposure*npix + darkcurrent*npix*exptime*nexposure)
# flux**2 - 25*flux -25*( sky_adu*npix*gain
#                         + readnoise**2*nexposure*npix
#                         + darkcurrent*npix*exptime*nexposure)
#                     = 0
# flux_e = 10^(-0.4*(maglim-zp))*gain
# sky_adu*npix*gain =  (flux_e**2 / 25 - flux_e - (readnoise**2*nexposure*npix
#                              + darkcurrent*npix*exptime*nexposure))
def compute_sky_ztfsn_maglimit(row):
    gain = _ztfcam_ccd_gain
    nea = GAUSS_EFF_AREA2FWHM_SQ * (row["fwhm"]) ** 2
    flux = np.power(10., -0.4*(row['maglimit'] - row['zp_abmag'])) * gain
    sky = (flux**2 / 25 - flux - _ztfcam_readout_noise**2 * nea) / nea / gain
    return sky    

def compute_sky_ztfmeta_maglim(row):
    gain = _ztfcam_ccd_gain
    nea = GAUSS_EFF_AREA2FWHM_SQ * (row["fwhm"]) ** 2
    flux = np.power(10., -0.4*(row['maglim'] - row['zp_abmag'])) * gain
    sky = (flux**2 / 25 - flux - _ztfcam_readout_noise**2 * nea) / nea / gain
    return sky   