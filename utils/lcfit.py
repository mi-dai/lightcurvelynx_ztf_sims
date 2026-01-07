import logging

import pandas as pd
import sncosmo
from astropy.table import Table
from sfdmap2 import sfdmap
from sncosmo.fitting import flatten_result

logger = logging.getLogger(__name__)

lc_colmap = {
    "flux": "flux",
    "fluxerr": "fluxerr",
    "mwebv": "mwebv",
    "redshift": "z",
}


def fit_single_lc(
    lc,
    modelsource="salt3",
    modelpars=None,
    mpbounds=None,
    modelcov=False,
    usebands="all",
    mwebv_from_coord=True,
    passbands=None,
    **kwargs,
):
    """Fit a single light curve given single row of a NestedFrame"""

    if not isinstance(lc, pd.Series):
        raise ValueError("This function takes a NestedFrame with single row")

    if modelpars is None:
        modelpars = ["t0", "x0", "x1", "c"]
    if mpbounds is None:
        mpbounds = {}

    dustmap = sfdmap.SFDMap("/Users/mi/sfddata-master")
    model = sncosmo.Model(
        source=modelsource, effects=[sncosmo.F99Dust()], effect_names=["mw"], effect_frames=["obs"]
    )

    lc["lightcurve"]["zp"] = 31.4
    lc["lightcurve"]["zpsys"] = "ab"

    z = lc[lc_colmap["redshift"]]

    if mwebv_from_coord:
        ra = lc.ra
        dec = lc.dec
        mwebv = dustmap.ebv(ra, dec)
    else:
        mwebv = lc[lc_colmap["mwebv"]]
    logger.info(f"fitting {lc.id}, z={z}, mwebv={mwebv}.")
    model.set(z=z, mwebv=mwebv)

    if usebands != "all":
        lc = lc.query(f"lightcurve.filter in list{usebands}").dropna()
        if len(lc) == 0:
            logger.info(f"No data in selected bands:{usebands}")

    lc["lightcurve"]["flux"] = lc["lightcurve"][f"{lc_colmap['flux']}"]
    lc["lightcurve"]["fluxerr"] = lc["lightcurve"][f"{lc_colmap['fluxerr']}"]
    lc["lightcurve"]["filter"] = "ztf" + lc["lightcurve"]["filter"]

    try:
        result, fitted_model = sncosmo.fit_lc(
            Table.from_pandas(lc["lightcurve"]),
            model,
            modelpars,
            modelcov=modelcov,
            bounds=mpbounds.copy(),
            **kwargs,
        )

        res = flatten_result(result)
        res["id"] = int(lc.id)
        res["fit_error"] = None

        return pd.Series(res)
    except Exception as e:
        return pd.Series({"id": int(lc.id), "fit_error": repr(e)})
