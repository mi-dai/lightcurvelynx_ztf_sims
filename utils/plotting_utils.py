import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

def plot_snr_distr(data_list, labels=None, colors=["C0","C1"],**kwargs):
    ax = plt.subplot(1,1,1)
    if labels is None:
        labels = [f'data{i}' for i in range(0,len(data_list))]
    for i, data in enumerate(data_list):
        ax.hist(data['lc.snr'], label=labels[i], color=colors[i],**kwargs)
    ax.legend()

def convert_flux_to_njy(flux,fluxerr,zp=0.):
    zp_njy = 31.4
    #-2.5*log10(flux_njy) + zp_njy = -2.5*log10(flux) + zp
    flux_njy = flux*np.power(10., -0.4*(zp-zp_njy))
    fluxerr_njy = fluxerr*np.power(10., -0.4*(zp-zp_njy))         
    return flux_njy,fluxerr_njy

def plot_logflux_vs_logfluxerr_corner(sim, data, labels=('sim','data'), zp=30., smooth_sigma=0.6):
    def _safe_log10(x):
        x = np.asarray(x, float)
        m = x > 0
        out = np.full_like(x, np.nan, dtype=float)
        out[m] = np.log10(x[m])
        return out

    # ---- Convert flux to nJy for data ----
    lcdata_flux_njy, lcdata_fluxerr_njy = convert_flux_to_njy(
        data['lc.flux'], data['lc.flux_err'], zp=zp
    )

    # ---- Take logs ----
    lcsim_logflux     = _safe_log10(sim['lc.flux'])
    lcsim_logfluxerr  = _safe_log10(sim['lc.fluxerr'])
    lcdata_logflux    = _safe_log10(lcdata_flux_njy)
    lcdata_logfluxerr = _safe_log10(lcdata_fluxerr_njy)

    # ---- Clean finite values ----
    m_sim  = np.isfinite(lcsim_logflux) & np.isfinite(lcsim_logfluxerr)
    m_data = np.isfinite(lcdata_logflux) & np.isfinite(lcdata_logfluxerr)
    xs, ys = lcsim_logflux[m_sim], lcsim_logfluxerr[m_sim]
    xd, yd = lcdata_logflux[m_data], lcdata_logfluxerr[m_data]

    # ---- Define bins ----
    xb = np.linspace(3.6, 6, 20)
    yb = np.linspace(3.0, 4.25, 20)

    # ---- Compute 2D histograms ----
    Hs, xe, ye = np.histogram2d(xs, ys, bins=[xb, yb])
    Hd, _, _   = np.histogram2d(xd, yd, bins=[xb, yb])
    if smooth_sigma:
        Hs = gaussian_filter(Hs, smooth_sigma)
        Hd = gaussian_filter(Hd, smooth_sigma)

    X, Y = np.meshgrid(0.5*(xe[:-1]+xe[1:]), 0.5*(ye[:-1]+ye[1:]))

    # ---- Create figure layout ----
    fig = plt.figure(figsize=(7,7))
    gs  = fig.add_gridspec(2,2, width_ratios=(4,1), height_ratios=(1,4),
                           wspace=0.05, hspace=0.05)

    ax_top   = fig.add_subplot(gs[0,0])
    ax_right = fig.add_subplot(gs[1,1])
    ax_main  = fig.add_subplot(gs[1,0])

    # ---- Main 2D contour ----
    levels_s = np.linspace(np.nanmin(Hs[Hs>0]), np.nanmax(Hs), 7)[1:-1]  # skip outermost
    levels_d = np.linspace(np.nanmin(Hd[Hd>0]), np.nanmax(Hd), 7)[1:-1]

    cs1 = ax_main.contour(X, Y, Hs.T, colors='C0', levels=levels_s, alpha=0.5, lw=2)
    cs2 = ax_main.contour(X, Y, Hd.T, colors='C1', levels=levels_d, alpha=0.5, lw=2) 
    proxies = [Line2D([],[],color='C0'), Line2D([],[],color='C1')]
    ax_main.legend(proxies, labels, loc='upper left')
    ax_main.set_xlabel(r"$\mathrm{log}_{10}\mathrm{(Flux [nJy])}$")
    ax_main.set_ylabel(r"$\mathrm{log}_{10}\mathrm{(Flux Error [nJy])}$")

    # ---- Top histogram ----
    ax_top.hist(xs, bins=xb, color='C0', alpha=0.5, density=True,histtype='step',lw=2)
    ax_top.hist(xd, bins=xb, color='C1', alpha=0.5, density=True,histtype='step',lw=2)
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    # ---- Right histogram ----
    ax_right.hist(ys, bins=yb, orientation='horizontal', color='C0', alpha=0.5, 
                  density=True,histtype='step',lw=2)
    ax_right.hist(yd, bins=yb, orientation='horizontal', color='C1', alpha=0.5,
                  density=True,histtype='step',lw=2)
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    # ---- Final layout ----
    for ax in [ax_top, ax_right]:
        for spine in ['top','right']:
            ax.spines[spine].set_visible(True)


def plot_logflux_vs_logfluxerr(sim, data, labels=['sim','data'],**kwargs):

    lc_to_plot = sim

    lcdata_plot = data

    lcdata_flux_njy,lcdata_fluxerr_njy = convert_flux_to_njy(lcdata_plot['lc.flux'],lcdata_plot['lc.flux_err'],zp=30.)

    lcdata_logflux = np.log10(lcdata_flux_njy)
    lcdata_logfluxerr = np.log10(lcdata_fluxerr_njy)
    lcsim_logflux = np.log10(lc_to_plot['lc.flux'])
    lcsim_logfluxerr = np.log10(lc_to_plot['lc.fluxerr'])
    logbins_flux = np.linspace(3.6,6,20)
    logbins_fluxerr = np.linspace(3,4.25,20)

    plt.hist(lcsim_logflux,bins=logbins_flux,alpha=0.3,density=True)
    plt.hist(lcdata_logflux,bins=logbins_flux,alpha=0.3,density=True)
    plt.xlabel("log10flux")
    plt.show()

    plt.hist(lcsim_logfluxerr,bins=logbins_fluxerr,alpha=0.3,density=True)
    plt.hist(lcdata_logfluxerr,bins=logbins_fluxerr,alpha=0.3,density=True)
    plt.xlabel("log10fluxerr")
    plt.show()

    lcsim_count,lcsim_x_edges,lcsim_y_edges, _ = stats.binned_statistic_2d(lcsim_logflux,lcsim_logfluxerr,
                                                 np.ones(len(lcsim_logflux)),
                                                 statistic='sum',bins=[logbins_flux,logbins_fluxerr])
    lcdata_count,lcdata_x_edges,lcdata_y_edges,_ = stats.binned_statistic_2d(lcdata_logflux,lcdata_logfluxerr,
                                             np.ones(len(lcdata_logflux)),
                                             statistic='sum',bins=[logbins_flux,logbins_fluxerr])
    lcsim_x = 0.5* (lcsim_x_edges[:-1]+lcsim_x_edges[1:])
    lcsim_y = 0.5* (lcsim_y_edges[:-1]+lcsim_y_edges[1:])
    lcdata_x = 0.5* (lcdata_x_edges[:-1]+lcdata_x_edges[1:])
    lcdata_y = 0.5* (lcdata_y_edges[:-1]+lcdata_y_edges[1:])
    lcsim_x_plot,lcsim_y_plot = np.meshgrid(lcsim_x, lcsim_y)
    lcdata_x_plot,lcdata_y_plot = np.meshgrid(lcdata_x, lcdata_y)
    CS = plt.contour(lcsim_x_plot.T,lcsim_y_plot.T,lcsim_count,alpha=0.5,label='sim',levels=10,colors='C0')
    CS = plt.contour(lcdata_x_plot.T,lcdata_y_plot.T,lcdata_count,alpha=0.5,label='data',levels=10,colors='C1')
    proxies = [Line2D([],[],color=c) for c in ['C0','C1']]
    plt.legend(proxies,['sim', 'data'])
    plt.xlabel('log10flux')
    plt.ylabel('log10fluxerr')

def plot_flux_vs_fluxerr(sim, data, labels=['sim','data'],**kwargs):

    lc_to_plot = sim

    lcdata_plot = data

    lcdata_flux_njy,lcdata_fluxerr_njy = convert_flux_to_njy(lcdata_plot['lc.flux'],lcdata_plot['lc.flux_err'],zp=30.)

    bins_flux = np.linspace(0,2e5,30)
    bins_fluxerr = np.linspace(0,1e4,30)

    plt.hist(lc_to_plot['lc.flux'],bins=bins_flux,alpha=0.3,density=True)
    plt.hist(lcdata_flux_njy,bins=bins_flux,alpha=0.3,density=True)
    # plt.xlabel("flux_nJy")
    plt.show()

    plt.hist(lc_to_plot['lc.fluxerr'],bins=bins_fluxerr,alpha=0.3,density=True)
    plt.hist(lcdata_fluxerr_njy,bins=bins_fluxerr,alpha=0.3,density=True)
    # plt.xlabel("fluxerr_nJy")
    plt.show()

    lcsim_count,lcsim_x_edges,lcsim_y_edges, _ = stats.binned_statistic_2d(lc_to_plot['lc.flux'],lc_to_plot['lc.fluxerr'],
                                                 np.ones(len(lc_to_plot['lc.flux'])),
                                                 statistic='sum',bins=[bins_flux,bins_fluxerr])
    lcdata_count,lcdata_x_edges,lcdata_y_edges,_ = stats.binned_statistic_2d(lcdata_flux_njy,lcdata_fluxerr_njy,
                                             np.ones(len(lcdata_flux_njy)),
                                             statistic='sum',bins=[bins_flux,bins_fluxerr])
    lcsim_x = 0.5* (lcsim_x_edges[:-1]+lcsim_x_edges[1:])
    lcsim_y = 0.5* (lcsim_y_edges[:-1]+lcsim_y_edges[1:])
    lcdata_x = 0.5* (lcdata_x_edges[:-1]+lcdata_x_edges[1:])
    lcdata_y = 0.5* (lcdata_y_edges[:-1]+lcdata_y_edges[1:])
    lcsim_x_plot,lcsim_y_plot = np.meshgrid(lcsim_x, lcsim_y)
    lcdata_x_plot,lcdata_y_plot = np.meshgrid(lcdata_x, lcdata_y)
    CS = plt.contour(lcsim_x_plot.T,lcsim_y_plot.T,lcsim_count,alpha=0.5,label='sim',levels=10,colors='C0')
    CS = plt.contour(lcdata_x_plot.T,lcdata_y_plot.T,lcdata_count,alpha=0.5,label='data',levels=10,colors='C1')
    proxies = [Line2D([],[],color=c) for c in ['C0','C1']]
    plt.legend(proxies,['sim', 'data'])
    plt.xlabel('flux_nJy')
    plt.ylabel('fluxerr_nJy')


def get_maxflux_and_err(flux,fluxerr):
    idx = np.argmax(flux)
    maxflux = flux[idx]
    maxfluxerr = fluxerr[idx]
    return {"maxflux":maxflux,"maxfluxerr":maxfluxerr}


def plot_logmaxflux_vs_logmaxfluxerr_corner(
    sim,
    data,
    labels=("sim","data"),
    zp=30.0,
    bins_flux=np.linspace(4.2, 6.4, 20),
    bins_fluxerr=np.linspace(3.0, 4.8, 20),
    smooth_sigma=0.6,       # set 0.0 to disable smoothing
    n_levels=7,             # number of raw contour levels before trimming
    figsize=(7,7),
):
    """
    Corner-plot layout:
      - Center: 2D contours of log10(maxflux [nJy]) vs log10(maxfluxerr [nJy])
      - Top:    1D hist of log10(maxflux)
      - Right:  1D hist of log10(maxfluxerr)

    Notes
    -----
    - `data` flux/err converted to nJy using `zp`; `sim` is assumed already in nJy.
    - Outermost (largest) contour level is removed.
    """

    def _safe_log10(x):
        x = np.asarray(x, float)
        out = np.full_like(x, np.nan, dtype=float)
        m = x > 0
        out[m] = np.log10(x[m])
        return out

    # ---- reduce to max flux/err ----
    # Expect .reduce(get_maxflux_and_err, "lc.flux", "lc.flux_err/fluxerr") -> {"maxflux", "maxfluxerr"}
    d_max = data.reduce(get_maxflux_and_err, "lc.flux", "lc.flux_err")
    d_maxflux_njy, d_maxfluxerr_njy = convert_flux_to_njy(
        d_max["maxflux"], d_max["maxfluxerr"], zp=zp
    )
    s_max = sim.reduce(get_maxflux_and_err, "lc.flux", "lc.fluxerr")

    # ---- log10 transform (safe) ----
    xd = _safe_log10(d_maxflux_njy)
    yd = _safe_log10(d_maxfluxerr_njy)
    xs = _safe_log10(s_max["maxflux"])
    ys = _safe_log10(s_max["maxfluxerr"])

    m_data = np.isfinite(xd) & np.isfinite(yd)
    m_sim  = np.isfinite(xs) & np.isfinite(ys)
    xd, yd = xd[m_data], yd[m_data]
    xs, ys = xs[m_sim], ys[m_sim]

    # ---- 2D histograms on shared bins ----
    Hs, xe, ye = np.histogram2d(xs, ys, bins=[bins_flux, bins_fluxerr])  # keep bins consistent
    Hd, _,  _  = np.histogram2d(xd, yd,         bins=[bins_flux, bins_fluxerr])

    if smooth_sigma and smooth_sigma > 0:
        Hs = gaussian_filter(Hs, smooth_sigma)
        Hd = gaussian_filter(Hd, smooth_sigma)

    # Centers for contour coordinates
    X = 0.5*(xe[:-1] + xe[1:])
    Y = 0.5*(ye[:-1] + ye[1:])
    XX, YY = np.meshgrid(X, Y)

    # ---- set levels and drop the outermost (largest) ----
    def _levels(H):
        positive = H[H > 0]
        if positive.size == 0:
            return None
        lv = np.linspace(np.nanmin(positive), np.nanmax(positive), n_levels)
        lv = lv[1:-1]  # drop the largest (outermost) level
        # make sure levels are strictly increasing and finite
        lv = lv[np.isfinite(lv)]
        lv = np.unique(lv)
        return lv if lv.size >= 2 else None

    levels_s = _levels(Hs)
    levels_d = _levels(Hd)

    # ---- figure layout (corner style) ----
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=(4,1), height_ratios=(1,4), wspace=0.05, hspace=0.05)
    ax_top   = fig.add_subplot(gs[0,0])
    ax_right = fig.add_subplot(gs[1,1])
    ax_main  = fig.add_subplot(gs[1,0])

    # ---- main 2D contours ----
    if levels_s is not None:
        cs1 = ax_main.contour(XX, YY, Hs.T, colors='C0', levels=levels_s, alpha=0.5,lw=2)
    if levels_d is not None:
        cs2 = ax_main.contour(XX, YY, Hd.T, colors='C1', levels=levels_d, alpha=0.5,lw=2)

    proxies = [Line2D([], [], color='C0'), Line2D([], [], color='C1')]
    ax_main.legend(proxies, labels, loc='upper left', frameon=False)
    ax_main.set_xlabel(r"$\mathrm{log}_{10}\mathrm{(Max Flux [nJy])}$")
    ax_main.set_ylabel(r"$\mathrm{log}_{10}\mathrm{(Max Flux Error [nJy])}$")

    # ---- top histogram: log10 max flux ----
    ax_top.hist(xs, bins=bins_flux, alpha=0.5, density=True, label=labels[0],
        histtype='step',lw=2)
    ax_top.hist(xd, bins=bins_flux, alpha=0.5, density=True, label=labels[1],
        histtype='step',lw=2)
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    # ---- right histogram: log10 max flux err ----
    ax_right.hist(ys, bins=bins_fluxerr, orientation='horizontal', alpha=0.5,
                  density=True, label=labels[0],histtype='step',lw=2)
    ax_right.hist(yd, bins=bins_fluxerr, orientation='horizontal', alpha=0.5,
                  density=True, label=labels[1],histtype='step',lw=2)
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    # subtle cleanup
    for ax in (ax_top, ax_right):
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)


def plot_logmaxflux_vs_logmaxfluxerr(sim, data, labels=['sim','data'],**kwargs):

    lc_to_plot = sim
    lcdata_plot = data

    lcdata_maxflux = lcdata_plot.reduce(get_maxflux_and_err,"lc.flux","lc.flux_err")
    lcdata_maxflux_njy,lcdata_maxfluxerr_njy = convert_flux_to_njy(lcdata_maxflux["maxflux"],lcdata_maxflux["maxfluxerr"],
                                                                   zp=30.)

    lcdata_logmaxflux = np.log10(lcdata_maxflux_njy)
    lcdata_logmaxfluxerr = np.log10(lcdata_maxfluxerr_njy)

    lcsim_maxflux = lc_to_plot.reduce(get_maxflux_and_err,"lc.flux","lc.fluxerr")
    lcsim_logmaxflux = np.log10(lcsim_maxflux["maxflux"])
    lcsim_logmaxfluxerr = np.log10(lcsim_maxflux["maxfluxerr"])
    logbins_flux = np.linspace(3.5,7,20)
    logbins_fluxerr = np.linspace(3,5,20)

    plt.hist(lcsim_logmaxflux,bins=logbins_flux,alpha=0.3,density=True)
    plt.hist(lcdata_logmaxflux,bins=logbins_flux,alpha=0.3,density=True)
    plt.show()

    plt.hist(lcsim_logmaxfluxerr,bins=logbins_fluxerr,alpha=0.3,density=True)
    plt.hist(lcdata_logmaxfluxerr,bins=logbins_fluxerr,alpha=0.3,density=True)
    plt.show()

    lcsim_count,lcsim_x_edges,lcsim_y_edges, _ = stats.binned_statistic_2d(lcsim_logmaxflux,lcsim_logmaxfluxerr,
                                                 np.ones(len(lcsim_logmaxflux)),
                                                 statistic='sum',bins=[logbins_flux,logbins_fluxerr])
    lcdata_count,lcdata_x_edges,lcdata_y_edges,_ = stats.binned_statistic_2d(lcdata_logmaxflux,lcdata_logmaxfluxerr,
                                             np.ones(len(lcdata_logmaxflux)),
                                             statistic='sum',bins=[logbins_flux,logbins_fluxerr])
    lcsim_x = 0.5* (lcsim_x_edges[:-1]+lcsim_x_edges[1:])
    lcsim_y = 0.5* (lcsim_y_edges[:-1]+lcsim_y_edges[1:])
    lcdata_x = 0.5* (lcdata_x_edges[:-1]+lcdata_x_edges[1:])
    lcdata_y = 0.5* (lcdata_y_edges[:-1]+lcdata_y_edges[1:])
    lcsim_x_plot,lcsim_y_plot = np.meshgrid(lcsim_x, lcsim_y)
    lcdata_x_plot,lcdata_y_plot = np.meshgrid(lcdata_x, lcdata_y)
    CS = plt.contour(lcsim_x_plot.T,lcsim_y_plot.T,lcsim_count,alpha=0.5,label='sim',levels=10,colors='C0')
    CS = plt.contour(lcdata_x_plot.T,lcdata_y_plot.T,lcdata_count,alpha=0.5,label='data',levels=10,colors='C1')
    proxies = [Line2D([],[],color=c) for c in ['C0','C1']]
    plt.legend(proxies,['sim', 'data'])
    plt.xlabel('log10maxflux')
    plt.ylabel('log10maxfluxerr')


def convert_flux_to_mag(flux,fluxerr,zp=0.):
    mag = -2.5*np.log10(flux) + zp
    magerr = 2.5/np.log(10.)*fluxerr/flux
    return mag,magerr

def plot_mag_vs_magerr(sim, data, labels=['sim','data'],**kwargs):
   
    lc_to_plot = sim
    lcdata_plot = data

    bins = np.linspace(10,23,30)

    lc_to_plot_mag,lc_to_plot_magerr = lc_to_plot['lc.mag'],lc_to_plot['lc.magerr']
    lcdata_mag,lcdata_magerr = lcdata_plot['lc.mag'],lcdata_plot['lc.magerr']

    plt.hist(lc_to_plot_mag,bins=bins,alpha=0.3,density=True)
    plt.hist(lcdata_mag,bins=bins,alpha=0.3,density=True)
    plt.show()

    bins = np.linspace(0,0.5,30)

    plt.hist(lc_to_plot_magerr,bins=bins,alpha=0.3,density=True)
    plt.hist(lcdata_magerr,bins=bins,alpha=0.3,density=True)
    plt.show()

    plt.plot(lc_to_plot_mag,lc_to_plot_magerr,'o',alpha=0.01,label='sim')
    plt.plot(lcdata_mag,lcdata_magerr,'o',alpha=0.01,label='data')
    # plt.axhline(y=0.01)
    plt.legend()
    plt.show()

def plot_coverage_map(ztf_obstable,lightcurves,plot_na_location=True,plot_all_location=False):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='mollweide')

    # Convert RA/Dec (deg) to Mollweide coords (radians)
    ra_rad = np.radians(ztf_obstable._table.drop_duplicates("fieldid").ra)
    dec_rad = np.radians(ztf_obstable._table.drop_duplicates("fieldid").dec)

    # Shift RA: matplotlib Mollweide expects longitude from -π to +π
    ra_moll = np.remainder(ra_rad + 2*np.pi, 2*np.pi)
    ra_moll[ra_moll > np.pi] -= 2*np.pi
    ra_moll = -ra_moll  # reverse RA (to match astronomy convention)

    # convert (quick) : 1 deg on x-axis spans total width of 2π rad = 360 deg
    # axis width in points:
    r_deg = 3.89
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in = bbox.width
    width_pt = width_in * fig.dpi * 72/72  # matplotlib already uses points; still compute for clarity
    # diameter in degrees -> as fraction of 360°
    diam_deg = 2*r_deg
    frac = diam_deg / 360.0
    # axis width in points (approx for marker size)
    axis_width_pts = fig.get_figwidth()*fig.dpi  # rough
    diam_pts = frac * axis_width_pts
    s = (diam_pts/2)**2 * np.pi  # area in points^2 for a circle marker

    # Plot
    ax.scatter(ra_moll, dec_rad, s=s, alpha=0.1, edgecolors="none",label='ztf_pointings')

    ra_rad = np.radians(lightcurves.ra)
    dec_rad = np.radians(lightcurves.dec)

    ra_moll = np.remainder(ra_rad + 2*np.pi, 2*np.pi)
    ra_moll[ra_moll > np.pi] -= 2*np.pi
    ra_moll = -ra_moll  # reverse RA (to match astronomy convention)

    if plot_all_location:
        ax.scatter(ra_moll, dec_rad, marker='*', alpha=0.8,lw=1,label='lightcurves')

    idx = lightcurves.lightcurve.isna()
    ra_rad = np.radians(lightcurves.loc[idx].ra)
    dec_rad = np.radians(lightcurves.loc[idx].dec)

    ra_moll = np.remainder(ra_rad + 2*np.pi, 2*np.pi)
    ra_moll[ra_moll > np.pi] -= 2*np.pi
    ra_moll = -ra_moll  # reverse RA (to match astronomy convention)

    if plot_na_location:
        ax.scatter(ra_moll, dec_rad, marker='*', alpha=1, lw=1, label='lightcurves_nan')

    # Set RA ticks (in hours)
    ra_tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    ax.set_xticks(np.radians(np.linspace(-150, 150, len(ra_tick_labels))))
    ax.set_xticklabels([f"{int(x/15)}h" for x in ra_tick_labels])  # RA in hours

    # Set Dec ticks (default is fine, but make sure labeled in degrees)
    ax.set_yticks(np.radians(np.arange(-75, 90, 15)))
    ax.set_yticklabels([f"{d}°" for d in np.arange(-75, 90, 15)])

    # Axis labels
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")

    ax.grid(True)
    plt.legend()
    plt.show()