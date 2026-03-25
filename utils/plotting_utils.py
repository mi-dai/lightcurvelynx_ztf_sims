import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import LogLocator

def plot_snr_distr(data_list, labels=None, colors=["C0","C1"],**kwargs):
    ax = plt.subplot(1,1,1)
    if labels is None:
        labels = [f'data{i}' for i in range(0,len(data_list))]
    for i, data in enumerate(data_list):
        # labels[i] = f"{labels[i]}(N={data['lc.snr'].count()})"
        ax.hist(data['lc.snr'], label=labels[i], color=colors[i],**kwargs)
    ax.legend()
    return ax

def convert_flux_to_njy(flux,fluxerr,zp=0.):
    zp_njy = 31.4
    #-2.5*log10(flux_njy) + zp_njy = -2.5*log10(flux) + zp
    flux_njy = flux*np.power(10., -0.4*(zp-zp_njy))
    fluxerr_njy = fluxerr*np.power(10., -0.4*(zp-zp_njy))         
    return flux_njy,fluxerr_njy

def plot_logflux_vs_logfluxerr_corner(sim, data, labels=('sim','data'), smooth_sigma=0.6, fraction=False):

    lcsim_flux     = sim['lc.flux']
    lcsim_fluxerr  = sim['lc.fluxerr']
    lcdata_flux    = data['lc.flux']
    lcdata_fluxerr = data['lc.flux_err']

    # ---- Clean finite values ----
    m_sim  = np.isfinite(lcsim_flux) & np.isfinite(lcsim_fluxerr)
    m_data = np.isfinite(lcdata_flux) & np.isfinite(lcdata_fluxerr)
    xs, ys = lcsim_flux[m_sim], lcsim_fluxerr[m_sim]
    xd, yd = lcdata_flux[m_data], lcdata_fluxerr[m_data]

    # ---- Define bins ----
    xb = np.logspace(2.8, 8., 20)
    yb = np.logspace(2.2, 5.7, 20)

    # ---- Compute 2D histograms ----
    Hs, xe, ye = np.histogram2d(xs, ys, bins=[xb, yb], density=False)
    Hd, _, _   = np.histogram2d(xd, yd, bins=[xb, yb], density=False)

    print("Nsim:",np.sum(Hs))
    print("Ndata",np.sum(Hd))

    Hs_s, Hd_s = Hs, Hd
    if smooth_sigma:
        # floor prevents empty bins from dominating log smoothing
        eps = 0.5   # half a count is a very standard choice

        Hs_s = np.exp(gaussian_filter(np.log(Hs + eps), smooth_sigma))
        Hd_s = np.exp(gaussian_filter(np.log(Hd + eps), smooth_sigma))

    if fraction:
        # ---- Convert smoothed counts -> density ----
        Hs = Hs_s / Hs_s.sum()
        Hd = Hd_s / Hd_s.sum()

    X, Y = np.meshgrid(0.5*(xe[:-1]+xe[1:]), 0.5*(ye[:-1]+ye[1:]))

    # ---- Create figure layout ----
    fig = plt.figure(figsize=(6,6))
    gs  = fig.add_gridspec(2,2, width_ratios=(4,1), height_ratios=(1,4),
                           wspace=0.05, hspace=0.05)

    ax_top   = fig.add_subplot(gs[0,0])
    ax_right = fig.add_subplot(gs[1,1])
    ax_main  = fig.add_subplot(gs[1,0])

    # ---- Main 2D contour ----
    # Set same levels for both contours
    levels = np.logspace(np.log10(np.nanmin(Hs[Hs>0])),np.log10(np.nanmax(Hs)),7)[1:-1]
    levels = np.power(10.,np.round(np.log10(levels),decimals=1))
    levels = np.power(10., np.arange(-5.,-1.,0.5))
    print(levels)
    print(np.log10(levels))

    cs1 = ax_main.contour(X, Y, Hs.T, colors='C0', levels=levels, alpha=0.8, lw=2)
    cs2 = ax_main.contour(X, Y, Hd.T, colors='C1', levels=levels, alpha=0.8, lw=2) 
    proxies = [Line2D([],[],color='C0'), Line2D([],[],color='C1')]
    ax_main.legend(proxies, labels, loc='upper left')
    ax_main.set_xlabel(r"$\mathrm{Flux [nJy]}$")
    ax_main.set_ylabel(r"$\mathrm{Flux Error [nJy]}$")
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')

    # ---- Top histogram ----
    xs_hist, bin_edges_xs = np.histogram(xs,bins=xb)
    xd_hist, bin_edges_xd = np.histogram(xd,bins=xb)
    if fraction:
        xs_hist = xs_hist/np.sum(xs_hist)
        xd_hist = xd_hist/np.sum(xd_hist)
    bins_xs = np.sqrt(bin_edges_xs[:-1] * bin_edges_xs[1:])
    bins_xd = np.sqrt(bin_edges_xd[:-1] * bin_edges_xd[1:])
    ax_top.step(bins_xs,xs_hist,color="C0", lw=2, alpha=0.8)
    ax_top.step(bins_xd,xd_hist,color="C1", lw=2, alpha=0.8)
    # ax_top.hist(xs, bins=xb, color='C0', alpha=0.8, density=False,histtype='step',lw=2)
    # ax_top.hist(xd, bins=xb, color='C1', alpha=0.8, density=False,histtype='step',lw=2)
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.yaxis.set_major_locator(LogLocator(base=10,numticks=3))
    ax_top.yaxis.minorticks_off()
    ax_top.set_ylabel("Count" if not fraction else "Fraction")
    ax_top.set_xticks([])

    # ---- Right histogram ----
    ys_hist, bin_edges_ys = np.histogram(ys,bins=yb)
    yd_hist, bin_edges_yd = np.histogram(yd,bins=yb)
    if fraction:
        ys_hist = ys_hist/np.sum(ys_hist)
        yd_hist = yd_hist/np.sum(yd_hist)
    bins_ys = np.sqrt(bin_edges_ys[:-1] * bin_edges_ys[1:])
    bins_yd = np.sqrt(bin_edges_yd[:-1] * bin_edges_yd[1:])
    ax_right.step(ys_hist,bins_ys,where="post",color="C0",lw=2, alpha=0.8)
    ax_right.step(yd_hist,bins_yd,where="post",color="C1",lw=2, alpha=0.8)
    # ax_right.hist(ys, bins=yb, orientation='horizontal', color='C0', alpha=0.8, 
    #               density=False,histtype='step',lw=2)
    # ax_right.hist(yd, bins=yb, orientation='horizontal', color='C1', alpha=0.8,
    #               density=False,histtype='step',lw=2)
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_xscale('log')
    ax_right.set_yscale('log')
    ax_right.xaxis.set_major_locator(LogLocator(base=10,numticks=3))
    ax_right.xaxis.minorticks_off()
    ax_right.set_xlabel("Count" if not fraction else "Fraction")
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
    CS = plt.contour(lcsim_x_plot.T,lcsim_y_plot.T,lcsim_count,alpha=0.8,label='sim',levels=10,colors='C0')
    CS = plt.contour(lcdata_x_plot.T,lcdata_y_plot.T,lcdata_count,alpha=0.8,label='data',levels=10,colors='C1')
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
    CS = plt.contour(lcsim_x_plot.T,lcsim_y_plot.T,lcsim_count,alpha=0.8,label='sim',levels=10,colors='C0')
    CS = plt.contour(lcdata_x_plot.T,lcdata_y_plot.T,lcdata_count,alpha=0.8,label='data',levels=10,colors='C1')
    proxies = [Line2D([],[],color=c) for c in ['C0','C1']]
    plt.legend(proxies,['sim', 'data'])
    plt.xlabel('flux_nJy')
    plt.ylabel('fluxerr_nJy')


def get_maxflux_and_err(flux,fluxerr):
    idx = np.argmax(flux)
    maxflux = flux[idx]
    maxfluxerr = fluxerr[idx]
    return {"maxflux":maxflux,"maxfluxerr":maxfluxerr}


def plot_logmaxflux_vs_logmaxfluxerr_corner(sim, data, labels=('sim','data'), smooth_sigma=0.6, fraction=False):

    d_max = data.reduce(get_maxflux_and_err, "lc.flux", "lc.flux_err")

    s_max = sim.reduce(get_maxflux_and_err, "lc.flux", "lc.fluxerr")

    lcsim_flux =  s_max["maxflux"]
    lcsim_fluxerr = s_max["maxfluxerr"]
    lcdata_flux = d_max["maxflux"]
    lcdata_fluxerr = d_max["maxfluxerr"]

    # ---- Clean finite values ----
    m_sim  = np.isfinite(lcsim_flux) & np.isfinite(lcsim_fluxerr)
    m_data = np.isfinite(lcdata_flux) & np.isfinite(lcdata_fluxerr)
    xs, ys = lcsim_flux[m_sim], lcsim_fluxerr[m_sim]
    xd, yd = lcdata_flux[m_data], lcdata_fluxerr[m_data]

    # ---- Define bins ----
    xb = np.logspace(3.5, 7.6, 20)
    yb = np.logspace(2.7, 5.2, 20)

    # ---- Compute 2D histograms ----
    Hs, xe, ye = np.histogram2d(xs, ys, bins=[xb, yb],density=False)
    Hd, _, _   = np.histogram2d(xd, yd, bins=[xb, yb],density=False)
    print("Nsim:",np.sum(Hs))
    print("Ndata",np.sum(Hd))

    Hs_s, Hd_s = Hs, Hd
    if smooth_sigma:
        # floor prevents empty bins from dominating log smoothing
        eps = 0.5   # half a count is a very standard choice

        Hs_s = np.exp(gaussian_filter(np.log(Hs + eps), smooth_sigma))
        Hd_s = np.exp(gaussian_filter(np.log(Hd + eps), smooth_sigma))

    if fraction:
        # ---- Convert smoothed counts -> density ----
        Hs = Hs_s / Hs_s.sum() 
        Hd = Hd_s / Hd_s.sum()

    X, Y = np.meshgrid(0.5*(xe[:-1]+xe[1:]), 0.5*(ye[:-1]+ye[1:]))

    # ---- Create figure layout ----
    fig = plt.figure(figsize=(6,6))
    gs  = fig.add_gridspec(2,2, width_ratios=(4,1), height_ratios=(1,4),
                           wspace=0.05, hspace=0.05)

    ax_top   = fig.add_subplot(gs[0,0])
    ax_right = fig.add_subplot(gs[1,1])
    ax_main  = fig.add_subplot(gs[1,0])

    # ---- Main 2D contour ----
    # Set same levels for both contours
    levels = np.logspace(np.log10(np.nanmin(Hs[Hs>0])),np.log10(np.nanmax(Hs)),7)[1:-1]
    levels = np.power(10.,np.round(np.log10(levels),decimals=1))
    levels = np.power(10., np.arange(-3.5,-1.,0.5))
    print(levels)
    print(np.log10(levels))

    cs1 = ax_main.contour(X, Y, Hs.T, colors='C0', levels=levels, alpha=0.8, lw=2)
    cs2 = ax_main.contour(X, Y, Hd.T, colors='C1', levels=levels, alpha=0.8, lw=2) 
    proxies = [Line2D([],[],color='C0'), Line2D([],[],color='C1')]
    ax_main.legend(proxies, labels, loc='upper left')
    ax_main.set_xlabel(r"$\mathrm{Max Flux [nJy]}$")
    ax_main.set_ylabel(r"$\mathrm{Max Flux Error [nJy]}$")
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')

    # ---- Top histogram ----
    xs_hist, bin_edges_xs = np.histogram(xs,bins=xb)
    xd_hist, bin_edges_xd = np.histogram(xd,bins=xb)
    if fraction:
        xs_hist = xs_hist/np.sum(xs_hist)
        xd_hist = xd_hist/np.sum(xd_hist)
    bins_xs = np.sqrt(bin_edges_xs[:-1] * bin_edges_xs[1:])
    bins_xd = np.sqrt(bin_edges_xd[:-1] * bin_edges_xd[1:])
    dx = np.diff(xb)
    ax_top.step(bins_xs,xs_hist,color="C0", lw=2, alpha=0.8)
    ax_top.step(bins_xd,xd_hist,color="C1", lw=2, alpha=0.8)
    # ax_top.hist(xs, bins=xb, color='C0', alpha=0.8, density=False,histtype='step',lw=2)
    # ax_top.hist(xd, bins=xb, color='C1', alpha=0.8, density=False,histtype='step',lw=2)
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.tick_params(which="minor", left=False, right=False)
    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_ylabel("Count" if not fraction else "Fraction")
    ax_top.set_xticks([])

    # ---- Right histogram ----
    ys_hist, bin_edges_ys = np.histogram(ys,bins=yb)
    yd_hist, bin_edges_yd = np.histogram(yd,bins=yb)
    if fraction:
        ys_hist = ys_hist/np.sum(ys_hist)
        yd_hist = yd_hist/np.sum(yd_hist)
    bins_ys = np.sqrt(bin_edges_ys[:-1] * bin_edges_ys[1:])
    bins_yd = np.sqrt(bin_edges_yd[:-1] * bin_edges_yd[1:])
    dy = np.diff(yb)
    ax_right.step(ys_hist,bins_ys,where="post",color="C0",lw=2, alpha=0.8)
    ax_right.step(yd_hist,bins_yd,where="post",color="C1",lw=2, alpha=0.8)
    # ax_right.hist(ys, bins=yb, orientation='horizontal', color='C0', alpha=0.8, 
    #               density=False,histtype='step',lw=2)
    # ax_right.hist(yd, bins=yb, orientation='horizontal', color='C1', alpha=0.8,
    #               density=False,histtype='step',lw=2)
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.tick_params(which="minor", bottom=False, top=False)
    ax_right.set_xscale('log')
    ax_right.set_yscale('log')
    ax_right.set_xlabel("Count" if not fraction else "Fraction")
    ax_right.set_yticks([])

    # ---- Final layout ----
    for ax in [ax_top, ax_right]:
        for spine in ['top','right']:
            ax.spines[spine].set_visible(True)


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
    CS = plt.contour(lcsim_x_plot.T,lcsim_y_plot.T,lcsim_count,alpha=0.8,label='sim',levels=10,colors='C0')
    CS = plt.contour(lcdata_x_plot.T,lcdata_y_plot.T,lcdata_count,alpha=0.8,label='data',levels=10,colors='C1')
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