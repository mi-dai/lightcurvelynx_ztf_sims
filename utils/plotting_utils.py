import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.lines import Line2D

def plot_snr_distr(data_list, labels=None, **kwargs):
    ax = plt.subplot(1,1,1)
    if labels is None:
        labels = [f'data{i}' for i in range(0,len(data_list))]
    for i, data in enumerate(data_list):
        ax.hist(data['lc.snr'], label=labels[i],**kwargs)
    ax.legend()

def convert_flux_to_njy(flux,fluxerr,zp=0.):
    zp_njy = 31.4
    #-2.5*log10(flux_njy) + zp_njy = -2.5*log10(flux) + zp
    flux_njy = flux*np.power(10., -0.4*(zp-zp_njy))
    fluxerr_njy = fluxerr*np.power(10., -0.4*(zp-zp_njy))         
    return flux_njy,fluxerr_njy

def plot_logflux_vs_logfluxerr(sim, data, labels=['sim','data'],**kwargs):

    lc_to_plot = sim

    lcdata_plot = data

    lcdata_flux_njy,lcdata_fluxerr_njy = convert_flux_to_njy(lcdata_plot['lc.flux'],lcdata_plot['lc.flux_err'],zp=30.)

    lcdata_logflux = np.log10(lcdata_flux_njy)
    lcdata_logfluxerr = np.log10(lcdata_fluxerr_njy)
    lcsim_logflux = np.log10(lc_to_plot['lc.flux'])
    lcsim_logfluxerr = np.log10(lc_to_plot['lc.fluxerr'])
    logbins_flux = np.linspace(3.5,7,30)
    logbins_fluxerr = np.linspace(3,5,20)

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
    plt.show()    



def get_maxflux_and_err(flux,fluxerr):
    idx = np.argmax(flux)
    maxflux = flux[idx]
    maxfluxerr = fluxerr[idx]
    return {"maxflux":maxflux,"maxfluxerr":maxfluxerr}


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
    plt.show()


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
    plt.legend()
    plt.show()