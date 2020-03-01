import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf


plt.rc('text', usetex=True)

outpath = "plots"
inpath = ""

currentfile = "data.txt"

# Read from file
fulln, fullW= np.loadtxt(path.join(inpath, currentfile), usecols=(0, 1), unpack=True)

# # Draw Raw data plot
# fig, ax = plt.subplots(figsize=(8, 3.8))
# ax.plot(n, W,  ls='-')
# plt.grid()
# plt.ylabel(r'$W_{n}$')
# plt.xlabel(r'$t_n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# #plt.title(r'$I(t) = I_0^A (1 - 2 \exp (\frac{-t }{T_1^A})) + I_0^B (1 - 2 \exp (\frac{-t }{T_1^B}))$')
# #ax.legend(loc='best', frameon=True)
# plt.draw()
# fig.savefig(path.join(outpath, "raw.png"))
# plt.clf()

n = fulln[65:1003]
W = fullW[65:1003]

# # Draw Raw data plot
# fig, ax = plt.subplots(figsize=(8, 3.8))
# ax.plot(n, W,  ls='-')
# plt.grid()
# plt.ylabel(r'$W_{n}$')
# plt.xlabel(r'$t_n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# #plt.title(r'$I(t) = I_0^A (1 - 2 \exp (\frac{-t }{T_1^A})) + I_0^B (1 - 2 \exp (\frac{-t }{T_1^B}))$')
# #ax.legend(loc='best', frameon=True)
# plt.draw()
# fig.savefig(path.join(outpath, "raw_cut.png"))
# plt.clf()

# nlags=200
# acf_val, confit_val, qstat_val, pvalues = acf(W, unbiased=True, nlags=nlags-1, qstat=True, alpha=.05)
# lags=np.arange(1, nlags+1, 1)


# # Draw acf plot
# fig, ax = plt.subplots(figsize=(8, 3.8))
# ax.fill_between(lags[1:], confit_val[1:, 0], confit_val[1:, 1], where=confit_val[1:, 1] >= confit_val[1:, 0], facecolor='gainsboro', interpolate=True)
# #ax.scatter(lags[1:], acf_val[1:], marker='+', color='crimson')
# ax.bar(lags[1:], acf_val[1:], color='crimson')
# plt.grid()
# plt.ylabel(r'$r_{\tau}$')
# plt.xlabel(r'$\tau$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# #plt.title(r'$I(t) = I_0^A (1 - 2 \exp (\frac{-t }{T_1^A})) + I_0^B (1 - 2 \exp (\frac{-t }{T_1^B}))$')
# #ax.legend(loc='best', frameon=True)
# plt.draw()
# fig.savefig(path.join(outpath, "acf200.png"))
# plt.clf()

# # Draw QLB p-values plot
# fig, ax = plt.subplots(figsize=(8, 3.8))
# ax.bar(lags[1:-1], pvalues[1:], color='crimson')
# plt.grid()
# plt.ylabel(r'Ljung–Box Q test p-values')
# plt.xlabel(r'$n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# #plt.title(r'$I(t) = I_0^A (1 - 2 \exp (\frac{-t }{T_1^A})) + I_0^B (1 - 2 \exp (\frac{-t }{T_1^B}))$')
# #ax.legend(loc='best', frameon=True)
# plt.draw()
# fig.savefig(path.join(outpath, "qlb200.png"))
# plt.clf()