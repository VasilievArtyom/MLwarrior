import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from scipy import signal
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA



plt.rc('text', usetex=True)

outpath = "plots"
inpath = ""

currentfile = "data.txt"

# Read from file
fulln, fullW= np.loadtxt(path.join(inpath, currentfile), usecols=(0, 1), unpack=True)

n = fulln[65:1003]
W = fullW[65:1003]

# fit model
model = ARIMA(W, order=(7,0,1))
model_fit = model.fit(disp=0)



# f, Pxx_den = signal.periodogram(W - np.average(W), window='hamming')

# # Draw QLB p-values plot
# fig, ax = plt.subplots(figsize=(8, 3.8))
# # ax.bar(lags[1:-1], pvalues[1:], color='crimson')
# ax.semilogy(f, Pxx_den, color='crimson')
# plt.grid()
# plt.ylabel(r'$\frac{1}{T} {|X_{T} (i \omega)|}^2 $')
# plt.xlabel(r'$\omega$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# # plt.title(r'Ljung–Box Q test p-values')
# #ax.legend(loc='best', frameon=True)
# plt.draw()
# fig.savefig(path.join(outpath, "psd.png"))
# plt.clf()
