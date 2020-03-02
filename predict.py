import math
import numpy as np
from numpy import *
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import itertools
import warnings


plt.rc('text', usetex=True)

outpath = "plots"
inpath = ""

currentfile = "data.txt"

# Read from file
fulln, fullW= np.loadtxt(path.join(inpath, currentfile), usecols=(0, 1), unpack=True)

n = fulln[65:1003]
W = fullW[65:1003]



# fit model
p = d = q = range(0, 15)
pdq = list(itertools.product(p, d, q))
warnings.filterwarnings("ignore")
f = open('hyperparam.txt','w') 
for param in pdq:
	try:
		model = sm.tsa.statespace.SARIMAX(W,
										  order=param,
										  seasonal_order=(0,0,0, 12),
										  enforce_stationarity=False,
										  enforce_invertibility=False)
		results = model.fit()
		print('ARIMA{} - AIC:{}'.format(param, results.aic), file=f)
	except:
		continue

# model = ARIMA(W, order=(p,d,q))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
# print(model_fit.conf_int(), shape(model_fit.conf_int()))


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
