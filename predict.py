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



# # fit model
# p = d = q = range(0, 15)
# pdq = list(itertools.product(p, d, q))
# warnings.filterwarnings("ignore")
# f = open('hyperparam.txt','w') 
# for param in pdq:
# 	try:
# 		model = sm.tsa.statespace.SARIMAX(W,
# 										  order=param,
# 										  seasonal_order=(0,0,0, 12),
# 										  enforce_stationarity=False,
# 										  enforce_invertibility=False)
# 		results = model.fit()
# 		print('ARIMA{} - AIC:{}'.format(param, results.aic), file=f)
# 	except:
# 		continue

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


# final_model = sm.tsa.statespace.SARIMAX(W,
# 										order=(2, 0, 14),
# 										seasonal_order=(0,0,0, 12),
# 										enforce_stationarity=False,
# 										enforce_invertibility=False)
# results = final_model.fit()
# print(results.summary().tables[1])

# pred = results.get_prediction(end=1023, dynamic=False)
# pred_vals = pred.predicted_mean
# pred_ci = pred.conf_int()

# extended_n = np.arange(0, 1024, 1)

# # Draw naive prediction plot
# fig, ax = plt.subplots(figsize=(8, 3.8))
# ax.scatter(fulln[1003:], pred_vals[1003:], 
# 			marker='+',
# 			color='crimson',
# 			label='Prediction',
# 			zorder=10)
# ax.fill_between(fulln[1003:],
#                 pred_ci[1003:, 0],
#                 pred_ci[1003:, 1],
#                 facecolor='gainsboro', 
#                 label='Confidence interval',
#                 interpolate=True,
#                 zorder=0)
# ax.plot(fulln[800:], fullW[800:],  ls='-', label='Raw signal', zorder=5)
# plt.grid()
# plt.ylabel(r'$W_{n}$')
# plt.xlabel(r'$t_n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# # plt.title(r'Ljung–Box Q test p-values')
# ax.legend(loc='best', frameon=True)
# plt.draw()
# fig.savefig(path.join(outpath, "validation.png"))
# plt.clf()


final_model = sm.tsa.statespace.SARIMAX(fullW[65:],
										order=(2, 0, 14),
										seasonal_order=(0,0,0, 12),
										enforce_stationarity=False,
										enforce_invertibility=False)
results = final_model.fit()
print(results.summary().tables[1])

pred = results.get_prediction(end=1043, dynamic=False)
pred_vals = pred.predicted_mean
pred_ci = pred.conf_int()

extended_n = np.arange(0, 1044, 1)

# Draw naive prediction plot
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.scatter(extended_n[1024:], pred_vals[1024:], 
			marker='+',
			color='crimson',
			label='Prediction',
			zorder=10)
ax.fill_between(extended_n[1024:],
                pred_ci[1024, 0],
                pred_ci[1024:, 1],
                facecolor='gainsboro', 
                label='Confidence interval',
                interpolate=True,
                zorder=0)
ax.plot(fulln[800:], fullW[800:],  ls='-', label='Raw signal', zorder=5)
plt.grid()
plt.ylabel(r'$W_{n}$')
plt.xlabel(r'$t_n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
ax.legend(loc='upper left', frameon=True)
plt.draw()
fig.savefig(path.join(outpath, "prediction.png"))
plt.clf()

f = open('prediction.txt','w')
print('#timestamp, value, Confidence interval bounds', file=f)
for index in range(1024, 1043):
	print(extended_n[index], pred_vals[index], pred_ci[index, 0], pred_ci[index, 1], file=f)
