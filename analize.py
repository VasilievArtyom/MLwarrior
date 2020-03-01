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

n = fulln[65:]
W = fullW[65:]

# Draw Raw data plot
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(n, W,  ls='-')
plt.grid()
plt.ylabel(r'$W_{n}$')
plt.xlabel(r'$t_n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
#plt.title(r'$I(t) = I_0^A (1 - 2 \exp (\frac{-t }{T_1^A})) + I_0^B (1 - 2 \exp (\frac{-t }{T_1^B}))$')
#ax.legend(loc='best', frameon=True)
plt.draw()
fig.savefig(path.join(outpath, "raw_cut.png"))
plt.clf()

#print(np.average(W[-900:]))