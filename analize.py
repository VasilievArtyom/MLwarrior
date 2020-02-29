import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from os import path
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

outpath = ""
inpath = ""

currentfile = "136T1"

#########################
# Define functions to fit

# I(t) = I_0A (1 - 2 exp (-t / T_1A)) + I_0B (1 - 2 exp (-t / T_1B))
def f_T1(_t, _I_0A, _T1A, _I_0B, _T1B):
    return _I_0A * (1 - 2 * exp(- _t / _T1A)) + _I_0B * (1 - 2 * exp(- _t / _T1B))

# I(t) = I_0A exp (-t / T_2A) + I_0B exp (-t / T_2B)
def f_T2(_t, _I_0A, _T2A, _I_0B, _T2B):
    return _I_0A * (exp(- _t / _T2A)) + _I_0B * (exp(- _t / _T2B))
#########################


# Read from file
TE, I, I_err = np.loadtxt(path.join(inpath, currentfile+".txt"), usecols=(0, 1, 2), unpack=True)

# Nonlinear curve fit
ftom = 0 # use 0 to start from the first element
to = -3 # use -1 to finish on last the last element
initial_guess = [5.5 , 0.025, 5.5 , 0.025] # guess for I_0A, T_iA, I_0B and T_iB, correspondently 
fit_vals, covar = curve_fit(f_T1, TE[ftom:to], I[ftom:to],
    absolute_sigma=True, sigma=I_err[ftom:to], maxfev=10000, p0=initial_guess)
fit_err = np.sqrt(np.diag(covar))

I_0A = fit_vals[0]
I_0A_err = fit_err[0]
TA = fit_vals[1]
TA_err = fit_err[1]
I_0B = fit_vals[2]
I_0B_err = fit_err[2]
TB = fit_vals[3]
TB_err = fit_err[3]

# Print results in console and file
file = open(path.join(outpath, currentfile+".txt"), "w")
print('# I_0A, I_0A_err, TA, TA_err, I_0B, I_0B_err, TB, TB_err', file=file)
print('# I_0A, I_0A_err, TA, TA_err, I_0B, I_0B_err, TB, TB_err')
print(I_0A, I_0A_err, TA, TA_err, I_0B, I_0B_err, TB, TB_err, file=file)
print(I_0A, I_0A_err, TA, TA_err, I_0B, I_0B_err, TB, TB_err)
file.close()

# Draw plot
# Simulate curve
drawpoits_num = 1000
TE_fit = np.linspace(TE[0], TE[-1], num=drawpoits_num)
I_fit = np.zeros(drawpoits_num)
for i in range(0, drawpoits_num):
    I_fit[i] = f_T1(TE_fit[i], I_0A, TA, I_0B, TB)

fig, ax = plt.subplots()
ax.errorbar(TE, I, yerr=I_err, fmt='o',
            marker='o', capsize=5, capthick=1, ecolor='black', color='r',
            label=(r'Raw'))
ax.plot(TE_fit, I_fit, label=(r'$I_0A =$' + str("{0:.2e}".format(I_0A)) +  r' $\pm$ ' + str("{0:.2e}".format(I_0A_err)) + 
                                r' $TA =$' + str("{0:.2e}".format(TA)) +  r' $\pm$ ' + str("{0:.2e}".format(TA_err)) + 
                                r'$I_0B =$' + str("{0:.2e}".format(I_0B)) +  r' $\pm$ ' + str("{0:.2e}".format(I_0B_err)) + 
                                r' $TB =$' + str("{0:.2e}".format(TB)) +  r' $\pm$ ' + str("{0:.2e}".format(TB_err)),  ls='--'))
plt.grid()
plt.ylabel(r'$I(t)$')
plt.xlabel(r'$t$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.title(r'$I(t) = I_0^A (1 - 2 \exp (\frac{-t }{T_1^A})) + I_0^B (1 - 2 \exp (\frac{-t }{T_1^B}))$')
#plt.title(r'$I(t) = I_0^B \exp (\frac{-t }{T_2^B}) + I_0^B \exp (\frac{-t }{T_2^B})$')
ax.legend(loc='best', frameon=True)
plt.draw()
fig.savefig(path.join(outpath, currentfile+".png"))