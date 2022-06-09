# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd
from funcs.ssvs_byquantile import SSVS_BQuant
from funcs.tvp_byquantile import TVP_BQuant
from matplotlib import pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg

# Data.

dfy = pd.read_csv('data\\yraw.csv', index_col=0)
dfx = pd.read_csv('data\\xraw.csv', index_col=0)
# dfx = dfx.iloc[:,1:]

# Vnames.
vNames = dfx.columns.tolist()
y = np.ascontiguousarray(dfy)
x = np.ascontiguousarray(dfx)
T,p = x.shape

# Normalize
# y = y - np.mean(y)
# x = (x - np.mean(x, axis = 0))/np.std(x, axis = 0)


nsave = 1000
quant = np.arange(0.05, 1, 0.1)
labels = []
for i in range(0,len(quant)):
    if np.mod(i,3) == 0:
        # labels.append(str(int(quant[i]*100)))
        labels.append(str(round(quant[i],2)))
    else:
        labels.append(' ')

# Quantile regression.
reg_mean = np.zeros((len(quant), p))
reg_lower = np.zeros((len(quant), p))
reg_upper = np.zeros((len(quant), p))
for q in range(len(quant)):
    mod = QuantReg(dfy, dfx)
    res = mod.fit(q=quant[q])
    pars = res.params
    sublower = res.conf_int()[0]
    subupper = res.conf_int()[1]
    reg_mean[q,:] = pars
    reg_lower[q,:] = sublower
    reg_upper[q,:] = subupper

# Estimation.

beta_draws, invdelta2_draws, gamma_draws, pi0_draws,  z_draws = SSVS_BQuant(x, y, nsave, quant)

# Graphs.

nsave, nquant, nexo = beta_draws.shape

def Draw_iteration(my, linewidth,filename):
    example = my
    fig = plt.figure(1, figsize=(20, 15))
    plt.rcParams['font.size'] = '25'
    plt.rcParams["font.family"] = "Times New Roman"
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(example,  linewidth = linewidth, color = 'blue')
    # ax.set_title(r'$\beta_{1, 0.05}$')
    plt.xlabel('Iteration', fontsize = 45)
    plt.savefig('graphs\\'+filename+'.pdf', bbox_inches = 'tight')
    plt.close()

# First, plot the sampling procedures.
# Example 1, beta_{0,0.05}
Draw_iteration(beta_draws[:,0,0], linewidth = 2, filename ='beta_0_0.05')

# Example 1.1, beta_{1,0.05}
Draw_iteration(beta_draws[:,0,1], linewidth = 2, filename ='beta_1_0.05')

# Example 2, invdelta2_{0,0.05}
Draw_iteration(invdelta2_draws[:,0,0],linewidth = 2, filename ='invdelta2_0_0.05')

# Example 3, Gamma_{0,0.05}
Draw_iteration(gamma_draws[:,0,0], linewidth = 0.5, filename ='gamma_0_0.05')


# Example 4, pi_{0.05}
Draw_iteration(pi0_draws[:,0], linewidth = 2, filename ='pi_0_0.05')

# Example 5, zt_{0,0.05}
Draw_iteration(z_draws[:,0,0], linewidth = 2, filename = 'z_0_0.05')



# Then, plot the results.

def DrawMedian_Confidence_exo(median, lower, upper, quant, labels, filename):
    fig = plt.figure(1, figsize=(25, 20))
    plt.rcParams['font.size'] = '15'
    plt.rcParams["font.family"] = "Times New Roman"
    nq, p = median.shape

    for i in range(p):
        sub_median = median[:, i]
        sub_lower= lower[:, i]
        sub_upper = upper[:, i]
        subpos = sub_upper-sub_median
        subneg = sub_median-sub_lower

        ax = fig.add_subplot(5, 4, i + 1)
        plt.errorbar(x=quant, y=sub_median, yerr=[subneg ,subpos], fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)
        ax.set_title(vNames[i], fontsize=25, y=1.02)
        ax.set_xticks(quant)
        ax.set_xticklabels(labels)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.22, hspace=0.45)
    plt.savefig('graphs\\'+filename+'.pdf', bbox_inches = 'tight')
    plt.close()

# 1. beta.
median = np.median(beta_draws,axis = 0)
lower = np.quantile(beta_draws,q=0.025, axis = 0)
upper = np.quantile(beta_draws,q=0.975, axis = 0)
DrawMedian_Confidence_exo(median, lower, upper, quant, labels, filename = 'betaquant')

# 2. invdelta^2.
delta2 = 1/invdelta2_draws
median = np.median(delta2,axis = 0)
lower = np.quantile(delta2,q=0.025, axis = 0)
upper = np.quantile(delta2,q=0.975, axis = 0)
DrawMedian_Confidence_exo(median, lower, upper, quant, labels, filename = 'deltaquant')

# 3. Comparison.
mean = np.mean(beta_draws,axis = 0)
lower = np.quantile(beta_draws,q=0.025, axis = 0)
upper = np.quantile(beta_draws,q=0.975, axis = 0)


fig = plt.figure(1, figsize=(25, 20))
plt.rcParams['font.size'] = '15'
plt.rcParams["font.family"] = "Times New Roman"
nq, p = mean.shape

for i in range(p):
    sub_mean = mean[:, i]
    sub_lower = lower[:, i]
    sub_upper = upper[:, i]
    subpos = sub_upper - sub_mean
    subneg = sub_mean- sub_lower

    sub_regmean = reg_mean[:,i]
    sub_regupper = reg_upper[:,i]
    sub_reglower = reg_lower[:,i]
    subpos2 = sub_regupper - sub_regmean
    subneg2 = sub_regmean - sub_reglower

    ax = fig.add_subplot(5, 4, i + 1)
    plt.errorbar(x=quant-0.015, y=sub_mean, yerr=[subneg, subpos], fmt='o', ecolor='r', color='b', elinewidth=2, capsize=3)
    plt.errorbar(x=quant +0.015, y=sub_regmean, yerr=[subneg2, subpos2], fmt='o', ecolor='g', color='orange', elinewidth=2,
                 capsize=3)
    ax.set_title(vNames[i], fontsize=25, y=1.02)
    ax.set_xticks(quant)
    ax.set_xticklabels(labels)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.22, hspace=0.45)
plt.savefig('graphs\\comparison.pdf', bbox_inches = 'tight')
plt.close()


beta_draws2, lam2_draws, xi_draws, psi2_draws, zeta_draws, sigma2_draws, z_draws = TVP_BQuant(x, y.reshape(T), nsave, quant, isFast = 1)


# 4. Comparison.
mean2 = np.mean(beta_draws2,axis = 0)
lower2 = np.quantile(beta_draws2,q=0.025, axis = 0)
upper2 = np.quantile(beta_draws2,q=0.975, axis = 0)


fig = plt.figure(1, figsize=(25, 20))
plt.rcParams['font.size'] = '15'
plt.rcParams["font.family"] = "Times New Roman"
nq, p = mean.shape

for i in range(p):
    sub_mean3 = mean2[:, i]
    sub_lower3 = lower2[:, i]
    sub_upper3 = upper2[:, i]
    subpos3 = sub_upper3 - sub_mean3
    subneg3 = sub_mean3 - sub_lower3

    sub_mean = mean[:, i]
    sub_lower = lower[:, i]
    sub_upper = upper[:, i]
    subpos = sub_upper - sub_mean
    subneg = sub_mean- sub_lower

    sub_regmean = reg_mean[:,i]
    sub_regupper = reg_upper[:,i]
    sub_reglower = reg_lower[:,i]
    subpos2 = sub_regupper - sub_regmean
    subneg2 = sub_regmean - sub_reglower

    ax = fig.add_subplot(5, 4, i + 1)
    plt.errorbar(x=quant-0.02, y=sub_mean, yerr=[subneg, subpos], fmt='o', ecolor='r', color='b', elinewidth=2, capsize=3)
    plt.errorbar(x=quant, y=sub_regmean, yerr=[subneg2, subpos2], fmt='o', ecolor='g', color='orange',
                 elinewidth=2,
                 capsize=3)
    plt.errorbar(x=quant +0.02, y=sub_mean3, yerr=[subneg3, subpos3], fmt='o', ecolor='y', color='m', elinewidth=2,
                 capsize=3)
    ax.set_title(vNames[i], fontsize=25, y=1.02)
    ax.set_xticks(quant)
    ax.set_xticklabels(labels)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.18, hspace=0.45)
plt.savefig('graphs\\comparisonHS.pdf', bbox_inches = 'tight')
plt.close()
