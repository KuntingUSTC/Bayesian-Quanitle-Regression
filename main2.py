'''

-- coding: utf-8 --

File : main2.py

Author : Shiqi Ye.

Email : 1353596298@qq.com

Time : 2022/5/28 22:47

'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from funcs.tvp_byquantile import GenX, TVP_BQuant, FastDraw, GenVbeta0, GenInvSqrtVy, TransTVP, RecoverTVP
from funcs.ssvs_byquantile import SSVS_BQuant


# Data.

dfy = pd.read_csv('data\\yraw.csv',index_col=0)
dfx = pd.read_csv('data\\xraw.csv',index_col=0)
# Start from 1990:Q1, to 2015:Q3.
# Consider time-varying intercept, AR(2) of CPI and
# RG, the Real Government Consumption & Gross Investment, Total.
# dfy = dfy.iloc[-103:,:]
# dfx = dfx.iloc[-103:,[1,2,9]]

dfy = dfy.iloc[:,:]
dfx = dfx.iloc[:,:3]
vsubNames = dfx.columns
vsubDates = dfx.index.tolist()


# Vnames.
vNames = dfx.columns.tolist()
y = np.ascontiguousarray(dfy)
x = np.ascontiguousarray(dfx)
T,N = x.shape

# # Normalize
# y = y - np.mean(y)
# x = (x - np.mean(x, axis = 0))/np.std(x, axis = 0)


# Generate TXTN matrix X.
X = GenX(x)
T,TN = X.shape
print(X.shape)
print(X)



# Preliminary
nsave = 20000
# quant = np.arange(0.05, 1, 0.1)
quant = np.array([0.05, 0.5, 0.95])
n_q = len(quant)
labels = []
for i in range(0,len(quant)):
    if np.mod(i,3) == 0:
        # labels.append(str(int(quant[i]*100)))
        labels.append(str(round(quant[i],2)))
    else:
        labels.append(' ')

# Estimation.
# beta_draws, invdelta2_draws, gamma_draws, pi0_draws,  z_draws = SSVS_BQuant(x, y, nsave, quant)
#
# def DrawMedian_Confidence_exo(median, lower, upper, quant, labels, filename):
#     fig = plt.figure(1, figsize=(20, 20))
#     plt.rcParams['font.size'] = '15'
#     plt.rcParams["font.family"] = "Times New Roman"
#     nq, p = median.shape
#
#     for i in range(p):
#         sub_median = median[:, i]
#         sub_lower= lower[:, i]
#         sub_upper = upper[:, i]
#         subpos = sub_upper-sub_median
#         subneg = sub_median-sub_lower
#
#         ax = fig.add_subplot(2, 1, i + 1)
#         plt.errorbar(x=quant, y=sub_median, yerr=[subneg ,subpos], fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)
#         ax.set_title(vNames[i], fontsize=25, y=1.02)
#         ax.set_xticks(quant)
#         ax.set_xticklabels(labels)
#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.6)
#     plt.show()
#
# # 1. beta.
# median = np.median(beta_draws,axis = 0)
# lower = np.quantile(beta_draws,q=0.025, axis = 0)
# upper = np.quantile(beta_draws,q=0.975, axis = 0)
# DrawMedian_Confidence_exo(median, lower, upper, quant, labels, filename = 'betaquant')
nsave2 = 5000
beta_draws, invdelta2_draws, gamma_draws, pi0_draws,  z_draws = SSVS_BQuant(x, y, nsave2, quant)
mean = np.mean(beta_draws, axis = 0)
std = np.std(beta_draws, axis = 0)
ss

betaD_draws, lam2_draws, xi_draws, psi2_draws, zeta_draws, sigma2_draws, z_draws = TVP_BQuant(X, y.reshape(T), nsave, quant, isFast=1)

# Recover beta_t.

betaD_draws = RecoverTVP(betaD_draws, T, N)

print(betaD_draws.shape)
median = np.median(betaD_draws,axis = 0)
mean = np.mean(betaD_draws, axis = 0)
print(mean.shape)
lower = np.quantile(betaD_draws,q=0.16, axis = 0)
upper = np.quantile(betaD_draws,q=0.84, axis = 0)



def DrawTVPBeta(meantotal, mediantotal, lowertotal, uppertotal, quant):
    fig = plt.figure(1, figsize=(25, 20))
    plt.rcParams['font.size'] = '20'
    plt.rcParams["font.family"] = "Times New Roman"
    n_q, T,N = mediantotal.shape

    plotlable = []
    count = [int(T/5*i) for i in range(1,5)]
    for idx in vsubDates:
        if vsubDates.index(idx) in count:
            plotlable.append(idx)

    for q in range(n_q):
        mean = meantotal[q]
        median = mediantotal[q]
        lower = lowertotal[q]
        upper = uppertotal[q]
        for i in range(N):
            ax = fig.add_subplot(n_q, N, q * N + i + 1)
            xfit = np.arange(0, T, 1)
            ax.plot(xfit, mean[:, i], '-k', linewidth=4)
            ax.plot(xfit, median[:, i], ':k', linewidth=4)
            ax.fill_between(xfit, lower[:, i], upper[:, i],
                             color='gray', alpha=0.2)
            ax.set_xticks(count)
            ax.set_xticklabels(plotlable, rotation = 30)
            if i == 0:
                ax.set_ylabel(r'$\tau = $ '+str(quant[q]), fontsize = 25, labelpad =15)
            if q == 0:
                ax.set_title('TVP of '+str(vsubNames[i]), fontsize=28, y=1.02)
            plt.grid()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.17, hspace=0.28)
    plt.savefig('graphsHS_fast\\TVP.pdf', bbox_inches = 'tight')

DrawTVPBeta(mean, median, lower, upper, quant)

# a test
# y = y.reshape(T)
# np.random.seed(123)
# n_q = 10
# z = np.random.rand(T * n_q).reshape(n_q, T)
# sigma2 = np.random.rand(n_q)
# lam2 = np.random.rand(n_q)
# psi2 = np.random.rand(TN*n_q).reshape(n_q, TN)
# xi = np.random.rand(n_q)
# zeta = np.random.rand(TN*n_q).reshape(n_q, TN)
# theta = np.zeros(n_q)
# tau_sq = np.zeros(n_q)
#
# q = 0
# theta[q] = (1 - 2 * quant[q]) / (quant[q] * (1 - quant[q]))
# tau_sq[q] = 2 / (quant[q] * (1 - quant[q]))
#
# Vbeta0= GenVbeta0(lam2[q], psi2[q])
# InvSqrtVy = GenInvSqrtVy(sigma2[q], tau_sq[q], z[q])
# Phi = InvSqrtVy@X
# alpha = np.multiply(y-theta[q]*z[q], np.diag(InvSqrtVy))
#
# Vyinv = np.diag(1/(sigma2[q]*tau_sq[q]*z[q]))
# posvar = np.linalg.inv(X.T@Vyinv@X+Vbeta0)
# posmean = posvar@X.T@Vyinv@(y-theta[q]*z[q])
#
# total = 200000
# betasave = np.zeros((total, TN))
# betasave2 = np.zeros((total, TN))
# for i in range(total):
#    betasave[i] = FastDraw(Vbeta0, Phi, alpha)
#    betasave2 = np.random.multivariate_normal(mean = posmean, cov = posvar)
#
# print(posmean - np.mean(betasave,axis = 0))
# print(posmean - np.mean(betasave2, axis = 0))