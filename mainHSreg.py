'''

-- coding: utf-8 --

File : mainHSreg.py

Author : Shiqi Ye.

Email : 1353596298@qq.com

Time : 2022/6/1 21:18

'''

import numpy as np
import pandas as pd
from funcs.horseshoe import HorseShoe, HorseShoeGamma
from funcs.tvp_byquantile import GenX

# Data.

dfy = pd.read_csv('data\\yraw.csv', index_col=0)
dfx = pd.read_csv('data\\xraw.csv', index_col=0)
# dfx = dfx.iloc[:,1:]

# Vnames.
vNames = dfx.columns.tolist()
y = np.ascontiguousarray(dfy)
x = np.ascontiguousarray(dfx)

y = y[158:]
x = x[158:,:3]
T,p = x.shape

X = GenX(x)

mcmc = 20000
burnin = 50000
thin = 1
type = 1

betaout, lamout, tauout, sigmaSqout = HorseShoeGamma(y.reshape(T), X, burnin, mcmc, thin, type)

pMean=np.mean(betaout,axis = 0)
pMedian=np.median(betaout,axis = 0)
pSigma=np.mean(sigmaSqout)
pLambda=np.mean(lamout,axis = 0)

