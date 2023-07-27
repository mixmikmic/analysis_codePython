import quantiacsToolbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
get_ipython().magic('matplotlib inline')

get_ipython().run_cell_magic('html', '', '<style>\ntable {float:left}\n</style>')

F_AD = pd.read_csv('./tickerData/F_AD.txt')
CLOSE = np.array(F_AD.loc[:252-1, [' CLOSE']])
plt.plot(CLOSE)

momentum = (CLOSE[20:] - CLOSE[:-20]) / CLOSE[:-20]
plt.plot(momentum)

X = np.concatenate([momentum[i:i+5] for i in range(252-20-5)], axis=1).T
y = np.sign((CLOSE[20+5:] - CLOSE[20+5-1: -1]).T[0])

clf = svm.SVC()
clf.fit(X, y)
clf.predict(momentum[-5:].T)

F_AD.loc[251:252, ['DATE', ' CLOSE']]

class myStrategy(object):
    
    def myTradingSystem(self, DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):

        def predict(momentum, CLOSE, lookback, gap, dimension):
            X = np.concatenate([momentum[i:i + dimension] for i in range(lookback - gap - dimension)], axis=1).T
            y = np.sign((CLOSE[dimension+gap:] - CLOSE[dimension+gap-1:-1]).T[0])
            y[y==0] = 1

            clf = svm.SVC()
            clf.fit(X, y)

            return clf.predict(momentum[-dimension:].T)

        nMarkets = len(settings['markets'])
        lookback = settings['lookback']
        dimension = settings['dimension']
        gap = settings['gap']

        pos = np.zeros((1, nMarkets), dtype=np.float)

        momentum = (CLOSE[gap:, :] - CLOSE[:-gap, :]) / CLOSE[:-gap, :]

        for market in range(nMarkets):
            try:
                pos[0, market] = predict(momentum[:, market].reshape(-1, 1),
                                         CLOSE[:, market].reshape(-1, 1),
                                         lookback,
                                         gap,
                                         dimension)
            except ValueError:
                pos[0, market] = .0
        return pos, settings


    def mySettings(self):
        """ Define your trading system settings here """

        settings = {}

        # Futures Contracts
        settings['markets'] = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
                               'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC',
                               'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
                               'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
                               'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX',
                               'F_YM']

        settings['lookback'] = 252
        settings['budget'] = 10 ** 6
        settings['slippage'] = 0.05

        settings['gap'] = 20
        settings['dimension'] = 5

        return settings

result = quantiacsToolbox.runts(myStrategy)

