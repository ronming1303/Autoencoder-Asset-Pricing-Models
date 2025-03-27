# used to calculate two dates difference
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
import pandas as pd
from .modelBase import modelBase
from utils import CHARAS_LIST
import sys
sys.path.append('../')


class FF(modelBase):
    def __init__(self, K):  # K is the number of factors
        super(FF, self).__init__(f'FF_{K}')
        self.K = K
        # ff5 data from FF website is only available from 196307
        self.train_period[0] = 19630731
        self.omit_char = []
        self.__prepare_FFf()  # This is a private method just for FF model, not for other models

    def __prepare_FFf(self):
        ff5 = pd.read_csv('data/ff5.csv', index_col=0)
        UMD = pd.read_csv('data/UMD.csv', index_col=0)
        UMD.columns = ['UMD']
        FFf = pd.concat([ff5, UMD.loc[196307:]], axis=1)
        self.fname = ['Mkt-RF', 'SMB', 'HML', 'CMA', 'RMW', 'UMD']
        self.FFf = FFf[self.fname]  # save factor return
        self.portfolio_ret = pd.read_pickle(
            'data/portfolio_ret.pkl')  # save portfolio return
        # change date from year-month-date to year-month, because FFf is in year-month format
        self.portfolio_ret['DATE'] = self.portfolio_ret['DATE'].apply(
            lambda x: x//100)

    def train_model(self):
        self.beta_matrix = []
        X = self.FFf[self.fname[:self.K]
                     ].loc[self.train_period[0]//100:self.train_period[1]//100]
        for col in CHARAS_LIST:  # 对n个portfolio分别回归
            y = self.portfolio_ret.set_index(
                'DATE')[col].loc[self.train_period[0]//100:self.train_period[1]//100]
            model = sm.OLS(y.values, X.values).fit()
            # save beta for each portfolio
            self.beta_matrix.append(model.params)
        self.beta_matrix = pd.DataFrame(
            self.beta_matrix, columns=self.fname[:self.K], index=CHARAS_LIST)

    def calBeta(self, month):  # beta is time invariant
        return self.beta_matrix  # N * K

    def calFactor(self, month):
        return self.FFf[self.fname[:self.K]].loc[month//100]  # K * 1

    def cal_delayed_Factor(self, month):
        last_mon = int(str(pd.to_datetime(
            str(month)) - relativedelta(months=1)).split(' ')[0].replace('-', '')[:-2])
        # return average of prevailing sample hat{f} (from 198701) up to t-1
        return self.FFf[self.fname[:self.K]].loc[198701:last_mon].mean()
