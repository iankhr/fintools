# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:04:29 2019

@author: Ian Khrashchevskyi

The code bellow performs Fama-MacBeth regressions as described in 

Fama, F. & Macbeth, J.D. (2003) "Risk, Return and Equilibrium: Empirical Tests"
    Journal of Political Economy Vol 81, No 3, pp. 607-636
    

"""
import numpy as np
import pandas as pd

class FamaMacBeth(object):
    def __init__(self, Y, X, add_constant = True, yLabel = None, xLabel = None):
        
        # check the input data
        self._checkInputData(Y, X, add_constant)
        
        # start dy identifying amount of stocks
        N = len(self._Y.T)
        T = len(self._Y)
        
        # write everything down in self
        self._N = N
        self._T = T
        
        # set empty output
        self.output = None
        
        # write down name of Y and X
        self.yLabel = yLabel
        self.xLabel = xLabel
        
    
    
    def fit(self, lags = None, NW = False):
        """
        The function performs Fama-MacBeth regression as in the paper and then
        presents the output.
        """ 
        try:              
            if lags<=0:
                raise TypeError('Lags must be a strictly positive number!')
        except:
            if lags is not None:
                raise TypeError('Lags must by numeric!')
        
        y = self._Y
        x = self._X
        T = self._T
        # run t-cross sectional regressions  and get betas
        if type(x) != tuple:           
            betas = [self._quickOLS(y.iloc[i,:].to_frame(), x.iloc[i,:].to_frame(),\
                       add_constant = True) for i in range(T)]
        else:
            betas = [self._quickOLS(y.iloc[i,:].to_frame(), pd.concat((\
                                    [x[j].iloc[i,:].to_frame() for j in range(len(x))]),\
                                     axis=1), add_constant = True)\
                                     for i in range(T)]
        # average betas over time
        avbeta = pd.concat(betas, axis=1)
        beta = avbeta.mean(axis=1)
        self.beta = beta
        """
        # calculate errors
        errors = [_quickOLSerrors(y.iloc[i,:].to_frame(), x.iloc[i,:].to_frame(),\
                           betas[i], add_constant = True) for i in range(len(y))]
        averrors = pd.concat(errors, axis=1)
        averror = averrors.mean(axis=1)
        """
        # to get errors
        parerror = avbeta.subtract(beta, axis=0)
        sterrs = parerror@parerror.T
        sterrs = sterrs/len(parerror.T)
        self.sterrs = pd.DataFrame(np.sqrt(np.diag(sterrs/(len(parerror.T)-1))),\
                                   columns = ['se'])
        self._parerror = parerror
        
        # get Newey-West errors
        if lags is None:
            lags = len(parerror.T)-1
           
        testNW = self._NeweyWest(parerror.T.values, lags = lags)
        self.NWerrs = pd.DataFrame(np.sqrt(np.diag(testNW/(len(parerror.T)-1))),\
                                   columns = ['se'])
        
        # create output
        if NW == False:
            self.output = pd.concat((beta, self.sterrs), axis=1)
        else:
            self.output = pd.concat((beta, self.NWerrs), axis=1)
            
        self.output.columns = ['estimate','se']
        self.output['t-stat'] = self.output['estimate'].divide(self.output['se'])
        
        

    def _checkInputData(self, Y, X, add_constant):
        """
        This function checks the input data and verifies that the data types
        are correct. If they are not, then it rises Type Error.
        """
        if type(Y) == pd.Series:
            Y = Y.to_frame()
        elif type(Y) == np.array:
            Y = pd.DataFrame(Y)
        elif type(Y) != pd.DataFrame:
            raise TypeError('Dependent variable must be DataFrame!')
        
        if type(X) == pd.Series:
            X = X.to_frame()
        elif type(X) == np.array:
            X = pd.DataFrame(X)
        elif (type(X) != pd.DataFrame) & (type(X) != tuple):
            raise TypeError('Independent variable must be DataFrame!')
        
        if type(add_constant) != bool:
            raise TypeError('Only True or False are allowed in add_constant')
                       
        self._Y = Y
        self._X = X
        self._const = add_constant           
            
     
    def _quickOLS(self, Y, X, add_constant = True):
        """
        This function performs OLS regression and spits out betas
        """
        # first dropna in X and Y
        X = X.dropna()
        Y = Y.dropna()
        commonCols = np.intersect1d(X.columns, Y.columns)
        commonTimes = np.intersect1d(X.index, Y.index)
        X = X.loc[commonTimes, commonCols]
        Y = Y.loc[commonTimes, commonCols]
        X = pd.concat((pd.DataFrame(1, index = X.index,\
                columns = ['Const']), X), axis=1, join='inner')
        if (len(X)>0) & (len(Y)>0):
            return pd.DataFrame(np.linalg.inv(X.T@X)@X.T@Y)

    
    def _quickOLSerrors(self, Y, X, beta, add_constant = True):
        """
        This function gets OLS errors
        """
        X = pd.concat((pd.DataFrame(1, index = X.index,\
                columns = ['Const']), X), axis=1, join='inner')
        return pd.DataFrame(Y.values - X.values@beta.values, index = Y.index, columns = Y.columns)
    
    
    def _NeweyWest(self, data, lags = 1):
        """
        requires demeaned data and was copied from Kevin Sheppard's code
        """
        T = np.shape(data)[0]
        dm = np.matrix(data)
        pdData = pd.DataFrame(data)
        w = (lags+1-np.arange(1,lags+1))/(lags+1)
        V = dm.T@dm/T
        for i in range(lags):
            lag = pdData.shift(i+1).dropna(how='all')
            tempData = pdData.loc[lag.index,:]
            gamma = np.asmatrix(tempData).T@np.asmatrix(lag)/T
            GplusG = gamma+gamma.T
            V = V+w[i]*GplusG
        return V