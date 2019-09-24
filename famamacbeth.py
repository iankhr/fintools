# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:04:29 2019

@author: Ian Khrashchevskyi

The code bellow performs Fama-MacBeth regressions as described in 

Fama, F. & Macbeth, J.D. (2003) "Risk, Return and Equilibrium: Empirical Tests"
    Journal of Political Economy Vol 81, No 3, pp. 607-636
    
Standard errors were checked against Petersen's simulated data for Standard Error
testing:
    https://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.htm

Petersen M. A. (2009) "Estimating Standard Errors in Finance Panel Data Sets:
    Comparing Approaches", The Review of Financial Studies, Vol 22, Issue 1, 
    pp 435-480
"""
import numpy as np
import pandas as pd
import shutil

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
        if yLabel is None:
            self.yLabel = 'y'
        else:
            self.yLabel = yLabel
        
        if xLabel is None:
            self.xLabel = xLabel
        else:
            # check whether it matches the length of X
            if len(xLabel) == len(X):
                self.xLabel = xLabel
            else:
                raise ValueError("Length of X labels does not match length of X")
        
    
    
    def fit(self, lags = None, NW = False):
        """
        The function performs Fama-MacBeth regression as in the paper and then
        presents the output.
        """
        self._NW = NW
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
                       add_constant = self._const) for i in range(T)]
        else:
            betas = [self._quickOLS(y.iloc[i,:].to_frame(), pd.concat((\
                                    [x[j].iloc[i,:].to_frame() for j in range(len(x))]),\
                                     axis=1), add_constant = self._const)\
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
        # estimate R2
        self._getRsquared()
        
        
        

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

    
    def summary(self):
        """
        Prints nice output of the regression
        """
        if self.output is None:
            raise ValueError("You need to fit the model first!")
        
        columns = shutil.get_terminal_size().columns
        title = "Fama-MacBeth estimation results"
        print(title.center(columns))
        print('='*columns)
        tab = 4
        smallCol = columns/2-tab
        
        # getting and printing small stats
        sts = self._smallStats()
        numItems = int(np.round(len(sts)/2))
        for i in range(numItems):
            item1 = sts[i]
            item2 = sts[i+numItems]
            print(self._cellStr(item1[0], item1[1], smallCol) + tab*' '
                  + self._cellStr(item2[0], item2[1], smallCol))
        
        print('='*columns)
        ocl = (columns)/4-tab
        # print output
        if self.xLabel is None:
            # use simply the values of the  output
            xLabels = self.output.index.tolist()
            if self._const:
                xLabels[0] = 'Constant'
        else:
            xLabels = self.xLabel
            if self._const:
                xLabels = ['Constant',]+xLabels
        self._tableOutput(self.output.values, xLabels, 
                               list(np.ones(len(xLabels))), tab, ocl)
        print('='*columns)
    
    
    def _cellStr(self, cellName, cellContent, length):
        resLen = int(length - len(cellName) - len(cellContent))
        if cellName !='':
            cellName = cellName+':'     
            if resLen<0:
                return cellName+' '+cellContent
            else:
                return cellName+' '*resLen + cellContent
        
        else:
            return ' '*int(length)+' '
    
    
    def _tableOutput(self, output, rowNames, reps, tab, ocl):
        columns = shutil.get_terminal_size().columns
        poq = np.cumsum(reps)
        pointer = 0
        counter = 0
        print(int(ocl)*' '+tab*' '
              + ' '*int(ocl-len('Estimate'))+'Estimate' +tab*' '
              + ' '*int(ocl-len('Std. Error'))+'Std. Error'+tab*' '
              + ' '*int(ocl-len('t-stat'))+'t-stat'
              )
        print('-'*columns)
        # remove names with zero reps       
        # build the table
        if np.shape(output)[1]>1:
            for i in range(len(output)):
                item = np.round(output[i], decimals = 3)
                # creating name
                if i>= poq[pointer]: 
                    pointer = pointer+1
                    while reps[pointer] == 0:
                        pointer = pointer+1
                    
                    if reps[pointer]>1:
                        counter = counter+1
                    else:
                        counter = 0
                
                elif counter >0:
                    counter = counter+1
                
                if counter == 0:
                    rowName = rowNames[pointer]
                else:
                    rowName = rowNames[pointer]+'['+str(counter)+']'
                
                tabLenName = ' '*int(ocl-len(str(rowName)))
                # putting the values
                est = str(item[0])
                se = str(item[1])
                tstat = str(item[2])
                print(str(rowName)+tabLenName+tab*' '
                      +' '*int(ocl-len(est)) + est+ tab*' '
                      +' '*int(ocl-len(se)) + se+tab*' '
                      +' '*int(ocl-len(tstat)) + tstat)
        else:
            tabLenName = ' '*int(ocl-len(str(rowName)))
            # putting the values
            est = str(output[0])
            se = str(output[1])
            tstat = str(output[2])
            print(str(rowName)+tabLenName+tab*' '
                  +' '*int(ocl-len(est)) + est+ tab*' '
                  +' '*int(ocl-len(se)) + se+tab*' '
                  +' '*int(ocl-len(tstat)) + tstat)
    
    
    def _smallStats(self):
        sts = []
        if self._NW:
            errors = 'Newey-West'
        else:
            errors = 'White'
        
        sts.append(['Dep Variable', str(self.yLabel)])
        sts.append(['Errors', errors])
        now = pd.to_datetime('today')
        sts.append(['Date', now.strftime("%a, %b %d %Y")])
        sts.append(['Time', now.strftime("%H:%M:%S")])     
        sts.append(['',''])
        sts.append(['Num obs T', str(self._T)])
        sts.append(['Avg Obs T',str(np.round(self._Y.mean(axis=1).mean(),2))])
        sts.append(['Min Obs T',str(np.round(self._Y.mean(axis=1).min(),2))])
        sts.append(['Max Obs T',str(np.round(self._Y.mean(axis=1).max(),2))])
        sts.append(['',''])
        sts.append(['Num obs N', str(self._N)])
        sts.append(['Avg Obs N',str(np.round(self._Y.mean(axis=0).mean(),2))])
        sts.append(['Min Obs N',str(np.round(self._Y.mean(axis=0).min(),2))])
        sts.append(['Max Obs N',str(np.round(self._Y.mean(axis=0).max(),2))])
        sts.append(['',''])
        sts.append(['Overall R-squared:',str(np.round(self.rsquared,2))])
        sts.append(['Between R-squared:',str(np.round(self.rsquaredbe,2))])
        sts.append(['Within R-squared:',str(np.round(self.rsquaredwithin,2))])
        sts.append(['F-stat','-'])
        sts.append(['p-value','-'])
        
        return sts   
        
     
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
    
       
    def _quickOLSerrors(self, Y, X, beta, add_constant = True, errors = True,\
                        crossSection = True):
        """
        This function gets OLS errors
        """
        # first dropna in X and Y
        X = X.dropna()
        Y = Y.dropna()

        if crossSection:
            commonCols = np.intersect1d(X.columns, Y.columns)
            
        commonTimes = np.intersect1d(X.index, Y.index)
        if crossSection:
            X = X.loc[commonTimes, commonCols]
            Y = Y.loc[commonTimes, commonCols]
        else:
            X = X.loc[commonTimes,:]
            Y = Y.loc[commonTimes,:]
          
        # add constant if needed
        if add_constant:
            X = pd.concat((pd.DataFrame(1, index = X.index,\
                    columns = ['Const']), X), axis=1, join='inner')

        if errors:
            return pd.DataFrame(Y.values - X.values@beta.values, index = Y.index,\
                            columns = Y.columns)
        else:
            return pd.DataFrame(X.values@beta.values, index = Y.index,\
                            columns = Y.columns)
    
    
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
    
    def _getRsquared(self):
        """
        The method of estimating of R-sqaures is taken from Stata
        """
        Y = self._Y
        X = self._X
        T = self._T
        """
        Calculating overall R-squared simply as a squared correlation
        """
        # get forecasts for general regression
        if type(X) != tuple:
            pred1 = [self._quickOLSerrors(Y.iloc[i,:].to_frame(),\
                                        X.iloc[i,:].to_frame(),\
                                        add_constant = self._const, errors = False)\
                                        for i in range(T)]
        else:
            pred1 = [self._quickOLSerrors(Y.iloc[i,:].to_frame(),\
                     pd.concat(([X[j].iloc[i,:].to_frame()\
                     for j in range(len(X))]), axis=1),\
                     self.beta, errors = False, add_constant = self._const)\
                     for i in range(T)]
        pred1 = pd.concat(pred1, axis=1, sort = True).T
        self._pred1 = pred1
        
        # compare with Yb
        yvector = self._Y.loc[pred1.index, pred1.columns]
        yvector = np.reshape(yvector.values, (yvector.shape[1]*yvector.shape[0]))
        predvector = np.reshape(pred1.values, (pred1.shape[1]*pred1.shape[0]))
        # combine in one frame
        vectorData = pd.DataFrame([yvector, predvector],\
                                  columns = range(len(yvector)),\
                                  index =['ys','preds']).T.dropna()
        # estimate r-sqaured as a fraction of variances
        self.rsquared = vectorData.corr().values[1,0]**2
        """
        Calculating between R squared
        """
        yavt = Y.mean(axis=0)
        if type(X) != tuple:
            xavt = X.mean(axis=0)
        else:
            xavt = pd.concat([X[j].mean(axis=0) for j in range(len(X))], axis=1)

        # get forecast for between regression
        pred2 = self._quickOLSerrors(yavt.to_frame(), xavt,\
                                     self.beta, errors = False,\
                                     add_constant = self._const,\
                                     crossSection=False)
        self._pred2 = pred2
        vectorData = pd.concat((yavt, pred2), axis=1, join='inner').dropna()
        self.rsquaredbe = vectorData.corr().values[1,0]**2
        
        """
        Calculating within R squared
        """
        cmnCols = np.intersect1d(pred1.columns, pred2.index)
        pred3 = pred1.loc[:, cmnCols] \
              - np.ones(pred1.loc[:, cmnCols].shape)\
              *pred2.loc[cmnCols].T.values
        self._pred3 = pred3
        # get vector of Y
        yvector = self._Y.loc[pred3.index, pred3.columns]
        yvector = np.reshape(yvector.values,\
                             (yvector.shape[1]*yvector.shape[0]))
        # get vector of predictions
        predvector = np.reshape(pred3.values, (pred3.shape[1]*pred3.shape[0]))
        # combine in one frame
        vectorData = pd.DataFrame([yvector, predvector],\
                                  columns = range(len(yvector)),\
                                  index =['ys','preds']).T.dropna()
        # estimate r-sqaured as a fraction of variances
        self.rsquaredwithin = vectorData.corr().values[1,0]**2