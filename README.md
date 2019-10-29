# Different tools for financial research

## famamacbeth
The code implements FamaMacbeth regression as in Fama, F. & Macbeth, J.D. (2003) "Risk, Return and Equilibrium: Empirical Tests"
Journal of Political Economy Vol 81, No 3, pp. 607-636

Standard errors were tested against Petersen's simulated data for Standard Error testing: https://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.htm

For more information on dataset and standard errors calculation read: Petersen M. A. (2009) "Estimating Standard Errors in Finance Panel Data Sets: Comparing Approaches", The Review of Financial Studies, Vol 22, Issue 1, pp 435-480

Example of code usage
```
from famamacbeth import FamaMacBeth as fm

model = fm(Y, (X1, X2, X3), add_constant = True, yLabel = 'Name of Y',\
           xLabel = ['X1 label','X2 label', 'X3 label'])
model.fit(NW = True)
model.summary()
```

