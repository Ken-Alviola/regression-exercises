import pandas as pd
from sklearn.metrics import mean_squared_error

def plot_residuals(y, yhat, baseline_residuals):
    plt.scatter(y, yhat)
    plt.axhline(y = 0, ls = ':')
    plt.title('OLS model residuals')
    plt.show()
    
    plt.scatter(y, baseline_residuals)
    plt.axhline(y = 0, ls = ':')
    plt.title('Baseline Residuals')
    plt.show()

def regression_errors(y, yhat, df):
    MSE = mean_squared_error(y,yhat)
    SSE = (mean_squared_error(y, yhat))*len(y)
    
    df['baseline'] = y.mean()
    df['baseline_residual'] = y - df.baseline
    df['baseline_residual^2'] = df.baseline_residual**2
    
    TSS = df['baseline_residual^2'].sum()
    
    ESS = TSS - SSE
    RMSE = mean_squared_error(y,yhat, squared=False)
    print (f'sum of squared errors (SSE): {SSE}')
    print (f'explained sum of squares (ESS): {ESS}')
    print (f'total sum of squares (TSS): {TSS}')
    print (f'mean squared error (MSE): {MSE}')
    print (f'root mean squared error (RMSE): {RMSE}')

def baseline_mean_errors(y,df):
    df['baseline'] = y.mean()
    df['baseline_residual'] = y - df.baseline
    df['baseline_residual^2'] = df.baseline_residual**2
    
    SSE = df['baseline_residual^2'].sum()
    MSE = mean_squared_error(y, df.baseline)
    RMSE = mean_squared_error(y, df.baseline, squared=False)
    
    print (f'sum of squared errors (SSE): {SSE}')
    print (f'mean squared error (MSE): {MSE}')
    print (f'root mean squared error (RMSE): {RMSE}')

def better_than_baseline(y, yhat, df): 
    SSE = (mean_squared_error(y, yhat))*len(y)
    
    df['baseline'] = y.mean()
    df['baseline_residual'] = y - df.baseline
    df['baseline_residual^2'] = df.baseline_residual**2
    
    SSE_baseline = df['baseline_residual^2'].sum()
    
    if SSE < SSE_baseline:
        print (f'Model SSE {SSE} is < Baseline SSE {SSE_baseline} so the model is better than baseline')
    else:
        print ('Model SSE is worse than baseline')