# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:15:03 2019

@author: Asus
"""
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
df=pd.read_csv('smport.csv', index_col='Date', parse_dates=True)
print(df.head(10))
df.info()
print(df.describe())

from pypfopt  import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
# Calculating expected returns mu 
mu = expected_returns.mean_historical_return(df)

# Calculating the covariance matrix S
Sigma = risk_models.sample_cov(df)

# Obtaining the efficient frontier
ef = EfficientFrontier(mu, Sigma)
print (mu, Sigma)
returns=df.pct_change()
covMatrix = returns.cov()*251
print(covMatrix)
# Getting the minimum risk portfolio for a target return 
weights = ef.efficient_return(0.2)
print (weights)
l=list(df.columns)
print(l)
size=list(weights.values())
print(size)
print(type(size))
plt.pie(size,labels=l,autopct='%1.1f%%')
plt.title('Return=20%')
plt.show()
# Showing portfolio performance 
ef.portfolio_performance(verbose=True)

# Calculating weights for the maximum Sharpe ratio portfolio
raw_weights_maxsharpe = ef.max_sharpe()
cleaned_weights_maxsharpe = ef.clean_weights()
print (raw_weights_maxsharpe, cleaned_weights_maxsharpe)
ef.portfolio_performance(verbose=True)
size=list(cleaned_weights_maxsharpe.values())
print(size)
plt.pie(size,labels=l,autopct='%1.1f%%')
plt.title('Max Return')
plt.show()
# Calculating weights for the minimum volatility portfolio
raw_weights_minvol = ef.min_volatility()
cleaned_weights_minvol = ef.clean_weights()

# Showing portfolio performance
print(cleaned_weights_minvol)
ef.portfolio_performance(verbose=True)
size=list(cleaned_weights_minvol.values())
print(size)
plt.pie(size,labels=l,autopct='%1.1f%%')
plt.title('Min Risk')
plt.show()

#Calculating an exponentially weighted portfolio
Sigma_ew = risk_models.exp_cov(df, span=180, frequency=252)
mu_ew = expected_returns.ema_historical_return(df, frequency=252, span=180)
# Calculate the efficient frontier
ef_ew = EfficientFrontier(mu_ew, Sigma_ew)
# Calculate weights for the maximum sharpe ratio optimization
raw_weights_maxsharpe_ew = ef_ew.max_sharpe()
# Show portfolio performance 
ef_ew.portfolio_performance(verbose=True)
size=list(raw_weights_maxsharpe_ew.values())
print(size)
plt.pie(size,labels=l,autopct='%1.1f%%')
plt.title('Max Return EW')
plt.show()



