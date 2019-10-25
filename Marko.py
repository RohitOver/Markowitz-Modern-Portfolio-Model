import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
df = web.DataReader(["TSLA", "AAPL", "AMZN"],'yahoo',"2010-1-1","2018-12-31")['Adj Close']
# df1 = web.DataReader('aapl','yahoo',"2014-2-2","2015-2-2")
# df2 = web.DataReader('amzn','yahoo',"2014-2-2","2015-2-2")
# df3 = web.DataReader('ibm','yahoo',"2014-2-2","2015-2-2")
# stocks = pd.concat([df,df1,df2,df3],axis=1)
# stocks.columns = ['a','b','c','d']
# print(stocks.head(5))
# data = df.pivot(index = 'Date',columns='ticker', values = 'Adj Close')
# print(df.head())
ldf = df.pct_change()

np.random.seed(42)
num_ports = 1500
all_weights = np.zeros((num_ports, len(df.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports):
    # Weights
    weights = np.array(np.random.random(3))
    weights = weights/np.sum(weights)
    
    # Save weights
    all_weights[x,:] = weights
    
    # Expected return
    ret_arr[x] = np.sum( (ldf.mean() * weights * 252))
    
    # Expected volatility
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(ldf.cov()*252, weights)))
    
    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x]/vol_arr[x]


# print(sharpe_arr.Max())
#get Max
def max1(a):
    xmax = a[0]
    for x in a:
        if x > xmax:
            xmax = x
    return xmax

mx = max1(sharpe_arr)

#get index
for i in range(num_ports): 
    if sharpe_arr[i] == mx:
        ind = i


print(ind)
mxvol = vol_arr[ind]
mxret = ret_arr[ind]


plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(mxvol, mxret,c='red', s=50) # red dot
plt.show()


    

















