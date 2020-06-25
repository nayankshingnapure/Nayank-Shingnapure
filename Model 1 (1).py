#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


stock=pd.read_csv('c:\\Nifty50.csv')


# In[3]:


print(stock)


# In[4]:


stock.head()


# In[5]:


stock.tail()


# In[6]:


stock.describe()


# In[7]:


stock.tail(90)


# In[8]:


stock.tail(90).min(axis=0)


# In[9]:


stock.tail(90).max(axis=0)


# In[10]:


stock.tail(90).mean(axis=0)


# In[11]:


stock.info()


# In[12]:


stock['Date']=pd.to_datetime(stock['Date'])


# In[13]:


stock.info()


# In[14]:


stock['Month']=pd.DatetimeIndex(stock['Date']).month


# In[15]:



stock['Year']=pd.DatetimeIndex(stock['Date']).year


# In[16]:


gym=stock.groupby(['Month','Year'])


# In[17]:


stock['vwap'] = (np.cumsum(stock['Shares Traded'] * stock['Close']) / np.cumsum(stock['Close'])).astype(int)



# In[18]:


stock


# In[19]:


N=int(input(""))
stock["avg"]=stock['Close'].rolling(window=N).mean()


# In[20]:


stock.head(1)


# In[21]:


def profit_loss_pct(N):
    stock['Profit/Loss'] = (stock['Close'] - stock['Close'].shift(1)) / stock['Close']
    total_days = len(stock['Profit/Loss'])
    calc_pnl = stock['Profit/Loss'][total_days-N:].sum()
    if stock["Profit/Loss"][N] < 0:
          print("Loss pct is: {:5.2f}%". format(stock["Profit/Loss"][N]*100));
    else:
         print("Profit pct is : {:5.2f}%". format(stock["Profit/Loss"][N]*100));
    return 


# In[22]:


profit_loss_pct(365)


# In[23]:


stock.fillna(0)


# In[24]:


stock['Day_Perc_Change']=stock['Close'].pct_change()


# In[25]:


stock


# In[26]:


stock.fillna(0)


# In[27]:


trend=pd.Series([])
for i in range(len(stock)):
    if stock['Day_Perc_Change'][i]>-0.5 or stock['Day_Perc_Change'][i]<0.5 :
        trend[i] = "Slight or No change"
    elif stock['Day_Perc_Change'][i]>0.5 or stock['Day_Perc_Change'][i]<1 :
        trend[i] = "Slight positive"
    elif stock['Day_Perc_Change'][i]>-1 or stock['Day_Perc_Change'][i]<-0.5 :
        trend[i] = "Slight negative"
    elif stock['Day_Perc_Change'][i]>1 or stock['Day_Perc_Change'][i]<3 :
        trend[i] = "Positive" 
    elif stock['Day_Perc_Change'][i]>-3 or stock['Day_Perc_Change'][i]<-1 :
        trend[i] = "Negative" 
    elif stock['Day_Perc_Change'][i]>3 or stock['Day_Perc_Change'][i]<7:
        trend[i] = "Among top gainers"
    elif stock['Day_Perc_Change'][i]>-7 or stock['Day_Perc_Change'][i]<-3 :
        trend[i] = "Among top losers" 
    elif stock['Day_Perc_Change'][i]>7:
        trend[i] = "Bull run" 
    elif stock['Day_Perc_Change'][i]<-7 :
        trend[i] = "Bear drop" 
stock.insert(10,"Trend",trend)        
    


# In[28]:


stock


# In[29]:


stock.fillna(0)


# In[30]:


stock.head(30)


# In[31]:


gtrend=stock.groupby(['Trend'])


# In[32]:


stock['Total Traded Quantity']=stock['Shares Traded']


# In[33]:


stock


# In[34]:


stock_trend = stock.groupby(["Trend"])
average_trades = stock_trend["Total Traded Quantity"].mean()
print("The average traded quantity is: ", average_trades)


# In[35]:


stock.groupby(stock.Trend).mean()['Total Traded Quantity'].astype(int)


# In[36]:


stock.groupby(stock.Trend).median()['Total Traded Quantity']


# In[ ]:





# In[ ]:




