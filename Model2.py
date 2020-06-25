#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


df=pd.read_csv(r'E:\\week2.csv')


# In[3]:


df


# In[4]:


df['Date']=pd.to_datetime(df["Date"])


# In[5]:


df.info()


# In[6]:


df.set_index("Date",inplace=True, drop=False)


# In[7]:


df


# In[8]:


df.fillna(0)


# In[9]:


close_of_day = df['Close'].groupby('Date').last().to_frame()
close_of_day['Date'] =close_of_day.index
close_of_day.head()


# In[10]:


sns.lineplot(x="Date", y="Close", data= close_of_day)


# In[11]:


df['Day_Perc_Change'].sort_values().head()


# In[12]:


df['Day_Perc_Change'].sort_values().tail()


# In[13]:


plt.stem(df['Date'],df['Day_Perc_Change'],use_line_collection=True)
plt.show()


# In[15]:


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

ax1.set_xlabel('Date')
ax2.set_ylabel('Total Traded Quantity', color='red')
ax2.plot(df['Total Traded Quantity'], 'r-')

ax1.plot(df['Day_Perc_Change'], 'b-')
ax1.set_ylabel('Day_Perc_Change', color='blue')


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)
align_yaxis(ax1, 0, ax2, 0)


# In[16]:


plt.plot(df['Date'],df['Total Traded Quantity'])


# In[11]:



plt.stem(df['Date'],df['Day_Perc_Change'])
plt.plot(df['Date'],df['Total Traded Quantity'])
plt.legend(loc='upper right')


# In[12]:


b=plt.plot(df['Shares Traded'])


# In[17]:


trend_aggregation = df.groupby('Trend').count()
trend_aggregation.index
trend_aggregation


# In[18]:


fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))
size=[35,33,73,249,98]
ax.pie( size, labels=trend_aggregation.index,autopct='%1.1f%%' )

ax.set_title("Trend Aggregation Pie Chart")
ax.legend(trend_aggregation.index,
          title="Trend",
          loc="center left",
          bbox_to_anchor=(1.1, 0, 0.5, 1))

plt.show()


# In[19]:


df.head(100)


# In[20]:


a=df.groupby(['Trend'],as_index=False)['Total Traded Quantity']
a


# In[21]:


ypos=np.arange(len(a))
ypos


# In[22]:


fig, axes = plt.subplots(figsize=(30, 10), nrows=1, ncols=2)
sns.barplot(ax=axes[0], x='Trend', y='Total Traded Quantity', data=a.mean())#.sort_values('Total Traded Quantity'))
sns.barplot(ax=axes[1], x='Trend', y='Total Traded Quantity', data=a.median())


# In[23]:


sns.distplot(df['Day_Perc_Change'].tolist(), kde=True, rug=False);


# In[24]:


ASHOKA=pd.read_csv("C://ASHOKA.csv")


# In[25]:


AXISBANK=pd.read_csv("C://AXISBANK.csv")


# In[26]:


APOLLOTYRE=pd.read_csv("C://APOLLOTYRE.csv")


# In[27]:


BAJAJELEC=pd.read_csv("C://BAJAJELEC.csv")


# In[28]:


BPCL=pd.read_csv("C://BPCL.csv")


# In[29]:


ASHOKA.set_index('Date')


# In[30]:


AXISBANK.set_index('Date')


# In[31]:


APOLLOTYRE.set_index('Date')


# In[32]:


BAJAJELEC


# In[33]:


BPCL


# In[34]:


df1=pd.DataFrame(columns=['ASHOKA','AXISBANK','APOLLOTYRE','BAJAJELEC','BPCL'])


# In[35]:


df1


# In[36]:


df1['ASHOKA']=ASHOKA['Close Price'].copy()


# In[37]:


df1['AXISBANK']=AXISBANK['Close Price'].copy()


# In[38]:


df1['APOLLOTYRE']=APOLLOTYRE['Close Price'].copy()


# In[39]:



df1['BAJAJELEC']=BAJAJELEC['Close Price'].copy()


# In[40]:


df1['BPCL']=BPCL['Close Price'].copy()


# In[41]:


df1


# In[42]:


df2=pd.DataFrame(columns=['ASHOKA','AXISBANK','APOLLOTYRE','BAJAJELEC','BPCL'])
df2['ASHOKA']=df1['ASHOKA'].pct_change()


# In[43]:


df2['AXISBANK']=df1['AXISBANK'].pct_change()


# In[44]:


df2['APOLLOTYRE']=df1['APOLLOTYRE'].pct_change()


# In[45]:


df2['BAJAJELEC']=df1['BAJAJELEC'].pct_change()


# In[46]:


df2['BPCL']=df1['BPCL'].pct_change()


# In[47]:


df2


# In[48]:


df2.dropna()


# In[49]:


sns.pairplot(df2)


# In[50]:


sns.pairplot(df1)


# In[51]:


df['Volatility'] = df.Close.pct_change().rolling(7).std() * np.sqrt(7)


# In[52]:


correlation_volatility = df2.rolling(7).std() * np.sqrt(7)


# In[53]:


correlation_volatility.plot()


# In[54]:


nifty=pd.read_csv('c:\\Nifty50.csv')


# In[55]:


nifty.set_index('Date', inplace=True, drop=False)
nifty['Volatility'] = nifty.Close.pct_change().rolling(7).std() * np.sqrt(7)
nifty.tail()


# In[56]:


correlation_volatility.plot()
nifty['Volatility'].plot()
plt.show()


# In[57]:


df["21_day_SMA"] = df.Close.rolling(21).mean()
df["34_day_SMA"] = df.Close.rolling(34).mean()
df[["21_day_SMA", "34_day_SMA"]].plot()


# In[58]:



prev_index       = df.index[0]
prev_row         = df[:prev_index]
short_sma_higher = prev_row["21_day_SMA"][0] > prev_row["34_day_SMA"][0]
df['Trade_Call'] = 'HODL'

for index, row in df.iterrows():
    if row["21_day_SMA"] and row["34_day_SMA"]:    
        if short_sma_higher       and row["21_day_SMA"] < row["34_day_SMA"]:
            result = "SELL"
        elif not short_sma_higher and row["21_day_SMA"] > row["34_day_SMA"]:
            result = "BUY"
        else:
            result = "HODL" 
        df.at[index, 'Trade_Call'] = result
        short_sma_higher = row["21_day_SMA"] > row["34_day_SMA"]                        
    
signals = df[["Close", "21_day_SMA", "34_day_SMA", "Trade_Call"]][ df['Trade_Call'] != "HODL" ]
signals


# In[59]:


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(111, ylabel="Price")

df[["Close", "21_day_SMA", "34_day_SMA"]].plot(ax=ax1)

ax1.plot( df["Close"][ df['Trade_Call'] == 'BUY' ].index, 
          df["Close"][ df['Trade_Call'] == 'BUY' ], 
          '^', markersize=15, color='g' )

ax1.plot( df["Close"][ df['Trade_Call'] == 'SELL' ].index, 
          df["Close"][ df['Trade_Call'] == 'SELL' ], 
          'v', markersize=15, color='r' )


# In[60]:


df["avg"]=(df['Close']+df['Low']+df['High'])/3


# In[61]:


df


# In[62]:


df["14_day_SMA"] = df.Close.rolling(14).mean()
df["14_day_STD"] = df.Close.rolling(14).std()
df["14_day_bollinger_upper"] = df["14_day_SMA"] + df["14_day_STD"] * 2
df["14_day_bollinger_lower"] = df["14_day_SMA"] - df["14_day_STD"] * 2


# In[63]:


ax = plt.gca()

df.plot(kind='line',x='Date', y='avg', ax=ax, color='black')
df.plot(kind='line',x='Date', y='14_day_SMA',    ax=ax, color='blue')
df.plot(kind='line',x='Date', y='14_day_bollinger_upper',    ax=ax, color='red')
df.plot(kind='line',x='Date', y='14_day_bollinger_lower',    ax=ax, color='green')


plt.title("14 day bollinger bands")
plt.ylabel("Price")
plt.show()


# In[64]:


df.to_csv(r'E:\\week2.csv')


# In[65]:


df


# In[ ]:




