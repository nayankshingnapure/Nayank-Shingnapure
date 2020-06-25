#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[2]:


titan=pd.read_csv('c:\\TITAN.csv')


# In[3]:


titan


# In[5]:


titan['Bollinger_mean']=titan['Close Price'].rolling(14).mean()


# In[6]:


titan['Bollinger_std']=titan['Close Price'].rolling(14).std()


# In[7]:


titan['Bollinger_upper_14days']=titan['Bollinger_mean'] + (titan['Bollinger_std'] * 2)


# In[8]:


titan['Bollinger_lower_14days']=titan['Bollinger_mean'] - (titan['Bollinger_std'] * 2)


# In[9]:


titan.head()


# In[10]:


def call(titan):
    if titan['Close Price'] <= titan['Bollinger_lower_14days']:
        return 'Buy'
    if titan['Close Price'] > titan['Bollinger_lower_14days'] and titan['Close Price'] < titan['Bollinger_mean']:
        return 'Hold Buy / Liquidate Short'
    if titan['Close Price'] < titan['Bollinger_upper_14days'] and titan['Close Price'] > titan['Bollinger_mean']: 
        return  'Hold Short / Liquidate Buy'
    if titan['Close Price'] >= titan['Bollinger_upper_14days']: 
        return 'Short'
titan=titan.assign(Call=titan.apply(call,axis=1))
titan


# In[11]:


titan=titan.dropna()


# In[12]:


titan


# In[13]:


fig = plt.figure(figsize=(20,10))
ax1 = plt.gca()
ax2 = ax1.twinx()


titan.plot(kind='line',x='Date', y='Close Price',             ax=ax1, color='black')
titan.plot(kind='line',x='Date', y='Bollinger_mean',              ax=ax1, color='blue', linestyle='--')
titan.plot(kind='line',x='Date', y='Bollinger_upper_14days',  ax=ax1, color='red',  linestyle='--')
titan.plot(kind='line',x='Date', y='Bollinger_lower_14days',  ax=ax1, color='red',  linestyle='--')
ax2.plot( titan['Call'] )


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
X=titan.dropna()[['Close Price','Bollinger_mean','Bollinger_upper_14days','Bollinger_lower_14days']]
y=titan.dropna()['Call']
X=StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
reg=LogisticRegression()
reg.fit(X_train,y_train)
reg.score(X_train,y_train)


# In[15]:


import numpy as np
X=titan.dropna()[['Close Price','Bollinger_mean','Bollinger_upper_14days','Bollinger_lower_14days']]
#X=np.reshape(-1,1)
y=titan.dropna()['Call']
from sklearn.neighbors import KNeighborsClassifier
n=KNeighborsClassifier(n_neighbors=3)
n.fit(X_train,y_train)
n.score(X_train,y_train)


# In[16]:


from sklearn.svm import SVC
s=SVC(kernel='rbf',gamma=1,C=2)
s.fit(X_train,y_train)
s.score(X_train,y_train)


# In[17]:


from sklearn.naive_bayes import GaussianNB  
Na=GaussianNB()
Na.fit(X_train,y_train)
Na.score(X_train,y_train)


# In[18]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=5)
dt.fit(X_train,y_train)
dt.score(X_train,y_train)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=50, max_features=1)
random_forest.fit(X_train,y_train)
random_forest.score(X_train,y_train)


# In[20]:


from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
sgd.fit(X_train,y_train)
sgd.score(X_train,y_train)


# In[21]:


voltas=pd.read_csv("c:\\VOLTAS.csv")


# In[22]:


voltas['Date']=pd.to_datetime(voltas['Date'])
voltas=voltas.set_index('Date')


# In[23]:


voltas['Bollinger_mean']=voltas['Close Price'].rolling(14).mean()


# In[24]:


voltas['Bollinger_std']=voltas['Close Price'].rolling(14).std()


# In[25]:


voltas['Bollinger_upper_14days']=voltas['Bollinger_mean'] + (voltas['Bollinger_std'] * 2)


# In[26]:


voltas['Bollinger_lower_14days']=voltas['Bollinger_mean'] - (voltas['Bollinger_std'] * 2)


# In[27]:


voltas=voltas.dropna()


# In[28]:


voltas


# In[29]:


X=voltas.dropna()[['Close Price','Bollinger_mean','Bollinger_upper_14days','Bollinger_lower_14days']]
voltas['Call']=n.predict(X)


# In[30]:


voltas.head(6)


# In[31]:


sun=pd.read_csv("c:\\SUNTV.csv")


# In[32]:


sun.head(5)


# In[33]:


sun['perc_opcl_price']=((sun['Close Price']-sun['Open Price'])/(sun['Open Price']))*100


# In[34]:


sun['perc_opcl_price']


# In[35]:


sun['perc_lohi_price']=((sun['High Price']-sun['Low Price'])/(sun['Low Price']))*100


# In[36]:


sun['perc_lohi_price']


# In[37]:


sun['rolling_mean_5_clp']=sun['Close Price'].rolling(5).mean()


# In[38]:


sun['rolling_mean_5_clp']


# In[39]:


sun['rolling_std_5_clp']=sun['Close Price'].rolling(5).std()


# In[40]:


sun['rolling_std_5_clp']


# In[42]:


sun['Action']=np.where(sun['Close Price'].shift(-1)>sun['Close Price'],1,-1)


# In[43]:


sun.head(5)


# In[44]:


from sklearn.ensemble import RandomForestClassifier
X=sun.dropna()[['perc_opcl_price','perc_lohi_price','rolling_mean_5_clp','rolling_std_5_clp']]
y=sun.dropna()['Action']
X=StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, max_features=2)
random_forest


# In[45]:


random_forest.fit(X_train,y_train)
random_forest.score(X_train,y_train)


# In[46]:


voltas['Net_cummulative_returns']=((voltas['Open Price']-voltas['Close Price'])/voltas['Open Price'])


# In[48]:


plt.figure(figsize=(20,10))
plt.plot(voltas['Net_cummulative_returns'])


# In[ ]:




