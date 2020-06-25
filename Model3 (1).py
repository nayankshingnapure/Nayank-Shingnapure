#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('c:\\GOLD.csv')


# In[3]:


df1=df.set_index('Date')


# In[4]:


df1


# In[5]:


x_field=['Open','High','Low','Price']
y_field='Pred'
from sklearn.model_selection import train_test_split


# In[6]:


x_train=df1[-np.isnan(df1['Pred'])]
x_test=df1[np.isnan(df1['Pred'])]


# In[7]:


x_train.head()


# In[8]:


x_test.head()


# In[9]:


regobj=linear_model.LinearRegression()


# In[10]:


regobj.fit(x_train[x_field],x_train[y_field])


# In[11]:


print('Coefficients: \n', regobj.coef_) 
  


# In[12]:


regobj.score(x_train[x_field],x_train[y_field])


# In[13]:


x_train[y_field]=regobj.predict(x_train[x_field])
x_test[y_field]=regobj.predict(x_test[x_field])


# In[14]:


x_test.head()


# In[15]:


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

ax2.plot( x_test.Price, label='Price' ) 
ax2.plot( x_test.Open,  label='Open' )  
ax2.plot( x_test.High,  label='High' )  
ax2.plot( x_test.Low,   label='Low' )   
ax1.plot( x_test.Pred,  label='Pred', linestyle=':')
plt.title('Gold')
plt.ylabel('Price')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')


# In[16]:


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2=ax1.twinx()
#ax2 = fig.add_subplot(1, 2, 1)

ax1.hist( x_test.Price )
ax1.hist( x_test.Open )
ax1.hist( x_test.High )
ax1.hist( x_test.Low )
ax2.hist( x_test.Pred )


# In[17]:


a=linear_model.LinearRegression().fit(x_train[x_field],x_train['new'])


# In[18]:


a.score(x_train[x_field],x_train['new'])


# In[19]:


a.coef_


# In[20]:


a.intercept_


# In[21]:


from sklearn.preprocessing import PolynomialFeatures


# In[22]:


features = {
    "train": PolynomialFeatures(2).fit_transform( x_train[ x_field ] ),
    "test":  PolynomialFeatures(2).fit_transform( x_test[  x_field ] ),    
}
model_new_poly =linear_model.LinearRegression().fit( features['train'], x_train['new'] )
model_new_poly.score( features['test'], x_test['new'] )


# In[23]:


x_test['new_linear'] = a.predict( x_test[ x_field ] )
x_test['new_poly']   = model_new_poly.predict(   features['test'] )


# In[24]:


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot( x_test['new'],        label='new',        linestyle='-', );  
ax1.plot( x_test['new_linear'], label='new_linear', linestyle='-', );  
ax1.plot( x_test['new_poly'],   label='new_poly',   linestyle='-', );  
ax1.legend(loc='upper left')


# In[25]:


AMARAJABAT1=pd.read_csv(r"C://AMARAJABAT.csv")


# In[26]:


AMARAJABAT1['Date']=pd.to_datetime(AMARAJABAT1['Date'])
AMARAJABAT=AMARAJABAT1.set_index('Date')


# In[27]:


AMARAJABAT


# In[28]:


NIFTY501=pd.read_csv(r"C://NIFTY50.csv")

NIFTY501['Date']=pd.to_datetime(NIFTY501['Date'])
NIFTY50=NIFTY501.set_index('Date')


# In[29]:


NIFTY50


# In[30]:


import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


# In[33]:


prices = pd.concat([ AMARAJABAT['Close Price'], NIFTY50['Close'] ], axis=1)
prices.columns = ['AMARAJABAT', 'NIFTY50']
prices.head()


# In[34]:


prices.pct_change()


# In[35]:


returns = prices.pct_change().dropna(axis=0)
returns.head()


# In[36]:


X  = returns['NIFTY50']
Y  = returns['AMARAJABAT']
X1 = sm.add_constant(X)

model = OLS( Y, X1 )
model.fit().summary()


# In[38]:


NIFTY50['month']    = NIFTY50.index.map(   lambda date: f"{date.year}-{date.month}")
AMARAJABAT['month'] = AMARAJABAT.index.map(lambda date: f"{date.year}-{date.month}")


# In[40]:


NIFTY50_monthly    = NIFTY50.groupby('month').last()
AMARAJABAT_monthly =AMARAJABAT.groupby('month').last()


# In[41]:


AMARAJABAT_monthly


# In[42]:


returns_monthly = pd.concat([ AMARAJABAT_monthly['Close Price'], NIFTY50_monthly['Close'] ], axis=1).pct_change().dropna()
returns_monthly.columns = ['AMARAJABAT', 'NIFTY50']
returns_monthly.head()


# In[43]:


X  = returns_monthly['NIFTY50']
Y  = returns_monthly['AMARAJABAT']
X1 = sm.add_constant(X)

model = OLS( Y, X1 )
model.fit().summary()


# In[45]:


#IN LinearRegression we got pred_score value approx equal to 1 so its means that its aperfect match.This exact match suggests that Pred is column with a linear combination of inputs.


# In[46]:


#Beta can also be calculated using the linear regression covarience coefficent.
#Beta of AMARAJABAT vs NIFTY50 is 0.8572 for daily return
#It is less than 1 so its volatility is low for daily returns


# In[47]:


#Beta of AMARAJABAT vs NIFTY50 is 0.7272 for monthly return
#It is also leass than 1 means it also has low volatility for monthly return


# In[ ]:




