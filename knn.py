#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('C:\\Users\kants\Downloads/nba_2013.csv')
pd.set_option('display.max_columns',None)
df.head(3)


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isna().sum()


# In[6]:


df['fg.'].fillna(df['fg.'].median(),inplace=True)
df['x3p.'].fillna(df['x3p.'].mean(),inplace=True)
df['x2p.'].fillna(df['x2p.'].median(),inplace=True)
df['efg.'].fillna(df['efg.'].mean(),inplace=True)
df['ft.'].fillna(df['ft.'].median(),inplace=True)


# In[7]:


df.drop('player',axis=1,inplace=True)
df.drop('bref_team_id',axis=1,inplace=True)
df.drop('season',axis=1,inplace=True)
df.drop('season_end', axis=1, inplace=True)


# In[8]:


df.head(3)


# In[11]:


dummy = pd.get_dummies(df['pos'],prefix= 'pos',drop_first=True)
dummy.head(3)


# In[12]:


df.drop('pos',axis=1,inplace=True)
df.head(3)


# In[13]:


df2 = pd.concat([df,dummy],axis=1)
df2.head(2)


# In[14]:


feature = df2.drop('pts',axis=1)
feature.head(2)


# In[16]:


target = df2['pts']
target.head(2)


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(feature,target,test_size=0.3,random_state=42)


# In[18]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(x_train,y_train)


# In[19]:


knr.score(x_test,y_test)


# In[21]:


from sklearn.metrics import mean_squared_error
print(f'MSE:{mean_squared_error(y_test,y_pred)}')
print(f'RMSE:{np.sqrt(mean_squared_error(y_test,y_pred))}')


# In[23]:


data = pd.DataFrame({'Actual Points': y_test.tolist(), 'Predicted Points': y_pred.tolist()})
data.head()


# In[24]:


from sklearn.preprocessing import Normalizer
norm = Normalizer()
X_norm = norm.fit_transform(feature)


# In[25]:


x_train,x_test,y_train,y_test = train_test_split(X_norm,target,test_size=0.3,random_state=42)


# In[26]:


knn_norm = KNeighborsRegressor()
knn_norm.fit(x_train,y_train)


# In[27]:


knn_norm.score(x_test,y_test)


# In[29]:


y_new_pred = knn_norm.predict(x_test)


# In[30]:


print(f'MSE:{mean_squared_error(y_test,y_new_pred)}')
print(f'RMSE:{np.sqrt(mean_squared_error(y_test,y_new_pred))}')


# In[31]:


data = pd.DataFrame({'Actual Points': y_test.tolist(), 'Predicted Points': y_new_pred.tolist()})
data.head(10)


# In[33]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(feature)


# In[34]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled,target,test_size=0.3,random_state=42)


# In[35]:


knn_scaled = KNeighborsRegressor()
knn_scaled.fit(x_train,y_train)


# In[36]:


knn_scaled.score(x_test,y_test)


# In[38]:


y_pred_2 = knn_scaled.predict(x_test)


# In[39]:


print(f'MSE:{mean_squared_error(y_test,y_pred_2)}')
print(f'RMSE:{np.sqrt(mean_squared_error(y_test,y_pred_2))}')


# In[40]:


data = pd.DataFrame({'Actual Points': y_test.tolist(), 'Predicted Points': y_pred_2.tolist()})
data.head()


# In[ ]:




