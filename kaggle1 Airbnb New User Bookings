#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv(r"C:\Users\wei\Documents\碩士\碩一\機器學習\Kaggle2\train.csv",dtype={'date_first_booking': str,'date_account_created': str,'timestamp_first_active':float})
test = pd.read_csv(r"C:\Users\wei\Documents\碩士\碩一\機器學習\Kaggle2\test.csv",dtype={'date_first_booking': str,'date_account_created': str,'timestamp_first_active':float})
data = pd.concat([train , test] , axis = 0 , ignore_index=True)
data.head(20)


# In[3]:


data.gender.replace('-unknown-', np.nan, inplace=True)
print("New null value % in gender is: " + "{0:.2%}".format(sum(data.gender.isnull())/data.shape[0]))


# In[4]:


data.loc[data.age < 18, 'age'] = np.nan
data.loc[data.age > 80, 'age'] = np.nan
data.age = data.age.replace("NaN", np.nan)
print("Now the % of null values in age is: " + "{0:.2%}".format(sum(data.age.isnull())/data.shape[0]))


# In[5]:


data['signup_flow'] = data['signup_flow'].astype('object')


# In[6]:


data.dtypes


# In[7]:


data.affiliate_channel.value_counts()


# In[8]:


data.affiliate_channel.replace('api', 'other', inplace=True)
data.affiliate_channel.replace('content', 'other', inplace=True)
data.affiliate_channel.replace('remarketing', 'other', inplace=True)


# In[9]:


data.affiliate_provider.value_counts()


# In[10]:


data.affiliate_provider.replace('daum', 'other', inplace=True)
data.affiliate_provider.replace('wayn', 'other', inplace=True)
data.affiliate_provider.replace('yandex', 'other', inplace=True)
data.affiliate_provider.replace('baidu', 'other', inplace=True)
data.affiliate_provider.replace('naver', 'other', inplace=True)
data.affiliate_provider.replace('email-marketing', 'other', inplace=True)
data.affiliate_provider.replace('meetup', 'other', inplace=True)
data.affiliate_provider.replace('gsp', 'other', inplace=True)


# In[11]:


data.first_browser.value_counts()


# In[12]:


data.first_browser.replace('Chromium', 'other', inplace=True)
data.first_browser.replace('BlackBerry Browser', 'other', inplace=True)
data.first_browser.replace('IE Mobile ', 'other', inplace=True)
data.first_browser.replace('Silk', 'other', inplace=True)
data.first_browser.replace('Opera', 'other', inplace=True)
data.first_browser.replace('Mobile Firefox', 'other', inplace=True)
data.first_browser.replace('Maxthon', 'other', inplace=True)
data.first_browser.replace('Apple Mail', 'other', inplace=True)
data.first_browser.replace('Sogou Explorer', 'other', inplace=True)
data.first_browser.replace('SiteKiosk', 'other', inplace=True)
data.first_browser.replace('RockMelt', 'other', inplace=True)
data.first_browser.replace('Iron', 'other', inplace=True)
data.first_browser.replace('IceWeasel', 'other', inplace=True)
data.first_browser.replace('Avant Browser', 'other', inplace=True)
data.first_browser.replace('CoolNovo', 'other', inplace=True)
data.first_browser.replace('Camino', 'other', inplace=True)
data.first_browser.replace('CometBird', 'other', inplace=True)
data.first_browser.replace('Opera Mini', 'other', inplace=True)
data.first_browser.replace('SeaMonkey', 'other', inplace=True)
data.first_browser.replace('Pale Moon', 'other', inplace=True)
data.first_browser.replace('Yandex.Browser', 'other', inplace=True)
data.first_browser.replace('Crazy Browser', 'other', inplace=True)
data.first_browser.replace('Mozilla', 'other', inplace=True)
data.first_browser.replace('Opera Mobile', 'other', inplace=True)
data.first_browser.replace('wOSBrowser', 'other', inplace=True)
data.first_browser.replace('TenFourFox', 'other', inplace=True)
data.first_browser.replace('Googlebot', 'other', inplace=True)
data.first_browser.replace('Epic', 'other', inplace=True)
data.first_browser.replace('Arora', 'other', inplace=True)
data.first_browser.replace('Stainless', 'other', inplace=True)
data.first_browser.replace('UC Browser', 'other', inplace=True)
data.first_browser.replace('Google Earth ', 'other', inplace=True)
data.first_browser.replace('NetNewsWire', 'other', inplace=True)
data.first_browser.replace('IceDragon', 'other', inplace=True)
data.first_browser.replace('Outlook 2007', 'other', inplace=True)
data.first_browser.replace('PS Vita browser', 'other', inplace=True)
data.first_browser.replace('Kindle Browser', 'other', inplace=True)
data.first_browser.replace('Nintendo Browser', 'other', inplace=True)
data.first_browser.replace('IBrowse', 'other', inplace=True)
data.first_browser.replace('Conkeror', 'other', inplace=True)
data.first_browser.replace('Palm Pre web browser', 'other', inplace=True)
data.first_browser.replace('TheWorld Browser', 'other', inplace=True)
data.first_browser.replace('OmniWeb', 'other', inplace=True)
data.first_browser.replace('SlimBrowser', 'other', inplace=True)
data.first_browser.replace('EFlock', 'other', inplace=True)
data.first_browser.replace('Comodo Dragon', 'other', inplace=True)
data.first_browser.replace('Flock', 'other', inplace=True)


# In[13]:


columns = ['affiliate_channel', 'affiliate_provider',
        'first_affiliate_tracked',
       'first_browser', 'first_device_type',
       'signup_app', 'signup_method']

for c in columns:
    data_ohe = pd.get_dummies(data[c],prefix=c, dummy_na=True)
    data.drop([c], axis = 1, inplace = True)
    data = pd.concat((data, data_ohe), axis = 1)


# In[14]:


data.columns


# In[15]:


from sklearn.linear_model import LogisticRegression

def set_missing_ages(df):

    columns = df[['age','affiliate_channel_direct',
       'affiliate_channel_other', 'affiliate_channel_sem-brand',
       'affiliate_channel_sem-non-brand', 'affiliate_channel_seo',
       'affiliate_channel_nan', 'affiliate_provider_bing',
       'affiliate_provider_craigslist', 'affiliate_provider_direct',
       'affiliate_provider_facebook', 'affiliate_provider_facebook-open-graph',
       'affiliate_provider_google', 'affiliate_provider_other',
       'affiliate_provider_padmapper', 'affiliate_provider_vast',
       'affiliate_provider_yahoo', 'affiliate_provider_nan',
       'first_affiliate_tracked_linked', 'first_affiliate_tracked_local ops',
       'first_affiliate_tracked_marketing', 'first_affiliate_tracked_omg',
       'first_affiliate_tracked_product',
       'first_affiliate_tracked_tracked-other',
       'first_affiliate_tracked_untracked', 'first_affiliate_tracked_nan',
       'first_browser_-unknown-', 'first_browser_AOL Explorer',
       'first_browser_Android Browser', 'first_browser_Chrome',
       'first_browser_Chrome Mobile', 'first_browser_Firefox',
       'first_browser_Google Earth', 'first_browser_IE',
       'first_browser_IE Mobile', 'first_browser_Mobile Safari',
       'first_browser_Safari', 'first_browser_other', 'first_browser_nan',
       'first_device_type_Android Phone', 'first_device_type_Android Tablet',
       'first_device_type_Desktop (Other)', 'first_device_type_Mac Desktop',
       'first_device_type_Other/Unknown',
       'first_device_type_SmartPhone (Other)',
       'first_device_type_Windows Desktop', 'first_device_type_iPad',
       'first_device_type_iPhone', 'first_device_type_nan',
       'signup_app_Android', 'signup_app_Moweb', 'signup_app_Web',
       'signup_app_iOS', 'signup_app_nan', 'signup_method_basic',
       'signup_method_facebook', 'signup_method_google', 'signup_method_weibo',
       'signup_method_nan']]

    known_age = columns[columns.age.notnull()].as_matrix()
    unknown_age = columns[columns.age.isnull()].as_matrix()

    y = known_age[:, 0]

    X = known_age[:, 1:]

    lr=LogisticRegression()
    lr.fit(X, y)

    predictedages = lr.predict(unknown_age[:, 1:])
    df.loc[ (df.age.isnull()), 'age' ] = predictedages 

    return df, lr


# In[16]:


set_missing_ages(data)


# In[17]:


data.info()


# In[18]:


data.gender.replace('MALE', 0, inplace=True)
data.gender.replace('FEMALE', 1, inplace=True)
data.gender.replace('OTHER', 2, inplace=True)


# In[19]:


def set_missing_genders(df):

    columns = df[['gender','affiliate_channel_direct',
       'affiliate_channel_other', 'affiliate_channel_sem-brand',
       'affiliate_channel_sem-non-brand', 'affiliate_channel_seo',
       'affiliate_channel_nan', 'affiliate_provider_bing',
       'affiliate_provider_craigslist', 'affiliate_provider_direct',
       'affiliate_provider_facebook', 'affiliate_provider_facebook-open-graph',
       'affiliate_provider_google', 'affiliate_provider_other',
       'affiliate_provider_padmapper', 'affiliate_provider_vast',
       'affiliate_provider_yahoo', 'affiliate_provider_nan',
       'first_affiliate_tracked_linked', 'first_affiliate_tracked_local ops',
       'first_affiliate_tracked_marketing', 'first_affiliate_tracked_omg',
       'first_affiliate_tracked_product',
       'first_affiliate_tracked_tracked-other',
       'first_affiliate_tracked_untracked', 'first_affiliate_tracked_nan',
       'first_browser_-unknown-', 'first_browser_AOL Explorer',
       'first_browser_Android Browser', 'first_browser_Chrome',
       'first_browser_Chrome Mobile', 'first_browser_Firefox',
       'first_browser_Google Earth', 'first_browser_IE',
       'first_browser_IE Mobile', 'first_browser_Mobile Safari',
       'first_browser_Safari', 'first_browser_other', 'first_browser_nan',
       'first_device_type_Android Phone', 'first_device_type_Android Tablet',
       'first_device_type_Desktop (Other)', 'first_device_type_Mac Desktop',
       'first_device_type_Other/Unknown',
       'first_device_type_SmartPhone (Other)',
       'first_device_type_Windows Desktop', 'first_device_type_iPad',
       'first_device_type_iPhone', 'first_device_type_nan',
       'signup_app_Android', 'signup_app_Moweb', 'signup_app_Web',
       'signup_app_iOS', 'signup_app_nan', 'signup_method_basic',
       'signup_method_facebook', 'signup_method_google', 'signup_method_weibo',
       'signup_method_nan']]

    known_gender = columns[columns.gender.notnull()].as_matrix()
    unknown_gender = columns[columns.gender.isnull()].as_matrix()

    y = known_gender[:, 0]
    X = known_gender[:, 1:]

    lr=LogisticRegression()
    lr.fit(X, y)

    predictedgenders = lr.predict(unknown_gender[:, 1:])
    df.loc[ (df.gender.isnull()), 'gender' ] = predictedgenders 

    return df, lr


# In[20]:


set_missing_genders(data)


# In[21]:


bins=[17,25,35,45,55,65,81]

labels=['18-25','26-35','36-45','46-55','56-65','66-80',]

data['age']=pd.cut(
        data.age,
        bins,
        labels=labels
        )


# In[22]:


bins=[20081200000000,20090131246060,20090531246060,20090831246060
      ,20091130246060,20100131246060,20100531246060,20100831246060
      ,20101130246060,20110131246060,20110531246060,20110831246060
      ,20111130246060,20120131246060,20120531246060,20120831246060
      ,20121130246060,20130131246060,20130531246060,20130831246060
      ,20131130246060,20140131246060,20140531246060,20140831246060
      ,20141130246060,20150131246060]

labels=['0812-0902','0903-0905','0906-0908','0909-0911','0912-1002'
                   ,'1003-1005','1006-1008','1009-1011','1012-1102'
                   ,'1103-1105','1106-1108','1109-1111','1112-1202'
                   ,'1203-1205','1206-1208','1209-1211','1212-1302'
                   ,'1303-1305','1306-1308','1309-1311','1312-1402'
                   ,'1403-1405','1406-1408','1409-1411','1412-1502']

data['timestamp_first_active']=pd.cut(
        data.timestamp_first_active,
        bins,
        labels=labels
        )


# In[23]:


data.info()


# In[24]:


data.drop(labels=['date_first_booking'],axis='columns',inplace=True)
data.drop(labels=['id'],axis='columns',inplace=True)


# In[25]:


data.columns


# In[26]:


columns = ['age',  'date_account_created', 'gender',
       'language', 'signup_flow', 'timestamp_first_active']

for c in columns:
    data_ohe = pd.get_dummies(data[c],prefix=c, dummy_na=True)
    data.drop([c], axis = 1, inplace = True)
    data = pd.concat((data, data_ohe), axis = 1)


# In[27]:


data.info()


# In[28]:


y = train['country_destination'].values


# In[29]:


y


# In[30]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# In[31]:


y


# In[33]:


data.drop(labels=['country_destination'],axis='columns',inplace=True)


# In[34]:


X_train = data.values[:213451]
X_test = data.values[213451:]


# In[35]:


data.head()


# In[36]:


from sklearn.metrics import mean_squared_error, r2_score


# In[37]:


test_ids = test['id']
test_ids


# # xgboost-------------------------

# In[ ]:


import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# In[ ]:


xgb = XGBClassifier(max_depth=8, learning_rate=0.3, n_estimators=40,
                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0)               
xgb.fit(X_train, y)


# In[ ]:


y_pred_xgb = xgb.predict_proba(X_test)


# In[ ]:


y_pred_xgb


# In[ ]:


ids = []
cts = []  
for i in range(len(test_ids)):
    idx = test_ids[i]
    ids += [idx] * 5
    cts += labelencoder.inverse_transform(np.argsort(y_pred_xgb[i])[::-1])[:5].tolist()


# In[ ]:


sub_xgb = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub_xgb.to_csv('submission.csv',index=False)


# # Logistic---------------------------------------------------------------

# In[39]:


lr=LogisticRegression()
lr.fit(X_train, y)


# In[40]:


y_pred = lr.predict_proba(X_test)
y_train_pred = lr.predict_proba(X_train)


# In[41]:


y_pred


# In[42]:


ids = []  
cts = []  
for i in range(len(test_ids)):
    idx = test_ids[i]
    ids += [idx] * 5
    cts += labelencoder.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()


# In[43]:


sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])


# In[ ]:


sub.to_csv('logistic.csv',index=False)


# # Decision Tree-----------------------------------------------

# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[94]:


tree = DecisionTreeClassifier(criterion = 'gini',
                              random_state = 15,
                              max_depth = 5,)

tree.fit(X_train,y)


# In[95]:


y_pred_tree = tree.predict_proba(X_test)


# In[96]:


y_pred_tree


# In[97]:


ids = []  
cts = []  
for i in range(len(test_ids)):
    idx = test_ids[i]
    ids += [idx] * 5
    cts += labelencoder.inverse_transform(np.argsort(y_pred_tree[i])[::-1])[:5].tolist()


# In[98]:


sub_tree = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])


# In[100]:


sub_tree.to_csv('tree8.csv',index=False)


# # RandomForest -----------------------------------------

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, criterion='gini')
rf.fit(X_train, y)


# In[ ]:


y_pred_forest = rf.predict(X_test)


# In[ ]:


y_pred_forest


# In[ ]:


ids = []  
cts = []  
for i in range(len(test_ids)):
    idx = test_ids[i]
    ids += [idx] * 5 
    cts += labelencoder.inverse_transform(np.argsort(aa[i])[::-1])[:5].tolist()


# In[ ]:


sub_random2 = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])


# In[ ]:


cts += labelencoder.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()


# In[ ]:


sub_random2.to_csv('random.csv',index=False)

