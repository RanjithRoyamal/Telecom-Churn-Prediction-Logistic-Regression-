#!/usr/bin/env python
# coding: utf-8

# # Task 1
# 
# 1. Churn Prediction in Telecom Industry using Logistic   Regression
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import seaborn as sns


# In[6]:


churn_data = pd.read_csv('churn_data.csv')


# In[7]:


churn_data.head() 


# In[8]:


customer_data = pd.read_csv('customer_data.csv')
customer_data


# In[9]:


internet_data = pd.read_csv('internet_data.csv')
internet_data


# # Merging tables using customer id

# In[10]:


df = pd.merge(churn_data, customer_data, how='inner', on = 'customerID')


# In[11]:


telecom_df = pd.merge(df, internet_data, how='inner', on='customerID')


# In[12]:


telecom_df


# In[13]:


for col in telecom_df.columns:
    print(col)


# In[14]:


telecom_df.shape


# In[15]:


telecom_df.describe()


# In[16]:


telecom_df.info()


# # Data Cleaning

# In[17]:


telecom_df.isnull().sum()*100/telecom_df.shape[0]


# In[18]:


telecom_df['TotalCharges'].describe()


# In[19]:


print(telecom_df['MonthlyCharges'])


# In[20]:


telecom_df['TotalCharges'] = telecom_df['TotalCharges'].replace(' ', np.nan)
telecom_df['TotalCharges'] = pd.to_numeric(telecom_df['TotalCharges'])


# In[21]:


value = (telecom_df['TotalCharges']/telecom_df['MonthlyCharges']).median()*telecom_df['MonthlyCharges']


# In[22]:


telecom_df['TotalCharges'].describe()


# In[23]:


telecom_df['TotalCharges'] = value.where(telecom_df['TotalCharges'] == np.nan, other =telecom_df['TotalCharges'])


# In[24]:


telecom_df['TotalCharges'].describe()


# # Data Analysis

# In[25]:


telecom_df.Churn.describe()


# In[26]:


fig, axs = plt.subplots(1,2, figsize = (15,5))
plt1 = sns.countplot(telecom_df['Churn'], ax = axs[0])

pie_churn = pd.DataFrame(telecom_df['Churn'].value_counts())
pie_churn.plot.pie( subplots=True,labels = pie_churn.index.values, autopct='%1.1f%%', figsize = (15,5), startangle= 50, ax = axs[1])
# Unsquish the pie.

plt.gca().set_aspect('equal')
plt.show()


# # Tenure

# In[27]:


sns.boxplot(x = 'tenure', y = 'Churn', data = telecom_df)
plt.show()


# # Phone Service

# In[28]:


pie_PhoneService_Yes = pd.DataFrame(telecom_df[telecom_df['PhoneService'] == "Yes"]['Churn'].value_counts())
pie_PhoneService_Yes.plot.pie(subplots=True, labels = pie_PhoneService_Yes.index.values, autopct='%1.1f%%', startangle= 50 )
plt.title('Churn Rate for customers \n opted for Phone Service')
plt.gca().set_aspect('equal')

pie_PhoneService_No = pd.DataFrame(telecom_df[telecom_df['PhoneService'] == "No"]['Churn'].value_counts())
pie_PhoneService_No.plot.pie(subplots=True, labels = pie_PhoneService_Yes.index.values, autopct='%1.1f%%', startangle= 50)
plt.title('Churn Rate for customers \n that did not opted for Phone Service')
plt.gca().set_aspect('equal')

plt.show() 


# # Contract

# In[29]:


pie_Contract_m2m = pd.DataFrame(telecom_df[telecom_df['Contract'] == "Month-to-month"]['Churn'].value_counts())
pie_Contract_m2m.plot.pie(subplots=True, labels = pie_Contract_m2m.index.values, autopct='%1.1f%%', startangle= 75)
plt.title('Month to Month Contract')
plt.gca().set_aspect('equal')

pie_Contract_1y = pd.DataFrame(telecom_df[telecom_df['Contract'] == "One year"]['Churn'].value_counts())
pie_Contract_1y.plot.pie(subplots=True, labels = pie_Contract_1y.index.values, autopct='%1.1f%%', startangle= 20)
plt.title('One Year Contract')
plt.gca().set_aspect('equal')

pie_Contract_2y = pd.DataFrame(telecom_df[telecom_df['Contract'] == "Two year"]['Churn'].value_counts())
pie_Contract_2y.plot.pie(subplots=True, labels = pie_Contract_2y.index.values, autopct='%1.1f%%', startangle= 5)
plt.title('Two Year Contract')
plt.gca().set_aspect('equal')

plt.show()


# # Paperless Bills

# In[30]:


plt.figure(figsize=(15,5))

pie_PaperlessBilling_Yes = pd.DataFrame(telecom_df[telecom_df['PaperlessBilling'] == "Yes"]['Churn'].value_counts())
pie_PaperlessBilling_Yes.plot.pie(subplots=True, labels = pie_PaperlessBilling_Yes.index.values, autopct='%1.1f%%', startangle= 60)
plt.title('Churn Rate for customers \n opted for Paperless Billing')
plt.gca().set_aspect('equal')


pie_PaperlessBilling_No = pd.DataFrame(telecom_df[telecom_df['PaperlessBilling'] == "No"]['Churn'].value_counts())
pie_PaperlessBilling_No.plot.pie(subplots=True, labels = pie_PaperlessBilling_No.index.values, autopct='%1.1f%%', startangle= 30)
plt.title('Churn Rate for customers \n that did not opted for Paperless Billing')
plt.gca().set_aspect('equal')

plt.show()


# # Payment Method

# In[31]:


telecom_df.PaymentMethod.describe()


# In[32]:


plt.figure(figsize=(15,10))
pie_PaymentMethod_ec = pd.DataFrame(telecom_df[telecom_df['PaymentMethod'] == "Electronic check"]['Churn'].value_counts())
pie_PaymentMethod_ec.plot.pie(subplots=True, labels = pie_PaymentMethod_ec.index.values, autopct='%1.1f%%', startangle= 82)
plt.title('Electronic Check')
plt.gca().set_aspect('equal')

pie_PaymentMethod_mc = pd.DataFrame(telecom_df[telecom_df['PaymentMethod'] == "Mailed check"]['Churn'].value_counts())
pie_PaymentMethod_mc.plot.pie(subplots=True, labels = pie_PaymentMethod_mc.index.values, autopct='%1.1f%%', startangle= 35)
plt.title('Mailed check')
plt.gca().set_aspect('equal')

pie_PaymentMethod_bta = pd.DataFrame(telecom_df[telecom_df['PaymentMethod'] == "Bank transfer (automatic)"]['Churn'].value_counts())
pie_PaymentMethod_bta.plot.pie(subplots=True, labels = pie_PaymentMethod_bta.index.values, autopct='%1.1f%%', startangle= 30)
plt.title('Bank transfer (automatic)')
plt.gca().set_aspect('equal')

pie_PaymentMethod_cca = pd.DataFrame(telecom_df[telecom_df['PaymentMethod'] == "Credit card (automatic)"]['Churn'].value_counts())
pie_PaymentMethod_cca.plot.pie(subplots=True, labels = pie_PaymentMethod_cca.index.values, autopct='%1.1f%%', startangle= 30)
plt.title('Credit card (automatic)')
plt.gca().set_aspect('equal')

plt.show()


# # Monthly Charges

# In[33]:


sns.boxplot(x = 'MonthlyCharges', y = 'Churn', data = telecom_df)
plt.show()


# # Total Charges

# In[34]:


sns.boxplot(x = 'TotalCharges', y= 'Churn', data = telecom_df)
plt.show()


# # Gender

# In[35]:


plt.figure(figsize=(15,5))
pie_Gender_M = pd.DataFrame(telecom_df[telecom_df['gender'] == "Male"]['Churn'].value_counts())
pie_Gender_M.plot.pie(subplots = True, labels = pie_Gender_M.index.values, autopct='%1.1f%%', startangle= 50)
plt.title('Male')
plt.gca().set_aspect('equal')

pie_Gender_F = pd.DataFrame(telecom_df[telecom_df['gender'] == "Female"]['Churn'].value_counts())
pie_Gender_F.plot.pie(subplots = True,  labels = pie_Gender_F.index.values, autopct='%1.1f%%', startangle= 50)
plt.title('Female')
plt.gca().set_aspect('equal')

plt.show() 


# # Senior Citizen

# In[36]:


plt.figure(figsize=(15,5))
pie_SeniorCitizen_Y = pd.DataFrame(telecom_df[telecom_df['SeniorCitizen'] == 1]['Churn'].value_counts())
pie_SeniorCitizen_Y.plot.pie(subplots = True, labels = pie_SeniorCitizen_Y.index.values, autopct='%1.1f%%', startangle= 75)
plt.title('Senior Citizen')
plt.gca().set_aspect('equal')

pie_SeniorCitizen_N = pd.DataFrame(telecom_df[telecom_df['SeniorCitizen'] == 0]['Churn'].value_counts())
pie_SeniorCitizen_N.plot.pie(subplots = True, labels = pie_SeniorCitizen_N.index.values, autopct='%1.1f%%', startangle= 45)
plt.title('Non Senior Citizen')

plt.gca().set_aspect('equal')
plt.show() 


# # Dependents

# In[37]:


plt.figure(figsize=(15,5))
pie_Dependents_Y = pd.DataFrame(telecom_df[telecom_df['Dependents'] == 'Yes']['Churn'].value_counts())
pie_Dependents_Y.plot.pie(subplots = True,  labels = pie_Dependents_Y.index.values, autopct='%1.1f%%', startangle= 35)
plt.title('Has Dependents')
plt.gca().set_aspect('equal')

pie_Dependents_N = pd.DataFrame(telecom_df[telecom_df['Dependents'] == 'No']['Churn'].value_counts())
pie_Dependents_N.plot.pie(subplots = True,  labels = pie_Dependents_N.index.values, autopct='%1.1f%%', startangle= 60)
plt.title('No Dependents')

plt.gca().set_aspect('equal')
plt.show() 


# # Multiple Lines

# In[38]:


plt.figure(figsize=(15,5))
pie_MultipleLines_Y = pd.DataFrame(telecom_df[telecom_df['MultipleLines'] == 'Yes']['Churn'].value_counts())
pie_MultipleLines_Y.plot.pie(subplots = True,  labels = pie_MultipleLines_Y.index.values, autopct='%1.1f%%', startangle= 50)
plt.title('Multiple lines of internet connectivity')
plt.gca().set_aspect('equal')

pie_MultipleLines_N = pd.DataFrame(telecom_df[telecom_df['MultipleLines'] == 'No']['Churn'].value_counts())
pie_MultipleLines_N.plot.pie(subplots = True,  labels = pie_MultipleLines_N.index.values, autopct='%1.1f%%', startangle= 45)
plt.title('Single line of internet connectivity')

plt.gca().set_aspect('equal')
plt.show() 


# In[39]:


import jovian


# In[40]:


jovian.commit


# # Internet service

# In[41]:


plt.figure(figsize=(15,5))
pie_InternetService_fo = pd.DataFrame(telecom_df[telecom_df['InternetService'] == "Fiber optic"]['Churn'].value_counts())
pie_InternetService_fo.plot.pie(subplots = True, labels = pie_InternetService_fo.index.values, autopct='%1.1f%%', startangle= 75)
plt.title('Fiber Optic')
plt.gca().set_aspect('equal')

pie_InternetService_dsl = pd.DataFrame(telecom_df[telecom_df['InternetService'] == "DSL"]['Churn'].value_counts())
pie_InternetService_dsl.plot.pie(subplots = True, labels = pie_InternetService_dsl.index.values, autopct='%1.1f%%', startangle= 35)
plt.title('DSL')
plt.gca().set_aspect('equal')

pie_InternetService_no = pd.DataFrame(telecom_df[telecom_df['InternetService'] == "No"]['Churn'].value_counts())
pie_InternetService_no.plot.pie(subplots = True, labels = pie_InternetService_no.index.values, autopct='%1.1f%%', startangle= 13)
plt.title('No Internet Service')
plt.gca().set_aspect('equal')

plt.show()


# # Online Security

# In[42]:


plt.figure(figsize=(15,5))
pie_OnlineSecurity_Y = pd.DataFrame(telecom_df[telecom_df['OnlineSecurity'] == 'Yes']['Churn'].value_counts())
pie_OnlineSecurity_Y.plot.pie(subplots = True,  labels = pie_OnlineSecurity_Y.index.values, autopct='%1.1f%%', startangle= 25)
plt.title('Online Security')
plt.gca().set_aspect('equal')

pie_OnlineSecurity_N = pd.DataFrame(telecom_df[telecom_df['OnlineSecurity'] == 'No']['Churn'].value_counts())
pie_OnlineSecurity_N.plot.pie(subplots = True, labels = pie_OnlineSecurity_N.index.values, autopct='%1.1f%%', startangle= 75)
plt.title('Not opted for Online Security')
plt.gca().set_aspect('equal')
plt.show() 


# # online backup

# In[43]:


plt.figure(figsize=(15,5))
pie_OnlineBackup_Y = pd.DataFrame(telecom_df[telecom_df['OnlineBackup'] == 'Yes']['Churn'].value_counts())
pie_OnlineBackup_Y.plot.pie(subplots = True,  labels = pie_OnlineBackup_Y.index.values, autopct='%1.1f%%', startangle= 40)
plt.title('Online Backup')
plt.gca().set_aspect('equal')

pie_OnlineBackup_N = pd.DataFrame(telecom_df[telecom_df['OnlineBackup'] == 'No']['Churn'].value_counts())
pie_OnlineBackup_N.plot.pie(subplots = True, labels = pie_OnlineBackup_N.index.values, autopct='%1.1f%%', startangle= 75)
plt.title('Not opted for Online Backup')
plt.gca().set_aspect('equal')

plt.show() 


# # Device Protection

# In[44]:


plt.figure(figsize=(15,5))

pie_DeviceProtection_Y = pd.DataFrame(telecom_df[telecom_df['DeviceProtection'] == 'Yes']['Churn'].value_counts())
pie_DeviceProtection_Y.plot.pie(subplots = True, labels = pie_DeviceProtection_Y.index.values, autopct='%1.1f%%', startangle= 40)
plt.title('Online Backup')
plt.gca().set_aspect('equal')

pie_DeviceProtection_N = pd.DataFrame(telecom_df[telecom_df['DeviceProtection'] == 'No']['Churn'].value_counts())
pie_DeviceProtection_N.plot.pie(subplots = True, labels = pie_DeviceProtection_N.index.values, autopct='%1.1f%%', startangle= 75)
plt.title('Not opted for Online Backup')
plt.gca().set_aspect('equal')
plt.show()


# # Tech Support

# In[45]:


plt.figure(figsize=(15,5))
pie_TechSupport_Y = pd.DataFrame(telecom_df[telecom_df['TechSupport'] == 'Yes']['Churn'].value_counts())
pie_TechSupport_Y.plot.pie(subplots = True,labels = pie_TechSupport_Y.index.values, autopct='%1.1f%%', startangle= 30)
plt.title('Tech Support')
plt.gca().set_aspect('equal')

pie_TechSupport_N = pd.DataFrame(telecom_df[telecom_df['TechSupport'] == 'No']['Churn'].value_counts())
pie_TechSupport_N.plot.pie(subplots = True, labels = pie_TechSupport_N.index.values, autopct='%1.1f%%', startangle= 75)
plt.title('Not opted for Tech Support')

plt.gca().set_aspect('equal')
plt.show() 


# # Streaming Tv and Movies doesn't make such impact on churning.

# # Model Building
# 
# 
# DATA PREPRATION

# In[46]:


# List of variables to map

varlist =  ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
telecom_df[varlist] = telecom_df[varlist].apply(binary_map)


# In[47]:


telecom_df.head()


# # For categorical variables with multiple levels, create dummy features (one-hot encoded)

# In[48]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(telecom_df[['Contract', 'PaymentMethod', 'gender', 'InternetService']], drop_first=True)

# Adding the results to the master dataframe
telecom_df = pd.concat([telecom_df, dummy1], axis=1)


# In[49]:


telecom_df.head()


# In[50]:


# Creating dummy variables for the remaining categorical variables and dropping the level with big names.

# Creating dummy variables for the variable 'MultipleLines'
ml = pd.get_dummies(telecom_df['MultipleLines'], prefix='MultipleLines')
# Dropping MultipleLines_No phone service column
ml1 = ml.drop(['MultipleLines_No phone service'], 1)
#Adding the results to the master dataframe
telecom_df = pd.concat([telecom_df,ml1], axis=1)

# Creating dummy variables for the variable 'OnlineSecurity'.
os = pd.get_dummies(telecom_df['OnlineSecurity'], prefix='OnlineSecurity')
os1 = os.drop(['OnlineSecurity_No internet service'], 1)
# Adding the results to the master dataframe
telecom_df = pd.concat([telecom_df,os1], axis=1)

# Creating dummy variables for the variable 'OnlineBackup'.
ob = pd.get_dummies(telecom_df['OnlineBackup'], prefix='OnlineBackup')
ob1 = ob.drop(['OnlineBackup_No internet service'], 1)
# Adding the results to the master dataframe
telecom_df = pd.concat([telecom_df,ob1], axis=1)

# Creating dummy variables for the variable 'DeviceProtection'. 
dp = pd.get_dummies(telecom_df['DeviceProtection'], prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'], 1)
# Adding the results to the master dataframe
telecom_df = pd.concat([telecom_df,dp1], axis=1)

# Creating dummy variables for the variable 'TechSupport'. 
ts = pd.get_dummies(telecom_df['TechSupport'], prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'], 1)
# Adding the results to the master dataframe
telecom_df = pd.concat([telecom_df,ts1], axis=1)

# Creating dummy variables for the variable 'StreamingTV'.
st =pd.get_dummies(telecom_df['StreamingTV'], prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'], 1)
# Adding the results to the master dataframe
telecom_df = pd.concat([telecom_df,st1], axis=1)

# Creating dummy variables for the variable 'StreamingMovies'. 
sm = pd.get_dummies(telecom_df['StreamingMovies'], prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'], 1)
# Adding the results to the master dataframe
telecom_df = pd.concat([telecom_df,sm1], axis=1)


# In[51]:


telecom_df.head()


# In[52]:


# We have created dummies for the below variables, so we can drop them
telecom_df = telecom_df.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)


# In[53]:


telecom_df.info()


# In[54]:


# Checking for outliers in the continuous variables
num_telecom = telecom_df[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]


# In[55]:


# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
num_telecom.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# In[56]:


# Adding up the missing values (column-wise)
telecom_df.isnull().sum()


# In[57]:


# Checking the percentage of missing values
round(100*(telecom_df.isnull().sum()/len(telecom_df.index)), 2)


# In[58]:


# Removing NaN TotalCharges rows
telecom_df = telecom_df[~np.isnan(telecom_df['TotalCharges'])]


# In[59]:


# Checking percentage of missing values after removing the missing values
round(100*(telecom_df.isnull().sum()/len(telecom_df.index)), 2)


# In[60]:


from sklearn.model_selection import train_test_split


# In[62]:


# Putting feature variable to X
X = telecom_df.drop(['Churn','customerID'], axis=1)

X.head()


# In[63]:


# Putting response variable to y
y = telecom_df['Churn']

y.head()


# In[64]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[65]:


from sklearn.preprocessing import StandardScaler


# In[66]:


scaler = StandardScaler()

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()


# In[67]:


### Checking the Churn Rate
churn = (sum(telecom_df['Churn'])/len(telecom_df['Churn'].index))*100
churn


# In[68]:


# Let's see the correlation matrix 
plt.figure(figsize = (30,15))        # Size of the figure
sns.heatmap(telecom_df.corr(),annot = True)
plt.show()


# In[69]:


X_test = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                       'StreamingTV_No','StreamingMovies_No'], 1)
X_train = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                         'StreamingTV_No','StreamingMovies_No'], 1)


# In[70]:


plt.figure(figsize = (20,10))
sns.heatmap(X_train.corr(),annot = True)
plt.show()


# In[71]:


jovian.commit


# In[ ]:




