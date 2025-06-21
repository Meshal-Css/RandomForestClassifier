#!/usr/bin/env python
# coding: utf-8

# In[155]:


import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


# In[94]:


data = pd.read_csv('cardekho_data.csv')


# In[95]:


data.head(10)


# In[96]:


data.info()


# In[97]:


#cheack is null value 
data.isnull().sum()


# In[98]:


# cheack duplicated value 
data.duplicated().sum()


# In[99]:


# drop duplicated Value
data  = data.drop_duplicates()


# In[100]:


data


# In[101]:


# drop column car_name 
data = data.drop(['Car_Name'],axis = 1)


# In[102]:


# tabular table 
data_tabular = data.sort_values(by = 'Year',ascending = False)


# In[103]:


data_tabular.head(10)


# In[104]:


data_tabular.describe()


# In[105]:


# What is the difference between columns Selling_Price ,Present_Price ?
data_tabular['Price_Difference'] = data_tabular['Present_Price'] - data_tabular['Selling_Price']


# In[106]:


data_tabular


# In[107]:


print(data_tabular[['Present_Price', 'Selling_Price', 'Price_Difference']].head())


# In[108]:


print(data_tabular['Fuel_Type'].unique())


# In[109]:


fuel_price_diff = data_tabular.groupby('Fuel_Type')['Price_Difference'].mean().reset_index()


# In[110]:


fuel_price_diff.rename(columns={'Price_Difference': 'Average_Price_Difference'}, inplace=True)


# In[111]:


print(fuel_price_diff)


# In[112]:


plt.figure(figsize=(8, 5))
plt.bar(fuel_price_diff['Fuel_Type'], fuel_price_diff['Average_Price_Difference'], color=['skyblue', 'orange', 'green'])

plt.title('Average Price Difference by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Average Price Difference')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#  يعني أن السيارات العاملة بالديزل تفقد قيمتها بشكل أكبر أو أنها أغلى عند الشراء وتنخفض أكثر عند البيع

# In[113]:


data_tabular


# In[114]:


# When is the price difference affected for car models?
Year_difference= data_tabular.groupby('Year')['Price_Difference'].mean().reset_index()


# In[115]:


print(Year_difference)


# In[116]:


import seaborn as sns

plt.figure(figsize=(10,6))

 
sns.regplot(x='Year', y='Price_Difference', data=data_tabular, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

plt.title('Price Difference vs. Car Model Year')
plt.xlabel('Car Model Year')
plt.ylabel('Price Difference')
plt.grid(True)
plt.show()


# In[117]:


#Seller_Type indicates whether the car was sold by a dealer or an individual.
print(data_tabular['Seller_Type'].unique())


# In[118]:


print(data_tabular['Owner'].unique())


# In[119]:


print(data_tabular['Transmission'].unique())


# In[120]:


# Average price difference with vehicle carrier type?
avg_price_by_transmission = data_tabular.groupby('Transmission')['Price_Difference'].mean().reset_index()

plt.figure(figsize=(8,5))
plt.bar(avg_price_by_transmission['Transmission'], avg_price_by_transmission['Price_Difference'], color=['skyblue', 'salmon'])

plt.title('Average Price Difference by Transmission Type')
plt.xlabel('Transmission Type')
plt.ylabel('Average Price Difference')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[121]:


print(avg_price_by_transmission)


# In[122]:


data_tabular


# In[123]:


data_tabular['Transmission'].value_counts()


# In[124]:


data_tabular['Fuel_Type'].value_counts()# CNG


# In[125]:


data_tabular['Seller_Type'].value_counts()


# In[126]:


data_tabular = data_tabular[data_tabular['Fuel_Type'] != 'CNG']


# In[127]:


data_tabular['Fuel_Type'].value_counts()#  after drop CNG


# In[128]:


data_tabular


# In[129]:


# chouse youer coulmns 
data_Kms_Price = data_tabular[['Kms_Driven', 'Price_Difference']]


# In[130]:


X = data_Kms_Price['Kms_Driven']
y = data_Kms_Price['Price_Difference']
X = sm.add_constant(X) 
model = sm.OLS(y, X).fit()


# In[131]:


print(model.summary())


# In[132]:


plt.scatter(data_Kms_Price['Kms_Driven'], data_Kms_Price['Price_Difference'], color='blue')
plt.plot(data_Kms_Price['Kms_Driven'], model.predict(X), color='red')
plt.title('Kms Driven vs Price Difference')
plt.xlabel('Kms Driven')
plt.ylabel('Price Difference')
plt.show()


الرسمة تظهر علاقة إيجابية طفيفة بين "الكيلومترات المدفوعة" و"فارق السعر"، حيث تزداد قيمة فارق السعر مع زيادة الكيلومترات. ومع ذلك، تتجمع معظم النقاط في الجزء السفلي، مما يشير إلى أن السيارات ذات الكيلومترات المنخفضة تميل إلى أن يكون لديها فارق سعر أقل.

# preprocessing 
# In[133]:


# cheack youer data 
data_tabular


# In[134]:


data_tabular.info()


# In[135]:


# outliers
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

for column in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data[column])
    plt.title(f'Box Plot for {column}')
    plt.show()


# In[138]:


data_tabular


# In[139]:


# creata target 
data_tabular['Price_Drop_High'] = data_tabular['Price_Difference'].apply(lambda x: 1 if x > 2.0 else 0)


# In[140]:


data_tabular


# In[142]:


data_tabular['Price_Drop_High'].value_counts()


# In[145]:


#label_encoder

label_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
le = LabelEncoder()
for col in label_cols:
    data_tabular[col] = le.fit_transform(data_tabular[col])


# In[148]:


numeric_cols = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']

scaler = StandardScaler()
data_tabular[numeric_cols] = scaler.fit_transform(data_tabular[numeric_cols])


# In[149]:


data_tabular


# In[150]:


X = data_tabular[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type',
                  'Seller_Type', 'Transmission', 'Owner']]
y = data_tabular['Price_Drop_High']


# In[152]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[153]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[154]:


accuracy = model.score(X_test, y_test)
print("دقة النموذج:", accuracy)


# In[156]:


y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# In[157]:


y_proba = model.predict_proba(X_test)[:,1]  
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


# the end

