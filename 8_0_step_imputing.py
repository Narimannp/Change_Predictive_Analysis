# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:03:10 2023

@author: narim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
def read_df():
    #Read external datasets, 1-ch_orders,2-Canadacities

    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\7_data_prep_ch_orders.csv')
    return(ch_orders)
ch_orders=read_df()


def pearson_spearman(df):
    df=df[['ProjectBaseContractValue','DailyCost', 'TotalChPer', 'PrimeChPer',
           'Duration','ChangeDuration','DurationDiff_Divided_Duration']]
    for column in df:
       #Range normalizing attributes
       df[column]=(df[column]-df[column].min())/(df[column].max()-df[column].min())
    pearson=df.corr(method='pearson', min_periods=1)
    spearman=df.corr(method='spearman', min_periods=1)
    return(pearson,spearman)
def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

def duration_modification(ch_orders):
    ch_orders["DurationModified"]=abs(ch_orders["Duration"])
    ch_orders["DurationModified"]=np.where(ch_orders["DurationModified"]==0,np.nan,ch_orders["DurationModified"])   

    ch_orders["DurationModified"]=np.where(ch_orders["DurationDiff_Divided_Duration"]<-25,np.nan,ch_orders["DurationModified"])  
    ch_orders["DurationModified"]=np.where(((ch_orders["DailyCost"]>150000 )&(ch_orders["Duration"]<5)),np.nan,ch_orders["DurationModified"])      
    return(ch_orders)

def frequency_normalization(ch_orders):
    ch_orders["freq_p_prime_dur"]=ch_orders["PrimeChFreq_p"]/ch_orders["DurationModified"]
    ch_orders["freq_p_prime_sze"]=ch_orders["PrimeChFreq_p"]*1000/ch_orders["ProjectBaseContractValue"]   
    
    ch_orders["freq_n_prime_dur"]=ch_orders["PrimeChFreq_n"]/ch_orders["DurationModified"]
    ch_orders["freq_n_prime_sze"]=ch_orders["PrimeChFreq_n"]*1000/ch_orders["ProjectBaseContractValue"]
    return(ch_orders)
  
def ch_freq(ch_orders):
    projecttypes=["Construction","Services"]
    ch_df=pd.DataFrame()
    for projecttype in projecttypes:
        df=ch_orders[ch_orders["ProjectType"]==projecttype]
        dict_1_p_d=df.groupby("Classification_1")["freq_p_prime_dur"].mean()
        dict_1_p_s=df.groupby("Classification_1")["freq_p_prime_sze"].mean()    
        dict_1_n_d=df.groupby("Classification_1")["freq_n_prime_dur"].mean()
        dict_1_n_s=df.groupby("Classification_1")["freq_n_prime_sze"].mean()
        dict_2_p_d=df.groupby("Classification_2")["freq_p_prime_dur"].mean()
        dict_2_p_s=df.groupby("Classification_2")["freq_p_prime_sze"].mean()
        dict_2_n_d=df.groupby("Classification_2")["freq_n_prime_dur"].mean()
        dict_2_n_s=df.groupby("Classification_2")["freq_n_prime_sze"].mean()
        df["Freq_Class1_p_dur"]=df["Classification_1"].map(dict_1_p_d)
        df["Freq_Class1_p_sze"]=df["Classification_1"].map(dict_1_p_s)
        df["Freq_Class1_n_dur"]=df["Classification_1"].map(dict_1_n_d)
        df["Freq_Class1_n_sze"]=df["Classification_1"].map(dict_1_n_s)
        df["Freq_Class2_p_dur"]=df["Classification_2"].map(dict_2_p_d)
        df["Freq_Class2_p_sze"]=df["Classification_2"].map(dict_2_p_s)
        df["Freq_Class2_n_dur"]=df["Classification_2"].map(dict_2_n_d)
        df["Freq_Class2_n_sze"]=df["Classification_2"].map(dict_2_n_s)
        dict_prv_p_d=df.groupby("ProjectProvince")["freq_p_prime_dur"].mean()
        dict_prv_p_s=df.groupby("ProjectProvince")["freq_p_prime_sze"].mean()
        dict_prv_n_d=df.groupby("ProjectProvince")["freq_n_prime_dur"].mean()
        dict_prv_n_s=df.groupby("ProjectProvince")["freq_n_prime_sze"].mean()
        df["Freq_Prov_p_dur"]=df["ProjectProvince"].map(dict_prv_p_d)  
        df["Freq_Prov_p_sze"]=df["ProjectProvince"].map(dict_prv_p_s)  
        df["Freq_Prov_n_dur"]=df["ProjectProvince"].map(dict_prv_n_d)   
        df["Freq_Prov_n_sze"]=df["ProjectProvince"].map(dict_prv_n_s) 
        dict_cty_p_d=df.groupby("ProjectCity")["freq_p_prime_dur"].mean()
        dict_cty_p_s=df.groupby("ProjectCity")["freq_p_prime_sze"].mean()
        dict_cty_n_d=df.groupby("ProjectCity")["freq_n_prime_dur"].mean()
        dict_cty_n_s=df.groupby("ProjectCity")["freq_n_prime_sze"].mean()
        df["Freq_City_p_dur"]=df["ProjectCity"].map(dict_cty_p_d)  
        df["Freq_City_p_sze"]=df["ProjectCity"].map(dict_cty_p_s) 
        df["Freq_City_n_dur"]=df["ProjectCity"].map(dict_cty_n_d) 
        df["Freq_City_n_sze"]=df["ProjectCity"].map(dict_cty_n_s)
        ch_df=pd.concat([df,ch_df],axis=0)
    
    return(ch_df)    
ch_orders=duration_modification(ch_orders)
ch_orders=ch_orders.set_index("ProjectId")
ch_orders_for_impute=ch_orders[["DurationModified","ProjectBaseContractValue","Classification_1","Classification_2","ProjectClassification","ProjectOperatingUnit","ProjectType","ProjectProvince","ProjectCity","Population","Density","ProjectBillingType"]]
ch_orders_for_impute=pd.get_dummies(ch_orders_for_impute,columns=["Classification_1","Classification_2","ProjectClassification","ProjectOperatingUnit","ProjectType","ProjectProvince","ProjectCity","ProjectBillingType"],drop_first=True)
# Divide the data into two parts: one with missing values and one without
data_missing = ch_orders_for_impute[ch_orders_for_impute["DurationModified"].isna()]
data_not_missing = ch_orders_for_impute[~ch_orders_for_impute["DurationModified"].isna()]



# # Split the data into target variable and features
X_not_missing = data_not_missing.drop(columns=["DurationModified"])
y_not_missing = data_not_missing["DurationModified"]
X_missing = data_missing.drop(columns=["DurationModified"])
x_train,x_test,y_train,y_test=train_test_split(X_not_missing,y_not_missing,test_size=.1,random_state=0)
sc=StandardScaler().fit(x_train)
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)
X_missing_std=sc.transform(X_missing)
# # Train a linear regression model on the data without missing values
# reg = LinearRegression().fit(x_train_std, y_train)
reg=SVR(kernel="poly")
params={"C":(0.1,0.01),"degree":(1,2),"gamma":(1,0.1)}
svm_grid=GridSearchCV(reg,params,n_jobs=-1,cv=5,verbose=-1,scoring="r2")
svm_grid.fit(x_train_std,y_train)
best_params=svm_grid.best_params_
svm_clf=svm_grid.best_estimator_
print("Best F1 score: ", svm_grid.best_score_)
# # Use the trained model to predict the missing values
y_test_pred=svm_clf.predict(x_test_std)
y_train_pred=svm_clf.predict(x_train_std)

r_square_test=r2_score(y_test,y_test_pred)
r_square_train=r2_score(y_train,y_train_pred)

# imputed_values = reg.predict(X_missing)
imputed_values = svm_clf.predict(X_missing_std)
# # Replace the missing values in the original data with the imputed values
# for col, imputed_col in zip(missing_cols, imputed_values.T):
#     data_missing[col] = imputed_col
data_missing["DurationModified"]=imputed_values
dictionary=dict(zip(data_missing.index,data_missing["DurationModified"]))
ch_orders["DurationModified"]=np.where(ch_orders["DurationModified"].isna(),ch_orders.index.map(dictionary),ch_orders["DurationModified"])

ch_orders=frequency_normalization(ch_orders)
ch_orders=ch_freq(ch_orders)

ch_orders.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
# # Combine the two parts of the data to get the final data with no missing values
# final_data = pd.concat([data_missing, data_not_missing]).sort_index()
# print(ch_orders.info())

ch_orders=ch_orders[ch_orders["ProjectType"]=="Construction"]
ch_orders["TotalChFreq"]=ch_orders["PrimeChFreq_p"]+ch_orders["PrimeChFreq_n"]
ch_orders=ch_orders[ch_orders["TotalChFreq"]>25]
# ch_orders=outlier_remove(ch_orders, ["TotalChFreq"])
plt.hist(ch_orders["TotalChFreq"], bins='auto', alpha=0.7, rwidth=0.85, density=True)

# Adding labels and title
plt.xlabel('Number of Change Orders')
plt.ylabel('Percentage Of Projects')
plt.title('Distribution of projects by number of COs')

# Displaying the histogram
plt.show()