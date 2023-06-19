# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:38:10 2023

@author: narim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
print(tf.__version__)

def read_df():
    #Read external datasets, 1-Projects,2-Canadacities
    ch_orders_orig=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    ch_orders=ch_orders_orig[['ProjectId', 'ProjectBaseContractValue', 'ProjectProvince',
            'ProjectCity', 'Population', 'Density',
             'ProjectBillingType','ProjectOperatingUnit', \
            'ProjectType','DurationModified', 'TotalChPer', 'PrimeChPer', 'CommitChPer', 'SalesChPer',\
                    "Classification_1","Classification_2",\
                    "Freq_Class1_p_dur","Freq_Class1_p_sze","Freq_Class1_n_dur","Freq_Class1_n_sze","Freq_Class2_p_dur","Freq_Class2_p_sze",\
                    "Freq_Class2_n_dur","Freq_Class2_n_sze","Freq_Prov_p_dur","Freq_Prov_p_sze",\
                        "Freq_Prov_n_dur","Freq_Prov_n_sze","Freq_City_p_dur","Freq_City_p_sze","Freq_City_n_dur"\
                            ,"Freq_City_n_sze"]]
    ch_orders.rename(columns={"ProjectBaseContractValue":"BaseValue","ProjectOperatingUnit":"OperatingUnit","DurationModified":"Duration",\
                               "ProjectBillingType":"BillType","ProjectCity":"City","ProjectProvince":"Province"},inplace=True)
    # ch_orders.drop(columns=["ProjectCity","DailyCost","ChangeDuration","TotalChFreq","PrimeChFreq","CommitChFreq",\""
    #                  "CommitChFreq","SalesChFreq","TotalChPer","PrimeChPer","CommitChPer","SalesChPer"],axis=1,inplace=True)
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)

def label_target_atr(ch_orders,boundry,existance_or_lvl,prime_commit):
    target_atr=prime_commit+"Ch"+existance_or_lvl
    if existance_or_lvl=="Lvl":
        # ch_orders=ch_orders[ch_orders[prime_commit+"ChPer"]!=0]
        print("Change Level Classifiction")
        ch_orders[target_atr]=np.where(ch_orders[prime_commit+"ChPer"]<=boundry[0],0,1)
        ch_orders[target_atr]=np.where(((ch_orders[prime_commit+"ChPer"]<=boundry[1]) & (ch_orders["PrimeChPer"]>boundry[0])),1,ch_orders[target_atr])
        ch_orders[target_atr]=np.where((ch_orders[prime_commit+"ChPer"]>boundry[1]),2,ch_orders[target_atr])

    elif existance_or_lvl=="Existance":
        print("Change Existance Classifiction")
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]==0.0)),1,0)
    else:
        print("Wrong Input...")
        
    return(ch_orders)
        
    return(ch_orders)
def drop_atr(ch_orders,atr_list):
    ch_orders.drop(columns=atr_list,axis=1,inplace=True)
    return(ch_orders)
def project_filter(ch_orders,atr,atr_class_list):
    ch_orders=ch_orders[ch_orders[atr]==(atr_class_list)]
    return(ch_orders)
def split_x_y(ch_orders,change_category,existance_or_level):
    target_atr=change_category+"Ch"+existance_or_level
    y=ch_orders[target_atr]
    target_classification_list="PrimeChExistance,CommitChExistance,TotalChExistance,SalesChExistance,PrimeChLvl,CommitChLvl,TotalChLvl,SalesChLvl,PrimeChPer,CommitChPer,SalesChPer,TotalChPer".split(",")
    x=ch_orders.loc[:,~ch_orders.columns.isin(target_classification_list)]
    return(x,y)
def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

def svm_classification_prep(ch_orders,x,y,test_size):
 
    a=x.dtypes=="object"
    categorical_atr_list=list(a.loc[x.dtypes=="object"].index)
    x=pd.get_dummies(x,columns=categorical_atr_list,drop_first=True)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=1)
    sc=StandardScaler().fit(x_train)
    x_train_str=pd.DataFrame(data=sc.transform(x_train),index=x_train.index,columns=x_train.columns)
    x_test_str=pd.DataFrame(data=sc.transform(x_test),index=x_test.index,columns=x_test.columns)
    return(x_train,x_train_str,x_test_str,y_train,y_test)

def run_the_code(list_boundries,ch_existance_or_lvl,prime_or_commit,construction_or_service):
    ch_orders=read_df()
    # ch_orders=project_filter(ch_orders,"ProjectType",construction_or_service)
    ch_orders=label_target_atr(ch_orders,list_boundries,ch_existance_or_lvl,prime_or_commit)
    ch_orders=outlier_remove(ch_orders,["PrimeChPer"])
    x,y=split_x_y(ch_orders,prime_or_commit,ch_existance_or_lvl) 
    x_train,x_train_str,x_test_str,y_train,y_test=svm_classification_prep(ch_orders,x,y,.3)
    return(ch_orders,x_train_str,x_test_str,y_train,y_test)
ch_orders,x_train_str,x_test_str,y_train,y_test=run_the_code([2,10],"Existance","Prime","Construction")
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=185,activation="relu"))
ann.add(tf.keras.layers.Dense(units=3,activation="sigmoid"))
ann.add(tf.keras.layers.Dense(units=2,activation="sigmoid"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
ann.fit(x_train_str,y_train,batch_size=32,epochs=100)
y_pred_test=ann.predict(x_test_str)
y_pred_test=y_pred_test>0.5
y_pred_train=ann.predict(x_train_str)
y_pred_train=y_pred_train>0.5
dist=y_train.value_counts()
confusion_test=confusion_matrix(y_test,y_pred_test)
confusion_train=confusion_matrix(y_train,y_pred_train)
accuracy_test=accuracy_score(y_test,y_pred_test)
accuracy_train=accuracy_score(y_train,y_pred_train)
