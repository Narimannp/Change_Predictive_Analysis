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
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from step_9_0_FeatureSelection_PCA import atr_list
from step_9_0_FeatureSelection_ChiSquare import top_five_cat_atrs
print(tf.__version__)

def read_df():
    #Read external datasets, 1-Projects,2-Canadacities
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    # ch_orders=ch_orders[['ProjectId', 'ProjectBaseContractValue', 'ProjectProvince',
    #         'ProjectCity', 'Population', 'Density',
    #          'ProjectBillingType','ProjectOperatingUnit', \
    #         'ProjectType','DurationModified', 'TotalChPer', 'PrimeChPer', 'CommitChPer', 'SalesChPer',\
    #                 "ProjectClassification","Classification_1","Classification_2",\
    #                 "Freq_Class1_p_dur","Freq_Class1_p_sze","Freq_Class1_n_dur","Freq_Class1_n_sze","Freq_Class2_p_dur","Freq_Class2_p_sze",\
    #                 "Freq_Class2_n_dur","Freq_Class2_n_sze","Freq_Prov_p_dur","Freq_Prov_p_sze",\
    #                     "Freq_Prov_n_dur","Freq_Prov_n_sze","Freq_City_p_dur","Freq_City_p_sze","Freq_City_n_dur"\
    #                         ,"Freq_City_n_sze"]]

    ch_orders.rename(columns={"ProjectBaseContractValue":"BaseValue","ProjectOperatingUnit":"OperatingUnit",\
                               "ProjectBillingType":"BillType","ProjectCity":"City","ProjectProvince":"Province"},inplace=True)
    # ch_orders.drop(columns=["ProjectCity","DailyCost","ChangeDuration","TotalChFreq","PrimeChFreq","CommitChFreq",\""
    #                  "CommitChFreq","SalesChFreq","TotalChPer","PrimeChPer","CommitChPer","SalesChPer"],axis=1,inplace=True)
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)

def label_target_atr(ch_orders,boundry,existance_or_lvl,prime_commit):
    target_atr=prime_commit+"Ch"+existance_or_lvl
    if existance_or_lvl=="Lvl":
        ch_orders=ch_orders[ch_orders[prime_commit+"ChPer"]!=0]
        print("Change Level Classifiction")
        ch_orders[target_atr]=np.where(ch_orders[prime_commit+"ChPer"]<=boundry[0],0,1)
        ch_orders[target_atr]=np.where(((ch_orders[prime_commit+"ChPer"]<=boundry[1]) & (ch_orders["PrimeChPer"]>boundry[0])),1,ch_orders[target_atr])
        ch_orders[target_atr]=np.where((ch_orders[prime_commit+"ChPer"]>boundry[1]),2,ch_orders[target_atr])

    elif existance_or_lvl=="Existance":
        print("Change Existance Classifiction")
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]>boundry)),1,0)
    else:
        print("Wrong Input...")
        
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

def classification_prep(ch_orders,x,y,test_size):
    a=x.dtypes=="object"
    categorical_atr_list=list(a.loc[x.dtypes=="object"].index)
    x=pd.get_dummies(x,columns=categorical_atr_list,drop_first=True)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=1)
    sc=StandardScaler().fit(x_train)
    x_train_str=pd.DataFrame(data=sc.transform(x_train),index=x_train.index,columns=x_train.columns)
    x_test_str=pd.DataFrame(data=sc.transform(x_test),index=x_test.index,columns=x_test.columns)
    return(x_train,x_train_str,x_test_str,y_train,y_test)

#Takes DF and list of  attribute types to remove, and returns the DF with desired remaning attributes
def select_atrs(df,atr_types):
    loc_columns=[]
    temp_loc_columns=[]
    target_atrs=['PrimeChPer']
    orig_atrs="BaseValue,OperatingUnit,DurationModified,BillType,ProjectClassification,Classification_1,Classification_2,City,Province,ProjectType".split(",")
    loc_keys="City,Prov".split(",")
    loc_atrs_add="Population,Density".split(",")
    columns=df.columns
    columns=columns.astype(str).tolist()
    Freq_atrs=[x for x in columns if "Freq" in x ]
    for loc_key in loc_keys: 
        temp_loc_columns=[x for x in columns if loc_key in x ]
        loc_columns=loc_columns+temp_loc_columns
    df=df[target_atrs+orig_atrs+Freq_atrs]
    return(df,loc_columns,Freq_atrs)
    
def run_the_code(list_boundries,ch_existance_or_lvl,prime_or_commit,construction_or_service,include_categorical):
    ch_orders=read_df()
    ch_orders,loc_atrs,Freq_atrs=select_atrs(ch_orders,"")
    ch_orders=project_filter(ch_orders,"ProjectType",construction_or_service)
    ch_orders=label_target_atr(ch_orders,list_boundries,ch_existance_or_lvl,prime_or_commit)
    ch_orders=outlier_remove(ch_orders,["PrimeChPer"])
    x,y=split_x_y(ch_orders,prime_or_commit,ch_existance_or_lvl) 
    
    if include_categorical==True:
        columns=ch_orders.columns
        atrs=[atr for atr in atr_list if atr in columns]
        x=x[atrs]
        RunName="Model Accuracy-Numerical+"+str(len(top_five_cat_atrs))+"Categorical Attributes"
    else:
        x=x[atr_list]
        RunName="Model Accuracy-NumericalAttributes"
    x_train,x_train_str,x_test_str,y_train,y_test=classification_prep(ch_orders,x,y,.15)
    return(ch_orders,columns,x_train_str,x_test_str,y_train,y_test,RunName,loc_atrs,Freq_atrs)

ch_orders,columns,x_train_str,x_test_str,y_train,y_test,RunName,loc_atrs,Freq_atrs=run_the_code(4,"Existance","Prime","Construction", True)
input_size=(len(x_train_str.columns))
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=input_size,activation="relu"))
ann.add(tf.keras.layers.Dense(units=4,activation="sigmoid"))
ann.add(tf.keras.layers.Dense(units=4,activation="sigmoid"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
# def lr_scheduler(epoch, lr):
#     if epoch < 3:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)
initial_learning_rate = 0.01
# optimizer = SGD(learning_rate=initial_learning_rate)
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

history=ann.fit(x_train_str,y_train,validation_data=(x_test_str, y_test),batch_size=8,epochs=45)
# print(atr_list)
y_pred_test=ann.predict(x_test_str)
y_pred_test=y_pred_test>0.5
y_pred_train=ann.predict(x_train_str)
y_pred_train=y_pred_train>0.5
#creating the distrubution of data variable
dist=y_train.value_counts()
#performance measument variables
confusion_test=confusion_matrix(y_test,y_pred_test)
confusion_train=confusion_matrix(y_train,y_pred_train)
accuracy_test=accuracy_score(y_test,y_pred_test)
accuracy_train=accuracy_score(y_train,y_pred_train)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(RunName)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
