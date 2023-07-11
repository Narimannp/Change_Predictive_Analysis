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
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from step_9_0_FeatureSelection_PCA import atr_list
from step_9_0_FeatureSelection_ChiSquare import sorted_cat_atrs as sorted_cat_atrs
print(tf.__version__)

"Reads datasets with require attributes"
def read_df():
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    ch_orders.rename(columns={"ProjectBaseContractValue":"BaseValue","ProjectOperatingUnit":"OperatingUnit",\
                                   "ProjectBillingType":"BillType","ProjectCity":"City","ProjectProvince":"Province"},inplace=True)
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)

"Given the Df and types of attributes necessary, return the df with required attribute types"
def select_atrs(df,atr_types):
    loc_columns=[]
    temp_loc_columns=[]
    target_atrs=['PrimeChPer']
    orig_cat_atrs="OperatingUnit,BillType,ProjectClassification,Classification_1,Classification_2,City,Province,ProjectType".split(",")
    orig_num_atrs="BaseValue,DurationModified".split(",")
    loc_keys="City,Prov".split(",")
    loc_atrs_add="Population,Density".split(",")
    columns=df.columns
    columns=columns.astype(str).tolist()
    Freq_atrs=[x for x in columns if "Frq" in x ]
    for loc_key in loc_keys: 
        temp_loc_columns=[x for x in columns if loc_key in x ]
        loc_columns=loc_columns+temp_loc_columns
    atrs=target_atrs+orig_cat_atrs+orig_num_atrs+Freq_atrs+loc_atrs_add+loc_columns

    
    if (("freq" not in atr_types )and( "orig_cat" not in atr_types)) or (("freq" in atr_types )and( "orig_cat" in atr_types)):
        print("both")
        RunName="both"
        # raise ValueError("1 NOT BOTH or NONE of Orig-cat and Freq attributes should be presented")
    elif "orig_cat" in atr_types:
        df=df[[x for x in atrs if x not in Freq_atrs]]
        RunName="ATRs=Orignal"
    else:
        df=df[[x for x in atrs if x not in orig_cat_atrs]]
        RunName="ATRs=Freq"
    if "loc_add" not in atr_types:
        columns=df.columns.astype(str).tolist()
        df=df[[x for x in columns if x not in loc_atrs_add]]
        RunName=RunName+",No new-location atrs"
    if "loc_atrs" not in atr_types:
        columns=df.columns.astype(str).tolist()
        df=df[[x for x in columns if x not in loc_columns]]
        RunName=RunName+" ,No old-location atrs"
    return(df,RunName)

"Given the DF and boundry or list of boundries labels the target atrribute"
def label_target_atr(ch_orders,boundry):
    if type(boundry)==list:
        target_atr="PrimeChMulti-Class"
        print("Change Level Classifiction")
        ch_orders[target_atr]=np.where(ch_orders["PrimeChPer"]<=boundry[0],0,1)
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]<=boundry[1]) & (ch_orders["PrimeChPer"]>boundry[0])),1,ch_orders[target_atr])
        ch_orders[target_atr]=np.where((ch_orders["PrimeChPer"]>boundry[1]),2,ch_orders[target_atr])
    else: 
        target_atr="PrimeChBinary"
        print("Change Existance Classifiction")
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]>boundry)),1,0)
    return(ch_orders)
        

def drop_atr(ch_orders,atrs_list):
    ch_orders.drop(columns=atrs_list,axis=1,inplace=True)
    return(ch_orders)

def project_filter(ch_orders,atr,atr_class_list):
    ch_orders=ch_orders[ch_orders[atr]==(atr_class_list)]
    return(ch_orders)

def split_x_y(ch_orders):
    columns=ch_orders.columns
    target=[x for x in columns if x in ["PrimeChBinary","PrimeChMulti-Class"]]
    y=ch_orders[target]
    target_classification_list="PrimeChBinary,PrimeChMulti-Class,PrimeChPer,CommitChPer,SalesChPer,TotalChPer".split(",")
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
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=0)
    sc=StandardScaler().fit(x_train)
    x_train_str=pd.DataFrame(data=sc.transform(x_train),index=x_train.index,columns=x_train.columns)
    x_test_str=pd.DataFrame(data=sc.transform(x_test),index=x_test.index,columns=x_test.columns)
    return(x_train,x_train_str,x_test_str,y_train,y_test)

def ANN(x_train_str,y_train,x_test_str, y_test,num_epocs):
    input_size=(len(x_train_str.columns))
    ann=tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=input_size,activation="relu"))
    ann.add(tf.keras.layers.Dense(units=4,activation="sigmoid"))
    ann.add(tf.keras.layers.Dense(units=4,activation="sigmoid"))
    ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
    initial_learning_rate=.002
    def lr_scheduler(epoch, initial_learning_rate):
        if epoch < 100:
            return initial_learning_rate
        else:
            return initial_learning_rate*tf.math.exp(0.005)


    optimizer = Adam(learning_rate=initial_learning_rate)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    ann.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"],)
    
    "Creating dictionary for class weights"
    class_distribution=y_test.value_counts()
    weights=round(1/(class_distribution/sum(class_distribution)),3)
    class_indexes=weights.index
    dict_keys=[x[0] for x in class_indexes]
    weight_dictionary=dict(zip(dict_keys,weights))
    history=ann.fit(x_train_str,y_train,validation_data=(x_test_str, y_test),batch_size=16,epochs=num_epocs,class_weight=weight_dictionary,callbacks=[lr_callback])
    
    y_pred_test=ann.predict(x_test_str)
    y_pred_test=y_pred_test>0.5
    y_pred_train=ann.predict(x_train_str)
    y_pred_train=y_pred_train>0.5
    #creating the distrubution of data variable

    #performance measument variables
    confusion_test=confusion_matrix(y_test,y_pred_test)
    confusion_train=confusion_matrix(y_train,y_pred_train)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    accuracy_train=accuracy_score(y_train,y_pred_train)
    return(confusion_test,confusion_train,accuracy_test,accuracy_train,history)

def visulise(history,RunName):
    "summarize history for accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(RunName)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    "summarize history for loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def run_the_code(list_boundries,feature_eng,atr_types,epocs):
    ch_orders=read_df()
    ch_orders=project_filter(ch_orders,"ProjectType","Construction")
    ch_orders,RunName=select_atrs(ch_orders,atr_types)
    ch_orders=label_target_atr(ch_orders,list_boundries)
    ch_orders=outlier_remove(ch_orders,["PrimeChPer"])
    x,y=split_x_y(ch_orders) 
    if feature_eng==True:
        columns=ch_orders.columns
        atrs=[atr for atr in atr_list if atr in columns]
        cat_atrs=[atr for atr in sorted_cat_atrs if atr in columns]
        x=x[atrs+cat_atrs]
        RunName=RunName+"\n With FeatureEngineering"
    else:
        RunName=RunName+"\n Without FeatureEngineering"
    #     columns=ch_orders.columns
    #     x=x[[x for x in atr_list if x in columns]]
    #     RunName="Model Accuracy-NumericalAttributes"
    x_train,x_train_str,x_test_str,y_train,y_test=classification_prep(ch_orders,x,y,.25)
    confusion_test,confusion_train,accuracy_test,accuracy_train,history=ANN(x_train_str,y_train,x_test_str, y_test,epocs)
    visulise(history,RunName)
    return(ch_orders,confusion_test,confusion_train,x_train_str,y_train,y_test,RunName)


ch_orders,confusion_test,confusion_train,x_train_str,y_train,y_test,RunName=\
    run_the_code(list_boundries=4,feature_eng=True,atr_types=["freq"],epocs=200)
