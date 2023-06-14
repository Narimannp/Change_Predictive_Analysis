# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:38:10 2023

@author: narim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import sklearn as slr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer

def read_df():
    #Read external datasets, 1-Projects,2-Canadacities
    ch_orders_orig=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    ch_orders=ch_orders_orig[['ProjectId', 'ProjectBaseContractValue', 'ProjectProvince',
            'ProjectCity', 'DailyCost', 'Population', 'Density',
            'ProjectClassification', 'ProjectBillingType','ProjectOperatingUnit', \
            'ProjectType', 'ChangeDuration',
            'DurationModified', 'TotalChPer', 'PrimeChPer', 'CommitChPer', 'SalesChPer',
            'PrimeChFreq_p','PrimeChFreq_n', 'CommitChFreq_p', 'SalesChFreq_p', 'CommitChFreq_n', 'SalesChFreq_n']]
    # ch_orders.drop(columns=["ProjectCity","DailyCost","ChangeDuration","TotalChFreq","PrimeChFreq","CommitChFreq",\
    #                  "CommitChFreq","SalesChFreq","TotalChPer","PrimeChPer","CommitChPer","SalesChPer"],axis=1,inplace=True)
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)
def divide_classification(ch_orders):
    ch_orders["a"]=ch_orders["ProjectClassification"].apply(lambda x:len(x.split(".")))
    ch_orders["ProjectClassification"]=np.where(ch_orders["a"]==1,ch_orders["ProjectClassification"]+". ",ch_orders["ProjectClassification"])
    # ch_orders[['Column1_1', 'Column1_2']] = ch_orders["ProjectClassification"].str.split('.', expand=True)
    ch_orders["Classification_1"]=ch_orders["ProjectClassification"].apply(lambda x:x.split(".")[0])
    ch_orders["Classification_1"]=    ch_orders["Classification_1"].replace("COMMERCIAL","COMM")
    ch_orders["Classification_2"]=np.where(ch_orders["ProjectClassification"].apply(lambda x:len(x.split(".")))>1,ch_orders["ProjectClassification"].apply(lambda x:x.split(".")[1]),"")
    # ch_orders["classification_2"]=np.where(ch_orders["classification_2"]=="",ch_orders["classification_2"],ch_orders["classification_2"].apply(lambda x:x[1]))
    # ch_orders["classification_2"]=ch_orders["ProjectClassification"].apply(lambda x:len(x.split(".")))
    ch_orders.drop(columns=["ProjectClassification","a"],axis=1,inplace=True)
    return(ch_orders)

def ch_freq(ch_orders):
    dict_1_p=ch_orders.groupby("Classification_1")["PrimeChFreq_p"].mean()
    dict_1_n=ch_orders.groupby("Classification_1")["PrimeChFreq_n"].mean()
    dict_2_p=ch_orders.groupby("Classification_2")["PrimeChFreq_p"].mean()
    dict_2_n=ch_orders.groupby("Classification_2")["PrimeChFreq_n"].mean()
    ch_orders["Freq_Class_1_p"]=ch_orders["Classification_1"].map(dict_1_p)
    ch_orders["Freq_Class_1_n"]=ch_orders["Classification_1"].map(dict_1_n)
    ch_orders["Freq_Class_2_p"]=ch_orders["Classification_2"].map(dict_2_p)
    ch_orders["Freq_Class_2_n"]=ch_orders["Classification_2"].map(dict_2_n)
    dict_prv_p=ch_orders.groupby("ProjectProvince")["PrimeChFreq_p"].mean()
    dict_prv_n=ch_orders.groupby("ProjectProvince")["PrimeChFreq_n"].mean()
    ch_orders["Freq_Prov_p"]=ch_orders["ProjectProvince"].map(dict_prv_p)  
    ch_orders["Freq_Prov_n"]=ch_orders["ProjectProvince"].map(dict_prv_n)   
    dict_cty_p=ch_orders.groupby("ProjectCity")["PrimeChFreq_p"].mean()
    dict_cty_n=ch_orders.groupby("ProjectCity")["PrimeChFreq_n"].mean()
    ch_orders["Freq_p_City"]=ch_orders["ProjectCity"].map(dict_cty_p)  
    ch_orders["Freq_n_City"]=ch_orders["ProjectCity"].map(dict_cty_n) 
    return(ch_orders)

def label_target_atr(ch_orders,low_lvl,high_lvl,existance_or_lvl,prime_commit):
    target_atr=prime_commit+"Ch"+existance_or_lvl
    if existance_or_lvl=="Lvl":
        ch_orders=ch_orders[ch_orders[prime_commit+"ChPer"]!=0]
        print("Change Level Classifiction")
        ch_orders[target_atr]=np.where(ch_orders["PrimeChPer"]>high_lvl,2,0)
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]>low_lvl) & (ch_orders["PrimeChPer"]<high_lvl)),1,ch_orders[target_atr])

    elif existance_or_lvl=="Existance":
        print("Change Existance Classifiction")
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]>0.1)),1,0)
    else:
        print("Wrong Input...")
        
    return(ch_orders)

def project_filter(ch_orders,atr,atr_class_list):
    ch_orders=ch_orders[ch_orders[atr]==(atr_class_list)]
    return(ch_orders)

def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

def drop_atr(ch_orders,atr_list):
    ch_orders.drop(columns=atr_list,axis=1,inplace=True)
    return(ch_orders)

def split_x_y(ch_orders,change_category,existance_or_level):
    target_atr=change_category+"Ch"+existance_or_level
    y=ch_orders[target_atr]
    target_classification_list="PrimeChExistance,CommitChExistance,TotalChExistance,SalesChExistance,PrimeChLvl,CommitChLvl,TotalChLvl,SalesChLvl".split(",")
    x=ch_orders.loc[:,~ch_orders.columns.isin(target_classification_list)]
    return(x,y)


def svm_classification_prep(ch_orders,x,y,test_size):
 
    a=x.dtypes=="object"
    categorical_atr_list=list(a.loc[x.dtypes=="object"].index)
    x=pd.get_dummies(x,columns=categorical_atr_list,drop_first=True)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=1)
    sc=StandardScaler().fit(x_train)
    x_train_str=sc.transform(x_train)
    x_test_str=sc.transform(x_test)
    return(x_train,x_train_str,x_test_str,y_train,y_test)

# x_train,x_train_str,x_test_str,y_train,y_test=svm_classification_prep(ch_orders,x,y)

def svm_classification_wh_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input):
    cv_results="Not applicaable"
    clf_svm_l=svm.SVC(kernel=kernel_input,C=0.1, degree=2,gamma=0.1,class_weight="balanced")
    clf_svm_l.fit(x_train_str,y_train)
    y_test_pred=clf_svm_l.predict(x_test_str)
    y_train_pred=clf_svm_l.predict(x_train_str)
    accuracy_test=accuracy_score(y_test,y_test_pred)
    confusion_test=pd.DataFrame(data=confusion_matrix(y_test,y_test_pred))
    accuracy_train=accuracy_score(y_train,y_train_pred)
    confusion_train=pd.DataFrame(data=confusion_matrix(y_train,y_train_pred))
    return(cv_results,confusion_test,accuracy_test,confusion_train,accuracy_train)

def svm_classification_w_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input):
    clf_svm=svm.SVC(kernel=kernel_input,class_weight="balanced")
    f1_scorer = make_scorer(f1_score,average="macro")
    params={"C":(0.01,0.001,0.1),"degree":(2,3),"gamma":(0.01,0.1,0.001)}
    svm_grid=GridSearchCV(clf_svm,params,n_jobs=-1,cv=15,verbose=-1,scoring=f1_scorer)
    svm_grid.fit(x_train_str,y_train)
    cv_results=svm_grid.cv_results_
    best_params=svm_grid.best_params_
    svm_clf=svm_grid.best_estimator_
    print("Best F1 score: ", svm_grid.best_score_)
    accuracy_test=accuracy_score(y_test,svm_clf.predict(x_test_str))
    confusion_test=pd.DataFrame(data=confusion_matrix(y_test,svm_clf.predict(x_test_str)))
    accuracy_train=accuracy_score(y_train,svm_clf.predict(x_train_str))
    confusion_train=pd.DataFrame(data=confusion_matrix(y_train,svm_clf.predict(x_train_str)))
    # confusion_test=confusion_matrix(y_test,linsvm_clf.predict(x_test_std))
    # accuracy_test=accuracy_score(y_test,linsvm_clf.predict(x_test_std))
    # confusion_train=confusion_matrix(y_train,linsvm_clf.predict(x_train_std))
    # accuracy_train=accuracy_score(y_train,linsvm_clf.predict(x_train_std))
    
    return(cv_results,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params)

def run_the_code(grid_search_bool,kernel_str,prime_or_commit,ch_existance_or_lvl,construction_or_service):
    best_params=0
    kernel_input=kernel_str
    ch_orders=read_df()
    ch_orders=divide_classification(ch_orders)
    # ch_orders=ch_freq(ch_orders)
    ch_orders=project_filter(ch_orders,"ProjectType",construction_or_service)
    ch_orders=label_target_atr(ch_orders,0.01,0.09,ch_existance_or_lvl,prime_or_commit)

    ch_orders=drop_atr(ch_orders,["ProjectType","ProjectCity","DailyCost","ChangeDuration","PrimeChFreq_p","PrimeChFreq_n","CommitChFreq_p","CommitChFreq_n"\
         ,"SalesChFreq_p","SalesChFreq_n","TotalChPer","CommitChPer","SalesChPer","PrimeChPer"])
    ch_orders=outlier_remove(ch_orders,["DurationModified"])
    x,y=split_x_y(ch_orders,prime_or_commit,ch_existance_or_lvl) 
    x_train,x_train_str,x_test_str,y_train,y_test=svm_classification_prep(ch_orders,x,y,.2)
    if grid_search_bool==False:
       cv_results,confusion_test,accuracy_test,confusion_train,accuracy_train=svm_classification_wh_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input)
    else:
       cv_results,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params=svm_classification_w_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input)
    return(cv_results,ch_orders,x_train,y_train,y_test,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params)
cv_results,ch_orders,x_train,y_train,y_test,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params=\
    run_the_code(True,"poly","Prime","Existance","Construction")
projects,ch_orders=run_the_code()
a=y_train.describe()
train_distribution=y_train.value_counts()
test_distribution=y_test.value_counts()
calss_recall_train=[confusion_train.loc[0,0]/train_distribution.loc[0],confusion_train.loc[1,1]/train_distribution.loc[1]]
calss_recall_test=[confusion_test.loc[0,0]/test_distribution.loc[0],confusion_test.loc[1,1]/test_distribution.loc[1]]
# calss_accuraacy_train=[confusion_train.loc[0,0]/train_distribution.loc[0],confusion_train.loc[1,1]/train_distribution.loc[1],confusion_train.loc[2,2]/train_distribution.loc[2]]

# b=ch_orders["City"].value_counts()
