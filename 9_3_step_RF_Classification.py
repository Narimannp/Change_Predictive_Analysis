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
from sklearn.ensemble import RandomForestClassifier

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
    if existance_or_lvl=="Multi-CLass":
        # ch_orders=ch_orders[ch_orders[prime_commit+"ChPer"]!=0]
        print("Multi-Class Classifiction")
        ch_orders[target_atr]=np.where(ch_orders[prime_commit+"ChPer"]<=boundry[0],0,1)
        ch_orders[target_atr]=np.where(((ch_orders[prime_commit+"ChPer"]<=boundry[1]) & (ch_orders["PrimeChPer"]>boundry[0])),1,ch_orders[target_atr])
        ch_orders[target_atr]=np.where((ch_orders[prime_commit+"ChPer"]>boundry[1]),2,ch_orders[target_atr])

    elif existance_or_lvl=="Binary":
        print("Binary Classifiction")
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]>boundry)),1,0)
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
    target_classification_list="PrimeChBinary,CommitChBinary,TotalChBinary,SalesChBinary,PrimeChMulti-Class,CommitChMulti-Class,TotalChMulti-Class,SalesChMulti-Class,PrimeChPer,CommitChPer,SalesChPer,TotalChPer".split(",")
    x=ch_orders.loc[:,~ch_orders.columns.isin(target_classification_list)]
    return(x,y)

def random_forest_classification(x_train,x_test,y_train,y_test):
    rfc = RandomForestClassifier(n_estimators=1000, random_state=42,criterion="entropy",max_depth=5,class_weight="balanced_subsample")
    rfc.fit(x_train,y_train)
    y_test_pred = rfc.predict(x_test)
    y_train_pred=rfc.predict(x_train)
    accuracy_test=accuracy_score(y_test,y_test_pred)
    confusion_test=pd.DataFrame(data=confusion_matrix(y_test,y_test_pred))
    accuracy_train=accuracy_score(y_train,y_train_pred)
    confusion_train=pd.DataFrame(data=confusion_matrix(y_train,y_train_pred))
    importances = pd.DataFrame(data=rfc.feature_importances_,index=x_train.columns)
    return(importances,confusion_test,accuracy_test,confusion_train,accuracy_train)

def svm_classification_prep(ch_orders,x,y,test_size):
 
    a=x.dtypes=="object"
    categorical_atr_list=list(a.loc[x.dtypes=="object"].index)
    x=pd.get_dummies(x,columns=categorical_atr_list,drop_first=True)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=1)
    sc=StandardScaler().fit(x_train)
    x_train_str=sc.transform(x_train)
    x_test_str=sc.transform(x_test)
    return(x_train,x_test,x_train_str,x_test_str,y_train,y_test)

# x_train,x_train_str,x_test_str,y_train,y_test=svm_classification_prep(ch_orders,x,y)

def svm_classification_wh_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input):
    clf_svm_l=svm.SVC(kernel=kernel_input)
    clf_svm_l.fit(x_train_str,y_train)
    y_test_pred=clf_svm_l.predict(x_test_str)
    y_train_pred=clf_svm_l.predict(x_train_str)
    accuracy_test=accuracy_score(y_test,y_test_pred)
    confusion_test=pd.DataFrame(data=confusion_matrix(y_test,y_test_pred))
    accuracy_train=accuracy_score(y_train,y_train_pred)
    confusion_train=pd.DataFrame(data=confusion_matrix(y_train,y_train_pred))
    return(confusion_test,accuracy_test,confusion_train,accuracy_train)

def svm_classification_w_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input):
    clf_svm=svm.SVC(kernel=kernel_input,class_weight={0:.1,2:5})
    f1_scorer = make_scorer(f1_score,average="micro")
    params={"C":(0.01,0.001,0.1),"degree":(3,4,5),"gamma":(0.01,0.1,0.001)}
    svm_grid=GridSearchCV(clf_svm,params,n_jobs=-1,cv=15,verbose=-1,scoring="accuracy")
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
    ch_orders=project_filter(ch_orders,"ProjectType",construction_or_service)
    ch_orders=label_target_atr(ch_orders,4,ch_existance_or_lvl,prime_or_commit)

    ch_orders=outlier_remove(ch_orders,["PrimeChPer"])
    x,y=split_x_y(ch_orders,prime_or_commit,ch_existance_or_lvl) 
    x_train,x_test,x_train_str,x_test_str,y_train,y_test=svm_classification_prep(ch_orders,x,y,.3)
    importance, confusion_test,accuracy_test,confusion_train,accuracy_train=random_forest_classification(x_train,x_test,y_train,y_test)
    # if grid_search_bool==False:
    #    confusion_test,accuracy_test,confusion_train,accuracy_train=svm_classification_wh_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input)
    # else:
    #    cv_results,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params=svm_classification_w_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input)
    return(importance,ch_orders,x_train,y_train,y_test,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params)
importance,ch_orders,x_train,y_train,y_test,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params=\
    run_the_code(False,"poly","Prime","Binary","Construction")
# projects,ch_orders=run_the_code()
a=y_train.describe()
test_distribution=y_test.value_counts()
train_distribution=y_train.value_counts()
pred_test_distribution=confusion_test.sum()
pred_train_distribution=confusion_train.sum()
calss_recall_train=[confusion_train.loc[0,0]/train_distribution.loc[0],confusion_train.loc[1,1]/train_distribution.loc[1]]
calss_recall_test=[confusion_test.loc[0,0]/test_distribution.loc[0],confusion_test.loc[1,1]/test_distribution.loc[1]]
calss_precision_train=[confusion_train.loc[0,0]/pred_train_distribution.loc[0],confusion_train.loc[1,1]/pred_train_distribution.loc[1]]
calss_precision_test=[confusion_test.loc[0,0]/pred_test_distribution.loc[0],confusion_test.loc[1,1]/pred_test_distribution.loc[1]]

# distribution=ch_orders.groupby("PrimeChExistance").count()["Density"]
# class_dist_weight=distribution[0]/(distribution[0]+distribution[1])
# ch_orders=pd.get_dummies(ch_orders,columns=["ProjectClassification","ProjectDepartment","ProjectOperatingUnit","ProjectType","ProjectBillingType"],drop_first=True)
# ch_orders=ch_orders.drop(columns="ProjectId,ProjectProvince,ProjectCity,DailyCost,ChangeDuration,TotalChFreq,TotalChPer,CommitChPer,SalesChPer,PrimeChFreq,PrimeChPer,CommitChFreq,SalesChFreq".split(","))
# x=ch_orders.loc[:,ch_orders.columns!="PrimeChLvl"]
# y=ch_orders["PrimeChLvl"]
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=0)
# sc=StandardScaler().fit(x_train)
# x_train_std=sc.transform(x_train)
# x_test_std=sc.transform(x_test)
# clf_svm_l=svm.SVC(kernel="linear")
# # clf_svm_l.fit(x_train_std,y_train)
# # y_test_pred=clf_svm_l.predict(x_test_std)
# # y_train_pred=clf_svm_l.predict(x_train_std)
# from sklearn.model_selection import GridSearchCV
# params={"C":(0.001,0.01,0.1,0.5,1,10,100,1000)}
# svm_grid_linear=GridSearchCV(clf_svm_l,params,n_jobs=-1,cv=10,verbose=-1,scoring=("recall"))
# svm_grid_linear.fit(x_train_std,y_train)
# svm_grid_linear.best_params_
# linsvm_clf=svm_grid_linear.best_estimator_

# 
# # a=r2_score(y_test,y_test_pred)
# # b=r2_score(y_train,y_train_pred)
# confusion_test=confusion_matrix(y_test,linsvm_clf.predict(x_test_std))
# accuracy_test=accuracy_score(y_test,linsvm_clf.predict(x_test_std))
# confusion_train=confusion_matrix(y_train,linsvm_clf.predict(x_train_std))
# accuracy_train=accuracy_score(y_train,linsvm_clf.predict(x_train_std))

# ch_dist_target=ch_orders_freq.groupby("ch_LVL").count()
# ch_dist_target=ch_dist_target.rename(columns={"ProjectBaseContractValue":"Number_of_Projects"})["Number_of_Projects"]
# 
# a=(ch_orders_freq.describe())
