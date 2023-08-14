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
from sklearn.metrics import f1_score,make_scorer,balanced_accuracy_score


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
    loc_orig=["City","Province"]
    orig_cat_atrs="OperatingUnit,BillType,ProjectClassification,Classification_1,Classification_2,ProjectType".split(",")
    orig_num_atrs="BaseValue,DurationModified".split(",")
    loc_keys="City,Prov".split(",")
    loc_atrs_add="Population,Density".split(",")
    columns=df.columns
    columns=columns.astype(str).tolist()
    Freq_atrs=[x for x in columns if "Frq" in x ]
    Freq_atrs_wh_loc=[x for x in Freq_atrs if not any(str_loc_keys in x for str_loc_keys in loc_keys) ]
    loc_frq=[x for x in Freq_atrs if  any(str_loc_keys in x for str_loc_keys in loc_keys) ]
    RunName="Num"


    atrs=target_atrs+orig_num_atrs
    for atr in atr_types:
        if atr not in ["freq","cat_no_loc","loc_add","cat_loc"]:
            raise ValueError("Wrong Atribute Type Input")
    if ("freq" in atr_types ):
        RunName=RunName+"+Frq_wh_loc"
        atrs=atrs+Freq_atrs_wh_loc
        # raise ValueError("1 NOT BOTH or NONE of Orig-cat and Freq attributes should be presented")
    if "cat_no_loc" in atr_types:
        atrs=atrs+orig_cat_atrs
        RunName=RunName+"+cat_wh_loc"
    if "cat_loc" in atr_types:
        atrs=atrs+loc_orig
        RunName=RunName+"+cat_loc"
    if "loc_add" in atr_types:
        atrs=atrs+loc_atrs_add
        RunName=RunName+"+loc_add"
    if (any(loc_type in atr_types for loc_type in ["cat_loc","loc_add"])) and ("freq" in atr_types) :
        atrs=atrs+loc_frq
        RunName=RunName+"+Frq_loc"

    df=df[atrs]
    return(df,RunName)

"Given the DF and boundry or list of boundries labels the target atrribute"
def label_target_atr(ch_orders,boundry):
    if type(boundry)==list:
        target_atr="PrimeChMulti-Class"
        print("Change Multi-Class Classifiction")
        ch_orders[target_atr]=np.where(ch_orders["PrimeChPer"]<=boundry[0],0,1)
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]<=boundry[1]) & (ch_orders["PrimeChPer"]>boundry[0])),1,ch_orders[target_atr])
        ch_orders[target_atr]=np.where((ch_orders["PrimeChPer"]>boundry[1]),2,ch_orders[target_atr])
    else: 
        target_atr="PrimeChBinary"
        print("Change Binary Classifiction")
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]>boundry)),1,0)
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

def split_x_y(ch_orders):
    columns=ch_orders.columns
    target=[x for x in columns if x in ["PrimeChBinary","PrimeChMulti-Class"]]
    y=ch_orders[target].values.flatten()
    target_classification_list="PrimeChBinary,PrimeChMulti-Class,PrimeChPer,CommitChPer,SalesChPer,TotalChPer".split(",")
    x=ch_orders.loc[:,~ch_orders.columns.isin(target_classification_list)]
    return(x,y)
def custom_scorer(y_true, y_pred):
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0)
    f1_class_1 = f1_score(y_true, y_pred, pos_label=1)
    
    # Calculate the average F1-score for both classes
    avg_f1_score = (f1_class_0 + f1_class_1) / 2.0
    
    # Minimize the F1-score of the minority class (class 1)
    minority_class_penalty = 1 - min(f1_class_0, f1_class_1)
    
    # Combine the average F1-score and the minority class penalty
    custom_score = avg_f1_score
    
    return custom_score

def svm_classification_prep(ch_orders,x,y,test_size):
 
    a=x.dtypes=="object"
    categorical_atr_list=list(a.loc[x.dtypes=="object"].index)
    x=pd.get_dummies(x,columns=categorical_atr_list,drop_first=True)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=1)
    sc=StandardScaler().fit(x_train)
    x_train_str=pd.DataFrame(data=sc.transform(x_train),index=x_train.index,columns=x_train.columns)
    x_test_str=pd.DataFrame(data=sc.transform(x_test),index=x_test.index,columns=x_test.columns)
    return(x_train,x_train_str,x_test_str,y_train,y_test)



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
    clf_svm=svm.SVC(kernel=kernel_input,class_weight="balanced")
    weighted_accuracy_scorer=make_scorer(balanced_accuracy_score)
    params={"C":(0.1,0.011,2),"degree":(2,3,4,5),"gamma":(.001,0.05,.5)}
    custom_scorer2 = make_scorer(custom_scorer)
    svm_grid=GridSearchCV(clf_svm,params,n_jobs=-1,cv=10,verbose=-1,scoring=custom_scorer2)
    svm_grid.fit(x_train_str,y_train)
    cv_results=svm_grid.cv_results_
    best_params=svm_grid.best_params_
    best_model = svm_grid.best_estimator_
    svm_clf=svm_grid.best_estimator_
    y_test_pred = best_model.predict(x_test_str)
    y_train_pred = best_model.predict(x_train_str)
    f1_score_train=custom_scorer(y_train,y_train_pred)
    f1_score_test=custom_scorer(y_test,y_test_pred)
    accuracy_test=accuracy_score(y_test,y_test_pred)
    confusion_test=pd.DataFrame(data=confusion_matrix(y_test,y_test_pred))
    accuracy_train=accuracy_score(y_train,y_train_pred)
    confusion_train=pd.DataFrame(data=confusion_matrix(y_train,y_train_pred))
    # confusion_test=confusion_matrix(y_test,linsvm_clf.predict(x_test_std))
    # accuracy_test=accuracy_score(y_test,linsvm_clf.predict(x_test_std))
    # confusion_train=confusion_matrix(y_train,linsvm_clf.predict(x_train_std))
    # accuracy_train=accuracy_score(y_train,linsvm_clf.predict(x_train_std))
    
    return(cv_results,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params,svm_grid.best_score_,f1_score_train,f1_score_test)
def prediction_output(atr_types,kernel_str,best_params,feature_eng,list_boundries,y_train,accuracy_train,accuracy_test,f1_train,f1_test):
    output=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\SVM_Binary-Class_results.csv')
    output = output.reset_index(drop=True)
    run_results=pd.DataFrame()

    all_atr_types=["numerical","cat_no_loc","cat_loc","loc_add","freq"]

    for atr_type in all_atr_types:
        if atr_type in atr_types+["numerical"]:
            run_results.loc[0,atr_type]=1
        else:
            run_results.loc[0,atr_type]=0

    run_results.loc[0,"FE"]=feature_eng
    a=y_train.shape
    # train_distribution=list(y_train).value_counts()/a  
    run_results.loc[0,"target_classes"]=list_boundries

    # run_results.loc[0,"target_distribution"]=str(train_distribution)

    run_results.loc[0,"accuracy_train"]=round(accuracy_train,3)
    run_results.loc[0,"accuracy_test"]=round(accuracy_test,3)
    run_results.loc[0,"f1_train"]=round(f1_train,3)
    run_results.loc[0,"f1_test"]=round(f1_test,3)
    run_results.loc[0,"kernel"]=kernel_str
    run_results.loc[0,"C_best"]=best_params["C"]
    run_results.loc[0,"Degree_best"]=best_params["degree"]
    run_results.loc[0,"gamma_best"]=best_params["gamma"]
    if  (output==run_results.loc[0,:]).all(axis=1).any():
        print("ALREADY_LOGED")
    else:
        print("New Run Results")
        output=pd.concat([output,run_results])    

    output.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\SVM_Binary-Class_results.csv',index=False)
    return(run_results,output)

def run_the_code(grid_search_bool,kernel_str,prime_or_commit,list_boundries,atr_types,FE):
    cv_results="not applicable"
    best_params=0
    kernel_input=kernel_str
    ch_orders=read_df()
    ch_orders=project_filter(ch_orders,"ProjectType","Construction")
    ch_orders,RunName=select_atrs(ch_orders,atr_types)
    ch_orders=label_target_atr(ch_orders,list_boundries)

    ch_orders=outlier_remove(ch_orders,["PrimeChPer"])

    x,y=split_x_y(ch_orders) 
    x_train,x_train_str,x_test_str,y_train,y_test=svm_classification_prep(ch_orders,x,y,.3)
    if grid_search_bool==False:
       confusion_test,accuracy_test,confusion_train,accuracy_train=svm_classification_wh_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input)
    else:
       cv_results,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params,best_score,f1_train,f1_test=svm_classification_w_gridsearch(x_train_str,x_test_str,y_train,y_test,kernel_input,)
    run_results,output=prediction_output(atr_types,kernel_str,best_params,FE,list_boundries,y_train,accuracy_train,accuracy_test,f1_train,f1_test)
    return(run_results,output,cv_results,ch_orders,x_train,y_train,y_test,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params,f1_train,f1_test)
run_results,output,cv_results,ch_orders,x_train,y_train,y_test,confusion_test,accuracy_test,confusion_train,accuracy_train,best_params,f1_train,f1_test=\
    run_the_code(True,"rbf","Prime",4,["cat_loc","cat_no_loc","freq"],True)
# ch_orders=read_df()
# ch_orders=project_filter(ch_orders,"ProjectType","Construction")
# ch_orders=label_target_atr(ch_orders,4)
# ch_orders,RunName=select_atrs(ch_orders,["freq"])
# x,y=split_x_y(ch_orders) 