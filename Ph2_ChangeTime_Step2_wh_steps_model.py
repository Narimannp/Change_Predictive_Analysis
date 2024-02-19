# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:57:07 2023

@author: narim
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
def read_df():
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\ph2_ChangeTime_Step1_DataPrep.csv')
    # ch_orders=ch_orders[ch_orders[]]
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)
a=read_df()
def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

def labling(datapoint):
    if datapoint<=0:
         return("0-minus")
    elif datapoint<5:
         return("1-0-5%")
    else:
         return("2-5%-inf")
 
"Given the DF and boundry or list of boundries labels the target atrribute"
def label_target_atr(ch_orders):
    target_atr_list=[atr for atr in ch_orders.columns if "Percentage" in atr]
    for i in range(len(target_atr_list)):
       ch_orders["ChMagnitude_"+str(i+1)]=ch_orders["ChPercentage_"+str(i+1)].apply(labling)
    return(ch_orders,target_atr_list)

def select_atrs(ch_orders,target_stage,include_pre_stages):
    target_atrs=["ChMagnitude_"+str(i+1) for i in range(5)]
    commun_atrs=["DurationNew","ContractValue","City","OperatingUnit","ProjectType"]
    frq_atrs=[atr+"@"+str(target_stage) for atr in ["Frq_City_p_sze","Frq_City_p_dur","Frq_ProjectType_p_sze","Frq_ProjectType_n_sze","Frq_IndustryGroup_p_sze",\
    "Frq_OperatingUnit_p_dur","Frq_Province_p_dur","Frq_IndustryGroup_n_sze","Frq_Province_n_dur","Frq_IndustryGroup_n_dur"]]
    if include_pre_stages:
        atr_list=commun_atrs+frq_atrs+target_atrs[0:target_stage-1]
    else:
        if target_stage==1:
           atr_list=commun_atrs+frq_atrs
        else:
           atr_list=commun_atrs+frq_atrs+[target_atrs[target_stage-2]]
    y=ch_orders[["ChMagnitude_"+str(target_stage),"ChMagnitude_5"]]
    x=ch_orders[atr_list]
    return(x,y)
def classification_prep(ch_orders,x,y,test_size):
    a=x.dtypes=="object"
    categorical_atr_list=list(a.loc[x.dtypes=="object"].index)
    num=x.drop(categorical_atr_list,axis=1)
    enc=OrdinalEncoder()
    cat=enc.fit_transform(x[categorical_atr_list])
    cat=pd.DataFrame(cat,columns=categorical_atr_list,index=x.index)
    x=pd.concat([num,cat],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=0)
    
    return(x_train,x_test,y_train,y_test)

def custom_f1_score(y_true, y_pred):
    return (f1_score(y_true, y_pred, average='micro'))

def confusion_MX(y_true, y_pred):
    confusion=pd.DataFrame(data=confusion_matrix(y_true,y_pred,labels=(list(set(y_true)))),columns=(list(set(y_true))),index=(list(set(y_true)))).sort_index(axis=0).sort_index(axis=1)
    confusion.columns=[column+"-pred" for column in confusion.columns]
    return(confusion)
    
def RF_stage(x_train,x_test,y_train,y_test):
    # hyperparameter grid to search
    param_grid = {
    'n_estimators': (500,),           # Number of decision trees in the forest
    'max_depth': (8,),              # Maximum depth of each decision tree
    'min_samples_split': (9, ),          # Minimum number of samples required to split an internal node
    'min_samples_leaf': (5, ),
    'criterion':('gini' ,) ,          # Minimum number of samples required in a leaf node
    # 'max_features': ['auto', 'sqrt'],        # Number of features to consider when looking for the best split
    # 'bootstrap': (True, False),               # Whether to use bootstrapped samples
    }
    custom_scorer = make_scorer(custom_f1_score)
    rfc = RandomForestClassifier( random_state=1,class_weight="balanced")
    grid_search = GridSearchCV(rfc, param_grid,cv=15,scoring=custom_scorer)
    grid_search.fit(x_train, y_train.iloc[:,0])
    rf_best=grid_search.best_estimator_
    # the best hyperparameters and best model from grid search
    best_params = grid_search.best_params_
    best_rf_model = grid_search.best_estimator_
    best_tree = best_rf_model.estimators_[0]
    #predictions on the test set using the best model
    y_test_pred = best_rf_model.predict(x_test)
    y_train_pred = best_rf_model.predict(x_train)

    accuracy_test=accuracy_score(y_test.iloc[:,0],y_test_pred)
    confusion_test=confusion_MX(y_test.iloc[:,0],y_test_pred)
    accuracy_train=accuracy_score(y_train.iloc[:,0],y_train_pred)
    confusion_train=confusion_MX(y_train.iloc[:,0],y_train_pred)
    importances = pd.DataFrame(data=best_rf_model.feature_importances_,index=x_train.columns).sort_values(by=0,axis=0,ascending=False)
    average="macro"
    f1_train=f1_score(y_train.iloc[:,0], y_train_pred, average=average)
    f1_test=f1_score(y_test.iloc[:,0], y_test_pred, average=average)
    return(best_params,y_test_pred,confusion_test,accuracy_test,confusion_train,accuracy_train,f1_train,f1_test,average,importances)

def RF_end(x_train,x_test,y_train,y_test,average):
    # hyperparameter grid to search
    param_grid = {
    'n_estimators': (450,),           # Number of decision trees in the forest
    'max_depth': (12,),              # Maximum depth of each decision tree
    'min_samples_split': (9, ),          # Minimum number of samples required to split an internal node
    'min_samples_leaf': (3, ),
    'criterion':('gini' ,) ,          # Minimum number of samples required in a leaf node
    # 'max_features': ['auto', 'sqrt'],        # Number of features to consider when looking for the best split
    # 'bootstrap': (True, False),               # Whether to use bootstrapped samples
    }
    custom_scorer = make_scorer(custom_f1_score)
    rfc = RandomForestClassifier( random_state=1,class_weight="balanced")
    grid_search = GridSearchCV(rfc, param_grid,cv=15,scoring=custom_scorer)
    grid_search.fit(x_train, y_train.iloc[:,1])
    rf_best=grid_search.best_estimator_
    # the best hyperparameters and best model from grid search
    best_params = grid_search.best_params_
    best_rf_model = grid_search.best_estimator_
    best_tree = best_rf_model.estimators_[0]
    #predictions on the test set using the best model
    y_test_pred = best_rf_model.predict(x_test)
    y_train_pred = best_rf_model.predict(x_train)

    accuracy_test=accuracy_score(y_test.iloc[:,1],y_test_pred)
    confusion_test=confusion_MX(y_test.iloc[:,1],y_test_pred)
    accuracy_train=accuracy_score(y_train.iloc[:,1],y_train_pred)
    confusion_train=confusion_MX(y_train.iloc[:,1],y_train_pred)
    importances = pd.DataFrame(data=best_rf_model.feature_importances_,index=x_train.columns).sort_values(by=0,axis=0,ascending=False)
    f1_train=(y_train.iloc[:,1] , y_train_pred)
    f1_test=(y_test.iloc[:,1],y_test_pred)
    f1_train=f1_score(y_train.iloc[:,1], y_train_pred, average=average)
    f1_test=f1_score(y_test.iloc[:,1], y_test_pred, average=average)
    return(best_params,confusion_test,accuracy_test,confusion_train,accuracy_train,f1_train,f1_test)

def prediction_output(average,best_params_stage,best_params_end,stage,include_pre_stages,accuracy_train_stage,accuracy_test_stage,f1_train_stage,f1_test_stage,accuracy_train_end,accuracy_test_end,f1_train_end,f1_test_end):
    # output=pd.DataFrame(columns="stage,with-pre-stages,no_tress,max_depth,min_s_leaf,min_s_split,accuracy_train_stage,accuracy_test_stage,f1_train_stage,f1_test_stage".split(","))
    output=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\Stage_results.csv')
    output = output.reset_index(drop=True)
    run_results=pd.DataFrame()
    #General Model Specs
    run_results.loc[0,"stage"]=stage
    run_results.loc[0,"with-pre-stages"]=include_pre_stages
    # run_results.loc[0,"target_classes"]=list_boundries

    # run_results.loc[0,"target_distribution"]=str(train_distribution)
    #Stage Results concatation
    run_results.loc[0,"no_tress_stage"]=round(best_params_stage["n_estimators"],3)
    run_results.loc[0,"max_depth_stage"]=round(best_params_stage["max_depth"],3)
    run_results.loc[0,"min_s_leaf_stage"]=round(best_params_stage["min_samples_leaf"],3)
    run_results.loc[0,"min_s_split_stage"]=round(best_params_stage["min_samples_split"],3)

    run_results.loc[0,"accuracy_train_stage"]=round(accuracy_train_stage,3)
    run_results.loc[0,"accuracy_test_stage"]=round(accuracy_test_stage,3)
    run_results.loc[0,"f1_train_stage"]=round(f1_train_stage,3)
    run_results.loc[0,"f1_test_stage"]=round(f1_test_stage,3)
    #End results concatation
    run_results.loc[0,"no_tress_end"]=round(best_params_end["n_estimators"],3)
    run_results.loc[0,"max_depth_end"]=round(best_params_end["max_depth"],3)
    run_results.loc[0,"min_s_leaf_end"]=round(best_params_end["min_samples_leaf"],3)
    run_results.loc[0,"min_s_split_end"]=round(best_params_end["min_samples_split"],3)

    run_results.loc[0,"accuracy_train_end"]=round(accuracy_train_end,3)
    run_results.loc[0,"accuracy_test_end"]=round(accuracy_test_end,3)
    run_results.loc[0,"f1_train_end"]=round(f1_train_end,3)
    run_results.loc[0,"f1_test_end"]=round(f1_test_end,3)
    run_results.loc[0,"average"]=average
    if  (output==run_results.loc[0,:]).all(axis=1).any():
        print("ALREADY_LOGED")
    else:
        print("New Run Results")
        output=pd.concat([output,run_results])    

    output.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\Stage_results.csv',index=False)
    return(run_results,output)
def run(target_stage,include_pre_stages):
    ch_orders=read_df()
    ch_orders,target_atr_list=label_target_atr(ch_orders)
    x,y=select_atrs(ch_orders,target_stage,include_pre_stages)
    x_train,x_test,y_train,y_test=classification_prep(ch_orders,x,y,0.25)
    best_params_stage,y_test_pred_stage,confusion_test_stage,accuracy_test_stage,confusion_train_stage,accuracy_train_stage,f1_train_stage,f1_test_stage,average,importances_stage=RF_stage(x_train,x_test,y_train,y_test)
    best_params_end,confusion_test_end,accuracy_test_end,confusion_train_end,accuracy_train_end,f1_train_end,f1_test_end=RF_end(x_train,x_test,y_train,y_test,average)
    prediction_output(average,best_params_stage,best_params_end,target_stage,include_pre_stages,accuracy_train_stage,accuracy_test_stage,f1_train_stage,f1_test_stage,accuracy_train_end,accuracy_test_end,f1_train_end,f1_test_end)
    return(y_test,x_train,y_test_pred_stage,ch_orders,confusion_test_stage,accuracy_test_stage,confusion_train_stage,accuracy_train_stage,f1_train_stage,f1_test_stage,confusion_test_end,accuracy_test_end,confusion_train_end,accuracy_train_end,f1_train_end,f1_test_end,importances_stage)
y_test,x_train,y_test_pred_stage,ch_orders,confusion_test_stage,accuracy_test_stage,confusion_train_stage,accuracy_train_stage,f1_train_stage,f1_test_stage,confusion_test_end,accuracy_test_end,confusion_train_end,accuracy_train_end,f1_train_end,f1_test_end,importances_stage=\
    run(3,True)
#y_pred_test_stage=pd.DataFrame(y_test).groupby(0)[0].count()
a=ch_orders["DurationNew"].min()