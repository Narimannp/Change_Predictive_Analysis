# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:38:10 2023

@author: narim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from step_9_0_FeatureSelection_PCA import atr_list
from step_9_0_FeatureSelection_ChiSquare import sorted_cat_atrs as sorted_cat_atrs

def read_df():
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    ch_orders.rename(columns={"ProjectBaseContractValue":"BaseValue","ProjectOperatingUnit":"OperatingUnit",\
                                   "ProjectBillingType":"BillType","ProjectCity":"City","ProjectProvince":"Province"},inplace=True)
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)


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


def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

"Given the Df and types of attributes necessary, return the df with required attribute types"
def select_atrs(df,atr_types):
    loc_columns=[]
    temp_loc_columns=[]
    target_atrs=['PrimeChPer']
    loc_orig=["City","Province"]
    orig_cat_atrs="OperatingUnit,BillType,ProjectClassification,Classification_1,Classification_2".split(",")
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
    return(df,RunName,)

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

def custom_scoring(estimator, x_train, x_test,y_train, y_test):
    y_pred_train = estimator.predict(x_train)
    y_pred_test = estimator.predict(x_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    diff_accuracy = abs(train_accuracy - test_accuracy)
    if diff_accuracy <= 0.05:
        return test_accuracy
    else:
        return test_accuracy - diff_accuracy
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
   
def random_forest_classification(x_train,x_test,y_train,y_test):
    # hyperparameter grid to search
    param_grid = {
    'n_estimators': (80,),           # Number of decision trees in the forest
    'max_depth': (8,),              # Maximum depth of each decision tree
    'min_samples_split': (15, ),          # Minimum number of samples required to split an internal node
    'min_samples_leaf': (9, ),
    'criterion':('gini' ,) ,          # Minimum number of samples required in a leaf node
    # 'max_features': ['auto', 'sqrt'],        # Number of features to consider when looking for the best split
    # 'bootstrap': (True, False),               # Whether to use bootstrapped samples
    }
    custom_scorer2 = make_scorer(custom_scorer)
    rfc = RandomForestClassifier( random_state=1,class_weight="balanced")
    grid_search = GridSearchCV(rfc, param_grid,n_jobs=-1,cv=15,verbose=-1,  scoring=custom_scorer2)
    grid_search.fit(x_train, y_train)
    rf_best=grid_search.best_estimator_
    # the best hyperparameters and best model from grid search
    best_params = grid_search.best_params_
    best_rf_model = grid_search.best_estimator_
    best_tree = best_rf_model.estimators_[0]
    #predictions on the test set using the best model
    y_test_pred = best_rf_model.predict(x_test)
    y_train_pred = best_rf_model.predict(x_train)

    accuracy_test=accuracy_score(y_test,y_test_pred)
    confusion_test=pd.DataFrame(data=confusion_matrix(y_test,y_test_pred),columns=["Pred_Low","Pred_High"],index=["True_Low","True_High"])
    accuracy_train=accuracy_score(y_train,y_train_pred)
    confusion_train=pd.DataFrame(data=confusion_matrix(y_train,y_train_pred),columns=["Pred_Low","Pred_High"],index=["True_Low","True_High"])
    importances = pd.DataFrame(data=best_rf_model.feature_importances_,index=x_train.columns).sort_values(by=0,axis=0,ascending=False)
    f1_score_train=custom_scorer(y_train,y_train_pred)
    f1_score_test=custom_scorer(y_test,y_test_pred)
    return(best_params,best_tree,importances,confusion_test,accuracy_test,confusion_train,accuracy_train,f1_score_train,f1_score_test)

def classification_prep(ch_orders,x,y,test_size):
    a=x.dtypes=="object"
    categorical_atr_list=list(a.loc[x.dtypes=="object"].index)
    num=x.drop(categorical_atr_list,axis=1)
    enc=OrdinalEncoder()
    cat=enc.fit_transform(x[categorical_atr_list])
    cat=pd.DataFrame(cat,columns=categorical_atr_list,index=x.index)
    x=pd.concat([num,cat],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=0)
    y_train=np.array(y_train).ravel()
    return(x_train,x_test,y_train,y_test)

def prediction_output(atr_types,best_params,feature_eng,list_boundries,y_train,accuracy_train,accuracy_test,f1_train,f1_test):
    output=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\RF_results.csv')
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


    run_results.loc[0,"max_depth"]=round(best_params["max_depth"],3)
    run_results.loc[0,"min_s_leaf"]=round(best_params["min_samples_leaf"],3)
    run_results.loc[0,"min_s_split"]=round(best_params["min_samples_split"],3)
    run_results.loc[0,"no_tress"]=round(best_params["n_estimators"],3)
    run_results.loc[0,"criterion"]=best_params["criterion"]
    run_results.loc[0,"accuracy_train"]=round(accuracy_train,3)
    run_results.loc[0,"accuracy_test"]=round(accuracy_test,3)
    run_results.loc[0,"f1_train"]=round(f1_train,3)
    run_results.loc[0,"f1_test"]=round(f1_test,3)
    if  (output==run_results.loc[0,:]).all(axis=1).any():
        print("ALREADY_LOGED")
    else:
        print("New Run Results")
        output=pd.concat([output,run_results])    

    output.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\RF_results.csv',index=False)
    return(run_results,output)

def visulize_tree(best_model,x_train,y_train):
    first_tree = best_model
    # Plot the decision tree
    plt.figure(figsize=(120, 80))
    tree.plot_tree(first_tree, feature_names=x_train.columns, class_names=["low","high"], filled=True)
    plt.show()
    
def run_the_code(list_boundries,feature_eng,atr_types):
    ch_orders=read_df()
    ch_orders=drop_atr(ch_orders,["ProjectType"])
    ch_orders,RunName=select_atrs(ch_orders,atr_types)
    ch_orders=label_target_atr(ch_orders,list_boundries)
    ch_orders=outlier_remove(ch_orders,["PrimeChPer"])


    x,y=split_x_y(ch_orders) 
    if feature_eng==True:
        columns=ch_orders.columns
        atrs=[atr for atr in atr_list if atr in columns]
        cat_atrs=[atr for atr in sorted_cat_atrs if atr in columns]
        x=x[atrs+cat_atrs]
        RunName=RunName+"\n w FE"
    else:
        RunName=RunName+"\n wh FE"
 

    x_train,x_test,y_train,y_test=classification_prep(ch_orders,x,y,.25)
    # x_train=drop_atr(x_train,atr_to_drop)
    # x_test=drop_atr(x_test,atr_to_drop)
    # x_train=drop_atr(x_train,atr2_to_drop)
    # x_test=drop_atr(x_test,atr2_to_drop)
    best_params,best_tree,importances,confusion_test,accuracy_test,confusion_train,accuracy_train,f1_train,f1_test=random_forest_classification(x_train,x_test,y_train,y_test)
    run_results,output=prediction_output(atr_types,best_params,feature_eng,list_boundries,y_train,accuracy_train,accuracy_test,f1_train,f1_test)
    visulize_tree(best_tree,x_train,y)
    return(ch_orders,importances,best_params,accuracy_train,accuracy_test,confusion_test,confusion_train,x_train,y_train,y_test,RunName,f1_train,f1_test)


ch_orders,importances,best_params,accuracy_train,accuracy_test,confusion_test,confusion_train,x_train,y_train,y_test,RunName,f1_score_train,f1_score_test=\
    run_the_code(list_boundries=4,feature_eng=False,atr_types=["freq","cat_no_loc"])

