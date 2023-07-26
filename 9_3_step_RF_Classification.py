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

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer
from sklearn.ensemble import RandomForestClassifier
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
    df=df[['BaseValue', 'ProjectClassification', 'BillType',
           'OperatingUnit', 'ProjectType',
           'TotalChPer', 'PrimeChPer', 'CommitChPer', 'SalesChPer',
           'Classification_1', 'Classification_2', 'DurationModified','Frq_Classification_p_dur',
           'Frq_Classification_p_sze', 'Frq_Classification_n_dur',
           'Frq_Classification_n_sze', 'Frq_OPU_p_dur', 'Frq_OPU_p_sze',
           'Frq_OPU_n_dur', 'Frq_OPU_n_sze', 'Frq_Class1_p_dur',
           'Frq_Class1_p_sze', 'Frq_Class1_n_dur', 'Frq_Class1_n_sze',
           'Frq_Class2_p_dur', 'Frq_Class2_p_sze', 'Frq_Class2_n_dur',
           'Frq_Class2_n_sze']]
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

# def custom_scoring(estimator, x_train, x_test,y_train, y_test):
#     y_pred_train = estimator.predict(x_train)
#     y_pred_test = estimator.predict(x_test)
#     train_accuracy = accuracy_score(y_train, y_pred_train)
#     test_accuracy = accuracy_score(y_test, y_pred_test)
#     diff_accuracy = abs(train_accuracy - test_accuracy)
#     if diff_accuracy <= 0.05:
#         return test_accuracy
#     else:
#         return test_accuracy - diff_accuracy
def custom_scorer(y_true, y_pred):
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0)
    f1_class_1 = f1_score(y_true, y_pred, pos_label=1)
    
    # Calculate the average F1-score for both classes
    avg_f1_score = (f1_class_0 + f1_class_1) / 2.0
    
    # Minimize the F1-score of the minority class (class 1)
    minority_class_penalty = 1 - min(f1_class_0, f1_class_1)
    
    # Combine the average F1-score and the minority class penalty
    custom_score = avg_f1_score - minority_class_penalty
    
    return custom_score
   
def random_forest_classification(x_train,x_test,y_train,y_test):
    # hyperparameter grid to search
    param_grid = {
    'n_estimators': ( 500,700,1000),           # Number of decision trees in the forest
    'max_depth': (5,8,10),              # Maximum depth of each decision tree
    'min_samples_split': (5, 10),          # Minimum number of samples required to split an internal node
    'min_samples_leaf': (4, 8),            # Minimum number of samples required in a leaf node
    # 'max_features': ['auto', 'sqrt'],         33# Number of features to consider when looking for the best split
    # 'bootstrap': (True, False),               # Whether to use bootstrapped samples
    }
    custom_scorer2 = make_scorer(custom_scorer)
    rfc = RandomForestClassifier( random_state=1,class_weight="balanced")
    grid_search = GridSearchCV(rfc, param_grid,n_jobs=-1,cv=10,verbose=-1,  scoring=custom_scorer2)
    grid_search.fit(x_train, y_train)
    rf_best=grid_search.best_estimator_
    # the best hyperparameters and best model from grid search
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    #predictions on the test set using the best model
    y_test_pred = best_model.predict(x_test)
    y_train_pred = best_model.predict(x_train)

    accuracy_test=accuracy_score(y_test,y_test_pred)
    confusion_test=pd.DataFrame(data=confusion_matrix(y_test,y_test_pred))
    accuracy_train=accuracy_score(y_train,y_train_pred)
    confusion_train=pd.DataFrame(data=confusion_matrix(y_train,y_train_pred))
    importances = pd.DataFrame(data=best_model.feature_importances_,index=x_train.columns)
    return(best_params,importances,confusion_test,accuracy_test,confusion_train,accuracy_train)

def classification_prep(ch_orders,x,y,test_size):
    a=x.dtypes=="object"
    categorical_atr_list=list(a.loc[x.dtypes=="object"].index)
    x=pd.get_dummies(x,columns=categorical_atr_list,drop_first=True)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=0)
    y_train=np.array(y_train).ravel()
    return(x_train,x_test,y_train,y_test)


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
    x_train,x_test,y_train,y_test=classification_prep(ch_orders,x,y,.25)
    best_params,importances,confusion_test,accuracy_test,confusion_train,accuracy_train=random_forest_classification(x_train,x_test,y_train,y_test)

    return(ch_orders,best_params,importances,accuracy_train,accuracy_test,confusion_test,confusion_train,x_train,y_train,y_test,RunName)


ch_orders,best_params,importances,accuracy_train,accuracy_test,confusion_test,confusion_train,x_train,y_train,y_test,RunName=\
    run_the_code(list_boundries=4,feature_eng=False,atr_types=["freq","orig_cat","loc_atrs","loc_add"],epocs=300)
# projects,ch_orders=run_the_code()
# a=y_train.describe()
# test_distribution=y_test.value_counts()
# train_distribution=y_train.value_counts()
# pred_test_distribution=confusion_test.sum()
# pred_train_distribution=confusion_train.sum()
# calss_recall_train=[confusion_train.loc[0,0]/train_distribution.loc[0],confusion_train.loc[1,1]/train_distribution.loc[1]]
# calss_recall_test=[confusion_test.loc[0,0]/test_distribution.loc[0],confusion_test.loc[1,1]/test_distribution.loc[1]]
# calss_precision_train=[confusion_train.loc[0,0]/pred_train_distribution.loc[0],confusion_train.loc[1,1]/pred_train_distribution.loc[1]]
# calss_precision_test=[confusion_test.loc[0,0]/pred_test_distribution.loc[0],confusion_test.loc[1,1]/pred_test_distribution.loc[1]]

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
