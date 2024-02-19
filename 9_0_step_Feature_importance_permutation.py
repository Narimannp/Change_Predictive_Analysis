# -*-- coding: utf-8 -*-
"""
Created on Tue Aug  8 07:44:08 2023

@author: narim
"""
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
from sklearn.inspection import permutation_importance
from sklearn import tree
# from step_9_0_FeatureSelection_PCA import atr_list
# from step_9_0_FeatureSelection_ChiSquare import sorted_cat_atrs as sorted_cat_atrs
from collections import defaultdict
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance


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
    'n_estimators': (30,),           # Number of decision trees in the forest
    'max_depth': (10,),              # Maximum depth of each decision tree
    'min_samples_split': (16, ),          # Minimum number of samples required to split an internal node
    'min_samples_leaf': (8, ),
    # 'criterion':('entropy' ,)           # Minimum number of samples required in a leaf node
    # 'max_features': ['auto', 'sqrt'],        # Number of features to consider when looking for the best split
    # 'bootstrap': (True, False),               # Whether to use bootstrapped samples
    }
    custom_scorer2 = make_scorer(custom_scorer)
    rfc = RandomForestClassifier( random_state=1,class_weight="balanced")
    grid_search = GridSearchCV(rfc, param_grid,n_jobs=-1,cv=10,  scoring=custom_scorer2)
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
    return(best_params,best_tree,best_rf_model,importances,confusion_test,accuracy_test,confusion_train,accuracy_train,f1_score_train,f1_score_test)


def permutation_importances(best_rf_model,x_train,y_train,):
    permutation_result = permutation_importance(best_rf_model, x_train, y_train, n_repeats=10, random_state=42)
    perm_sorted_idx = permutation_result.importances_mean.argsort()
    tree_importance_sorted_idx = np.argsort(best_rf_model.feature_importances_)
    tree_indices = np.arange(0, len(best_rf_model.feature_importances_)) + 0.5
    plt.rcParams['axes.linewidth'] = 15
    plt.rcParams['ytick.labelsize'] = 80
    plt.rcParams['xtick.labelsize'] = 80   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(120, 80))
    ax1.barh(tree_indices, best_rf_model.feature_importances_[tree_importance_sorted_idx], height=0.3)
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(x_train.columns[tree_importance_sorted_idx])
    ax1.set_ylim((0, len(best_rf_model.feature_importances_)))
    boxplot=ax2.boxplot(
        permutation_result.importances[perm_sorted_idx].T,
        vert=False,
        labels=x_train.columns[perm_sorted_idx]
    )
    for box in boxplot['boxes']:
       box.set_linewidth(10)  # Adjust line width here
    for cap in boxplot['caps']:
       cap.set(color ='#8B008B',linewidth = 12)
    ax2.tick_params(axis='both', which='major', labelsize=80)  # Adjust font size here
    fig.tight_layout()
    plt.show()
    return(perm_sorted_idx,permutation_result)
def higherarchi_clustring(x_train):
    # ax1 = plt.subplots(1, 2, figsize=(480, 360))
    pearson,spearman = pearson_spearman(x_train)

    # # Ensure the correlation matrix is symmetric
    # corr = (spearman + spearman.T) / 2
    # np.fill_diagonal(corr, 1)

     # We convert the correlation matrix to a distance matrix before performing
     # hierarchical clustering using Ward's linkage.
    fig,ax3 = plt.subplots(figsize=(360, 150))
    plt.rcParams['lines.linewidth'] = 10
    plt.rcParams['axes.linewidth'] = 12
    plt.rcParams['ytick.labelsize'] = 200
    distance_matrix = 1 - np.abs(np.array(spearman))
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
    dist_linkage,ax=ax3, labels=x_train.columns.tolist(),  leaf_rotation=90,leaf_font_size=100  
    )
    plt.show()

    return(spearman)
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

def pearson_spearman(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    for column in df:
       "Range normalizing attributes"
       df[column]=(df[column]-df[column].min())/(df[column].max()-df[column].min())
    pearson=df.corr(method='pearson', min_periods=1).round(2)
    spearman=df.corr(method='spearman', min_periods=1).round(2)
    return(pearson,spearman)


def visulize_tree(best_model,x_train,y_train):
    first_tree = best_model
    # Plot the decision tree
    plt.figure(figsize=(120, 80))
    tree.plot_tree(first_tree, feature_names=x_train.columns, class_names=["low","high"], filled=True)
    plt.show()
    
def run_the_code(list_boundries,feature_eng,atr_types):
    ch_orders=read_df()
    perm_sorted_idx,permutation_result=[],[]
    spearman=pd.DataFrame()
    ch_orders=drop_atr(ch_orders, ["ProjectType"])
    ch_orders,RunName=select_atrs(ch_orders,atr_types)
    ch_orders=label_target_atr(ch_orders,list_boundries)
    ch_orders=outlier_remove(ch_orders,["PrimeChPer"])
    x,y=split_x_y(ch_orders) 
    if feature_eng==True:
        columns=ch_orders.columns
        # atrs=[atr for atr in atr_list if atr in columns]
        # cat_atrs=[atr for atr in sorted_cat_atrs if atr in columns]
        # x=x[atrs+cat_atrs]
        RunName=RunName+"\n w FE"
    else:
        RunName=RunName+"\n wh FE"
    cor_atrs_1=["Frq_Classification_n_sze","Frq_Classification_n_dur","Frq_Class1_p_dur","Frq_Class1_p_sze","Frq_OPU_n_sze","Frq_OPU_n_dur","Province","Frq_Prov_n_sze"]
    cor_atrs_2=["Frq_Class2_p_dur","Frq_Prov_p_sze"]
    cor_atrs_3=["Classification_1","Frq_OPU_p_sze"]
    cor_atrs_4=["ProjectClassification","Frq_Class2_p_sze","Frq_Class2_n_dur","Frq_City_n_dur","Frq_City_n_sze"]
    cor_atrs_5=["Frq_City_p_sze","Frq_Class2_n_sze","DurationModified"]
    cor_atrs_6=["Frq_Class1_n_dur","City"]
    cor_atrs_7=["OperatingUnit","Frq_Prov_p_dur","Classification_2","BaseValue"]
    cor_atrs_list=[cor_atrs_1,cor_atrs_2,cor_atrs_3,cor_atrs_4,cor_atrs_5,cor_atrs_6,cor_atrs_7]
    redun_atrs_permut=["BillType","Frq_Prov_n_dur","Frq_City_p_dur","Frq_Prov_p_dur","Frq_Classification_p_dur","Frq_OPU_p_dur","City","Frq_Class1_n_dur","Frq_Class1_n_sze","Classification_2","OperatingUnit","Frq_Class2_n_sze"]
    x_train,x_test,y_train,y_test=classification_prep(ch_orders,x,y,.25)
    for i in range(4):      
        x_train=drop_atr(x_train,cor_atrs_list[i])
        x_test =drop_atr(x_test,cor_atrs_list[i])
    x_train=drop_atr(x_train,redun_atrs_permut[:3])
    x_test =drop_atr(x_test,redun_atrs_permut[:3])
      

    best_params,best_tree,best_rf_model,importances,confusion_test,accuracy_test,confusion_train,accuracy_train,f1_train,f1_test=\
    random_forest_classification(x_train,x_test,y_train,y_test)
    perm_sorted_idx,permutation_result=permutation_importances(best_rf_model,x_train,y_train,)
    spearman=higherarchi_clustring(x_train)

    return(spearman,ch_orders,importances,perm_sorted_idx,permutation_result,best_params,accuracy_train,accuracy_test,confusion_test,confusion_train,x_train,y_train,y_test,RunName,f1_train,f1_test)


spearman,ch_orders,importances,perm_sorted_idx,permutation_result,best_params,accuracy_train,accuracy_test,confusion_test,confusion_train,x_train,y_train,y_test,RunName,f1_score_train,f1_score_test=\
    run_the_code(list_boundries=4,feature_eng=False,atr_types=["cat_no_loc","freq","cat_loc"])
print(x_train.columns)