# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:50:04 2023

@author: narim
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


from step_9_0_Feature_importance_permutation import x_test,x_train,y_test,y_train
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer
from sklearn.ensemble import RandomForestClassifier
# from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_prep():
    x=pd.concat([x_train,x_test],axis=0)

    sc=StandardScaler().fit(x)
    x_str=sc.transform(x)
    x_str=pd.DataFrame(data=x_str,columns=x.columns,index=x.index)
    x_str_train=x_str.loc[x_train.index]
    x_str=pd.DataFrame(data=x_str,columns=x.columns,index=x.index)
    x_str_test=x_str.loc[x_test.index]
    return(x_str,x_str_train,x_str_test)
def rf_transform_prep(trans_data):

    x_transform_train=trans_data.loc[x_train.index]
    x_transform_test=trans_data.loc[x_test.index]

    return(x_transform_train,x_transform_test)
"given a DF and number of PCAs; already normalised, outlier removed,and filtered to the numeric attributes without the target atr, runs PCA dimentionality reduction algorythm\
     returns"
def pca(df,n):
    pca = PCA(n_components=n)
    pca.fit(df)
    atr_list=df.columns
    PCs= ["PC-" + str(i) for i in range(1, n + 1)]
    eigenvectors = abs(pd.DataFrame(data=pca.components_.transpose(),index=atr_list,columns=PCs))
    #variance Ratio is the amount of variance explained by each principal component
    variance_ratios = pd.DataFrame(data=pca.explained_variance_ratio_,index=PCs,columns=["VariationRatio"])
    #singular values are square roots of the eigenvalues
    singular_values = pd.DataFrame(data=pca.singular_values_,columns=["SquareRoot_EigenValues"],index=PCs)
    #data in the new feature space
    transformed_data = pd.DataFrame(data=pca.transform(df),columns=PCs,index=df.index)
    cumulative_variance = np.cumsum(variance_ratios)
    # plt.scatter(range(1, len(variance_ratios) + 1), cumulative_variance)
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.title('Cumulative PCs Variance')
    # custom_x_ticks = np.arange(1, len(variance_ratios) + 1)  # Specify the tick positions
    # custom_x_labels = [str(i) for i in custom_x_ticks]
    # plt.xticks(custom_x_ticks, custom_x_labels, fontsize=10)
    # custom_y_ticks = list(cumulative_variance)  # Specify the tick positions
    # custom_y_labels = [str(cumulative_variance)]
    # plt.xticks(custom_y_ticks, custom_y_labels, fontsize=10)
    # plt.xticks(fontsize=10)  # Adjust the fontsize for x-axis tick labels
    # plt.yticks(fontsize=10)  # Adjust the fontsize for y-axis tick labels
    # plt.show()
    return(eigenvectors,variance_ratios,singular_values,transformed_data,transformed_data,variance_ratios,cumulative_variance)
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
    custom_score = avg_f1_score
    
    return custom_score
def random_forest_classification(x_train,x_test,y_train,y_test):
    # hyperparameter grid to search
    param_grid = {
    'n_estimators': (410,),           # Number of decision trees in the forest
    'max_depth': (8,),              # Maximum depth of each decision tree
    'min_samples_split': (15, ),          # Minimum number of samples required to split an internal node
    'min_samples_leaf': (7, ),
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
# def classification_prep(ch_orders,x,y,test_size):
#     a=x.dtypes=="object"
#     categorical_atr_list=list(a.loc[x.dtypes=="object"].index)
#     num=transformed_data
#     enc=OrdinalEncoder()
#     cat=enc.fit_transform(x[categorical_atr_list])
#     cat=pd.DataFrame(cat,columns=categorical_atr_list,index=x.index)
#     x=pd.concat([num,cat],axis=1)
#     x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=0)
#     y_train=np.array(y_train).ravel()
#     return(x_train,x_test,y_train,y_test)

def run_the_code():
   
    x_str,x_str_train,x_str_test=pca_prep()
    eigenvectors,variance_ratios,singular_values,transformed_data,transformed_data,variance_ratios,cumulative_variance=pca(x_str,9)
    x_transform_train,x_transform_test=rf_transform_prep(transformed_data)
    best_params,best_tree,best_rf_model,importances,confusion_test,accuracy_test,confusion_train,accuracy_train,f1_score_train,f1_score_test=random_forest_classification(x_transform_train,x_transform_test,y_train,y_test)
    return(x_str,x_str_train,x_str_test,eigenvectors,variance_ratios,singular_values,transformed_data,transformed_data,variance_ratios,cumulative_variance,confusion_test,accuracy_test,confusion_train,accuracy_train,f1_score_train,f1_score_test)
x_str,x_str_train,x_str_test,eigenvectors,variance_ratios,singular_values,transformed_data,transformed_data,variance_ratios,cumulative_variance,confusion_test,accuracy_test,confusion_train,accuracy_train,f1_score_train,f1_score_test=run_the_code()
