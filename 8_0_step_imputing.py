# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:03:10 2023

@author: narim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense
def read_df():
    #Read datasets,ch_orders
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\7_data_prep_ch_orders.csv')
    return(ch_orders)



def pearson_spearman(df):
    df=df[['ProjectBaseContractValue','DailyCost', 'TotalChPer', 'PrimeChPer',
           'Duration','ChangeDuration','DurationDiff_Divided_Duration']]
    for column in df:
       #Range normalizing attributes
       df[column]=(df[column]-df[column].min())/(df[column].max()-df[column].min())
    pearson=df.corr(method='pearson', min_periods=1)
    spearman=df.corr(method='spearman', min_periods=1)
    return(pearson,spearman)
def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

def duration_modification(ch_orders):
    ch_orders["DurationModified"]=abs(ch_orders["Duration"])
    ch_orders["DurationModified"]=np.where(ch_orders["DurationModified"]==0,np.nan,ch_orders["DurationModified"])   

    ch_orders["DurationModified"]=np.where(ch_orders["DurationDiff_Divided_Duration"]<-25,np.nan,ch_orders["DurationModified"])  
    ch_orders["DurationModified"]=np.where(((ch_orders["DailyCost"]>150000 )&(ch_orders["Duration"]<5)),np.nan,ch_orders["DurationModified"])      
    return(ch_orders)

def frequency_normalization(ch_orders):
    ch_orders["freq_p_prime_dur"]=ch_orders["PrimeChFreq_p"]/ch_orders["DurationModified"]
    ch_orders["freq_p_prime_sze"]=ch_orders["PrimeChFreq_p"]*1000/ch_orders["ProjectBaseContractValue"]   
    
    ch_orders["freq_n_prime_dur"]=ch_orders["PrimeChFreq_n"]/ch_orders["DurationModified"]
    ch_orders["freq_n_prime_sze"]=ch_orders["PrimeChFreq_n"]*1000/ch_orders["ProjectBaseContractValue"]
    ch_orders["freq_t_prime_sze"]=ch_orders["freq_n_prime_sze"]+ch_orders["freq_p_prime_sze"]
    return(ch_orders)
  
def ch_freq(ch_orders):
    projecttypes=["Construction","Services"]
    ch_df=pd.DataFrame()
    for projecttype in projecttypes:
        df=ch_orders[ch_orders["ProjectType"]==projecttype]
        dict_cls_p_d=df.groupby("ProjectClassification")["freq_p_prime_dur"].mean()
        dict_cls_p_s=df.groupby("ProjectClassification")["freq_p_prime_sze"].mean()
        dict_cls_n_d=df.groupby("ProjectClassification")["freq_p_prime_dur"].mean()
        dict_cls_n_s=df.groupby("ProjectClassification")["freq_p_prime_sze"].mean()
        df["Frq_Classification_p_dur"]=df["ProjectClassification"].map(dict_cls_p_d)
        df["Frq_Classification_p_sze"]=df["ProjectClassification"].map(dict_cls_p_s)
        df["Frq_Classification_n_dur"]=df["ProjectClassification"].map(dict_cls_n_d)
        df["Frq_Classification_n_sze"]=df["ProjectClassification"].map(dict_cls_n_s)
        

        dict_cls_p_d=df.groupby("ProjectOperatingUnit")["freq_p_prime_dur"].mean()
        dict_cls_p_s=df.groupby("ProjectOperatingUnit")["freq_p_prime_sze"].mean()
        dict_cls_n_d=df.groupby("ProjectOperatingUnit")["freq_p_prime_dur"].mean()
        dict_cls_n_s=df.groupby("ProjectOperatingUnit")["freq_p_prime_sze"].mean()
        df["Frq_OPU_p_dur"]=df["ProjectOperatingUnit"].map(dict_cls_p_d)
        df["Frq_OPU_p_sze"]=df["ProjectOperatingUnit"].map(dict_cls_p_s)
        df["Frq_OPU_n_dur"]=df["ProjectOperatingUnit"].map(dict_cls_n_d)
        df["Frq_OPU_n_sze"]=df["ProjectOperatingUnit"].map(dict_cls_n_s)

        dict_1_p_d=df.groupby("Classification_1")["freq_p_prime_dur"].mean()
        dict_1_p_s=df.groupby("Classification_1")["freq_p_prime_sze"].mean()    
        dict_1_n_d=df.groupby("Classification_1")["freq_n_prime_dur"].mean()
        dict_1_n_s=df.groupby("Classification_1")["freq_n_prime_sze"].mean()
        dict_2_p_d=df.groupby("Classification_2")["freq_p_prime_dur"].mean()
        dict_2_p_s=df.groupby("Classification_2")["freq_p_prime_sze"].mean()
        dict_2_n_d=df.groupby("Classification_2")["freq_n_prime_dur"].mean()
        dict_2_n_s=df.groupby("Classification_2")["freq_n_prime_sze"].mean()
        df["Frq_Class1_p_dur"]=df["Classification_1"].map(dict_1_p_d)
        df["Frq_Class1_p_sze"]=df["Classification_1"].map(dict_1_p_s)
        df["Frq_Class1_n_dur"]=df["Classification_1"].map(dict_1_n_d)
        df["Frq_Class1_n_sze"]=df["Classification_1"].map(dict_1_n_s)
        df["Frq_Class2_p_dur"]=df["Classification_2"].map(dict_2_p_d)
        df["Frq_Class2_p_sze"]=df["Classification_2"].map(dict_2_p_s)
        df["Frq_Class2_n_dur"]=df["Classification_2"].map(dict_2_n_d)
        df["Frq_Class2_n_sze"]=df["Classification_2"].map(dict_2_n_s)
        dict_prv_p_d=df.groupby("ProjectProvince")["freq_p_prime_dur"].mean()
        dict_prv_p_s=df.groupby("ProjectProvince")["freq_p_prime_sze"].mean()
        dict_prv_n_d=df.groupby("ProjectProvince")["freq_n_prime_dur"].mean()
        dict_prv_n_s=df.groupby("ProjectProvince")["freq_n_prime_sze"].mean()
        df["Frq_Prov_p_dur"]=df["ProjectProvince"].map(dict_prv_p_d)  
        df["Frq_Prov_p_sze"]=df["ProjectProvince"].map(dict_prv_p_s)  
        df["Frq_Prov_n_dur"]=df["ProjectProvince"].map(dict_prv_n_d)   
        df["Frq_Prov_n_sze"]=df["ProjectProvince"].map(dict_prv_n_s) 
        dict_cty_p_d=df.groupby("ProjectCity")["freq_p_prime_dur"].mean()
        dict_cty_p_s=df.groupby("ProjectCity")["freq_p_prime_sze"].mean()
        dict_cty_n_d=df.groupby("ProjectCity")["freq_n_prime_dur"].mean()
        dict_cty_n_s=df.groupby("ProjectCity")["freq_n_prime_sze"].mean()
        df["Frq_City_p_dur"]=df["ProjectCity"].map(dict_cty_p_d)  
        df["Frq_City_p_sze"]=df["ProjectCity"].map(dict_cty_p_s) 
        df["Frq_City_n_dur"]=df["ProjectCity"].map(dict_cty_n_d) 
        df["Frq_City_n_sze"]=df["ProjectCity"].map(dict_cty_n_s)
        ch_df=pd.concat([df,ch_df],axis=0)
    return(ch_df)   
def output(algorythm,Freq,kernel_str,best_params,r2_train,r2_test,mean_error_train,mean_error_test):
    params_keys=list(best_params.keys())
    NA_keys=[params for params in ["K","C","degree","gamma","max_depth","n_estimators","min_samples_split","min_samples_leaf"] if params not in params_keys ]
    for NA_key in NA_keys:
        best_params[NA_key]="X"
    output=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\Imputation_Results.csv')
    output = output.reset_index(drop=True)
    run_results=pd.DataFrame()

    all_atr_types=["freq_project","freq_class","cat"]

    for atr_type in all_atr_types:
        if atr_type in Freq:
            run_results.loc[0,atr_type]=1
        else:
            run_results.loc[0,atr_type]=0

    run_results.loc[0,"algorythm"]=algorythm
    if algorythm=="SVM":
       run_results.loc[0,"kernel"]=kernel_str
    else:
       run_results.loc[0,"kernel"]="X"
    run_results.loc[0,"K"]=str(best_params["K"])
    run_results.loc[0,"C"]=str(best_params["C"])
    run_results.loc[0,"Degree"]=str(best_params["degree"])
    run_results.loc[0,"gamma"]=str(best_params["gamma"])
    run_results.loc[0,"max_depth"]=str(best_params["max_depth"])
    run_results.loc[0,"n_estimators"]=str(best_params["n_estimators"])
    run_results.loc[0,"min_samples_split"]=str(best_params["min_samples_split"])
    run_results.loc[0,"min_samples_leaf"]=str(best_params["min_samples_leaf"])
    run_results.loc[0,"r2_train"]=round(r2_train,3)
    run_results.loc[0,"r2_test"]=round(r2_test,3)
    run_results.loc[0,"mean_error_train"]=round(mean_error_train,3)
    run_results.loc[0,"mean_error_test"]=round(mean_error_test,3)
    if  (output==run_results.loc[0,:]).all(axis=1).any():
        print("ALREADY_LOGED")
    else:
        print("New Run Results")
        print(run_results)

        output=pd.concat([output,run_results])    

    output.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\Imputation_Results.csv',index=False)
    return(run_results,output)
def imputation_prep(ch_orders,Freq):
    ch_orders=ch_orders.set_index("ProjectId")
    a=ch_orders.dtypes=="object"
    columns=ch_orders.columns
    freq_project_atrs=[x for x in columns if "sze" in x and "freq" in x]

    freq_class_atrs=[x for x in columns if "Frq" in x and "sze" in x]
    categorical_atr_list=list(a.loc[ch_orders.dtypes=="object"].index)
    # ch_orders_for_impute=ch_orders[["freq_t_prime_sze","freq_p_prime_sze","DurationModified","ProjectBaseContractValue","Classification_1","Classification_2","ProjectClassification","ProjectOperatingUnit","ProjectProvince","ProjectCity"]]
    imputation_atrs=["ProjectBaseContractValue","DurationModified"]
    ch_orders_for_impute=ch_orders[freq_project_atrs+freq_class_atrs+categorical_atr_list+["ProjectBaseContractValue","DurationModified"]]
    for atr_type in Freq:
        if atr_type=="freq_project":
            imputation_atrs=imputation_atrs+freq_project_atrs
        if atr_type=="freq_class":
            imputation_atrs=imputation_atrs+freq_class_atrs
        if atr_type=="cat":
            imputation_atrs=imputation_atrs+categorical_atr_list    
        if atr_type not in ["freq_class","freq_project","cat"]:
            raise ValueError("...Wrong Frequency Input...")

    ch_orders_for_impute=ch_orders_for_impute[imputation_atrs]
    b=ch_orders_for_impute.dtypes=="object"
    impute_cat_list=list(b.loc[ch_orders_for_impute.dtypes=="object"].index)
    ch_orders_for_impute=pd.get_dummies(ch_orders_for_impute,columns=impute_cat_list,drop_first=True)
    
    # Divide the data into two parts: one with missing values and one without
    data_missing = ch_orders_for_impute[ch_orders_for_impute["DurationModified"].isna()]
    data_not_missing = ch_orders_for_impute[~ch_orders_for_impute["DurationModified"].isna()]
    # # Split the data into target variable and features
    X_not_missing = data_not_missing.drop(columns=["DurationModified"])
    y_not_missing = data_not_missing["DurationModified"]
    x_missing = data_missing.drop(columns=["DurationModified"])
    x_train,x_test,y_train,y_test=train_test_split(X_not_missing,y_not_missing,test_size=.1,random_state=0)
    return(ch_orders,data_missing,x_test,x_train,x_missing,y_train,y_test)
def SVM_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel):
    sc=StandardScaler().fit(x_train)
    x_train_std=sc.transform(x_train)
    x_test_std=sc.transform(x_test)
    x_missing_std=sc.transform(x_missing)
    # # Train a linear regression model on the data without missing values
    # reg = LinearRegression().fit(x_train_std, y_train)
    reg=SVR(kernel=kernel)
    params={"C":(1,0.1,0.01),"degree":(2,3,4),"gamma":(1,0.1)}
    svm_grid=GridSearchCV(reg,params,n_jobs=-1,cv=2,verbose=-1,scoring="r2")
    svm_grid.fit(x_train_std,y_train)
    best_params=svm_grid.best_params_
    svm_clf=svm_grid.best_estimator_
    # # Use the trained model to predict the missing values
    y_test_pred=svm_clf.predict(x_test_std)
    y_train_pred=svm_clf.predict(x_train_std)

    r_square_test=r2_score(y_test,y_test_pred)
    r_square_train=r2_score(y_train,y_train_pred)
    mean_error_train=mean_squared_error(y_train,y_train_pred)
    mean_error_test=mean_squared_error(y_test,y_test_pred)
    # imputed_values = reg.predict(X_missing)
    imputed_values = svm_clf.predict(x_missing_std)

    return(r_square_train,r_square_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values)
def MLR_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel):
    best_params={}
    sc=StandardScaler().fit(x_train)
    x_train_std=sc.transform(x_train)
    x_test_std=sc.transform(x_test)
    x_missing_std=sc.transform(x_missing)
    # # Train a linear regression model on the data without missing values
    # reg = LinearRegression().fit(x_train_std, y_train)
    mlr_model = LinearRegression()
    mlr_model.fit(x_train_std,y_train)

    # # Use the trained model to predict the missing values
    y_test_pred=mlr_model.predict(x_test_std)
    y_train_pred=mlr_model.predict(x_train_std)

    r_square_test=r2_score(y_test,y_test_pred)
    r_square_train=r2_score(y_train,y_train_pred)
    mean_error_train=mean_squared_error(y_train,y_train_pred)
    mean_error_test=mean_squared_error(y_test,y_test_pred)
    # imputed_values = reg.predict(X_missing)
    imputed_values = mlr_model.predict(x_missing_std)

    return(r_square_train,r_square_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values)
def RF_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel):
    sc=StandardScaler().fit(x_train)
    x_train_std=sc.transform(x_train)
    x_test_std=sc.transform(x_test)
    x_missing_std=sc.transform(x_missing)
     # Training a linear regression model on the data without missing values
    param_grid = {
       'n_estimators': [12],
       'max_depth': [9],
        'min_samples_split': [6],
        'min_samples_leaf': [ 3],
        # 'max_features': ['auto', 'sqrt', 'log2']
    }
    rf_regressor = RandomForestRegressor(random_state=42)
    RF_grid=GridSearchCV(rf_regressor,param_grid,n_jobs=-1,cv=2,scoring='r2')
    RF_grid.fit(x_train,y_train)
    best_params=RF_grid.best_params_
    RF_best=RF_grid.best_estimator_
     # Useing the trained model to predict the missing values
    y_test_pred=RF_best.predict(x_test)
    y_train_pred=RF_best.predict(x_train)

    r_square_test=r2_score(y_test,y_test_pred)
    r_square_train=r2_score(y_train,y_train_pred)
    mean_error_train=mean_squared_error(y_train,y_train_pred)
    mean_error_test=mean_squared_error(y_test,y_test_pred)
    imputed_values = RF_best.predict(x_missing)

    return(r_square_train,r_square_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values)

def KNN_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel):
    sc=StandardScaler().fit(x_train)
    x_train_std=sc.transform(x_train)
    x_test_std=sc.transform(x_test)
    x_missing_std=sc.transform(x_missing)
    # # Train a linear regression model on the data without missing values
    param_grid = {'n_neighbors': [ 3,] }
    
    KNN_regressor = KNeighborsRegressor()
    KNN_grid=GridSearchCV(KNN_regressor,param_grid,n_jobs=-1,cv=2,verbose=-1,scoring='r2')
    KNN_grid.fit(x_train_std,y_train)
    best_params=KNN_grid.best_params_
    KNN_best=KNN_grid.best_estimator_
    # # Use the trained model to predict the missing values
    y_test_pred=KNN_best.predict(x_test_std)
    y_train_pred=KNN_best.predict(x_train_std)

    r_square_test=r2_score(y_test,y_test_pred)
    r_square_train=r2_score(y_train,y_train_pred)
    mean_error_train=mean_squared_error(y_train,y_train_pred)
    mean_error_test=mean_squared_error(y_test,y_test_pred)
    # imputed_values = reg.predict(X_missing)
    imputed_values = KNN_best.predict(x_missing_std)

    return(r_square_train,r_square_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values)
def ANN_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel):
    best_params={}
    sc=StandardScaler().fit(x_train)
    x_train_std=sc.transform(x_train)
    x_test_std=sc.transform(x_test)
    x_missing_std=sc.transform(x_missing)
    # # Train a linear regression model on the data without missing values
    # Create ANN model
    model = Sequential()
    model.add(Dense(units=125, activation='relu', input_dim=x_train_std.shape[1]))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))  # Output layer for regression

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train_std, y_train, epochs=50, batch_size=32, verbose=1)
    # # Use the trained model to predict the missing values
    y_test_pred=model.predict(x_test_std)
    y_train_pred=model.predict(x_train_std)

    r_square_test=r2_score(y_test,y_test_pred)
    r_square_train=r2_score(y_train,y_train_pred)
    mean_error_train=mean_squared_error(y_train,y_train_pred)
    mean_error_test=mean_squared_error(y_test,y_test_pred)
    # imputed_values = reg.predict(X_missing)
    imputed_values = model.predict(x_missing_std)
    print(model.summary())
    return(r_square_train,r_square_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values)
def run_the_code(algorythm,kernel_str,Freq):
    
    ch_orders=read_df()   
    ch_orders=duration_modification(ch_orders)
    ch_orders=frequency_normalization(ch_orders)
    ch_orders=ch_freq(ch_orders)
    ch_orders,data_missing,x_test,x_train,x_missing,y_train,y_test=imputation_prep(ch_orders,Freq)
    if algorythm=="SVM":
        r2_train,r2_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values=SVM_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel_str)
    elif algorythm=="MLR":
        r2_train,r2_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values=MLR_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel_str)
    elif algorythm=="KNN":
        r2_train,r2_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values=KNN_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel_str)
    elif algorythm=="RF":
        r2_train,r2_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values=RF_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel_str)
    elif algorythm=="ANN":
        r2_train,r2_test,mean_error_train,mean_error_test,best_params,x_train,imputed_values=ANN_regression(data_missing,x_test,x_train,x_missing,y_train,y_test,kernel_str)
    else:
        raise ValueError("Wrong Algorythm Input")
    
    run,outpu=output(algorythm,Freq,kernel_str,best_params,r2_train,r2_test,mean_error_train,mean_error_test)
    # Replace the missing values in the original data with the imputed values
    data_missing["DurationModified"]=imputed_values
    dictionary=dict(zip(data_missing.index,data_missing["DurationModified"]))
    ch_orders["DurationModified"]=np.where(ch_orders["DurationModified"].isna(),ch_orders.index.map(dictionary),ch_orders["DurationModified"])
    return(run,outpu,dictionary,r2_train,r2_test,mean_error_train,mean_error_test,best_params,ch_orders,x_train,imputed_values)


run,outpu,dictionary,r2_train,r2_test,mean_error_train,mean_error_test,best_params,ch_orders,x_train,imputed_values=\
    run_the_code(algorythm="RF",kernel_str="rbf",Freq=["freq_project","freq_class"])


ch_orders=ch_orders.loc[(ch_orders["missing_per2_up"]<0.25) & (ch_orders["missing_per2_low"]<0.25)]



ch_orders=frequency_normalization(ch_orders)
ch_orders=ch_freq(ch_orders)
print(ch_orders.info())
ch_orders.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_2_imputed_duration.csv')
a=ch_orders["ProjectBillingType"].value_counts()