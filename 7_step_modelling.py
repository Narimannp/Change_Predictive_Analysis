# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:38:10 2023

@author: narim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def read_df():
    #Read external datasets, 1-Projects,2-Canadacities
    projects =pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\6_data_prep_project.csv')
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\6_data_prep_ch_orders.csv')
    return(projects,ch_orders)


def prep_for_classification(ch_orders):

    ch_orders["CountChange"]=0
    ch_orders["CountChange_p"]=0
    ch_orders["CountChange_n"]=0
    ch_orders=ch_orders[ch_orders["Amount"]!=0]
    ch_orders["sign"]=np.where(ch_orders["Amount"]>0,"p","n")
    ch_orders=ch_orders.groupby(["ProjectId","Type","sign"],as_index=False).agg({
           'ProjectBaseContractValue':"last",
              'ProjectProvince':"last","DailyCost":"last", 'ProjectCity':"last", 'population':"last", 'density':"last",
              'ProjectClassification':"last", 'ProjectBillingType':"last","missing_per2_up":"last","missing_per2_low":"last",'DurationDiff_Divided_Duration':"last",
              'ProjectOperatingUnit':"last", 'ProjectType':"last","ChangeDuration":"last","CountChange":"count", "Duration":"last",'Amount' :"sum" })
    ch_orders["ChPercentage"]=ch_orders["Amount"]*100/ch_orders["ProjectBaseContractValue"]
    a=ch_orders
    ch_orders_prime_p=ch_orders[(ch_orders["Type"]=="PrimeContractChangeOrder") & (ch_orders["sign"]=="p")]
    ch_orders_prime_t=ch_orders[(ch_orders["Type"]=="PrimeContractChangeOrder")]
    ch_orders_prime_n=ch_orders[(ch_orders["Type"]=="PrimeContractChangeOrder") & (ch_orders["sign"]=="n")]
    ch_orders_prime_t=ch_orders_prime_t.groupby(["ProjectId"],as_index=False).agg({"ChPercentage":"sum"})
    prime_percentage_dict=dict(zip(ch_orders_prime_t["ProjectId"],ch_orders_prime_t["ChPercentage"]))
    prime_freq_p_dict=dict(zip(ch_orders_prime_p["ProjectId"],ch_orders_prime_p["CountChange"]))
    prime_freq_n_dict=dict(zip(ch_orders_prime_n["ProjectId"],ch_orders_prime_n["CountChange"]))
    
    ch_orders_commit_t=ch_orders[ch_orders["Type"]=="CommitmentContractChangeOrder"]
    ch_orders_commit_t=ch_orders_commit_t.groupby(["ProjectId"],as_index=False).agg({"ChPercentage":"sum"})
    ch_orders_commit_p=ch_orders[(ch_orders["Type"]=="CommitmentContractChangeOrder")& (ch_orders["sign"]=="p")]
    ch_orders_commit_n=ch_orders[(ch_orders["Type"]=="CommitmentContractChangeOrder")& (ch_orders["sign"]=="n")]
    commit_percentage_dict=dict(zip(ch_orders_commit_t["ProjectId"],ch_orders_commit_t["ChPercentage"]))
    commit_freq_p_dict=dict(zip(ch_orders_commit_p["ProjectId"],ch_orders_commit_p["CountChange"]))
    commit_freq_n_dict=dict(zip(ch_orders_commit_n["ProjectId"],ch_orders_commit_n["CountChange"])) 
    
    ch_orders_sales_t=ch_orders[ch_orders["Type"]=="Sales Order"]
    ch_orders_sales_t=ch_orders_sales_t.groupby(["ProjectId"],as_index=False).agg({"ChPercentage":"sum"})
    ch_orders_sales_p=ch_orders[(ch_orders["Type"]=="Sales Order")& (ch_orders["sign"]=="p")]    
    ch_orders_sales_n=ch_orders[(ch_orders["Type"]=="Sales Order")& (ch_orders["sign"]=="n")]
    sales_percentage_dict=dict(zip(ch_orders_sales_t["ProjectId"],ch_orders_sales_t["ChPercentage"]))
    sales_freq_p_dict=dict(zip(ch_orders_sales_p["ProjectId"],ch_orders_sales_p["CountChange"]))
    sales_freq_n_dict=dict(zip(ch_orders_sales_n["ProjectId"],ch_orders_sales_n["CountChange"]))
   
    ch_orders=ch_orders.groupby(["ProjectId"],as_index=False).agg({
           'ProjectBaseContractValue':"last",
              'ProjectProvince':"last", 'ProjectCity':"last","DailyCost":"last", 'population':"last", 'density':"last",
              'ProjectClassification':"last", 'ProjectBillingType':"last", "missing_per2_up":"last","missing_per2_low":"last",'DurationDiff_Divided_Duration':"last",
              'ProjectOperatingUnit':"last", 'ProjectType':"last","ChangeDuration":"last","Duration":"last",'ChPercentage' :"sum" })
    ch_orders["PrimeChPer"]=np.where(ch_orders["ProjectId"].isin(prime_percentage_dict.keys()),ch_orders["ProjectId"].map(prime_percentage_dict),0)
    ch_orders["CommitChPer"]=np.where(ch_orders["ProjectId"].isin(commit_percentage_dict.keys()),ch_orders["ProjectId"].map(commit_percentage_dict),0)
    ch_orders["SalesChPer"]=np.where(ch_orders["ProjectId"].isin(sales_percentage_dict.keys()),ch_orders["ProjectId"].map(sales_percentage_dict),0)
    
    ch_orders["PrimeChFreq_p"]=np.where(ch_orders["ProjectId"].isin(prime_freq_p_dict.keys()),ch_orders["ProjectId"].map(prime_freq_p_dict),0)
    ch_orders["CommitChFreq_p"]=np.where(ch_orders["ProjectId"].isin(commit_freq_p_dict.keys()),ch_orders["ProjectId"].map(commit_freq_p_dict),0)
    ch_orders["SalesChFreq_p"]=np.where(ch_orders["ProjectId"].isin(sales_freq_p_dict.keys()),ch_orders["ProjectId"].map(sales_freq_p_dict),0)

    ch_orders["PrimeChFreq_n"]=np.where(ch_orders["ProjectId"].isin(prime_freq_n_dict.keys()),ch_orders["ProjectId"].map(prime_freq_n_dict),0)
    ch_orders["CommitChFreq_n"]=np.where(ch_orders["ProjectId"].isin(commit_freq_n_dict.keys()),ch_orders["ProjectId"].map(commit_freq_n_dict),0)
    ch_orders["SalesChFreq_n"]=np.where(ch_orders["ProjectId"].isin(sales_freq_n_dict.keys()),ch_orders["ProjectId"].map(sales_freq_n_dict),0)
    
    ch_orders_cate_attr=["ProjectProvince","ProjectCity","ProjectClassification","ProjectOperatingUnit","ProjectType","ProjectBillingType"]
    ch_orders[ch_orders_cate_attr]=ch_orders[ch_orders_cate_attr].astype("category")
    # ch_orders_for_freq_classification.set_index(keys=["ProjectId","Type"],inplace=True)
    # ch_orders_for_freq_classification["Change"]
    ch_orders=ch_orders.rename(columns={"density":"Density","population":"Population","ChPercentage":"TotalChPer"})
    # ch_orders["TotalChFreq"]=np.where(ch_orders["ChangeDuration"].isna(),0,    ch_orders["TotalChFreq"])
    return(ch_orders,a)

def project_filter(projects,ch_orders,atr,atr_class_list):
    projects=projects[projects[atr].isin(atr_class_list)]
    projects_filter_list=projects["ProjectId"]
    ch_orders=ch_orders[ch_orders["ProjectId"].isin(projects_filter_list)]
    return(projects,ch_orders)
    
def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

def pearson_spearman(df):
    df=df["Date_Diff,Population".split(",")]
    for column in df:
       #Range normalizing attributes
       df[column]=(df[column]-df[column].min())/(df[column].max()-df[column].min())
    pearson=df.corr(method='pearson', min_periods=1)
    spearman=df.corr(method='spearman', min_periods=1)
    return(pearson,spearman)
# def 
# def label_target_atr(ch_orders,low_lvl,high_lvl):
#     ch_orders["PrimeChLvl"]=np.where(ch_orders["PrimeChPer"]>high_lvl,"high","low")
#     ch_orders["PrimeChLvl"]=np.where(((ch_orders["PrimeChPer"]>low_lvl) & (ch_orders["PrimeChPer"]<high_lvl)),"med",ch_orders["PrimeChLvl"])
    
#     ch_orders["CommitChLvl"]=np.where(ch_orders["CommitChPer"]>high_lvl,"high","low")
#     ch_orders["CommitChLvl"]=np.where(((ch_orders["CommitChPer"]>low_lvl) & (ch_orders["CommitChPer"]<high_lvl)),"med",ch_orders["CommitChLvl"])
    
#     ch_orders["SalesChLvl"]=np.where(ch_orders["SalesChPer"]>high_lvl,"high","low")
#     ch_orders["SalesChLvl"]=np.where(((ch_orders["SalesChPer"]>low_lvl) & (ch_orders["SalesChPer"]<high_lvl)),"med",ch_orders["SalesChLvl"])
    
#     ch_orders["TotalChLvl"]=np.where(ch_orders["TotalChPer"]>high_lvl,"high","low")
#     ch_orders["TotalChLvl"]=np.where(((ch_orders["TotalChPer"]>low_lvl) & (ch_orders["TotalChPer"]<high_lvl)),"med",ch_orders["TotalChLvl"])
#     return(ch_orders)
def divide_classification(ch_orders):
    ch_orders["a"]=ch_orders["ProjectClassification"].apply(lambda x:len(x.split(".")))
    ch_orders["ProjectClassification"]=ch_orders["ProjectClassification"].astype(str)
    ch_orders["ProjectClassification_new"]=np.where(ch_orders["a"]==1,ch_orders["ProjectClassification"]+". ",ch_orders["ProjectClassification"])
    # ch_orders[['Column1_1', 'Column1_2']] = ch_orders["ProjectClassification"].str.split('.', expand=True)
    ch_orders["Classification_1"]=ch_orders["ProjectClassification_new"].apply(lambda x:x.split(".")[0])
    ch_orders["Classification_1"]=    ch_orders["Classification_1"].replace("COMMERCIAL","COMM")
    ch_orders["Classification_2"]=np.where(ch_orders["ProjectClassification_new"].apply(lambda x:len(x.split(".")))>1,ch_orders["ProjectClassification_new"].apply(lambda x:x.split(".")[1]),"")
    # ch_orders["classification_2"]=np.where(ch_orders["classification_2"]=="",ch_orders["classification_2"],ch_orders["classification_2"].apply(lambda x:x[1]))
    # ch_orders["classification_2"]=ch_orders["ProjectClassification"].apply(lambda x:len(x.split(".")))
    # ch_orders.drop(columns=["ProjectClassification","a"],axis=1,inplace=True)
    return(ch_orders)



def run_the_code():

    projects,ch_orders=read_df()  
    ch_orders,a=prep_for_classification(ch_orders)
    ch_orders=divide_classification(ch_orders)


    ch_orders.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\7_data_prep_ch_orders.csv',index=False)
    return(projects,ch_orders,a)
projects,ch_orders,a=run_the_code()
# dist_prime_ch_lvl=ch_orders.groupby("PrimeChLvl").count()
dist_project_oper_unit=ch_orders.groupby("ProjectOperatingUnit").count()
oper_analys=projects.groupby("ProjectOperatingUnit").agg({"DailyCost":["count","mean","min","max"],"Duration":["count","mean","min","max"],"ProjectBaseContractValue":["count","mean","min","max"]})


