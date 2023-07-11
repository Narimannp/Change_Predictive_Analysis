# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:54:22 2023

@author: narim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

    #Reads Existing datasets, 1-Projects,2-ChangeOrders
def read_df():
    projects =pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\5_data_prep_project.csv')
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\5_data_prep_ch_orders.csv')
    return(projects,ch_orders)

# Prepers the attributes' types and missing datapoints
def attribute_preperation(projects):
    projects["ProjectClassification"]=projects["ProjectClassification"].apply(lambda x:x.upper())
    projects["first_ch_date"]=pd.to_datetime(projects["first_ch_date"],format="%Y-%m-%d")
    projects["last_ch_date"]=pd.to_datetime(projects["last_ch_date"],format="%Y-%m-%d")
    projects["ChangeDuration"]=projects["last_ch_date"]-projects["first_ch_date"]
    projects_w_ch=projects[projects["first_ch_date"].notna()]
    projects_wh_ch=projects[projects["first_ch_date"].isna()]
    projects_wh_ch["ChangeDuration"]=np.nan
    projects_w_ch["ChangeDuration"]= projects_w_ch["ChangeDuration"].apply(lambda x:int(str(x).split(" ")[0]))
    projects=pd.concat([projects_w_ch,projects_wh_ch],ignore_index=True)
    projects=projects[["ProjectId","ProjectBaseContractValue","Duration","ChangeDuration","ProjectExpectedStartDate","ProjectExpectedEndDate","first_ch_date","last_ch_date","missing_per2_up","missing_per2_low","ProjectProvince","ProjectCity","population","density","ProjectClassification","ProjectBillingType","ProjectDepartment","ProjectOperatingUnit","ProjectType",]]
    return (projects)

#Removes the projects with start and end point not aligned with our analysis
def projects_out_of_date_boundry(ch_orders,ch_events,projects):
    projects["ProjectExpectedStartDate"]=projects["ProjectExpectedStartDate"].apply(lambda x:str(x).split(" ")[0])
    projects["ProjectExpectedStartDate"]=pd.to_datetime(projects["ProjectExpectedStartDate"],format="%Y-%m-%d")
    projects["ProjectExpectedEndDate"]=projects["ProjectExpectedEndDate"].apply(lambda x:str(x).split(" ")[0])
    projects["ProjectExpectedEndDate"]=pd.to_datetime(projects["ProjectExpectedEndDate"],format="%Y-%m-%d")
    low_bound=pd.to_datetime("2017-05-05",format="%Y-%m-%d")
    upper_bound=pd.to_datetime("2022-06-07",format="%Y-%m-%d")
    projects_list=projects[(pd.to_datetime(projects["ProjectExpectedStartDate"])<low_bound) | (pd.to_datetime(projects["ProjectExpectedEndDate"])>upper_bound)]
    print("STEP_4 IS DONE ....")
    return(ch_orders,ch_events,projects)

#Given the DF and list of attributes, removes the ouyliers using the IQR method
def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

def duration_filter(projects):
    projects_w_ch=projects[projects["first_ch_date"].notna()]
    projects_wh_ch=projects[projects["first_ch_date"].isna()]
    projects["DailyCost"]=np.where(projects["Duration"]!=0,projects["ProjectBaseContractValue"]/projects["Duration"],projects["ProjectBaseContractValue"])
    projects["DailyCost"]=projects["DailyCost"].apply(lambda x:abs(round(x)))
    projects["DurationDiff"]=projects["Duration"]-projects["ChangeDuration"]
    projects["DurationDiff_Divided_Duration"]=np.where(projects["Duration"]!=0,projects["DurationDiff"]/(projects["Duration"]),projects["DurationDiff"])
    projects=projects[['ProjectId', 'ProjectBaseContractValue', 'Duration', 'ChangeDuration',
           'missing_per2_up', 'missing_per2_low', 'DailyCost', 'DurationDiff',
           'DurationDiff_Divided_Duration' ,
           'ProjectClassification', 'ProjectBillingType', 'ProjectDepartment',
           'ProjectOperatingUnit', 'ProjectType','ProjectProvince', 'ProjectCity', 'population', 'density',
           'ProjectExpectedStartDate', 'ProjectExpectedEndDate', 'first_ch_date',
            'last_ch_date']]
    # conflict_Period_Duration=projects_w_ch[projects_w_ch["DurationDiff"]<0]
    # no_conflict_period_duration=projects[~projects["DurationDiff"]<0]
    # conflict_Period_Duration["diff_exp_ch_durations"]=abs(conflict_Period_Duration["diff_exp_ch_durations"])
    # conflict_Period_Duration["Portion_Changes_Out_Duration"]=abs(conflict_Period_Duration["Portion_Changes_Out_Duration"])
    # Date_Difference=conflict_Period_Duration[["diff_exp_ch_durations","Portion_Changes_Out_Duration"]].describe()
    # print(no_conflict_period_duration.info())
    return(projects)


def merge_ch_orders_projects(ch_orders,projects):
    ch_orders=projects.merge(ch_orders,on="ProjectId",how="outer")
    ch_orders["Type"]=ch_orders["Type"].fillna("NoChangeOrder")
    return(ch_orders)
# ch_orders=merge_ch_orders_projects(ch_orders,projects)

def categorical_classes(projects):  
    Data=list()

    categorical_atr=["ProjectClassification","ProjectBillingType","ProjectDepartment","ProjectOperatingUnit","ProjectType"]
    for column in categorical_atr:
        Data.append(list(set(projects[column])))
    categorical_atr_classes=pd.DataFrame(data=Data,index=categorical_atr)      .transpose()
    return(categorical_atr_classes)
  
def run_the_code():
    projects,ch_orders=read_df()
    projects=attribute_preperation(projects)
    projects=duration_filter(projects)
    ch_orders=merge_ch_orders_projects(ch_orders,projects)
    ch_orders=ch_orders[ch_orders["ProjectType"]=="Construction"]
    ch_orders.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\6_data_prep_ch_orders.csv',index=False)
    projects.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\6_data_prep_project.csv',index=False)
    return(ch_orders,projects) 


ch_orders,projects=run_the_code()
# ch_orders=ch_orders[ch_orders["ProjectType"]=="Construction"]
Data=categorical_classes(projects)
# # print(projects.columns)
# # problematic=projects[(projects["DurationDiff_1Divided_Duration"]!=0) & (projects["Duration"]>0)]
# # plt.clf()
# # sns.histplot(data=problematic, x="DurationDiff_Divided_Duration")
# a=projects["DailyCost"].describe()
# projects=projects[(projects["ProjectType"]=="Construction")]
# # minusduration=projects[(projects["Duration"]<0) & (projects["ProjectType"]=="Services")]
# projects["Duration"]=abs(projects["Duration"])
# projects=projects[projects["Duration"]>1]
# projects=outlier_remove(projects,["DailyCost"])
# pearson_con,spearman_con=pearson_spearman(projects,"ProjectBaseContractValue,DailyCost,Duration,ChangeDuration,DurationDiff")
# pearson_ser,spearman_ser=pearson_spearman(projects,"ProjectBaseContractValue,DailyCost,Duration,ChangeDuration,DurationDiff")
# plt.clf()
# sns.histplot(data=projects_out_boundry[projects_out_boundry["missing_per2_low"]!=0], x="missing_per2_low")