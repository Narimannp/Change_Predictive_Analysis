# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:28:28 2023

@author: narim
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
"Read external datasets, 1-Projects,2-Canadacities\
    removes Service projects+removes the projecttype atr+filters change orders to those with cost effect"
def read_df():
    
    projects =pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\6_data_prep_project.csv')
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\6_data_prep_ch_orders.csv')
    ch_imputed=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv',)
    projects=projects.loc[projects["ProjectType"]=="Construction",:]
    construction_ids=list(set(projects.index))
    projects.drop("ProjectType",axis=1,inplace=True)
    ch_orders=ch_orders[ch_orders["Amount"]!=0]
    construction_ids=list(set(projects["ProjectId"]))
    ch_orders=ch_orders[ch_orders["ProjectId"].isin(construction_ids) ]
    return(projects,ch_orders,ch_imputed)
"The function to map each change order to its proper time-snap of project duration"
def time_grouping(row):
    if row["DateCreated"]<row["12.5%"]:
        return("1")
    elif row["DateCreated"]<row["25%"]:
        return("2")
    elif row["DateCreated"]<row["37.5%"]:
        return("3")
    elif row["DateCreated"]<row["50%"]:
        return("4")
    elif row["DateCreated"]<row["62.5%"]:
        return("5")
    elif row["DateCreated"]<row["75%"]:
        return("6")
    elif row["DateCreated"]<row["87.5%"]:
        return("7")
    else:
        return("8")
"renames the new ATRS generated from old CLASSIFICATION Atr and renames other atrs\
    change the start and end time of projects to first and last change order for\
        projects that a change order is happening out of duration period\
            changes the duration to what imputed in step-8\
                creates the target devisions of duration and selects the atributes needed from project DF"
def data_prep(projects,ch_orders,ch_imputed):
    projects.set_index("ProjectId",inplace=True)
    ch_orders.set_index("ProjectId",inplace=True)
    projects.rename(columns={"Classification_1":"IndustryGroup","Classification_2":"ProjectType","ProjectExpectedStartDate":"Start","ProjectExpectedEndDate":"End","ProjectCity":"City","ProjectProvince":"Province","ProjectOperatingUnit":"OperatingUnit","ProjectBaseContractValue":"ContractValue"},inplace=True)
    projects["End"]=pd.to_datetime(projects["End"],format="%Y-%m-%d")
    projects["Start"]=pd.to_datetime(projects["Start"],format="%Y-%m-%d")
    ch_orders=ch_orders[["Amount","DateCreated","Type"]]
    ch_orders=ch_orders[ch_orders["Type"]=="PrimeContractChangeOrder"]
    projects.loc[:,"ImputedDuration"]=projects.index.map(dict(zip(ch_imputed["ProjectId"],ch_imputed["DurationModified"])))
    
    projects.loc[projects['first_ch_date'] < projects['Start'], 'Start'] = projects['first_ch_date']
    projects.loc[projects['last_ch_date'] > projects['End'], 'End'] = projects['last_ch_date']
    projects["DurationNew"]=projects["ImputedDuration"]

    projects["12.5%"]= (projects['Start'] + pd.to_timedelta(projects['DurationNew']/8, unit='D'))
    projects["25%"]= (projects['Start'] + pd.to_timedelta(projects['DurationNew']*2/8, unit='D'))
    projects["37.5%"]= (projects['Start'] + pd.to_timedelta(projects['DurationNew']*3/8, unit='D'))
    projects["50%"]= (projects['Start'] + pd.to_timedelta(projects['DurationNew']*4/8, unit='D'))
    projects["62.5%"]= (projects['Start'] + pd.to_timedelta(projects['DurationNew']*5/8, unit='D'))
    projects["75%"]= (projects['Start'] + pd.to_timedelta(projects['DurationNew']*6/8, unit='D'))
    projects["87.5%"]= (projects['Start'] + pd.to_timedelta(projects['DurationNew']*7/8, unit='D'))
    projects["Start"]=projects["Start"]
    projects["End"]=projects["End"]
    ch_orders["DateCreated"]=pd.to_datetime(ch_orders["DateCreated"],format="%Y-%m-%d")
    projects=projects[["ContractValue","DurationNew","IndustryGroup","ProjectType","OperatingUnit","Province","City","Start","12.5%","25%","37.5%","50%","62.5%","75%","87.5%","End"]]

    return(projects,ch_orders,ch_imputed)
"merges the atributes"
def merge(ch_orders,projects):
    merged=ch_orders.merge(projects, left_index=True,right_index=True, how="outer")
    return(merged)

"maps each change oder to its corresponding target time-frame of project and then\
    groups change orders of each project based on their time group and\
        uses pivot-table to calculate count and amount of change orders in each time group"
def group(ch_orders,projects):
    merged=ch_orders.merge(projects, left_index=True,right_index=True, how="outer")
    merged["TimeGroup"]=merged.apply(time_grouping,axis=1)
    merged["Sign"]=np.where(merged["Amount"]>0,"+","-")
    ch_grouped=merged.groupby([merged.index,"TimeGroup","Sign"]).size().reset_index(name="Count")
    pivot_table_count = ch_grouped.pivot_table(index=ch_grouped['ProjectId'], columns=["Sign","TimeGroup"], values=['Count'], aggfunc='sum', fill_value=0)
    pivot_table_count.columns=[col[1]+"_ChCount_"+col[2] for col in pivot_table_count.columns]
    ch_grouped_amount=merged.groupby([merged.index,"TimeGroup"])["Amount"].sum().reset_index(name="Amount")
    pivot_table_amount = ch_grouped_amount.pivot_table(index=ch_grouped_amount['ProjectId'], columns=["TimeGroup"], values=['Amount'], aggfunc='sum', fill_value=0)
    pivot_table_amount.columns=["Ch"+col[0]+"_"+col[1] for col in pivot_table_amount.columns]
    pivot_table_amount_2=pd.DataFrame()
    pivot_table_amount_2[0]=pivot_table_amount.iloc[:,0]
    for i in range (pivot_table_amount.shape[1]-1):
        pivot_table_amount_2[i+1]=pivot_table_amount_2.iloc[:,i]+pivot_table_amount.iloc[:,i+1]
    pivot_table_amount.iloc[:,:]=pivot_table_amount_2.iloc[:,:]
    merged_all=pd.concat([projects,pivot_table_amount,pivot_table_count],axis=1)
    for i in range (pivot_table_amount.shape[1]):
        merged_all["ChPercentage_"+str(i+1)]=100*merged_all["ChAmount_"+str(i+1)]/merged_all["ContractValue"]

    return(merged_all)

"divides the Classification atr to the two distinct atrs"
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

"normalise the number of change orders of each time group by its duration and project value"
def frequency_normalization(merged):
    freq_atrs=[x for x in merged.columns if "Count" in x]
    for atr in freq_atrs:
        merged["freq"+atr.split("_")[0]+"dur@"+atr.split("_")[2]]=merged[atr]/merged["DurationNew"]
        merged["freq"+atr.split("_")[0]+"sze@"+atr.split("_")[2]]=1000*merged[atr]/merged["ContractValue"]
 
    return(merged)
"creates the frequency atrs for all the categorical atributes and all the time frames"  
def ch_freq(df):
    categorical_attributes = df.select_dtypes(include=['object'])
    for atr in categorical_attributes.columns:
        for i in range (8):
           dict_cls_p_d=df.groupby(atr)["freq+dur@"+str(i+1)].mean()
           dict_cls_p_s=df.groupby(atr)["freq+sze@"+str(i+1)].mean()
           dict_cls_n_d=df.groupby(atr)["freq-dur@"+str(i+1)].mean()
           dict_cls_n_s=df.groupby(atr)["freq-sze@"+str(i+1)].mean()
           df["Frq_"+str(atr)+"_p_dur@"+str(i+1)]=df[atr].map(dict_cls_p_d)
           df["Frq_"+str(atr)+"_p_sze@"+str(i+1)]=df[atr].map(dict_cls_p_s)
           df["Frq_"+str(atr)+"_n_dur@"+str(i+1)]=df[atr].map(dict_cls_n_d)
           df["Frq_"+str(atr)+"_n_sze@"+str(i+1)]=df[atr].map(dict_cls_n_s)
    drop_atr_list=[atr for atr in df.columns if any(key in atr for key in ["Amount","freq","Count","%","Start","End"])]
    df.drop(drop_atr_list,axis=1,inplace=True)
    return(df,drop_atr_list) 
"runs the code"
def run():
    projects,ch_orders,ch_imputed=read_df()
    projects=divide_classification(projects)
    projects,ch_orders,ch_imputed=data_prep(projects,ch_orders,ch_imputed)
    merged=group(ch_orders,projects)
    merged=frequency_normalization(merged)
    merged,drop_atr_list=ch_freq(merged)
    merged.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\ph2_ChangeTime_Step1_DataPrep_8steps.csv',index=True)
    return(projects,ch_orders,ch_imputed,merged,drop_atr_list)
projects,ch_orders,ch_imputed,merged,drop_atr_list=run()

