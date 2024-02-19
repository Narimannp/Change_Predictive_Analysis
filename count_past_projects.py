# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:03:00 2023

@author: narim
"""
import pandas as pd

"Reads datasets"
def read_df():
    projects =pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\5_data_prep_project.csv')
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    ch_orders.rename(columns={"ProjectBaseContractValue":"BaseValue","ProjectOperatingUnit":"OperatingUnit",\
                                   "ProjectBillingType":"BillType","ProjectCity":"City","ProjectProvince":"Province"},inplace=True)
    ch_orders=ch_orders.set_index("ProjectId")
    return(projects,ch_orders)
def pearson_spearman(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    for column in df:
       "Range normalizing attributes"
       df[column]=(df[column]-df[column].min())/(df[column].max()-df[column].min())
    pearson=df.corr(method='pearson', min_periods=1).round(2)
    spearman=df.corr(method='spearman', min_periods=1).round(2)
    return(pearson,spearman)
def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 
projects,ch_orders=read_df()
ch_orders["freq_t_prime_sze"]=100*(ch_orders["freq_p_prime_sze"]+ch_orders["freq_n_prime_sze"])
start_dict=dict(zip(projects["ProjectId"],projects["ProjectExpectedStartDate"]))
end_dict=dict(zip(projects["ProjectId"],projects["ProjectExpectedEndDate"]))
ch_orders["StartDate"]=ch_orders.index.map(start_dict)
ch_orders["EndDate"]=ch_orders.index.map(end_dict)
sorted_ch_or=ch_orders.sort_values(by=["City","StartDate"])
sorted_ch_or_2=ch_orders.sort_values(by=["City","EndDate"])
grouped=sorted_ch_or.groupby(by="City")
sorted_ch_or["CountPastProjects"]=0
count_dict={}
count_end_dict={}
for group,group_df in grouped:
    for idx,row in group_df.iterrows():
        count=group_df.loc[group_df["StartDate"]<row["StartDate"]]["StartDate"].count()
        count_end=group_df.loc[group_df["EndDate"]<row["EndDate"]]["EndDate"].count()
        count_dict[idx]=count
        count_end_dict[idx]=count_end
ch_orders["CountPastProjects"]=ch_orders.index.map(count_dict)
ch_orders["CountPastProjects_end"]=ch_orders.index.map(count_end_dict)
sorted_ch_or=sorted_ch_or[["City","StartDate","CountPastProjects"]]
ch_orders=ch_orders.loc[ch_orders["freq_t_prime_sze"]>0]

ch_oders=ch_orders.sort_values(by=["City","CountPastProjects"])
ch_oders=ch_oders[["BaseValue","DurationModified","ProjectClassification","Classification_1","Classification_2","OperatingUnit","DailyCost","PrimeChPer","CountPastProjects","City","freq_p_prime_sze","freq_n_prime_sze","freq_t_prime_sze","freq_p_prime_dur","freq_n_prime_dur"]]
ch_oders=outlier_remove(ch_oders, ["PrimeChPer"])
pearson,spearman=pearson_spearman(ch_oders)