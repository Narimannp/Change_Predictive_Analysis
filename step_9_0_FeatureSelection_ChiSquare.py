# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:13:40 2023

@author: narim
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Read and store the DataFrame
def read_df():
    ch_orders_orig=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    ch_orders=ch_orders_orig[['ProjectId', 'ProjectBaseContractValue', 'ProjectProvince',
            'ProjectCity', 'Population', 'Density',
             'ProjectBillingType','ProjectOperatingUnit', \
            'ProjectType','DurationModified', 'TotalChPer', 'PrimeChPer', 'CommitChPer', 'SalesChPer',\
                    "ProjectClassification","Classification_1","Classification_2",\
                    "Freq_Class1_p_dur","Freq_Class1_p_sze","Freq_Class1_n_dur","Freq_Class1_n_sze","Freq_Class2_p_dur","Freq_Class2_p_sze",\
                    "Freq_Class2_n_dur","Freq_Class2_n_sze","Freq_Prov_p_dur","Freq_Prov_p_sze",\
                        "Freq_Prov_n_dur","Freq_Prov_n_sze","Freq_City_p_dur","Freq_City_p_sze","Freq_City_n_dur"\
                            ,"Freq_City_n_sze"]]
    ch_orders.rename(columns={"ProjectBaseContractValue":"BaseValue","ProjectOperatingUnit":"OperatingUnit","DurationModified":"Duration",\
                                   "ProjectBillingType":"BillType","ProjectCity":"City","ProjectProvince":"Province"},inplace=True)
#adding project id as index to be excluded from analysis
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)
ch_or=read_df()
def categorical_numerical_atributes(df):
    categorical=df.dtypes=="object"
    categorical_atrs=list(df.columns[categorical])
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numerics)
    numeric_atrs=list(numeric_df.columns)
    return(numeric_atrs,categorical_atrs)

def label_target_atr(ch_orders,boundry,existance_or_lvl,prime_commit):
    target_atr=prime_commit+"Ch"+existance_or_lvl
    if existance_or_lvl=="Multi-CLass":
        # ch_orders=ch_orders[ch_orders[prime_commit+"ChPer"]!=0]
        print("Multi-Class Classifiction")
        ch_orders[target_atr]=np.where(ch_orders[prime_commit+"ChPer"]<=boundry[0],0,1)
        ch_orders[target_atr]=np.where(((ch_orders[prime_commit+"ChPer"]<=boundry[1]) & (ch_orders["PrimeChPer"]>boundry[0])),1,ch_orders[target_atr])
        ch_orders[target_atr]=np.where((ch_orders[prime_commit+"ChPer"]>boundry[1]),2,ch_orders[target_atr])

    elif existance_or_lvl=="Binary":
        print("Binary Classifiction")
        ch_orders[target_atr]=np.where(((ch_orders["PrimeChPer"]>boundry)),1,0)
    else:
        print("Wrong Input...")
        
    return(ch_orders)

def project_filter(ch_orders,atr,atr_class_list):
    ch_orders=ch_orders[ch_orders[atr]==(atr_class_list)]
    return(ch_orders)

# Perform the chi-square test for each categorical attribute against the target variable
def chi_square(df,target_atr):
    chi_square_result=pd.DataFrame()
    for column in df.columns:
        if column != target_atr:
            contingency_table = pd.crosstab(df[column], df[target_atr])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            chi_square_result.loc[str(column),"chi_value"]=chi2
            chi_square_result.loc[str(column),"p_value"]=p_value
    chi_square_result=chi_square_result.sort_values(by="chi_value",ascending=False)
    top_five_atrs=chi_square_result.head(2).index.tolist()
    return(chi_square_result,top_five_atrs)

def run_the_code():
    ch_orders=read_df()
    ch_orders=project_filter(ch_orders,"ProjectType","Construction")
    numeric_atrs,categorical_atrs=categorical_numerical_atributes(ch_orders)
    ch_orders=label_target_atr(ch_orders,2,"Binary","Prime")
    categorical_atrs.append("PrimeChPer")
    ch_orders=ch_orders[categorical_atrs]
    chi_square_result,top_five_atrs=chi_square(ch_orders,"PrimeChPer")
    return (ch_orders,chi_square_result,top_five_atrs)
ch_orders,chi_square_result,top_five_cat_atrs=run_the_code()


