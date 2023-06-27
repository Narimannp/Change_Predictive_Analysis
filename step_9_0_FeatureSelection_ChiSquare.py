# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:13:40 2023

@author: narim
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

"Reads datasets"
def read_df():
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    ch_orders.rename(columns={"ProjectBaseContractValue":"BaseValue","ProjectOperatingUnit":"OperatingUnit",\
                                   "ProjectBillingType":"BillType","ProjectCity":"City","ProjectProvince":"Province"},inplace=True)
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)

"Given the Df and types of attributes necessary, return the df with required attribute types"
def select_atrs(df,atr_types):
    loc_columns=[]
    temp_loc_columns=[]
    target_atrs=['PrimeChPer']
    orig_atrs="BaseValue,OperatingUnit,DurationModified,BillType,ProjectClassification,Classification_1,Classification_2,City,Province,ProjectType".split(",")
    loc_keys="City,Prov".split(",")
    loc_atrs_add="Population,Density".split(",")
    columns=df.columns
    columns=columns.astype(str).tolist()
    Freq_atrs=[x for x in columns if "Frq" in x ]
    for loc_key in loc_keys: 
        temp_loc_columns=[x for x in columns if loc_key in x ]
        loc_columns=loc_columns+temp_loc_columns
    df=df[target_atrs+orig_atrs]
    return(df)

"Given the df returns two list of categorical and numerical atrs"
def categorical_numerical_atributes(df):
    categorical=df.dtypes=="object"
    categorical_atrs=list(df.columns[categorical])
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numerics)
    numeric_atrs=list(numeric_df.columns)
    return(numeric_atrs,categorical_atrs)

"Given the DF, boundry or list of boundries, type of analysis, and Prime or Commit, Labels the target attribute and returns the DF"
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

"given the DF, the attribute, and specific CLass of the attribute,removes other classes from the atribute"
def project_filter(ch_orders,atr,atr_class_list):
    ch_orders=ch_orders[ch_orders[atr]==(atr_class_list)]
    return(ch_orders)

"Given the Df, number of best attributes,  and the target atribute, Performs the chi-square test for each categorical attribute against the target variable and returns the results with selected attributes"
def chi_square(df,target_atr,number_of_atrs):
    chi_square_result=pd.DataFrame()
    for column in df.columns:
        if column != target_atr:
            contingency_table = pd.crosstab(df[column], df[target_atr])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            chi_square_result.loc[str(column),"chi_value"]=chi2
            chi_square_result.loc[str(column),"p_value"]=p_value
    chi_square_result=chi_square_result.sort_values(by="chi_value",ascending=False)
    top_atrs=chi_square_result.head(number_of_atrs).index.tolist()
    return(chi_square_result,top_atrs)

def run_the_code(number_of_atrs):
    ch_orders=read_df()
    ch_orders=select_atrs(ch_orders,"")
    ch_orders=project_filter(ch_orders,"ProjectType","Construction")
    numeric_atrs,categorical_atrs=categorical_numerical_atributes(ch_orders)
    ch_orders=label_target_atr(ch_orders,2,"Binary","Prime")
    categorical_atrs.append("PrimeChPer")
    ch_orders=ch_orders[categorical_atrs]
    chi_square_result,top_five_atrs=chi_square(ch_orders,"PrimeChPer",number_of_atrs)
    return (ch_orders,chi_square_result,top_five_atrs)
ch_orders,chi_square_result,top_five_cat_atrs=run_the_code(2)


