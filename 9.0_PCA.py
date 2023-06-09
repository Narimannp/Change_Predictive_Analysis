# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:38:10 2023

@author: narim
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def read_df():
    #Read external datasets, 1-Projects,2-Canadacities
    ch_orders_orig=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    ch_orders=ch_orders_orig[['ProjectId', 'ProjectBaseContractValue', 'ProjectProvince',
            'ProjectCity', 'Population', 'Density',
             'ProjectBillingType','ProjectOperatingUnit', \
            'ProjectType','DurationModified', \
                    'ProjectClassification',"Classification_1","Classification_2",\
                    "Freq_Class_1_p_dur","Freq_Class_1_p_sze","Freq_Class_1_n_dur","Freq_Class_1_n_sze","Freq_Class_2_p_dur","Freq_Class_2_p_sze",\
                    "Freq_Class_2_n_dur","Freq_Class_2_n_sze","Freq_Prov_p_dur","Freq_Prov_p_sze",\
                        "Freq_Prov_n_dur","Freq_Prov_n_sze","Freq_p_City_dur","Freq_p_City_sze","Freq_n_City_dur"\
                            ,"Freq_n_City_sze"]]
    # ch_orders.drop(columns=["ProjectCity","DailyCost","ChangeDuration","TotalChFreq","PrimeChFreq","CommitChFreq",\
    #                  "CommitChFreq","SalesChFreq","TotalChPer","PrimeChPer","CommitChPer","SalesChPer"],axis=1,inplace=True)
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)



def project_filter(ch_orders,atr,atr_class_list):
    ch_orders=ch_orders[ch_orders[atr]==(atr_class_list)]
    return(ch_orders)

def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

def drop_atr(ch_orders,atr_list):
    ch_orders.drop(columns=atr_list,axis=1,inplace=True)
    return(ch_orders)
def numeric_atr_list(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    numeric_atr=list(df.columns)
    return(numeric_atr)


ch_orders=read_df()
ch_orders=project_filter(ch_orders,"ProjectType","Construction")
num_atr=numeric_atr_list(ch_orders)
ch_orders=outlier_remove(ch_orders,["ProjectBaseContractValue","DurationModified",'Population','Density'])
ch_orders=ch_orders[num_atr]
sc=StandardScaler().fit(ch_orders)
ch_orders_str=sc.transform(ch_orders)
ch_orders_str=pd.DataFrame(data=ch_orders_str,columns=ch_orders.columns,index=ch_orders.index)
pca = PCA(n_components=7)
pca.fit(ch_orders_str)
components = pca.components_
variance_ratios = pca.explained_variance_ratio_
singular_values = pca.singular_values_
transformed_data = pca.transform(ch_orders_str)
variance_ratios = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratios)
plt.scatter(range(1, len(variance_ratios) + 1), cumulative_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.show()
ch_orders_str.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\9.0_PCA.csv')



