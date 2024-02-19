# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:36:36 2024

@author: narim
"""
from scipy.stats import chi2_contingency,chi2
import pandas as pd
def read_df():
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\ch_magnitude_for_markov.csv')

    return(ch_orders)

ch=read_df()
classes=["0-minus","1-0-5%","2-5%-inf"]
result=pd.DataFrame(columns=["2","3","4","5","count"])
for i in range(2,5):
   for category in classes:
       temp2=ch[ch["ChMagnitude_"+str(i)]==category]
       for cat in classes:
           temp3=temp2[temp2["ChMagnitude_"+str(i+1)]==cat]
           count=pd.DataFrame(temp3.groupby("ChMagnitude_"+str(i-1))["ProjectId"].count()).transpose().reset_index()
           count[str(i)]=category
           count[str(i+1)]=cat
           count["count"]=temp3.shape[0]
           result=pd.concat([result,count.drop(["index"],axis=1)])
result=result[result["count"]!=0]
result["nonsequence"]=result[classes].max(axis=1)/result["count"]

p=1 - chi2.cdf(5.097, df=1)
        