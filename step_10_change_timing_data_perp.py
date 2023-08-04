# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:38:10 2023

@author: narim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
# from step_9_0_FeatureSelection_PCA import atr_list
# from step_9_0_FeatureSelection_ChiSquare import sorted_cat_atrs as sorted_cat_atrs
print(tf.__version__)

"Reads datasets with require attributes"
def read_df():
    projects=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\7_data_prep_projects.csv')
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    ch_orders.rename(columns={"ProjectBaseContractValue":"BaseValue","ProjectOperatingUnit":"OperatingUnit",\
                                   "ProjectBillingType":"BillType","ProjectCity":"City","ProjectProvince":"Province"},inplace=True)
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders,projects)

ch_orders,projects=read_df()

def prep_1(ch_orders,projects):
    projects=projects[["first_ch_date","last_ch_date","ProjectId","ProjectExpectedStartDate","ProjectExpectedEndDate"]]
    ch_orders=ch_orders.merge(projects,on="ProjectId",how="left")
    return(ch_orders)
ch_orders=prep_1(ch_orders,projects)
ch_orders=ch_orders[~ch_orders["first_ch_date"].isna()]
ch_orders["ProjectExpectedStartDate"]=pd.to_datetime(ch_orders["ProjectExpectedStartDate"],format="%Y-%m-%d")
ch_orders["EndDate"]=ch_orders.apply(lambda x:x["ProjectExpectedStartDate"]+pd.Timedelta(days=int(x["DurationModified"])),axis=1)
a=ch_orders[ch_orders["EndDate"]!=ch_orders["ProjectExpectedEndDate"]]
with_desgin=ch_orders[ch_orders["first_ch_date"]<ch_orders["ProjectExpectedStartDate"]]
schedule_overrrun=ch_orders[ch_orders["last_ch_date"]>ch_orders["ProjectExpectedEndDate"]]
schedule_overrrun2=ch_orders[ch_orders["last_ch_date"]>ch_orders["EndDate"]]
both=ch_orders[(ch_orders["last_ch_date"]>ch_orders["ProjectExpectedEndDate"])&(ch_orders["first_ch_date"]<ch_orders["ProjectExpectedStartDate"])]