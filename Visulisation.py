# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:02:07 2023

@author: narim
"""

## lets start data visulisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def read_df():
    #Read external datasets, 1-Projects,2-Canadacities
    projects =pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\6_data_prep_project.csv')
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\6_data_prep_ch_orders.csv')
    return(projects,ch_orders)

def project_filter(projects,ch_orders,atr,atr_class_list):
    projects=projects[projects[atr].isin(atr_class_list)]
    projects_filter_list=projects["ProjectId"]
    ch_orders=ch_orders[ch_orders["ProjectId"].isin(projects_filter_list)]
    return(projects,ch_orders)


projects,ch_orders=read_df()
projects,ch_orders=project_filter(projects,ch_orders,"ProjectType",["Construction"])
ch_orders=ch_orders[ch_orders["Type"]=="PrimeContractChangeOrder"]
ch_orders=ch_orders[ch_orders["Amount"]!=0]
ch_orders["freq"]=0
projects=projects[projects["ProjectId"].isin(ch_orders["ProjectId"])]
project_level=ch_orders.groupby("ProjectId",as_index=False).agg({"ProjectBaseContractValue":"last","freq":"count","Amount":"sum",})

# project_level_pos=project_level[project_level["Amount"]>0]
ch_orders=ch_orders[ch_orders["ProjectId"].isin(project_level["ProjectId"])]
def sort_group(group):
    return group.sort_values(by='DateCreated')
ch_orders=ch_orders.sort_values(by="DateCreated")
ch_orders_group=ch_orders.groupby("ProjectId")
# ch_orders["Ch_Number"]=0

# sortd_groups=ch_orders_group.apply(lambda x:x.sort_values("DateCreated",as_index=False))
ch_orders['MemberNumber'] = ch_orders_group.cumcount() + 1
ch_orders["CumAmount"]=ch_orders_group["Amount"].cumsum()
ch_orders=ch_orders.sort_values(["ProjectId","MemberNumber"])
dict_total_change=dict(zip(project_level["ProjectId"],project_level["Amount"]))
ch_orders["TotalChange"]=ch_orders["ProjectId"].map(dict_total_change)
ch_orders["CumChangePer"]=ch_orders["CumAmount"]/ch_orders["ProjectBaseContractValue"]
ch_orders_group=ch_orders.groupby("ProjectId")
for group_name, group_data in ch_orders_group:
    a=group_data[["MemberNumber","CumChangePer"]]
    plt.scatter(a['MemberNumber'], a['CumChangePer'],s=50)
projectIds=list(set(ch_orders["ProjectId"]))
# ch_orders=ch_orders[ch_orders["ProjectId"]==projectIds[25]]
plt.scatter(ch_orders['MemberNumber'], ch_orders['CumChangePer'])
ch_orders.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\for_viso.csv')