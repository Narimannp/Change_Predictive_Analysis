# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:48:30 2024

@author: narim
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency,chi2

def read_df():
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\ph2_ChangeTime_Step1_DataPrep.csv')
    # ch_orders=ch_orders[ch_orders[]]
    ch_orders=ch_orders.set_index("ProjectId")
    return(ch_orders)

def none_cumulative(ch_orders):
    target_atr_list=[atr for atr in ch_orders.columns if "Percentage" in atr]
    for i in range(len(target_atr_list)-1):
       ch_orders["ChPerPure_"+str(5-i)]=ch_orders["ChPercentage_"+str(5-i)]-ch_orders["ChPercentage_"+str(5-i-1)]
    ch_orders["ChPerPure_"+str(1)]=ch_orders["ChPercentage_"+str(1)]
    return(ch_orders)

def durationfilter(ch_orders,durGroup):
    duration_dict={0:[0,5000],1:[0,150],2:[150,365],3:[365,5000]}
    ch_orders=ch_orders[(ch_orders["DurationNew"]<duration_dict[durGroup][1])&(ch_orders["DurationNew"]>=duration_dict[durGroup][0])].copy()
    return(ch_orders)

def labling(datapoint):
    if datapoint<=0:
         return("0-minus")
    elif datapoint<5:
         return("1-0-5%")
    else:
         return("2-5%-inf")
     
"Given the DF and boundry or list of boundries labels the target atrribute"
def label_target_atr(ch_orders):
    target_atr_list=[atr for atr in ch_orders.columns if "Percentage" in atr]
    for i in range(len(target_atr_list)):
       ch_orders["ChMagnitude_"+str(i+1)]=ch_orders["ChPerPure_"+str(i+1)].apply(labling)
    return(ch_orders,target_atr_list)

def transition_matrixes(ch_orders):
    count_initial=ch_orders.groupby("ChMagnitude_1")["ContractValue"].count()
    
    states=count_initial.index
    N_list=[]
    P_list=[]
    #creating 1 step transition matrixes
    for i in range(1,5):
        N=pd.DataFrame(columns=[count_initial.index])
        
        for state in states:
            temp=ch_orders[ch_orders["ChMagnitude_"+str(i)]==state]
            NS_dict=dict(zip(temp.groupby("ChMagnitude_"+str(i+1))["ContractValue"].count().index,temp.groupby("ChMagnitude_"+str(i+1))["ContractValue"].count()))
            for key in NS_dict.keys():
                N.loc[state,key]=NS_dict[key]
        N=N.fillna(0)
        NSum=N.sum(axis=1)
        N["sum"]=list(NSum)
        P=N.loc[:,states].div(list(NSum),axis=0)
        N_list.append(N)
        P_list.append(P)
    #creating 4 step transition matrix from 1 to 5 and adding it to the end of N and P matrixes
    N=pd.DataFrame(columns=[count_initial.index])
    for state in states:
        temp=ch_orders[ch_orders["ChMagnitude_"+str(1)]==state]
        NS_dict=dict(zip(temp.groupby("ChMagnitude_"+str(5))["ContractValue"].count().index,temp.groupby("ChMagnitude_"+str(5))["ContractValue"].count()))
        for key in NS_dict.keys():
            N.loc[state,key]=NS_dict[key]   
    N=N.fillna(0)
    NSum=N.sum(axis=1)
    N["sum"]=list(NSum)
    P=N.loc[:,states].div(list(NSum),axis=0)
    N_list.append(N)
    P_list.append(P)    

    return(count_initial,N_list,P_list)

def product(p_df_list,n_df_list,count_1):
    P15_ex=p_df_list[0]
    for i in range(1,len(p_df_list)-1):
        P15_ex=pd.DataFrame(np.array(P15_ex) @ np.array(p_df_list[i]),columns=count_1.index,index=count_1.index)
    E15=P15_ex.mul(count_1,axis=0)
    #N_ws: N without sum
    N_ws=np.array(n_df_list[-1].drop("sum",axis=1))
    U=((N_ws-E15).pow(2)).div(E15)
    USum=U.sum(axis=1)
    U["sum"]=list(USum)
    U=U.fillna(0)
    uSum=U["sum"].sum()
    dgFr=U.shape[0]*(U.shape[0]-1)
    p_value = 1 - chi2.cdf(uSum, df=dgFr)
    return(P15_ex,E15,U,uSum,p_value)

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
    top_atrs=chi_square_result.index.tolist()
    return(chi_square_result,top_atrs)
        
def run(targetstate,durationgroup):
    ch_orders=read_df()
    ch_orders_filtered=durationfilter(ch_orders,durationgroup)
    ch_orders_filtered=none_cumulative(ch_orders_filtered)
    ch_orders_filtered,target_atr_list=label_target_atr(ch_orders_filtered)
    count_1,N_list,P_list=transition_matrixes(ch_orders_filtered)
    expected_p15,E15,U,uSum,p_value=product(P_list,N_list,count_1)
    return(ch_orders,ch_orders_filtered,count_1,N_list,P_list,expected_p15,E15,U,uSum,p_value)
ch_orders,ch_orders_filtered,count_1,N_list,P_list,expected_p15,E15,U,uSum,p_value=run(5,0)
