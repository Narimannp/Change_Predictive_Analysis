# -*- coding: utf-8 -*-Prc
"""
Created on Sat Jan 14 11:27:43 2023

@author: narim
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


def read_external_df(projects,cities):
    #Read external datasets, 1-Projects,2-Canadacities
    projects=pd.read_csv(projects)
    projects=projects["ProjectId,ParentProjectId,ProjectBaseContractValue,ProjectCity,ProjectProvince,ProjectClassification,ProjectBillingType,ProjectDepartment,ProjectExpectedStartDate,ProjectExpectedEndDate,ProjectOperatingUnit,ProjectType".split(",")]
    canada_cities=pd.read_csv(cities)
    return(projects,canada_cities)



"STEP 1"
def ch_or_ev_available_in_projects(ch_orders,ch_events,projects):
    projects_list=projects["ProjectId"]
    ch_orders=ch_orders[ch_orders["ProjectId"].isin(projects_list)]
    ch_events=ch_events[ch_events["ProjectId"].isin(projects_list)]
    print("STEP_1 IS DONE ....")
    return(ch_orders,ch_events,projects)


"STEP 2"
def ch_or_ev_filter_with_parents(ch_orders,ch_events,projects):
    projects_list=projects["ProjectId"]
    individual_projects=projects[projects["ParentProjectId"].isna()]
    ch_orders=ch_orders[ch_orders["ProjectId"].isin(individual_projects["ProjectId"])]
    ch_events=ch_events[ch_events["ProjectId"].isin(individual_projects["ProjectId"])]
    print("STEP_2 IS DONE ....")
    return(ch_orders,ch_events,individual_projects)

def ch_or_filter_draft_ch_orders(ch_orders):
    ch_orders=ch_orders[ch_orders["co_Status"]=="3_approved"]
    return(ch_orders)
"STEP 3"
def ch_or_ev_filter_zero_contract(ch_orders,ch_events,projects):
    projects["ProjectBaseContractValue"]=abs(projects["ProjectBaseContractValue"])
    projects=projects[projects["ProjectBaseContractValue"]>101]
    ch_orders=ch_orders[ch_orders["ProjectId"].isin(projects["ProjectId"])]
    ch_events=ch_events[ch_events["ProjectId"].isin(projects["ProjectId"])]
    print("STEP_3 IS DONE ....")
    return(ch_orders,ch_events,projects)
"STEP 4"
def canada_cities_projects_city_province_prep(projects,canada_cities):
    canada_cities=canada_cities["city,population,density,province_id".split(",")]
    canada_cities=canada_cities.rename(columns={"city":"ProjectCity", "province_id":"ProjectProvince"}) 

    province_dict={"Medicine Hat":"AB","Manassas":"VR_USA","Saskatoon":"SK","Abbotsford":"BC","Barrie":"ON","Brampton":"ON","Markham":"ON","St. Catharines":"ON","Port Hope":"ON","Winchester":"ON","Vaughan":"ON","Belleville":"ON","Ajax":"ON","Burnaby":"BC","Calgary":"AB","Edmonton":"AB","Jasper":"AB","Kingston":"ON",\
                   "Lac La Biche":"AB","Leduc":"AB","Ottawa":"ON","Mississauga":"ON","New Westminster":"BC",\
                       "Niagara Falls":"ON","Nisku":"AB","North Vancouver":"BC","Ponoka":"AB","Richmond":"BC",\
                           "Surrey":"BC","Toronto":"ON","Vancouver":"BC","Victoria":"BC","White Rock":"BC"}
    Data=np.array([["Kananaskis",210,0.1,"AB"],["Manassas",42000,1640,"VR_USA"],["Paris",12600,970,"ON"],["Balzac",39407,10.3,"AB"],["Enterprise",106,0.4,"ON"],["Carp",1477,721,"ON"],["Lancaster Park",1501,1637,"AB"],["Lac La Biche",2294,791,"AB"],["De Winton",23000,6.3,"AB"],["Canfield",45600,36.5,"ON"],["Richmond",4055,968,"ON"],["Selby",15900,34.5,"ON"],["Stayner",4000,1114,"ON"],["Winchester",2394,1058,"ON"],["Yellowhead County",11000,0.5,"AB"],["Kingston",124000,274,"ON"],["Acheson",812000,1187,"AB"],["Alert",0,0,"NU"],["Almonte",5040,1325,"ON"],["Bath",1180,845,"ON"]])
    projects["ProjectCity"]=projects["ProjectCity"].apply(lambda x:str(x).strip())
    city_rename={"Tupperville":"Chatham","Lindbergh":"St. Paul","Utopia":"Barrie","Whitchurch-Stouffville":"Stouffville","Greely":"Ottawa","Kingson":"Kingston","Georgetown":"Halton Hills","mississauga":"Mississauga","Trenton":"Quinte West","Vaugahan":"Vaughan","Metcalfe":"Ottawa","West Kingston":"Kingston","KINGSTON":"Kingston","Battersea":"South Frontenac","TORONTO":"Toronto","ORANGEVILLE":"Orangeville","Sherwood Park":"Edmonton","Dundas":"Hamilton","St Thomas":"St. Thomas","Weston":"Toronto","Bobcaygeon":"Kawartha Lakes","Elmira":"Woolwich","Elora":"Centre Wellington","Vendor# 437702":"Ottawa","Vendor#437702":"Ottawa","St Catherines":"St. Catharines","Boyle":"Athabasca","Bradford":"Bradford West Gwillimbury","Fergus":"Wellington","Mount Albert":"East Gwillimbury","VAUGHAN":"Vaughan","Alberta":"Calgary","Calagry":"Calgary","ottawa":"Ottawa","Manotick":"Ottawa","Maple":"Vaughan","Glenburnie":"Kingston","Kemptville":"North Grenville","Fort McMurray":"Wood Buffalo","East York":"Toronto","Courtice":"Clarington","Concord":"Vaughan","Mississauaga":"Mississauga","Quinte":"Quinte West","Mount Forest":"Wellington","Odessa":"Loyalist","Rexdale":"Toronto","Stittsville":"Ottawa","Orleans":"Ottawa","SCARBOROUGH":"Toronto","Strathcona County":"Edmonton","Unionville":"Markham","York":"Toronto","Scaroborough":"Toronto","Kingston - Frontenac - Kingston":"Kingston","Scarborough":"Toronto","North York":"Toronto","Nisku":"Leduc","Nepean":"Ottawa","Vanier":"Ottawa","Torono":"Toronto","Vaughn":"Vaughan","Woodbridge":"Vaughan","Amherst Island":"Amherstburg","Campbellvill":"Milton","Etobicoke":"Toronto","Gloucester":"Ottawa","Kanata":"Ottawa"}
    missing_cities=pd.DataFrame(data=Data,columns=["ProjectCity","population","density","ProjectProvince"])
    canada_cities=pd.concat([canada_cities,missing_cities],ignore_index=True)
    projects["ProjectCity"]=np.where(projects["ProjectCity"].isin(city_rename.keys()),projects["ProjectCity"].map(city_rename),projects["ProjectCity"])
    projects["ProjectProvince"]=np.where((projects["ProjectProvince"]=="No Province Assigned"),projects["ProjectCity"].map(province_dict),projects["ProjectProvince"])
    projects["ProjectProvince"]=np.where((projects["ProjectCity"].isin(["Medicine Hat","Manassas","Saskatoon","Richmond","Mississauga","Ottawa","Calgary","Vaughan","Barrie"])),projects["ProjectCity"].map(province_dict),projects["ProjectProvince"])
    projects["ProjectProvince"]=np.where(projects["ProjectProvince"].isna(),projects["ProjectCity"].map(province_dict),projects["ProjectProvince"])
    
    projects["ProjectProvince"]=projects["ProjectProvince"].replace({"AL":"AB"})
    mode_city_per_province=(projects[projects["ProjectCity"]!="No City Assigned"]).groupby("ProjectProvince",as_index=False).agg({"ProjectCity":pd.Series.mode})
    city_na_dict=dict (zip(mode_city_per_province['ProjectProvince'], mode_city_per_province['ProjectCity']))
    province_na_dict=dict (zip(mode_city_per_province['ProjectCity'],mode_city_per_province['ProjectProvince'], ))
    projects["ProjectCity"]=np.where(projects["ProjectCity"].isin(["No City Assigned","nan"]),np.nan,projects["ProjectCity"])
    projects["ProjectCity"]=np.where(projects["ProjectCity"].isna(),projects["ProjectProvince"].map(city_na_dict),projects["ProjectCity"])
    projects["ProjectProvince"]=np.where(projects["ProjectProvince"].isna(),projects["ProjectCity"].map(province_na_dict),projects["ProjectProvince"])
    projects["ProjectProvince"].fillna(projects["ProjectProvince"].mode()[0],inplace=True)
    projects["ProjectCity"].fillna(projects["ProjectCity"].mode()[0],inplace=True)
    print("Step 4 is done ...")
    return(projects,canada_cities)

"STEP 7"
def deal_missing_categorical_atr(projects):
    projects.replace(to_replace="No Job Classification Assigned", value=np.nan, inplace=True)
    mode_project_classification_per_city=(projects[~projects["ProjectClassification"].isna()]).groupby("ProjectProvince",as_index=False).agg({"ProjectClassification":pd.Series.mode})
    classification_na_dict=dict (zip(mode_project_classification_per_city['ProjectProvince'], mode_project_classification_per_city['ProjectClassification']))
    projects["ProjectClassification"]=np.where(projects["ProjectClassification"].isna(),projects["ProjectProvince"].map(classification_na_dict),projects["ProjectClassification"])
    projects.replace(to_replace="Unassigned BillingType", value=np.nan, inplace=True)
    projects["ProjectBillingType"].fillna(projects["ProjectBillingType"].mode()[0],inplace=True)
    projects.replace(to_replace="No Operating Unit Assigned", value=np.nan, inplace=True)
    projects["ProjectOperatingUnit"].fillna(projects["ProjectOperatingUnit"].mode()[0],inplace=True)
    return(projects)
"STEP 5"
def attribute_selection(ch_orders,ch_events,projects):
    df=projects
    df["ProjectExpectedEndDate"]=pd.to_datetime(df["ProjectExpectedEndDate"],format="%Y-%m-%d")
    df["ProjectExpectedStartDate"]=pd.to_datetime(df["ProjectExpectedStartDate"],format="%Y-%m-%d")
    df["Duration"]=(df["ProjectExpectedEndDate"]-df["ProjectExpectedStartDate"]).apply(lambda x:int(str(x).split(" ")[0]))
    low_bound=pd.to_datetime("2017-05-05",format="%Y-%m-%d")
    upper_bound=pd.to_datetime("2022-06-07",format="%Y-%m-%d")
    df["missing_dates_up"]=np.where(df["ProjectExpectedStartDate"]<upper_bound,(df["ProjectExpectedEndDate"]-upper_bound).apply(lambda x:int(str(x).split(" ")[0])),(df["ProjectExpectedEndDate"]-df["ProjectExpectedStartDate"]).apply(lambda x:int(str(x).split(" ")[0])))
    df["missing_dates_low"]=(low_bound-df["ProjectExpectedStartDate"]).apply(lambda x:int(str(x).split(" ")[0]))
    df["missing_per_up"]=np.where(df["Duration"]!=0,df["missing_dates_up"]/df["Duration"],df["missing_dates_up"])
    df["missing_per_low"]=np.where(df["Duration"]!=0,df["missing_dates_low"]/df["Duration"],df["missing_dates_low"])
    df["missing_per2_up"]=np.where(df["missing_dates_up"]<0,0,df["missing_per_up"])
    df["missing_per2_low"]=np.where(df["missing_dates_low"]<0,0,df["missing_per_low"])
    first_ch_or_date=ch_orders.groupby("ProjectId").min()["DateCreated"]
    last_ch_or_date=ch_orders.groupby("ProjectId").max()["DateCreated"]
    df=df.merge(first_ch_or_date,on="ProjectId",how="outer")
    df=df.merge(last_ch_or_date,on="ProjectId",how="outer")
    df=df.rename(columns={"DateCreated_x":"first_ch_date","DateCreated_y":"last_ch_date"})
    df=df.drop("ParentProjectId",axis=1)
    df["first_ch_date"]=pd.to_datetime(df["first_ch_date"],format="%Y-%m-%d")
    df["last_ch_date"]=pd.to_datetime(df["last_ch_date"],format="%Y-%m-%d")
    projects=df
    print("STEP_5 IS DONE ....")
    return(ch_orders,ch_events,projects)

"STEP 6"
def merge_ch_orders_projects(ch_orders,ch_events,projects,canada_cities):
    projects=projects.merge(canada_cities,on=["ProjectCity","ProjectProvince"],how="left")
    print("STEP_6 IS DONE ....")
    return(ch_orders,ch_events,projects)


    

def Run_Prep_Steps(steps_list):
    ch_events=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\3_ce_without_duplicates_cleaned.csv')
    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\2_v2_co_without_duplicates.csv')
    ch_orders=ch_or_filter_draft_ch_orders(ch_orders)
    projects,canada_cities=read_external_df(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\Projects_LastUpdate_220620.csv',r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\canadacities.csv')
    # Number associated with each step is defind befor each function
    if 1 in steps_list:
        ch_orders,ch_events,projects=ch_or_ev_available_in_projects(ch_orders,ch_events,projects)
    if 2 in steps_list:
        ch_orders,ch_events,projects=ch_or_ev_filter_with_parents(ch_orders,ch_events,projects) 
    if 3 in steps_list:
        ch_orders,ch_events,projects=ch_or_ev_filter_zero_contract(ch_orders,ch_events,projects)
    if 4 in steps_list:
        projects,canada_cities=canada_cities_projects_city_province_prep(projects,canada_cities)
    projects=deal_missing_categorical_atr(projects)
    if 5 in steps_list:
        ch_orders,ch_events,projects=attribute_selection(ch_orders,ch_events,projects)
    if 6 in steps_list:
        ch_orders_merged,ch_events,projects=merge_ch_orders_projects(ch_orders,ch_events,projects,canada_cities)

    ch_orders_merged.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\5_data_prep_ch_orders.csv',index=False)
    ch_events.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\5_data_prep_ch_events.csv',index=False)
    projects.to_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\5_data_prep_project.csv',index=False)
    return(ch_orders_merged,ch_events,projects,canada_cities)
df_or,df_ev,df_pr,canada_cities=Run_Prep_Steps([1,2,3,4,5,6])       
# df_pr_ch_or_out_duration=df_pr_with_ch_or[(df_pr_with_ch_or["first_ch_date"]<df_pr_with_ch_or["ProjectExpectedStartDate"])|(df_pr_with_ch_or["last_ch_or"]<df_pr_with_ch_or["ProjectExpectedEndDate"])]

###projects_final=df_pr_valid_duration_w_ch[(df_pr_valid_duration_w_ch["missing_per2_up"]<0.25)]
###print(projects_final.info())

pr_w_ch=df_pr[~df_pr["first_ch_date"].isna()]
###projects_w_ch_dist_projecttype=pr_w_ch.groupby("ProjectType")["ProjectType"].count()
###projects_wh_ch_dist_projecttype=pr_wh_ch.groupby("ProjectType")["ProjectType"].count()
###pr_w_ch_ser=pr_w_ch[pr_w_ch["ProjectType"]=="Services"].describe()
###pr_wh_ch_ser=pr_wh_ch[pr_wh_ch["ProjectType"]=="Services"].describe()
###pr_w_ch_co=pr_w_ch[pr_w_ch["ProjectType"]=="Construction"].describe()
###pr_wh_ch_co=pr_wh_ch[pr_wh_ch["ProjectType"]=="Construction"].describe()
###pr_w_wh_BCV=pd.DataFrame(data=[pr_w_ch_ser["ProjectBaseContractValue"],pr_w_ch_co["ProjectBaseContractValue"],pr_wh_ch_ser["ProjectBaseContractValue"],pr_wh_ch_co["ProjectBaseContractValue"]],index=["Value_With_CH_Services","Value_With_CH_Construction","Value_Without_CH_Services","Value_Without_CH_Construction"])
###pr_w_wh_DUR=pd.DataFrame(data=[pr_w_ch_ser["Duration"],pr_w_ch_co["Duration"],pr_wh_ch_ser["Duration"],pr_wh_ch_co["Duration"]],index=["Duration_With_CH_Services","Duration_With_CH_Construction","Duration_Without_CH_Services","DurationWithout_CH_Construction"])
###pr_w_ch_city_unmatched=pr_w_ch[pr_w_ch["population"].isna()]
###pr_w_ch_city_matched=pr_w_ch[~pr_w_ch["population"].isna()]
# projects_w_ch_dist_projecttype=projects_w_ch_dist_projecttype.rename(columns={"ProjectType":"With_ch_or"})
###comparison_w_wh_ch_or_projecttype=pd.DataFrame(data=[projects_w_ch_dist_projecttype,projects_wh_ch_dist_projecttype],index=["With_Ch_Or","Without_Ch_Or"])