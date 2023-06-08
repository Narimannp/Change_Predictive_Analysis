# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:48:03 2023

@author: narim
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def read_df():
    #Read ch_orders dataset from the 7th step

    ch_orders=pd.read_csv(r'D:\Concordia\Master_Of_Science\Dataset_aedo_june_2022\Text_Mining\allprojects\8_imputed_duration.csv')
    return(ch_orders)
ch_orders=read_df()

def outlier_remove(dataset,atrlist):
    for attribute in atrlist:
       describe=dataset.describe()
       IQR=describe.loc["75%",attribute]-describe.loc["25%",attribute]
       lowerfence=describe.loc["25%",attribute]-1.5*IQR
       higherfence=describe.loc["75%",attribute]+1.5*IQR
       dataset=dataset[(lowerfence<dataset[attribute]) & (dataset[attribute]<higherfence)]
    return(dataset) 

def pearson_spearman(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    for column in df:
       #Range normalizing attributes
       df[column]=(df[column]-df[column].min())/(df[column].max()-df[column].min())
    pearson=df.corr(method='pearson', min_periods=1).round(2)
    spearman=df.corr(method='spearman', min_periods=1).round(2)
    return(pearson,spearman)


#  my bin thresholds
describe=ch_orders["PrimeChPer"].describe()
type_construction=ch_orders[ch_orders["ProjectType"]=="Construction"]
zero_construction=type_construction[type_construction["PrimeChPer"]==0]
dp_neg,atr=type_construction[type_construction["PrimeChPer"]<0].shape
dp_pos,atr=type_construction[type_construction["PrimeChPer"]>0].shape
dp_zero,atr=zero_construction.shape
nonzero_construction=type_construction[type_construction["PrimeChPer"]!=0]
a,b=np.histogram(nonzero_construction["PrimeChPer"], bins=3, density=False)

a=np.percentile(nonzero_construction["PrimeChPer"],[1,10,33.3,66.6])
# data=ch_orders["PrimeChPer"]
# hist, bin_edges = np.histogram(data, bins=bin_thresholds)

# # Plot histogram
# plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')

# # Add labels and title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram')

# # Add text annotations for each bin
# for i, count in enumerate(hist):
#     plt.text(bin_edges[i], count, str(count), ha='center', va='bottom')

# # Display the histogram
# plt.show()


# labels = ['No_Change', 'Negative_Change', 'Positive_Change']
# sizes = [dp_zero, dp_neg, dp_pos]  # Corresponding sizes or proportions for each category
# colors = ['red', 'blue', 'green']  # Custom colors for each category

# # Create the pie chart
# plt.pie(sizes, labels=labels, colors=colors, autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*sum(sizes))})", startangle=90)

# # Add a title
# plt.title('Project-Level COs in (1029) Construction Projects\n 2nd Approach; More Duplicates')

# # Equal aspect ratio ensures that pie is drawn as a circle
# plt.axis('equal')

# # Display the chart
# plt.show()

# data = nonzero_construction["PrimeChPer"]
# data=outlier_remove(nonzero_construction,["PrimeChPer"])["PrimeChPer"]
# data_2=data
# data_2=nonzero_construction[nonzero_construction["ProjectBaseContractValue"]>600000]
# Create a figure and axis
# fig, ax = plt.subplots()

# Create the box plot
# ax.boxplot([data,data_2])
# ax.boxplot(data_2)
# # Add labels and title
# ax.set_xticklabels(['Construction Projects',"2"])
# ax.set_xlabel('Distribution of Project-Level COs in Construction Projects')
# ax.set_ylabel('Change Percentage')
# ax.set_title('Box Plot')

# Display the plot
# plt.show()

data=nonzero_construction
# columns=nonzero_construction.columns
# # data=data[["PrimeChPer","TotalChPer","CommitChPer","Freq_Class_1_p","Freq_Class_1_n","Freq_Class_2_p","Freq_Class_2_n","Freq_Prov_p","Freq_Prov_n","Freq_p_City","Freq_n_City"]]
# g=sns.scatterplot(data=data,x="ProjectBaseContractValue",y="PrimeChFreq_p",hue="Classification_1",size="DurationModified",sizes=(10, 250),legend="brief")
# h,l = g.get_legend_handles_labels()
# plt.legend(h[0:3],l[0:3],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=13)
# plt.show(g)
pearson,spearman=pearson_spearman(data)