import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import cut

df = pd.read_csv("H:\\bank-full.csv")

#Calulating Age
ageMean = df['age'].mean()
ageStd = df['age'].std()
ageMedian = df['age'].median()
minAge = df['age'].min()
maxAge = df['age'].max()

#This code prints out a formatted version of the Age Attribute from Bank-Full.csv
step = 10
bin_range = np.arange(18,95+step, step+1)
out, bins = pd.cut(df['age'], bins=bin_range, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Age of Respondents')
#agePlot = plt.plot(df['age'])
#plt.xlabel("Age of Clients")
#plt.ylabel("Number of Clients")
#plt.show()

# Using Min-Max Normalization for Age Attribute
normalized_df = (((df['age'] - minAge) / (maxAge - minAge))*(1.0-0)+0)
#print(normalized_df.min())
#print(normalized_df.max())
#print(normalized_df.mean())
#print(normalized_df.median())

#Prints Min-Max Normalized Histogram of the Age Attribute
step = 0.1
bin_range_normalized = np.arange(0.0, 1.0+step, step+0.1)
out, bins = pd.cut(normalized_df, bins=bin_range_normalized, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Age of Respondents')
#normalizedAgePlot = plt.plot(normalized_df)
#plt.xlabel("Age of Clients")
#plt.ylabel("Number of Clients")
#plt.show()

# Using z-score Normalization of the Age Attribute
zScore_Normalized_df = ((df['age'] - ageMedian) / ageStd)
#print(zScore_Normalized_df.min())
#print(zScore_Normalized_df.max())
#print(zScore_Normalized_df.mean())
#print(zScore_Normalized_df.median())

#Prints z-score Normalized Histogram of the Age Attribute
step = 0.5
bin_range_zScorenormalized = np.arange(-2, 0.5+step, step)
out, bins = pd.cut(zScore_Normalized_df, bins=bin_range_zScorenormalized, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Age of Respondents')
#zScoreNormalizedAgePlot = plt.plot(normalized_df)
#plt.xlabel("Age of Clients")
#plt.ylabel("Number of Clients")
#plt.show()

###################################################################
#Calulating Balance
balanceMean = df['balance'].mean()
balanceMedian = df['balance'].median()
balanceStd = df['balance'].std()
balanceMax = df['balance'].max()
balanceMin = df['balance'].min()

#This code prints out a formatted version of the Balance Attribute from Bank-Full.csv
#NOT FINISHED
#for index, row in df.itertuples():
#    if df['balance'] < 0:
#        df['balance'] = df['balance'].replace([df.balance], value=0)

#step = 1000
#balance_bin_range = np.arange(balanceMin,balanceMax+step, step)
#out, bins = pd.cut(df['balance'], bins=balance_bin_range, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Balance of Respondents')
#balancePlot = plt.plot(df['balance'])
#plt.xlabel("balance of Clients")
#plt.ylabel("Number of Clients")
#plt.show()

# Using Min-Max Normalization for Age Attribute
normalized_df = (((df['age'] - minAge) / (maxAge - minAge))*(1.0-0)+0)
#print(normalized_df.min())
#print(normalized_df.max())
#print(normalized_df.mean())
#print(normalized_df.median())

#Prints Min-Max Normalized Histogram
step = 0.1
bin_range_normalized = np.arange(0.0, 1.0+step, step+0.1)
out, bins = pd.cut(normalized_df, bins=bin_range_normalized, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Age of Respondents')
#normalizedAgePlot = plt.plot(normalized_df)
#plt.xlabel("Age of Clients")
#plt.ylabel("Number of Clients")
#plt.show()

# Using z-score Normalization
zScore_Normalized_df = ((df['age'] - ageMedian) / ageStd)
#print(zScore_Normalized_df.min())
#print(zScore_Normalized_df.max())
#print(zScore_Normalized_df.mean())
#print(zScore_Normalized_df.median())

#Prints z-score Normalized Histogram
step = 0.5
bin_range_zScorenormalized = np.arange(-2, 0.5+step, step)
out, bins = pd.cut(zScore_Normalized_df, bins=bin_range_zScorenormalized, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Age of Respondents')
#zScoreNormalizedAgePlot = plt.plot(normalized_df)
#plt.xlabel("Age of Clients")
#plt.ylabel("Number of Clients")
#plt.show()

###################################################################
#Calulating Day and Months NOT FINISHED
dayMean = df['day'].mean()
dayMedian = df['day'].median()
dayStd = df['day'].std()
dayMax = df['day'].max()
dayMin = df['day'].min()
i = 0

dayPlot = plt.plot(df['day'])
#for i in df['month']:
#    if df['month']
#        plt.title("Days of Week for May Unformatted")
#        plt.xlabel('number of clients')
#        plt.ylabel('day')
#        plt.show()
#while df['month'] == 'june':
#    plt.title("Days of Week for June Unformatted")
#    plt.xlabel('number of clients')
 #   plt.ylabel('day')
  #  plt.show()

#This code prints out a formatted version of the Age Attribute from Bank-Full.csv
step = dayMin
bin_range = np.arange(dayMin, dayMax+step, step)
out, bins = pd.cut(df['day'], bins=bin_range, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Day of Week')
#dayPlotFormatted = plt.plot(df['day'])
#plt.xlabel("Day of Week")
#plt.ylabel("Number of Clients")
#plt.show()

# Using Min-Max Normalization for Day Attribute
normalized_df = (((df['age'] - minAge) / (maxAge - minAge))*(1.0-0)+0)
#print(normalized_df.min())
#print(normalized_df.max())
#print(normalized_df.mean())
#print(normalized_df.median())

#Prints Min-Max Normalized Histogram of the Day Attribute
step = 0.1
bin_range_normalized = np.arange(0.0, 1.0+step, step+0.1)
out, bins = pd.cut(normalized_df, bins=bin_range_normalized, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Age of Respondents')
#normalizedAgePlot = plt.plot(normalized_df)
#plt.xlabel("Age of Clients")
#plt.ylabel("Number of Clients")
#plt.show()

# Using z-score Normalization of the Day Attribute
#zScore_Normalized_df = ((df['age'] - ageMedian) / ageStd)
#print(zScore_Normalized_df.min())
#print(zScore_Normalized_df.max())
#print(zScore_Normalized_df.mean())
#print(zScore_Normalized_df.median())

#Prints z-score Normalized Histogram of the Day Attribute
step = 0.5
bin_range_zScorenormalized = np.arange(-2, 0.5+step, step)
out, bins = pd.cut(zScore_Normalized_df, bins=bin_range_zScorenormalized, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Age of Respondents')
#zScoreNormalizedAgePlot = plt.plot(normalized_df)
#plt.xlabel("Age of Clients")
#plt.ylabel("Number of Clients")
#plt.show()

#########################################################################
#Calulating Duration
durationStd = df['duration'].std()
durationMean = df['duration'].mean()
durationMedian = df['duration'].median()
durationMax = df['duration'].max()
durationMin = df['duration'].min()

#prints raw version of column
durationPlot = plt.plot(df['duration'])
plt.title('Call Durations for Customers (Raw Data)')
plt.xlabel("Number of Calls")
plt.ylabel("Call Duration (in Seconds)")
plt.show()

#This code prints out a formatted version of the Duration Attribute from Bank-Full.csv
step = 50
duration_bin_range = np.arange(durationMin,durationMax+step, step)
out, bins = pd.cut(df['duration'], bins=duration_bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Call Durations for Customers')
durationPlotFormatted = plt.plot(df['duration'])
plt.xlabel("Call Duration (in Seconds)")
plt.ylabel("Number of Calls")
plt.show()

# Using Min-Max Normalization for Duration Attribute
duration_normalized_df = (((df['duration'] - durationMin) / (durationMax - durationMin))*(1.0-0)+0)
#print(normalized_df.min())
#print(normalized_df.max())
#print(normalized_df.mean())
#print(normalized_df.median())

#Prints Min-Max Normalized Histogram of the Age Attribute
step = 0.1
duration_bin_range_normalized = np.arange(duration_normalized_df.min(), duration_normalized_df.max()+step, step+0.1)
out, bins = pd.cut(duration_normalized_df, bins=duration_bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Call Durations for Customers (Min/Max Normalization)')
durationPlotMinMaxNormalization = plt.plot(df['duration'])
plt.xlabel("Call Duration (in Seconds)")
plt.ylabel("Number of Calls")
plt.show()

# Using z-score Normalization of the Age Attribute
duration_zScore_Normalized_df = ((df['duration'] - durationMedian) / durationStd)

#Prints z-score Normalized Histogram of the Age Attribute
step = 0.5
bin_range_durationzScorenormalized = np.arange(duration_zScore_Normalized_df.min(), duration_zScore_Normalized_df.max()+step, step)
out, bins = pd.cut(duration_zScore_Normalized_df, bins=bin_range_durationzScorenormalized, include_lowest=True, right=False, retbins=True)
durationHistogram = out.value_counts(sort=False).plot(kind = 'bar', title='Call Durations for Customers (zScore Normalization)')

durationPlotzScoreNormalization = plt.plot()
plt.title('Call Durations for Customers (zScore Normalization)')
plt.xlabel("Call Duration (in Seconds)")
plt.ylabel("Number of Calls")
plt.show()
print(duration_zScore_Normalized_df)
###################################################################
#Calulating poutcome
#counts = df['poutcome'].value_counts().plot.bar()
#poutcomeResult = plt.pyplot.show()

