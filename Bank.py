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

#Prints raw verison of age attribute
plt.plot(df['age'])
plt.title('Age of Respondents (Raw Data)')
plt.xlabel("Age of Clients")
plt.ylabel("Number of Clients")
plt.show()

#This code prints out a formatted version of the Age Attribute from Bank-Full.csv
step = 10
bin_range = np.arange(18,95+step, step+1)
out, bins = pd.cut(df['age'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Age of Respondents')
agePlot = plt.plot()
plt.xlabel("Age of Clients")
plt.ylabel("Number of Clients")
plt.show()

# Using Min-Max Normalization for Age Attribute
normalized_df = (((df['age'] - minAge) / (maxAge - minAge))*(1.0-0)+0)

#Prints Min-Max Normalized Histogram of the Age Attribute
step = 0.1
bin_range_normalized = np.arange(0.0, 1.0+step, step+0.1)
out, bins = pd.cut(normalized_df, bins=bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Age of Respondents (Min-Max Normalized)')
normalizedAgePlot = plt.plot(normalized_df)
plt.xlabel("Age of Clients")
plt.ylabel("Number of Clients")
plt.show()

# Using z-score Normalization of the Age Attribute
zScore_Normalized_df = ((df['age'] - ageMedian) / ageStd)

#Prints z-score Normalized Histogram of the Age Attribute
step = 0.5
bin_range_zScorenormalized = np.arange(-2, 0.5+step, step)
out, bins = pd.cut(zScore_Normalized_df, bins=bin_range_zScorenormalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Age of Respondents (z-score Normalized)')
zScoreNormalizedAgePlot = plt.plot(normalized_df)
plt.xlabel("Age of Clients")
plt.ylabel("Number of Clients")
plt.show()

###################################################################
#Calculating job
jobCounts = df['job'].value_counts().plot.bar()
plt.title('Jobs of Clients')
plt.xlabel('Jobs')
plt.ylabel('Frequency')
plt.show()

###################################################################
#Calculating marital
maritalCounts = df['marital'].value_counts().plot.bar()
plt.title('Marital Status of Clients')
plt.xlabel('Marital Status')
plt.ylabel('Frequency')
plt.show()

###################################################################
#Calculating education
educationCounts = df['education'].value_counts().plot.bar()
plt.title('Education Level of Clients')
plt.xlabel('Education Levels')
plt.ylabel('Frequency')
plt.show()

###################################################################
#Calculating default
defaultCounts = df['default'].value_counts().plot.bar()
plt.title('Customer has Credit in default')
plt.xlabel('Credit in Default')
plt.ylabel('Frequency')
plt.show()

###################################################################
#Calulating Balance
balanceMean = df['balance'].mean()
balanceMedian = df['balance'].median()
balanceStd = df['balance'].std()
balanceMax = df['balance'].max()
balanceMin = df['balance'].min()

#prints raw version of column
balancePlot = plt.plot(df['balance'])
plt.title('Balance of Respondents (Raw Data)')
plt.xlabel("Balance of Clients")
plt.ylabel("Number of Clients")
plt.show()

#This code prints out a formatted version of the Balance Attribute from Bank-Full.csv
#NOT FINISHED
#for index, row in df.itertuples():#
#    if df['balance'] < 0:
#        df['balance'] = df['balance'].replace([df.balance], value=0)

step = 1000
balance_bin_range = np.arange(balanceMin, balanceMax+step, step-1)
out, bins = pd.cut(df['balance'], bins=balance_bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot(kind='bar', title='Balance of Respondents (Formatted)')
balancePlot = plt.plot(kind='bar')
plt.xlabel("Balance of Clients")
plt.ylabel("Number of Clients")
plt.show()

# Using Min-Max Normalization for Balance Attribute
#Most data values supersede the new 1.0 Max, due to the negative integers
normalizedBalance_df = ((df['balance'] - balanceMin / (balanceMax - balanceMin))*(1.0-0)+0)

#Prints Min-Max Normalized Histogram for Balance Attribute#
step = 0.01
bin_range_Balancenormalized = np.arange(0.0, 1.0+step, step)
out, bins = pd.cut(normalizedBalance_df, bins=bin_range_Balancenormalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot(kind='bar', title='Balance of Respondents (Normalized Max/Min)')
zScoreNormalizedBalancePlot = plt.plot(kind='bar')
plt.xlabel("Balance of Clients")
plt.ylabel("Number of Clients")
plt.show()

# Using z-score Normalization
zScore_BalanceNormalized_df = ((df['balance'] - balanceMedian) / balanceStd)

#Prints z-score Normalized Histogram
step = 0.5
bin_range_zScoreBalancenormalized = np.arange(-2, 0.5+step, step)
out, bins = pd.cut(zScore_BalanceNormalized_df, bins=bin_range_zScoreBalancenormalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Balance of Respondents (Z-Score)')
zScoreNormalizedAgePlot = plt.plot()
plt.xlabel("Balance of Clients")
plt.ylabel("Number of Clients")
plt.show()

###################################################################
#Calculating housing
housingCounts = df['housing'].value_counts().plot.bar()
plt.title('Housing of Client')
plt.xlabel('Housing')
plt.ylabel('Frequency')
plt.show()

###################################################################
#Calculating contact
contactCounts = df['contact'].value_counts().plot.bar()
plt.title('Contact Method used for Client')
plt.xlabel('Contact')
plt.ylabel('Frequency')
plt.show()

###################################################################
#Calulating Day and Months NOT FINISHED
dayMean = df['day'].mean()
dayMedian = df['day'].median()
dayStd = df['day'].std()
dayMax = df['day'].max()
dayMin = df['day'].min()
i = 0

#dayPlot = plt.plot(df['day'])
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

#This code prints out a formatted version of the Day Attribute from Bank-Full.csv
step = dayMin
bin_range = np.arange(dayMin, dayMax+step, step)
#out, bins = pd.cut(df['day'], bins=bin_range, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Day of Week')
#dayPlotFormatted = plt.plot(df['day'])
#plt.xlabel("Day of Week")
#plt.ylabel("Number of Clients")
#plt.show()

# Using Min-Max Normalization for Day Attribute
#days_normalized_df = (((df['day'] - dayMin) / (dayMax - dayMin))*(1.0-0)+0)

#Prints Min-Max Normalized Histogram of the Day Attribute
step = 0.1
days_bin_range_normalized = np.arange(0.0, 1.0+step, step+0.1)
#out, bins = pd.cut(days_normalized_df, bins=days_bin_range_normalized, include_lowest=True, right=False, retbins=True)
#out.value_counts(sort=False).plot.bar(title='Age of Respondents')
#normalizedDayPlot = plt.plot()
#plt.xlabel("Age of Clients")
#plt.ylabel("Number of Clients")
#plt.show()

# Using z-score Normalization of the Day Attribute
days_zScore_Normalized_df = ((df['day'] - dayMedian) / dayStd)

#Prints z-score Normalized Histogram of the Day Attribute
step = 0.5
days_bin_range_zScorenormalized = np.arange(-2, 0.5+step, step)
out, bins = pd.cut(days_zScore_Normalized_df, bins=days_bin_range_zScorenormalized, include_lowest=True, right=False, retbins=True)
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
#PRINTS AS LINE GRAPH
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
durationPlotFormatted = plt.plot()
plt.xlabel("Call Duration (in Seconds)")
plt.ylabel("Number of Calls")
plt.show()

# Using Min-Max Normalization for Duration Attribute
duration_normalized_df = (((df['duration'] - durationMin) / (durationMax - durationMin))*(1.0-0)+0)

#Prints Min-Max Normalized Histogram of the Duration Attribute
step = 0.1
duration_bin_range_normalized = np.arange(duration_normalized_df.min(), duration_normalized_df.max()+step, step+0.1)
out, bins = pd.cut(duration_normalized_df, bins=duration_bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Call Durations for Customers (Min/Max Normalization)')
durationPlotMinMaxNormalization = plt.plot(df['duration'])
plt.xlabel("Call Duration (in Seconds)")
plt.ylabel("Number of Calls")
plt.show()

# Using z-score Normalization of the Duration Attribute
duration_zScore_Normalized_df = ((df['duration'] - durationMedian) / durationStd)

#Prints z-score Normalized Histogram of the Duration Attribute
step = 0.5
bin_range_durationzScorenormalized = np.arange(duration_zScore_Normalized_df.min(), duration_zScore_Normalized_df.max()+step, step)
out, bins = pd.cut(duration_zScore_Normalized_df, bins=bin_range_durationzScorenormalized, include_lowest=True, right=False, retbins=True)
durationHistogram = out.value_counts(sort=False).plot(kind = 'bar', title='Call Durations for Customers (zScore Normalization)')

durationPlotzScoreNormalization = plt.plot()
plt.title('Call Durations for Customers (zScore Normalization)')
plt.xlabel("Call Duration (in Seconds)")
plt.ylabel("Number of Calls")
plt.show()

###################################################################
#Calulating Campaign
campaignStd = df['campaign'].std()
campaignMean = df['campaign'].mean()
campaignMedian = df['campaign'].median()
campaignMax = df['campaign'].max()
campaignMin = df['campaign'].min()

#prints raw version of column
#PRINTS AS LINE GRAPH
campaignPlot = plt.plot(df['campaign'])
plt.title('Number of contacts performed during this campaign and for this client (Raw Data)')
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

#This code prints out a formatted version of the Campaign Attribute from Bank-Full.csv
step = 1
duration_bin_range = np.arange(campaignMin, campaignMax+step, step)
out, bins = pd.cut(df['campaign'], bins=duration_bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of contacts performed during this campaign and for this client')
campaignPlotFormatted = plt.plot()
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

# Using Min-Max Normalization for Campaign Attribute
campaign_normalized_df = (((df['campaign'] - campaignMin) / (campaignMax - campaignMin))*(1.0-0)+0)

#Prints Min-Max Normalized Histogram of the Campaign Attribute
step = 0.1
campaign_bin_range_normalized = np.arange(duration_normalized_df.min(), duration_normalized_df.max()+step, step)
out, bins = pd.cut(campaign_normalized_df, bins=campaign_bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of contacts performed during this campaign and for this client (Min/Max Normalization)')
campaignPlotMinMaxNormalization = plt.plot()
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

# Using z-score Normalization of the Campaign Attribute
campaign_zScore_Normalized_df = ((df['campaign'] - campaignMedian) / campaignStd)

#Prints z-score Normalized Histogram of the Campaign Attribute
step = 0.5
bin_range_campaignzScorenormalized = np.arange(campaign_zScore_Normalized_df.min(), campaign_zScore_Normalized_df.max()+step, step)
out, bins = pd.cut(campaign_zScore_Normalized_df, bins=bin_range_campaignzScorenormalized, include_lowest=True, right=False, retbins=True)
campaignHistogram = out.value_counts(sort=False).plot(kind = 'bar', title='Number of contacts performed during this campaign and for this client (zScore Normalization)')
campaignPlotzScoreNormalization = plt.plot()
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

#Frequency of days for people
campaigngCounts = df['campaign'].value_counts().plot.bar()
plt.title('Housing of Client (Frequency in Excel Sheet)')
plt.xlabel('Housing')
plt.ylabel('Frequency')
plt.show()

###################################################################
#Calulating previous
previousStd = df['previous'].std()
previousMean = df['previous'].mean()
previousMedian = df['previous'].median()
previousMax = df['previous'].max()
previousMin = df['previous'].min()

#prints raw version of column
previousPlot = plt.plot(df['previous'])
plt.title('Number of Contacts Performed Before this campaign and for this client (Raw Data)')
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

#This code prints out a formatted version of the previous Attribute from Bank-Full.csv
step = 1
previous_bin_range = np.arange(previousMin, previousMax+step, step)
out, bins = pd.cut(df['previous'], bins=previous_bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of Contacts Performed Before this campaign and for this client')
previousPlotFormatted = plt.plot()
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

# Using Min-Max Normalization for previous Attribute
previous_normalized_df = (((df['previous'] - previousMin) / (previousMax - previousMin))*(1.0-0)+0)

#Prints Min-Max Normalized Histogram of the previous Attribute
step = 0.1
previous_bin_range_normalized = np.arange(previous_normalized_df.min(), previous_normalized_df.max()+step, step+0.1)
out, bins = pd.cut(previous_normalized_df, bins=previous_bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of Contacts Performed Before this campaign and for this client (Min/Max Normalization)')
previousPlotMinMaxNormalization = plt.plot()
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

# Using z-score Normalization of the previous Attribute
previous_zScore_Normalized_df = ((df['previous'] - previousMedian) / previousStd)

#Prints z-score Normalized Histogram of the previous Attribute
step = 0.5
bin_range_pdayszScorenormalized = np.arange(previous_zScore_Normalized_df.min(), previous_zScore_Normalized_df.max()+step, step)
out, bins = pd.cut(previous_zScore_Normalized_df, bins=bin_range_pdayszScorenormalized, include_lowest=True, right=False, retbins=True)
previousHistogram = out.value_counts(sort=False).plot(kind = 'bar', title='Number of Contacts Performed Before this campaign and for this client (zScore Normalization)')
previousPlotzScoreNormalization = plt.plot()
plt.title('Number of Contacts Performed Before this campaign and for this client (zScore Normalization)')
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

###################################################################
#Calulating poutcome
pOutcomeCounts = df['poutcome'].value_counts().plot.bar()
plt.title('pOutcome')
plt.xlabel('Outcomes')
plt.ylabel('Frequency')
plt.show()

###################################################################
#Calculating pdays
pdaysStd = df['pdays'].std()
pdaysMean = df['pdays'].mean()
pdaysMedian = df['pdays'].median()
pdaysMax = df['pdays'].max()
pdaysMin = df['pdays'].min()

#prints raw version of column
pdaysPlot = plt.plot(df['pdays'])
plt.title('Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign (Raw Data)')
plt.xlabel("Number of Days Passed")
plt.ylabel("Frequency")
plt.show()

#This code prints out a formatted version of the pdays Attribute from Bank-Full.csv
step = 50
pdays_bin_range = np.arange(pdaysMin,pdaysMax+step, step)
out, bins = pd.cut(df['pdays'], bins=pdays_bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign')
pdaysPlotFormatted = plt.plot()
plt.xlabel("Number of Days Passed")
plt.ylabel("Frequency")
plt.show()

df['pdays'].replace([-1], 0)

# Using Min-Max Normalization for pdays Attribute
pdays_normalized_df = (((df['pdays'] - pdaysMin) / (pdaysMax - pdaysMin))*(1.0-0)+0)

#Prints Min-Max Normalized Histogram of the pdays Attribute
step = 0.1
pdays_bin_range_normalized = np.arange(pdays_normalized_df.min(), pdays_normalized_df.max()+step, step+0.1)
out, bins = pd.cut(pdays_normalized_df, bins=pdays_bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign (Min/Max Normalization)')
pdaysPlotMinMaxNormalization = plt.plot()
plt.xlabel("Number of Days Passed")
plt.ylabel("Frequency")
plt.show()

# Using z-score Normalization of the Age Attribute
pdays_zScore_Normalized_df = ((df['pdays'] - pdaysMedian) / pdaysStd)

#Prints z-score Normalized Histogram of the pdays Attribute
step = 0.5
bin_range_pdayszScorenormalized = np.arange(pdays_zScore_Normalized_df.min(), pdays_zScore_Normalized_df.max()+step, step)
out, bins = pd.cut(pdays_zScore_Normalized_df, bins=bin_range_pdayszScorenormalized, include_lowest=True, right=False, retbins=True)
pdaysHistogram = out.value_counts(sort=False).plot(kind = 'bar', title='Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign (zScore Normalization)')
pdaysPlotzScoreNormalization = plt.plot()
plt.title('Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign')
plt.xlabel("Number of Days Passed")
plt.ylabel("Frequency")
plt.show()

###################################################################
#Calculating y
yCounts = df['y'].value_counts().plot.bar()
plt.title('y')
plt.xlabel('Answers')
plt.ylabel('Frequency')
plt.show()
