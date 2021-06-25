import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import cut
from scipy import stats

df = pd.read_csv("H:\\bank-full.csv")

df['job'] =df['job'].astype('category')
df['marital'] =df['marital'].astype('category')
df['education'] =df['education'].astype('category')
df['default'] =df['default'].astype('category')
df['housing'] =df['housing'].astype('category')
df['loan'] =df['loan'].astype('category')
df['contact'] =df['contact'].astype('category')
df['month'] =df['month'].astype('category')
df['poutcome'] =df['poutcome'].astype('category')
df['y'] =df['y'].astype('category')
print(df.info())

df.isnull().sum().sum()

#Calulating Age
ageMean = df['age'].mean()
ageStd = df['age'].std()
ageMedian = df['age'].median()
minAge = df['age'].min()
maxAge = df['age'].max()

#This code prints out a formatted version of the Age Attribute from Bank-Full.csv
step = 10
bin_range = np.arange(minAge,maxAge+step, step+1)
out, bins = pd.cut(df['age'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Age of Respondents')
agePlot = plt.plot()
plt.xlabel("Age of Clients")
plt.ylabel("Number of Clients")
plt.show()

# Using Min-Max Normalization for Age Attribute
def MinMaxScalar(X):
    return (X - minAge)/(maxAge-minAge)

df['minMaxNormalization_Age'] = df['age'].apply(MinMaxScalar)

MinMaxNormalized_AgeMin = df['minMaxNormalization_Age'].min()
MinMaxNormalized_AgeMax = df['minMaxNormalization_Age'].max()

#Prints Min-Max Normalized Histogram of the Age Attribute
step = 0.1
bin_range_normalized = np.arange(0.0, 1.0+step, step+0.1)
out, bins = pd.cut(df['minMaxNormalization_Age'], bins=bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Age of Respondents (Min-Max Normalized)')
normalizedAgePlot = plt.plot()
plt.xlabel("Age of Clients")
plt.ylabel("Number of Clients")
plt.show()

# Using z-score Normalization of the Age Attribute
df['z_score_age'] = stats.zscore(df['age'])
#zScore_Age_df = df.loc[df['z_score'].abs()<=3]

zScore_AgeMin = df['z_score_age'].min()
zScore_AgeMax = df['z_score_age'].max()

#Prints z-score Normalized Histogram of the Age Attribute
step = 0.5
bin_range_zScoreAgenormalized = np.arange(zScore_AgeMin, zScore_AgeMax+step, step)
out, bins = pd.cut(df['z_score_age'], bins=bin_range_zScoreAgenormalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Age of Respondents (z-score Normalized)')
zScoreNormalizedAgePlot = plt.plot()
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
def MinMaxScalar(X):
    return (X - balanceMin)/(balanceMax-balanceMin)

df['minMaxNormalization_Balance'] = df['balance'].apply(MinMaxScalar)

MinMaxNormalized_BalanceMin = df['minMaxNormalization_Balance'].min()
MinMaxNormalized_BalanceMax = df['minMaxNormalization_Balance'].max()

print(MinMaxNormalized_BalanceMin)
print(MinMaxNormalized_BalanceMax)

#Prints Min-Max Normalized Histogram of the Balance Attribute
step = 0.1
bin_range_normalized = np.arange(0.0, 1.0+step, step+0.1)
out, bins = pd.cut(df['minMaxNormalization_Balance'], bins=bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Balance of Respondents (Min-Max Normalized)')
normalizedAgePlot = plt.plot()
plt.xlabel("Balance of Clients")
plt.ylabel("Number of Clients")
plt.show()

# Using z-score Normalization
#zScore_BalanceNormalized_df = ((df['balance'] - balanceMedian) / balanceStd)
df['z_score_balance'] = stats.zscore(df['balance'])

zScore_AgeMin = df['z_score_balance'].min()
zScore_AgeMax = df['z_score_balance'].max()

#Prints z-score Normalized Histogram
step = 0.5
bin_range_zScoreBalancenormalized = np.arange(zScore_AgeMin, zScore_AgeMax+step, step)
out, bins = pd.cut(df['z_score_balance'], bins=bin_range_zScoreBalancenormalized, include_lowest=True, right=False, retbins=True)
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

#This code prints out a formatted version of the Duration Attribute from Bank-Full.csv
step = 50
duration_bin_range = np.arange(durationMin, durationMax+step, step)
out, bins = pd.cut(df['duration'], bins=duration_bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Call Durations for Customers')
durationPlotFormatted = plt.plot()
plt.xlabel("Call Duration (in Seconds)")
plt.ylabel("Number of Calls")
plt.show()

# Using Min-Max Normalization for Duration Attribute
def MinMaxScalar(X):
    return (X - durationMin)/(durationMax-durationMin)

df['minMaxNormalization_Duration'] = df['duration'].apply(MinMaxScalar)

MinMaxNormalized_DurationMin = df['minMaxNormalization_Duration'].min()
MinMaxNormalized_DurationMax = df['minMaxNormalization_Duration'].max()

#Prints Min-Max Normalized Histogram of the Duration Attribute
step = 0.1
duration_bin_range_normalized = np.arange(MinMaxNormalized_DurationMin, MinMaxNormalized_DurationMax+step, step+0.1)
out, bins = pd.cut(df['minMaxNormalization_Duration'], bins=duration_bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Call Durations for Customers (Min/Max Normalization)')
durationPlotMinMaxNormalization = plt.plot()
plt.xlabel("Call Duration (in Seconds)")
plt.ylabel("Number of Calls")
plt.show()

# Using z-score Normalization of the Duration Attribute
df['z_score_duration'] = stats.zscore(df['duration'])

zScore_DurationMin = df['z_score_duration'].min()
zScore_DurationMax = df['z_score_duration'].max()

#Prints z-score Normalized Histogram of the Duration Attribute
step = 0.5
duration_bin_range_normalized = np.arange(zScore_DurationMin, zScore_DurationMax+step, step)
out, bins = pd.cut(df['z_score_duration'], bins=duration_bin_range_normalized, include_lowest=True, right=False, retbins=True)
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
def MinMaxScalar(X):
    return (X - campaignMin)/(campaignMax-campaignMin)

df['minMaxNormalization_Campaign'] = df['campaign'].apply(MinMaxScalar)

MinMaxNormalized_CampaignMin = df['minMaxNormalization_Campaign'].min()
MinMaxNormalized_CampaignMax = df['minMaxNormalization_Campaign'].max()

#Prints Min-Max Normalized Histogram of the Campaign Attribute
step = 0.1
campaign_bin_range_normalized = np.arange(MinMaxNormalized_CampaignMin, MinMaxNormalized_CampaignMax+step, step)
out, bins = pd.cut(df['minMaxNormalization_Campaign'], bins=campaign_bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of contacts performed during this campaign and for this client (Min/Max Normalization)')
campaignPlotMinMaxNormalization = plt.plot()
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

# Using z-score Normalization of the Campaign Attribute
df['z_score_campaign'] = stats.zscore(df['campaign'])

zScore_CampaignMin = df['z_score_campaign'].min()
zScore_CampaignMax = df['z_score_campaign'].max()

#Prints z-score Normalized Histogram of the Campaign Attribute
step = 0.5
bin_range_campaignzScorenormalized = np.arange(zScore_CampaignMin , zScore_CampaignMax+step, step)
out, bins = pd.cut(df['z_score_campaign'], bins=bin_range_campaignzScorenormalized, include_lowest=True, right=False, retbins=True)
campaignHistogram = out.value_counts(sort=False).plot(kind = 'bar', title='Number of contacts performed during this campaign and for this client (zScore Normalization)')
campaignPlotzScoreNormalization = plt.plot()
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

###################################################################
#Calulating previous
previousStd = df['previous'].std()
previousMean = df['previous'].mean()
previousMedian = df['previous'].median()
previousMax = df['previous'].max()
previousMin = df['previous'].min()

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
def MinMaxScalar(X):
    return (X - previousMin)/(previousMax-previousMin)

df['minMaxNormalization_Previous'] = df['previous'].apply(MinMaxScalar)

MinMaxNormalized_PreviousMin = df['minMaxNormalization_Previous'].min()
MinMaxNormalized_PreviousMax = df['minMaxNormalization_Previous'].max()

#Prints Min-Max Normalized Histogram of the previous Attribute
step = 0.1
previous_bin_range_normalized = np.arange(MinMaxNormalized_PreviousMin, MinMaxNormalized_PreviousMax+step, step)
out, bins = pd.cut(df['minMaxNormalization_Previous'], bins=previous_bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of Contacts Performed before this campaign and for this client (Min/Max Normalization)')
previousPlotMinMaxNormalization = plt.plot()
plt.xlabel("Number of Contacts Performed")
plt.ylabel("Frequency")
plt.show()

# Using z-score Normalization of the previous Attribute
df['z_score_previous'] = stats.zscore(df['previous'])

zScore_PreviousMin = df['z_score_previous'].min()
zScore_PreviousMax = df['z_score_previous'].max()

#Prints z-score Normalized Histogram of the previous Attribute
step = 0.5
bin_range_pdayszScorenormalized = np.arange(zScore_PreviousMin, zScore_PreviousMax+step, step)
out, bins = pd.cut(df['z_score_previous'], bins=bin_range_pdayszScorenormalized, include_lowest=True, right=False, retbins=True)
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

#This code prints out a formatted version of the pdays Attribute from Bank-Full.csv
step = 50
pdays_bin_range = np.arange(pdaysMin,pdaysMax+step, step)
out, bins = pd.cut(df['pdays'], bins=pdays_bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign')
pdaysPlotFormatted = plt.plot()
plt.xlabel("Number of Days Passed")
plt.ylabel("Frequency")
plt.show()

#df['pdays'].replace([-1], 0)

# Using Min-Max Normalization for pdays Attribute
def MinMaxScalar(X):
    return (X - pdaysMin)/(pdaysMax-pdaysMin)

df['minMaxNormalization_Pdays'] = df['pdays'].apply(MinMaxScalar)

MinMaxNormalized_PdaysMin = df['minMaxNormalization_Pdays'].min()
MinMaxNormalized_PdaysMax = df['minMaxNormalization_Pdays'].max()

#Prints Min-Max Normalized Histogram of the pdays Attribute
step = 0.1
pdays_bin_range_normalized = np.arange(MinMaxNormalized_PdaysMin, MinMaxNormalized_PdaysMax+step, step)
out, bins = pd.cut(df['minMaxNormalization_Pdays'], bins=pdays_bin_range_normalized, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar(title='Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign (Min/Max Normalization)')
pdaysPlotMinMaxNormalization = plt.plot()
plt.xlabel("Number of Days Passed")
plt.ylabel("Frequency")
plt.show()

# Using z-score Normalization of the Age Attribute
df['z_score_pdays'] = stats.zscore(df['pdays'])

zScore_PdaysMin = df['z_score_pdays'].min()
zScore_PdaysMax = df['z_score_pdays'].max()

#Prints z-score Normalized Histogram of the pdays Attribute
step = 0.5
bin_range_pdayszScorenormalized = np.arange(zScore_PdaysMin , zScore_PdaysMax+step, step)
out, bins = pd.cut(df['z_score_pdays'], bins=bin_range_pdayszScorenormalized, include_lowest=True, right=False, retbins=True)
pdaysHistogram = out.value_counts(sort=False).plot(kind = 'bar', title='Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign (zScore Normalization)')
pdaysPlotzScoreNormalization = plt.plot()
plt.title('Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign')
plt.xlabel("Number of Days Passed")
plt.ylabel("Frequency")
plt.show()

###################################################################
#Calculating y
df['yCounts'] = df['y'].value_counts()
yCountsPlot = df['y'].value_counts().plot.bar()
plt.title('y')
plt.xlabel('Answers')
plt.ylabel('Frequency')
plt.show()

yMin = df['yCounts'].min()
yMax = df['yCounts'].max()
