import seaborn as sns
get_ipython().magic('matplotlib inline')

flights = sns.load_dataset('flights')

tips = sns.load_dataset('tips')

tips.head()

flights.head()

tips.head()

# Matrix form for correlation data
tips.corr()

sns.heatmap(tips.corr())

sns.heatmap(tips.corr(),cmap='coolwarm',annot=True)

flights.pivot_table(values='passengers',index='month',columns='year')

pvflights = flights.pivot_table(values='passengers',index='month',columns='year')
sns.heatmap(pvflights)

sns.heatmap(pvflights,cmap='magma',linecolor='white',linewidths=1)

sns.clustermap(pvflights)

# More options to get the information a little clearer like normalization
sns.clustermap(pvflights,cmap='coolwarm',standard_scale=1)

