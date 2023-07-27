# Printing the content of git_log_excerpt.csv
with open("datasets/git_log_excerpt.csv", "r") as file:
    print(file.read())

# Loading in the pandas module
import pandas as pd

# Reading in the log file
git_log = pd.read_csv("datasets/git_log.gz", sep='#', encoding='latin-1', header=None, names=['timestamp', 'author'], compression='gzip')

# Printing out the first 5 rows
git_log.head(5)

# calculating number of commits
number_of_commits = len(git_log)

# calculating number of authors
number_of_authors = len(git_log.query("author != ''").groupby('author'))

# printing out the results
print("%s authors committed %s code changes." % (number_of_authors, number_of_commits))

# Listing top 10 authors
top_10_authors = git_log.groupby('author').count().apply(lambda x: x.sort_values(ascending=False)).head(10)

top_10_authors

# converting the timestamp column
git_log.timestamp = pd.to_datetime(git_log.timestamp, unit='s')

# summarizing the converted timestamp column
git_log.timestamp.describe()

# determining the first real commit timestamp
first_commit_timestamp = git_log.iloc[-1].timestamp

# determining the last sensible commit timestamp
last_commit_timestamp = pd.to_datetime('now')

# filtering out wrong timestamps
corrected_log = git_log[(first_commit_timestamp <= git_log.timestamp) & (git_log.timestamp <= last_commit_timestamp)]

# summarizing the corrected timestamp column
corrected_log['timestamp'].describe()

# Counting the no. commits per year
commits_per_year = corrected_log.groupby(pd.Grouper(key='timestamp', freq='AS')).count()
commits_per_year.rename(columns={'author': 'num_commits'}, inplace=True)

# Listing the first rows
commits_per_year.head(5)

# Setting up plotting in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the data
years = commits_per_year.index.year
ax = commits_per_year.plot.bar(legend=False)
ax.set_title('Commits per year on github')
ax.set_xticklabels(years)
ax.set_xlabel('years')
ax.set_ylabel('number of commits');

# calculating or setting the year with the most commits to Linux
year_with_most_commits = commits_per_year[commits_per_year == commits_per_year.max()].sort_values(by='num_commits').head(1).reset_index()['timestamp'].dt.year
print(year_with_most_commits[0])

