import sklearn.metrics.pairwise as skm
import sklearn.preprocessing as skp
import pandas as pd
import numpy as np
import fastdtw
from timeit import default_timer as timer
import sklearn.cluster as skc
from sklearn.model_selection import train_test_split

def dtw(arr1, arr2):
    error, _ = fastdtw.fastdtw(arr1, arr2, radius=1, dist=2)
    return error

rawData_df = pd.read_csv('../input/train_1.csv')
print(rawData_df.shape)
# Fill all NaN with zeros
rawData_df.fillna(value=0.0,inplace=True)
rawData_df.drop('Page',axis=1,inplace=True)

# Shuffle the dataframe but do not reset indices
rawData_df = rawData_df.sample(frac=1)

# Could also scale [0,1/N] or z-normalize
scaled_data = skp.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(rawData_df)

scaledData_df = pd.DataFrame(data=scaled_data,  # values
                index=rawData_df.index.values,  # 1st column as index
                columns=rawData_df.columns)     # 1st row as the column names

dataRows_df = scaledData_df.iloc[:1000]

# Write some sample rows to file
dataRows_df.to_csv(path_or_buf='../Processing/FrameContainer/DataRows/Data_Rows.txt', header=False, sep=' ',index=False, index_label=False, line_terminator=' ', na_rep=0.0)

dataRow_key = open('../Processing/FrameContainer/DataRows/Data_Keys.csv', 'w')

for item in dataRows_df.index.tolist():
    dataRow_key.write("%d\n" % item)

dataRow_key.close()

# Determine split points for processing sets
splitPoints = list(range(1000, len(scaledData_df), 10000))
print(splitPoints)

splitPoints = list(range(1000, len(scaledData_df), 10000))
for i, val in enumerate(splitPoints):
    tmpQueryRows = scaledData_df.iloc[val:val+10000]
    filePath = '../Processing/FrameContainer/QueryRows/Query_Rows_' + str(i) + '.txt' 
    tmpQueryRows.to_csv(path_or_buf=filePath, header=False, sep=' ',index=False, index_label=False, line_terminator=' ', na_rep=0.0)
    print(filePath)
    
    queryRow_key = open('../Processing/FrameContainer/QueryRows/Query_Keys_' + str(i) + '.csv', 'w')

    for item in tmpQueryRows.index.tolist():
        queryRow_key.write("%d\n" % item)
    
    queryRow_key.close()



