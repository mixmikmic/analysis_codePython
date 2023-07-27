# Time Series

import pandas as pd

df = pd.read_csv('https://github.com/hupili/python-for-data-and-media-communication/raw/master/w7-text/regular_reader_tweets.csv')

df.head()

from datetime import datetime
from dateutil import parser
import numpy

len(df)

df = df.sample(frac=0.1)

len(df)

def parse_datetime(x):
    try:
        return parser.parse(x)
    except:
        return numpy.nan
df['datetime'] = df['created_str'].apply(parse_datetime)

df.set_index('datetime').resample('1w').aggregate('count').plot()

def has_hillary(t):
    return 'hillary' in str(t).lower()
df['kw-hillary'] = df['text'].apply(has_hillary)

def has_trump(t):
    return 'trump' in str(t).lower()
df['kw-trump'] = df['text'].apply(has_trump)

df.set_index('datetime').resample('1w').aggregate('sum').plot()

df.set_index('datetime').resample('1w').aggregate('sum').tail()

df['kw-all'] = df['text'].apply(lambda x: 1)

df.set_index('datetime').resample('1w').aggregate('sum').plot()

df_kws = df.set_index('datetime').resample('1w').aggregate('sum')

(df_kws['kw-trump'] / df_kws['kw-all']).plot()

s_trump_ratio = (df_kws['kw-trump'] / df_kws['kw-all'])

s_hillary_ratio = (df_kws['kw-hillary'] / df_kws['kw-all'])

pd.DataFrame({'trump':s_trump_ratio, 'hillary': s_hillary_ratio}).plot()

df[
    (df['datetime'] > datetime(2016, 1, 1)) &
    (df['datetime'] < datetime(2017, 1, 1))
].set_index('datetime').resample('1w').aggregate('count').plot()

# This one is more close

df[
    (df['datetime'] > datetime(2016, 1, 1)) &
    (df['datetime'] < datetime(2017, 1, 1))
].set_index('datetime').resample('1d').aggregate('count').plot()

df['kw-cliton'] = df['text'].apply(lambda t: 'cliton' in str(t).lower())
df['kw-debate'] = df['text'].apply(lambda t: 'debate' in str(t).lower())
df['kw-blacklivesmatter'] = df['text'].apply(lambda t: 'blacklivesmatter' in str(t).lower())

df_kws = df.set_index('datetime').resample('1m').aggregate('sum')

df_kws.plot()

del df_kws['kw-all']

df_kws.plot()

df_kws.plot(kind='area')

df_kws.divide(df_kws.sum(axis=1), axis=0).plot(kind='area')



