# First, let's verify that the ATK client libraries are installed
import trustedanalytics as ta
print "ATK installation path = %s" % (ta.__path__)

# Next, look-up your ATK server URI from the TAP Console and enter the information below.
# This setting will be needed in every ATK notebook so that the client knows what server to communicate with.

# E.g. ta.server.uri = 'demo-atk-c07d8047.demotrustedanalytics.com'
ta.server.uri = 'ENTER URI HERE'

# This notebook assumes you have already created a credentials file.
# Enter the path here to connect to ATK
ta.connect('myuser-cred.creds')

# Create a new frame by uploading rows
data = [ ['a', 1], 
         ['b', 2], 
         ['c', 3], 
         ['b', 4],     
         ['a', 5] ]

schema = [ ('letter', str),
           ('number', ta.int64) ]

frame = ta.Frame(ta.UploadRows(data, schema))

# View the first few rows of a frame
frame.inspect()

# View a specfic number of rows of a frame
frame.inspect(2)

# Add a column to the frame
frame.add_columns(lambda row: row.number * 2, ('number_doubled', ta.int64))
frame.inspect()

# Get summary information for a column
frame.column_summary_statistics('number_doubled')

# Add a column with the cumulative sum of the number column
frame.cumulative_sum('number')
frame.inspect()

# Rename a column
frame.rename_columns({ 'number_doubled': "x2" })
frame.inspect()

# Sort the frame by column 'number' descending
frame.sort('number', False)
frame.inspect()

# Remove a column from the frame
frame.drop_columns("x2")
frame.inspect()

# Download a frame from ATK to pandas
pandas_frame = frame.download(columns=['letter', 'number'])
pandas_frame

# Calculate aggregations on the frame
results = frame.group_by('letter', ta.agg.count, {'number': [ta.agg.avg, ta.agg.sum, ta.agg.min] })
results.inspect()

# Count the number of rows satisfying a predicate
frame.count(lambda row: row.number > 2)



