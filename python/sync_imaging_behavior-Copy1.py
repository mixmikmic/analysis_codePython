import sys
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io as scio
import bandit_preprocessing as bp
import sys
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

record = pd.read_csv('/Users/celiaberon/GitHub/mouse_bandit/session_record.csv',index_col=0)
ca_data = scio.loadmat('/Volumes/Neurobio/MICROSCOPE/Celia/data/k7_03142017_test/neuron_results.mat',squeeze_me = True, struct_as_record = False)
neuron = ca_data['neuron_results'] 

record.head(5)

session_name  = '03142017_K7'
mouse_id = 'K7'

record[record['Session ID'] == session_name]

'''
load in trial data
'''
columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked',
           'Right Reward Prob','Left Reward Prob','Reward Given',
          'center_frame','decision_frame']

root_dir = '/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'

full_name = session_name + '_trials.csv'

path_name = os.path.join(root_dir,full_name)

trial_df = pd.read_csv(path_name,names=columns)

trial_df.head(11)

feature_matrix = bp.create_feature_matrix(trial_df,10,mouse_id,session_name,feature_names='Default',imaging=True)

feature_matrix.head(2)

feature_matrix[['10_Port','10_ITI','10_trialDuration']].head(5)

decisions = [0,1]
frames = ['center_frame','decision_frame']
imaging_frames = [] #initialize empty list
for decision in decisions:
    for frame in frames:
        imaging_frames.append(feature_matrix[((feature_matrix['Switch'] == 0) 
                                      & (feature_matrix['Decision'] == decision))][frame])

d_right = {'center_frame_right':imaging_frames[0],
     'decision_frame_right':imaging_frames[1],
    }
d_left = {'center_frame_left':imaging_frames[2],
     'decision_frame_left':imaging_frames[3]}

df_right = pd.DataFrame(data=d_right) #df_right.values[:,0]
df_left = pd.DataFrame(data=d_left)

switch_decision = feature_matrix[feature_matrix['Switch'] == 1]['decision_frame']
switch_center = feature_matrix[feature_matrix['Switch'] == 1]['center_frame']

preStart = switch_cen -10
postDecision = stay_decision_port0 +10
trialDecision = stay_decision_port0

def extract_frames(df, cond1_name, cond1=False, cond2_name=False, cond2=False, frame_type='decision_frame'):
    if type(cond2_name)==str:
        frames = (df[((df[cond1_name] == cond1) 
                    & (df[cond2_name] == cond2))][frame_type])
        return frames
    else:
        frames =(df[(df[cond1_name] == cond1)][frame_type])
        return frames

cond1_name = 'Reward'
cond1_a = 1
cond1_b = 0
cond2_name = 'Decision'
cond2 = 0

frames_center_a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2, 'center_frame')
frames_decision_a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2, 'decision_frame')

frames_center_b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2, 'center_frame')
frames_decision_b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2, 'decision_frame')

preStart_a = frames_center_a - 10 # start 10 frames before center poke
postDecision_a = frames_decision_a + 10 # end 10 frames after decision poke

start_stop_times_a = [[frames_center_a - 10], [frames_decision_a + 10]] # start 10 frames before center poke
start_stop_times_b = [[frames_center_b - 10], [frames_decision_b + 10]] # start 10 frames before center poke



#plt.plot(neuron.C_raw[0, preStart:trialDecision])
nNeurons = neuron.C.shape[0]

# remove neurons that have NaNs
nan_neurons = np.where(np.isnan(neuron.C_raw))[0]
nan_neurons = np.unique(nan_neurons)
good_neurons = [x for x in range(0, nNeurons) if x not in nan_neurons]

nNeurons = len(good_neurons) # redefine number of neurons
nTrials = [len(preStart_a), len(preStart_b)] # number of trials

# iterate through to determine duration between preStart and postDecision for each trial
window_length_a = []
window_length_b = []
for i in range(0,nTrials[0]):
    window_length_a.append((postDecision_a.iloc[i] - preStart_a.iloc[i]))
for i in range(0,nTrials[1]):
    window_length_b.append((postDecision_b.iloc[i] - preStart_b.iloc[i]))
window_length = [window_length_a, window_length_b]

# find longest window between preStart and postDecision and set as length for all trials
max_window = int(np.max((np.max(window_length[:]))))

start_stop_times

#start_stop_times = [start_stop_times_a, start_stop_times_b]
for i in [0,1]:

    aligned_start = np.zeros(nTrials[i], max_window, nNeurons)

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to center poke
    count = 0
    for iNeuron in good_neurons:
        for iTrial in range(0,nTrials[i]):
            aligned_start[iTrial,0:max_window, count] = neuron.C_raw[iNeuron, int(start_stop_times.iloc[i][iTrial]):(int(preStart_a.iloc[iTrial])+max_window)]
        count = count+1

#plt.imshow(aligned_start[:,:,0]), plt.colorbar()

# take mean of fluorescent traces across all trials for each neuron, then normalize for each neuron
mean_across_trials = np.mean(aligned_start, axis=0)
norm_mean = np.zeros((mean_across_trials.shape[0], nNeurons))

for iNeuron in range(0,nNeurons):
    norm_mean[:,iNeuron] = (mean_across_trials[:,iNeuron] - np.min(mean_across_trials, axis=0)[iNeuron])/(np.max(mean_across_trials,axis=0)[iNeuron] - np.min(mean_across_trials,axis=0)[iNeuron])
    #norm_mean = [x-np.min(mean_across_trials, axis=0) for x in mean_across_trials]/(np.max(mean_across_trials,axis=0) - np.min(mean_across_trials,axis=0))
plt.imshow(np.transpose(mean_across_trials)), plt.colorbar()

# heatmap for all neurons (each neuron represented by avg fluorescence across all trials)
plt.imshow(np.transpose(norm_mean)), plt.colorbar()
plt.axvline(x=10, color='k', linestyle = '--', linewidth=.9)

# heatmap for calcium traces of a single neuron across all trials
# white dashed line for center poke time
# white vertical lines for decision poke time -- need something more subtle
sample_neuron = 10

plt.figure(figsize=(8,8))
plt.imshow(aligned_start[:,:,sample_neuron])
plt.axvline(x=10, color='white', linestyle = '--', linewidth=.9)
#plt.scatter(trialDecision-preStart,range(0,nTrials), color='white', marker = '|', s=8)

aligned_decision = np.zeros((nTrials, max_window, nNeurons))

# create array containing segment of raw trace for each neuron for each trial 
# aligned to decision poke
count = 0
for iNeuron in good_neurons:
    for iTrial in range(0,nTrials):
        aligned_decision[iTrial,0:max_window, count] = neuron.C_raw[iNeuron, int(postDecision.iloc[iTrial])-max_window:(int(postDecision.iloc[iTrial]))]
    count = count+1

#plt.imshow(aligned_start[:,:,0]), plt.colorbar()

# take mean of fluorescent traces across all trials for each neuron, then normalize for each neuron
mean_decision = np.mean(aligned_decision, axis=0)
norm_mean_decision = np.zeros((mean_decision.shape[0], nNeurons))

for iNeuron in range(0,nNeurons):
    norm_mean_decision[:,iNeuron] = (mean_decision[:,iNeuron] - np.min(mean_decision, axis=0)[iNeuron])/(np.max(mean_decision,axis=0)[iNeuron] - np.min(mean_across_trials,axis=0)[iNeuron])
plt.imshow(np.transpose(mean_decision)), plt.colorbar()

# heatmap for all neurons (each neuron represented by avg fluorescence across all trials)
plt.imshow(np.transpose(norm_mean_decision)), plt.colorbar()
plt.axvline(x=max_window-10, color='k', linestyle = '--', linewidth=.9)

# save array of normalized means for one condition before re-running on second condition
norm_mean_1 = norm_mean
norm_mean_decision_1 = norm_mean_decision
max_window_1 = max_window

# plot the difference between two conditions for aligned to center poke
plt.imshow(np.transpose(norm_mean_1[0:62] - norm_mean[0:62])), plt.colorbar()
plt.axvline(x=10, color='white', linestyle = '--', linewidth=.9)

# plot the difference between two conditions for aligned to decision poke
max_windows = [max_window_1, max_window]
max_window = min(max_windows)
plt.imshow(np.transpose(norm_mean_decision_1[0:max_window] - norm_mean_decision[0:max_window])), plt.colorbar()
plt.axvline(x=max_window-10, color='white', linestyle = '--', linewidth=.9)

norm_mean_decision

