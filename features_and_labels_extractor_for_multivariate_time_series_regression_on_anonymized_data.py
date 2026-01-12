import datetime

import pandas as pd
import numpy as np
from numpy import inf

from scipy.signal import iirnotch, filtfilt
from scipy.signal import find_peaks

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, make_scorer
from sklearn.neural_network import MLPClassifier

import joblib
import pickle

from typing import Union


STORAGE_PATHS_DIR = {
	'P001_1st_session/': [
		'p001_1st_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P001_2nd_session/': [
		'p001_2nd_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P002_1st_session/': [
		'p002_1st_session_anonymized.log',
		['0/RawData/MuseS-4626_EEG_anonymized.csv', '1/RawData/MuseS-4626_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv', '1/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv', '1/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv', '1/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P002_2nd_session/': [
		'p002_2nd_session_anonymized.log',
		['0/RawData/MuseS-4626_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P003_1st_session/': [
		'p003_1st_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv', '1/RawData/MuseS-4646_EEG_anonymized.csv', '2/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv', '1/RawData/8413C6_BVP_anonymized.csv', '2/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv', '1/RawData/8413C6_GSR_anonymized.csv', '2/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv', '1/RawData/8413C6_TEMP_anonymized.csv', '2/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P003_2nd_session/': [
		'p003_2nd_session_anonymized.log',
		['0/RawData/MuseS-4626_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P004_1st_session/': [
		'p004_1st_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/A4880B_BVP_anonymized.csv'],
		['0/RawData/A4880B_GSR_anonymized.csv'],
		['0/RawData/A4880B_TEMP_anonymized.csv'],
	],
	'P004_2nd_session/': [
		'p004_2nd_session_anonymized.log',
		['0/RawData/EEG_anonymized.csv'],
		['0/RawData/BVP_anonymized.csv'],
		['0/RawData/GSR_anonymized.csv'],
		['0/RawData/TEMP_anonymized.csv'],
	],
	'P005_1st_session/': [
		'p005_1st_session_anonymized.log',
		['0/RawData/MuseS-4626_EEG_anonymized.csv', '1/RawData/MuseS-4626_EEG_anonymized.csv', '2/RawData/MuseS-4626_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv', '1/RawData/8413C6_BVP_anonymized.csv', '2/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv', '1/RawData/8413C6_GSR_anonymized.csv', '2/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv', '1/RawData/8413C6_TEMP_anonymized.csv', '2/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P005_2nd_session/': [
		'p005_2nd_session_anonymized.log',
		['0/RawData/EEG_anonymized.csv'],
		['0/RawData/BVP_anonymized.csv'],
		['0/RawData/GSR_anonymized.csv'],
		['0/RawData/TEMP_anonymized.csv'],
	],
	'P006_1st_session/': [
		'p006_1st_session_anonymized.log',
		['0/RawData/MuseS-4626_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P006_2nd_session/': [
		'p006_2nd_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/A4880B_BVP_anonymized.csv'],
		['0/RawData/A4880B_GSR_anonymized.csv'],
		['0/RawData/A4880B_TEMP_anonymized.csv'],
	],
	'P007_1st_session/': [
		'p007_1st_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P007_2nd_session/': [
		'p007_2nd_session_anonymized.log',
		['0/RawData/MuseS-4626_EEG_anonymized.csv'],
		['0/RawData/A4880B_BVP_anonymized.csv'],
		['0/RawData/A4880B_GSR_anonymized.csv'],
		['0/RawData/A4880B_TEMP_anonymized.csv'],
	],
	'P008_1st_session/': [
		'p008_1st_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P008_2nd_session/': [
		'p008_2nd_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P009_1st_session/': [
		'p009_1st_session_anonymized.log',
		['0/RawData/MuseS-4626_EEG_anonymized.csv'],
		['0/RawData/A4880B_BVP_anonymized.csv'],
		['0/RawData/A4880B_GSR_anonymized.csv'],
		['0/RawData/A4880B_TEMP_anonymized.csv'],
	],
	'P009_2nd_session/': [
		'p009_2nd_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/A4880B_BVP_anonymized.csv'],
		['0/RawData/A4880B_GSR_anonymized.csv'],
		['0/RawData/A4880B_TEMP_anonymized.csv'],
	],
	'P010_1st_session/': [
		'p010_1st_session_anonymized.log',
		['0/RawData/MuseS-4646_EEG_anonymized.csv'],
		['0/RawData/8413C6_BVP_anonymized.csv'],
		['0/RawData/8413C6_GSR_anonymized.csv'],
		['0/RawData/8413C6_TEMP_anonymized.csv'],
	],
	'P010_2nd_session/': [
		'p010_2nd_session_anonymized.log',
		['0/RawData/MuseS-4626_EEG_anonymized.csv'],
		['0/RawData/A4880B_BVP_anonymized.csv'],
		['0/RawData/A4880B_GSR_anonymized.csv'],
		['0/RawData/A4880B_TEMP_anonymized.csv'],
	],
}

EPOCH_DURATION_IN_SECONDS = 30

LABEL_NUM_TO_LABEL_DICT = {
	0: 'UNKNOWN',
	1: 'very, very low',
	2: 'low',
	3: 'nor',
	4: 'high',
	5: 'very, very high',
	6: 'UNKNOWN',
}

LABEL_FOR_FLAGGING_DICT = {
	'UNKNOWN': None,
	'very, very low': -2,
	'low': -1,
	'nor': 0,
	'high': 1,
	'very, very high': 2,
}


def extract_valid_windows_for_duration_of_n_seconds(data, assumed_srate, n_seconds=30):
	win_len = assumed_srate * n_seconds
	windowed_data = []
	for i in range(int(len(data)//win_len)):
		windowed_data.append(data.iloc[int(i*win_len):int((i+1)*win_len)])
	return windowed_data


def trim_data_to_length_of_experiment(data, experiment_start, experiment_end):
	start_rows_skipped_ctr = 0
	for timestamp in data['Timestamp']:
		if parse_relative_timestamp(timestamp) >= experiment_start: # .%f
			print('OVERALL, %d ROWS IN THE BEGINNING WERE WRITTEN BEFORE THE START OF THE FIRST TASK!' % start_rows_skipped_ctr)
			break
		else:
			start_rows_skipped_ctr += 1

	end_rows_skipped_ctr = 0
	for timestamp in data.iloc[::-1]['Timestamp']:
		if parse_relative_timestamp(timestamp) <= experiment_end: # .%f
			print('OVERALL, %d ROWS AT THE END WERE WRITTEN AFTER FINISHING THE LAST TASK!' % end_rows_skipped_ctr)
			break
		else:
			end_rows_skipped_ctr += 1

	data['Timestamp'] = parse_relative_timestamp(data['Timestamp'])
	EPOCH_TIMESTAMP = pd.to_datetime('1970-01-01 00:00:00')

	return data[(data['Timestamp'] - EPOCH_TIMESTAMP >= pd.Timedelta(experiment_start)) & (data['Timestamp'] - EPOCH_TIMESTAMP <= pd.Timedelta(experiment_end))] # data[(data['Timestamp'] >= pd.Timedelta(experiment_start)) & (data['Timestamp'] <= pd.Timedelta(experiment_end))] # data[data['Timestamp'].between(pd.Timedelta(experiment_start), pd.Timedelta(experiment_end))] # data.iloc[:,start_rows_skipped_ctr:-end_rows_skipped_ctr]


def roughly_same_starting_time(win_a, win_b):
	if (win_a is None) or (win_b is None) or win_a.empty or win_b.empty:
		return False
	win_a_start = win_a.iloc[0] # win_a_start = win_a.iloc[0,]['Timestamp']
	win_a_minus_ten_seconds = win_a_start - datetime.timedelta(seconds=10)
	win_a_plus_ten_seconds = win_a_start + datetime.timedelta(seconds=10)
	win_b_time = win_b.iloc[0] # win_b_time = win_b.iloc[0,]['Timestamp']
	return win_a_minus_ten_seconds <= win_b_time <= win_a_plus_ten_seconds


def get_label_for_timestamp(timestamp_str, labels, pid, session_idx):
	timestamp = timestamp_str.iloc[0] # parse_relative_timestamp(timestamp_str)
	datetime_obj = parse_relative_timestamp(str(timestamp))

	# Here, these were the last tasks, which apparently were not properly placed in the compiled log, resulting in loading issues. Here's the correct information now.
	if (pid == 'P008') and (session_idx == '_S2'):
		missing_task_datetime_obj_start = parse_relative_timestamp('0000-00-00 01:51:31')
		print('Checking for P008...')
		if datetime_obj > missing_task_datetime_obj_start:
			print('Found the match!')
			return ('0000-00-00 01:51:31', 'UNKNOWN', 80, 'nor')

	if (pid == 'P007') and (session_idx == '_S2'):
		missing_task_datetime_obj_start = parse_relative_timestamp('0000-00-00 01:45:42')
		print('Checking for P007...')
		if datetime_obj > missing_task_datetime_obj_start:
			print('Found the match!')
			return ('0000-00-00 01:45:42', 'UNKNOWN', 84, 'high')

	if (pid == 'P009') and (session_idx == '_S2'):
		missing_task_datetime_obj_start = parse_relative_timestamp('0000-00-00 01:39:01')
		print('Checking for P009...')
		if datetime_obj > missing_task_datetime_obj_start:
			print('Found the match!')
			return ('0000-00-00 01:39:01', 'UNKNOWN', 44, 'very, very high')

	if (pid == 'P002') and (session_idx == '_S1'):
		missing_task_datetime_obj_start = parse_relative_timestamp('0000-00-00 01:55:15')
		print('Checking for P002...')
		if datetime_obj > missing_task_datetime_obj_start:
			print('Found the match!')
			return ('0000-00-00 01:55:15', 'UNKNOWN', 100, 'nor')

	if (pid == 'P010') and (session_idx == '_S1'):
		missing_task_datetime_obj_start = parse_relative_timestamp('0000-00-00 01:38:02')
		print('Checking for P010...')
		if datetime_obj > missing_task_datetime_obj_start:
			print('Found the match!')
			return ('0000-00-00 01:38:02', 'UNKNOWN', 0, 'very, very low')

	if (pid == 'P004') and (session_idx == '_S1'):
		missing_task_datetime_obj_start = parse_relative_timestamp('0000-00-00 02:03:29')
		print('Checking for P004...')
		if datetime_obj > missing_task_datetime_obj_start:
			print('Found the match!')
			return ('0000-00-00 02:03:29', 'UNKNOWN', 10, 'low')

	for idx, important_time in enumerate(labels):
		start_time_this_task = important_time[0]
		end_time_this_task = important_time[1]

		# Convert the time-point (Timestamp) into a duration (timedelta)
		# relative to the experiment start.
		timestamp_duration = parse_relative_timestamp(timestamp)

		# Now, compare the durations (timedelta vs timedelta)
		if (start_time_this_task <= timestamp_duration) and (timestamp_duration <= end_time_this_task):
		# if (start_time_this_task <= timestamp) and (timestamp <= end_time_this_task):
			return (important_time[0], important_time[1], important_time[-2], LABEL_NUM_TO_LABEL_DICT[important_time[-1]]) # this_task_performance, cl_level

	return ('UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN')


def for_this_window_find_matching_data(timestamp_col, bvp_data, gsr_data, temp_data):
	bvp_idx, gsr_idx, temp_idx = None, None, None

	for idx, win in enumerate(bvp_data):
		if roughly_same_starting_time(timestamp_col, win['Timestamp']):
			bvp_idx = idx
			break
		elif win['Timestamp'].iloc[0] - datetime.timedelta(seconds=10) > timestamp_col.iloc[0]:
			break
	for idx, win in enumerate(gsr_data):
		if roughly_same_starting_time(timestamp_col, win['Timestamp']):
			gsr_idx = idx
			break
		elif win['Timestamp'].iloc[0] - datetime.timedelta(seconds=10) > timestamp_col.iloc[0]:
			break
	for idx, win in enumerate(temp_data):
		if roughly_same_starting_time(timestamp_col, win['Timestamp']):
			temp_idx = idx
			break
		elif win['Timestamp'].iloc[0] - datetime.timedelta(seconds=10) > timestamp_col.iloc[0]:
			break
	if (bvp_idx is not None) and (gsr_idx is not None) and (temp_idx is not None):
		return [bvp_data[bvp_idx], gsr_data[gsr_idx], temp_data[temp_idx]]
	else:
		return None


def combine_labels_and_data(eeg_data, bvp_data, gsr_data, temp_data, labels, pid, session_idx):
	start_times_to_data = {}

	labels_to_data_dict = {
		'very, very low': [],
		'low': [],
		'nor': [],
		'high': [],
		'very, very high': [],
		'UNKNOWN': []
	}

	if len(eeg_data) == len(bvp_data) == len(gsr_data) == len(temp_data):
		for idx, win in eeg_data.iterrows():
			task_start, task_end, perf, clevel = get_label_for_timestamp(win['Timestamp'], labels, pid, session_idx)
			if perf == 'UNKNOWN':
				continue
			labels_to_data_dict[clevel].append([eeg_data[idx], bvp_data[idx], gsr_data[idx], temp_data[idx]])
			key_all_tuple = str(task_start) + '/' + str(task_end) + '/' + str(perf) + '/' + str(clevel)
			if key_all_tuple in start_times_to_data.keys():
				start_times_to_data[key_all_tuple].append([eeg_data[idx], bvp_data[idx], gsr_data[idx], temp_data[idx]])
			else:
				start_times_to_data[key_all_tuple] = []
				start_times_to_data[key_all_tuple].append([eeg_data[idx], bvp_data[idx], gsr_data[idx], temp_data[idx]])
	else:
		min_len = min(len(eeg_data), len(bvp_data), len(gsr_data), len(temp_data))
		eeg_ctr = 0
		ctr = 0
		while (ctr < min_len) and (eeg_ctr < len(eeg_data)):
			if len(np.unique(eeg_data[eeg_ctr]['AF7'].to_numpy())) <= 3:
				eeg_ctr+=1
				continue
			task_start, task_end, perf, clevel = get_label_for_timestamp(bvp_data[ctr]['Timestamp'], labels, pid, session_idx)
			if perf == 'UNKNOWN':
				ctr += 1
				continue
			labels_to_data_dict[clevel].append([eeg_data[eeg_ctr], bvp_data[ctr], gsr_data[ctr], temp_data[ctr]])
			key_all_tuple = str(task_start) + '/' + str(task_end) + '/' + str(perf) + '/' + str(clevel)
			if key_all_tuple in start_times_to_data.keys():
				start_times_to_data[key_all_tuple].append([eeg_data[eeg_ctr], bvp_data[ctr], gsr_data[ctr], temp_data[ctr]])
			else:
				start_times_to_data[key_all_tuple] = []
				start_times_to_data[key_all_tuple].append([eeg_data[eeg_ctr], bvp_data[ctr], gsr_data[ctr], temp_data[ctr]])
			ctr += 1
			eeg_ctr+=1

	return labels_to_data_dict, start_times_to_data


def compute_band_powers(eegdata, fs):
	"""Extract the features (band powers) from the EEG.
	Args:
	eegdata (numpy.ndarray): array of dimension [number of samples, number of channels]
	fs (float): sampling frequency of eegdata
	Returns:
	(numpy.ndarray): feature matrix of shape [number of feature points, number of different features]
	"""
	# 1. Compute the PSD
	winSampleLength, nbCh = eegdata.shape

	# Apply Hamming window
	w = np.hamming(winSampleLength)
	dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
	dataWinCenteredHam = (dataWinCentered.T * w).T

	NFFT = nextpow2(winSampleLength)
	Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
	PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
	f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

	# SPECTRAL FEATURES
	# Average of band powers
	# Delta <4
	ind_delta, = np.where(f < 4)
	meanDelta = np.mean(PSD[ind_delta, :], axis=0)
	# Theta 4-8
	ind_theta, = np.where((f >= 4) & (f <= 8))
	meanTheta = np.mean(PSD[ind_theta, :], axis=0)
	# Alpha 8-12
	ind_alpha, = np.where((f >= 8) & (f <= 12))
	meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
	# Beta 12-30
	ind_beta, = np.where((f >= 12) & (f < 30))
	meanBeta = np.mean(PSD[ind_beta, :], axis=0)
	# Gamma 30-45
	ind_gamma, = np.where((f >= 30) & (f < 45))
	meanGamma = np.mean(PSD[ind_gamma, :], axis=0)

	feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha, meanBeta, meanGamma), axis=0)

	feature_vector = np.log10(feature_vector)

	return feature_vector


def extract_eeg_features(window):
	EEG_FS = 256
	# Start by notch-filtering the EEG
	f0 = 50
	b, a = iirnotch(f0, Q=30, fs=EEG_FS)
	if 'Timestamp' in window.columns:
		window.drop('Timestamp', inplace=True, axis=1)
	window['AF7'] = pd.to_numeric(window.AF7, errors='ignore')
	window['AF8'] = pd.to_numeric(window.AF8, errors='ignore')
	window['TP9'] = pd.to_numeric(window.TP9, errors='ignore')
	window['TP10'] = pd.to_numeric(window.TP10, errors='ignore')
	notchd_window = window # notch_filter(window, ['AF7', 'AF8', 'TP9', 'TP10'], b, a)

	# mean_channel = np.expand_dims(np.mean(np.asarray([notchd_window['AF7'], notchd_window['AF8'], notchd_window['TP9'], notchd_window['TP10']]), axis=0), axis=-1)
	mean_channel = np.expand_dims(np.mean(np.asarray([notchd_window['AF7'], notchd_window['AF8'], notchd_window['TP9'], notchd_window['TP10']]), axis=0), axis=-1)
	# prefrontal_mean_eeg = np.expand_dims(np.mean(np.asarray([notchd_window['AF7'], notchd_window['AF8']]), axis=0), axis=-1)
	prefrontal_mean_eeg = np.expand_dims(np.mean(np.asarray([notchd_window['AF7'], notchd_window['AF8']]), axis=0), axis=-1)
	# temporal_mean_eeg = np.expand_dims(np.mean(np.asarray([notchd_window['TP9'], notchd_window['TP10']]), axis=0), axis=-1)
	temporal_mean_eeg = np.expand_dims(np.mean(np.asarray([notchd_window['TP9'], notchd_window['TP10']]), axis=0), axis=-1)
	mean_feature_vector = compute_band_powers(mean_channel, EEG_FS)
	prefrontal_feature_vector = compute_band_powers(prefrontal_mean_eeg, EEG_FS)
	temporal_feature_vector = compute_band_powers(temporal_mean_eeg, EEG_FS)

	(DELTA_IDX, THETA_IDX, ALPHA_IDX, BETA_IDX, GAMMA_IDX) = (0, 1, 2, 3, 4)

	sample_features = [
		np.squeeze((mean_feature_vector[BETA_IDX] / (mean_feature_vector[THETA_IDX] + mean_feature_vector[ALPHA_IDX]))), # 'EngagementIndex'
		np.squeeze(prefrontal_feature_vector[THETA_IDX] / temporal_feature_vector[ALPHA_IDX]), # 'BrainBeat'
		np.squeeze(mean_feature_vector[THETA_IDX] / mean_feature_vector[ALPHA_IDX]), # 'CLI'
		np.mean((notchd_window['AF7'] + notchd_window['TP9']) - (notchd_window['AF8'] + notchd_window['TP10'])), # 'AsymmetryIndexAllChannels'
		np.mean((notchd_window['AF7']) - (notchd_window['AF8'])), # 'AsymmetryIndexPrefrontal'
		np.mean((notchd_window['TP9']) - (notchd_window['TP10'])), # 'AsymmetryIndexTemporal'
		np.squeeze(prefrontal_feature_vector[DELTA_IDX]), # 'PrefrontalDeltaPower'
		np.squeeze(prefrontal_feature_vector[THETA_IDX]), # 'PrefrontalThetaPower'
		np.squeeze(prefrontal_feature_vector[ALPHA_IDX]), # 'PrefrontalAlphaPower'
		np.squeeze(prefrontal_feature_vector[BETA_IDX]), # 'PrefrontalBetaPower'
		np.squeeze(prefrontal_feature_vector[GAMMA_IDX]), # 'PrefrontalGammaPower'
		np.squeeze(temporal_feature_vector[DELTA_IDX]), # 'TemporalDeltaPower'
		np.squeeze(temporal_feature_vector[THETA_IDX]), # 'TemporalThetaPower'
		np.squeeze(temporal_feature_vector[ALPHA_IDX]), # 'TemporalAlphaPower'
		np.squeeze(temporal_feature_vector[BETA_IDX]), # 'TemporalBetaPower'
		np.squeeze(temporal_feature_vector[GAMMA_IDX]), # 'TemporalGammaPower'
	]

	actual_features = []

	for feature in sample_features:
		actual_features.append(feature if feature is not None else 0)

	return actual_features


def bvp_to_hr(input_df):
	input_df['BVP'] = pd.to_numeric(input_df.BVP, errors='ignore')
	sampling_rate = 64 # bvp_assumed_srate = 64
	bvp_signal = input_df['BVP'] - input_df['BVP'].mean()
	peaks, _ = find_peaks(bvp_signal, distance=sampling_rate*0.5)
	peak_intervals_seconds = np.diff(input_df['Timestamp'].iloc[peaks].values).astype('timedelta64[ms]').astype('float') / 1000
	heart_rates = 60 / peak_intervals_seconds
	hr_df = pd.DataFrame({
		'Timestamp': input_df['Timestamp'].iloc[peaks[:-1]],
		'HR': heart_rates
	})

	result_df = pd.merge_asof(input_df, hr_df, on='Timestamp', direction='forward').bfill()

	return result_df


def extract_bvp_features(window):
	hr_df = bvp_to_hr(window)
	hr_numpy = hr_df['HR'].to_numpy()
	hr_numpy[hr_numpy == -inf] = 30
	hr_numpy[hr_numpy == inf] = 120
	hr_numpy[np.isnan(hr_numpy)] = 30
	return hr_df['HR'].mean()


def extract_gsr_features(window):
	window['GSR'] = pd.to_numeric(window.GSR, errors='ignore')
	return np.asarray(window['GSR']).mean()	# simply set it to mean for now, in lack of time...


def extract_temp_features(window):
	window['TEMP'] = pd.to_numeric(window.TEMP, errors='ignore')
	return np.asarray(window['TEMP']).mean()	# simply set it to mean for now, in lack of time...


def combine_labels_and_data_to_same_length_collections(labels_to_list_of_features_dict):
	label_list = []
	tuple_list = []
	# for each key, value run through to check the key and then assign respective label to it and store it in the label_list and tuple_list tmp_to_return
	for key, value in labels_to_list_of_features_dict.items():
		for dat in value:
			tuple_list.append(np.asarray(dat))
			label_list.append(LABEL_FOR_FLAGGING_DICT[key])
	return np.asarray(tuple_list), np.asarray(label_list)


def combine_labels_and_data_to_same_length_collections_for_all_data(labels_to_list_of_features_dict):
	label_list = []
	tuple_list = []
	# for each key, value run through to check the key and then assign respective label to it and store it in the label_list and tuple_list tmp_to_return
	for key, value in labels_to_list_of_features_dict.items():
		for dat in value:
			tuple_list.append(np.asarray(dat))
			label_list.append(key)
	return np.asarray(tuple_list), np.asarray(label_list)


def load_log_file_and_extract_relevant_times(path_to_log):

	log_markers_to_check = [
		'Routine Vocabulary Presenter Setup:',
		'Routine WILL NOW START THE TASK',
		'questionnaire_element Likert_Scale_Rating.mental effort',
		'questionnaire_element Likert_Scale_Rating.stress',
		'which is correct',
		'which is incorrect',
		'Routine Eye Closing Routine '
	]

	task_times_and_labels = []

	with open(path_to_log, 'r') as file:
		this_task_performance = 0
		correct_ctr = 0
		incorrect_ctr = 0
		current_task = None
		current_task_start = None
		current_task_end = None
		for line in file:
			log_line = line.rstrip()
			if any(marker in log_line for marker in log_markers_to_check):
				if 'questionnaire_element Likert_Scale_Rating.mental effort' in log_line:
					if (correct_ctr > 0) or (incorrect_ctr > 0):
						this_task_performance = (correct_ctr / (correct_ctr + incorrect_ctr)) * 100
						current_task_end = parse_relative_timestamp(log_line.split('INFO:logging_utilities:')[1].split(': Routine')[0])
						cl_level = round(float(log_line.split('with response ')[1]))
						task_times_and_labels.append([current_task_start, current_task_end, this_task_performance, cl_level])
					this_task_performance = 0
					correct_ctr = 0
					incorrect_ctr = 0
				elif 'which is correct' in log_line:
					correct_ctr += 1
				elif 'which is incorrect' in log_line:
					incorrect_ctr += 1
				elif 'Routine WILL NOW START THE TASK' in log_line:
					current_task = log_line.split('Routine WILL NOW START THE TASK ')[1]
					current_task_start = parse_relative_timestamp(log_line.split('INFO:logging_utilities:')[1].split(': Routine')[0])

	return task_times_and_labels


def nextpow2(i):
	"""
	Find the next power of 2 for number i
	"""
	n = 1
	while n < i:
		n *= 2
	return n


def extract_features_for_windows(label_to_raw_data_dict):
	features_dict = {}
	for key, value in label_to_raw_data_dict.items():
		samples = []
		for collection in value:
			temp_collection_copy = collection.copy()
			tmp_eeg_features = extract_eeg_features(temp_collection_copy[0])
			tmp_bvp_features = extract_bvp_features(temp_collection_copy[1])
			tmp_gsr_features = extract_gsr_features(temp_collection_copy[2])
			tmp_temp_features = extract_temp_features(temp_collection_copy[3])
			samples.append(np.concatenate((tmp_eeg_features, tmp_bvp_features, tmp_gsr_features, tmp_temp_features), axis=None))
		features_dict[key] = samples
	return features_dict


def extract_features_for_windows_with_all_data(label_to_raw_data_dict):
	features_dict = {}
	for key, value in label_to_raw_data_dict.items():
		samples = []
		for collection in value:
			temp_collection_copy = collection.copy()
			tmp_eeg_features = extract_eeg_features(temp_collection_copy[0])
			tmp_bvp_features = extract_bvp_features(temp_collection_copy[1])
			tmp_gsr_features = extract_gsr_features(temp_collection_copy[2])
			tmp_temp_features = extract_temp_features(temp_collection_copy[3])
			samples.append(np.concatenate((tmp_eeg_features, tmp_bvp_features, tmp_gsr_features, tmp_temp_features), axis=None))
		features_dict[key] = samples
	return features_dict


def parse_relative_timestamp(timestamp_str_or_series: Union[str, pd.Series]) -> Union[datetime.timedelta, pd.TimedeltaIndex]:
	"""
	Parses a relative timestamp string or a Pandas Series/Column, handling flexible 
	formats while working around the invalid '0000-00-00' date structure and the 
	newly encountered '0 days 00:05:42' Timedelta string format.

	This function programmatically resolves the 'ValueError: year 0 is out of range' 
	error and the current issue where the Timedelta string format is not recognized.
	"""

	# 1. Prepare the reference points
	BASE_DATE_STR = '1970-01-01'
	REFERENCE_POINT = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)

	is_series = isinstance(timestamp_str_or_series, pd.Series)

	# 2. String/Series preparation for '0000-00-00' replacement

	# Convert to string representation for safe manipulation
	safe_data = timestamp_str_or_series.astype(str) if is_series else timestamp_str_or_series

	if is_series:
		# Only replace '0000-00-00' entries, leaving '0 days ...' entries untouched.
		safe_data = safe_data.str.replace('0000-00-00', BASE_DATE_STR, regex=False)
	elif isinstance(safe_data, str):
		safe_data = safe_data.replace('0000-00-00', BASE_DATE_STR)


	# 3. Attempt parsing with flexible formats (using Pandas tools for robustness)

	def parse_with_formats(data):
		# 1. Try parsing as Datetime (high precision)
		try:
			return pd.to_datetime(data, format='%Y-%m-%d %H:%M:%S.%f', errors='raise')
		except (ValueError, TypeError, pd.core.tools.datetimes.DateParseError):
			# 2. Try parsing as Datetime (low precision)
			try:
				return pd.to_datetime(data, format='%Y-%m-%d %H:%M:%S', errors='raise')
			except (ValueError, TypeError, pd.core.tools.datetimes.DateParseError):
				# 3. Try parsing as Timedelta string (NEW: Handles "0 days 00:05:42" format)
				try:
					return pd.to_timedelta(data)
				except Exception as e:
					# If all formats fail, raise the final error
					raise ValueError(f"Time data could not be parsed with high or low precision formats.") from e

	parsed_time_object = parse_with_formats(safe_data)

	# 4. Calculate the pure duration (timedelta)

	# Check if the result is a Timestamp/Datetime type (needs subtraction)
	if isinstance(parsed_time_object, (pd.Timestamp, datetime.datetime, pd.DatetimeIndex)):
		# Subtract reference point to get the relative duration
		relative_timedelta_result = parsed_time_object - REFERENCE_POINT
	else:
		# If it's a Timedelta type (from pd.to_timedelta), it's already the duration
		relative_timedelta_result = parsed_time_object

    # 5. Return the correct type based on input
	if not is_series:
		# If the result is a Pandas container (TimedeltaIndex or Series), extract the single element
		if isinstance(relative_timedelta_result, (pd.TimedeltaIndex, pd.Series)):
			result_td = relative_timedelta_result.iloc[0]
		else:
			# It's either a single pd.Timedelta, pd.Timestamp, or a single datetime.timedelta
			result_td = relative_timedelta_result

		# *** NEW FIX: If a Pandas Timestamp sneaked out, convert it to a duration first ***
		if isinstance(result_td, pd.Timestamp):
			# Convert the remaining time-point into a duration relative to the epoch
			result_td = result_td - REFERENCE_POINT

		# If the result is a Pandas Timedelta (duration), convert it to standard Python timedelta
		if isinstance(result_td, pd.Timedelta):
			return result_td.to_pytimedelta()

		# If it was already a standard datetime.timedelta or another type
		return result_td

	# If the input was a Series, return the TimedeltaIndex/Series
	return relative_timedelta_result


if __name__ == '__main__':
	
	for PARTICIPANT_DIR, PARTICIPANT_DATA_STORAGE in STORAGE_PATHS_DIR.items():

		BASE_PATH = 'PATH_TO_DIR/Anonymized_E_Learning_data/' + PARTICIPANT_DIR
		LOG_STORAGE_PATH = PARTICIPANT_DATA_STORAGE[0]
		EEG_STORAGE_PATH = PARTICIPANT_DATA_STORAGE[1]
		BVP_STORAGE_PATH = PARTICIPANT_DATA_STORAGE[2]
		GSR_STORAGE_PATH = PARTICIPANT_DATA_STORAGE[3]
		TEMP_STORAGE_PATH = PARTICIPANT_DATA_STORAGE[4]

		print(BASE_PATH)

		# Load data
		individual_eeg_data = []
		individual_bvp_data = []
		individual_gsr_data = []
		individual_temp_data = []

		for eeg_storage_path in EEG_STORAGE_PATH:
			print(eeg_storage_path)
			eeg = pd.read_csv(BASE_PATH + eeg_storage_path, names=['Timestamp', 'AF7', 'AF8', 'TP9', 'TP10', 'FpZ'], dtype=str)
			eeg_important = eeg.loc[:, ['Timestamp', 'AF7', 'AF8', 'TP9', 'TP10']]
			mask = eeg_important.Timestamp.str.contains('.')
			eeg_important = eeg_important.assign(NewTimestamp=eeg_important.Timestamp[mask].str.split('.').str[0].values)
			eeg_important.drop(columns=['Timestamp'], inplace=True)
			eeg_important.rename(columns={'NewTimestamp':'Timestamp', 'AF7':'AF7', 'AF8':'AF8', 'TP9':'TP9', 'TP10':'TP10'}, inplace=True)
			eeg_important_reordered = eeg_important.loc[:, ['Timestamp', 'AF7', 'AF8', 'TP9', 'TP10']]# [5, 0, 1, 2, 3, 4]] #['Timestamp', 'AF7', 'AF8', 'TP9', 'TP10', 'FpZ']]
			eeg_important_reordered.dropna(inplace=True)
			individual_eeg_data.append(eeg_important_reordered)

		for bvp_storage_path in BVP_STORAGE_PATH:
			print(bvp_storage_path)
			bvp = pd.read_csv(BASE_PATH + bvp_storage_path, names=['Timestamp', 'BVP'], dtype=str)
			bvp_important = bvp.loc[:, ['Timestamp', 'BVP']]
			mask = bvp_important.Timestamp.str.contains('.')
			bvp_important = bvp_important.assign(NewTimestamp=bvp_important.Timestamp[mask].str.split('.').str[0].values)
			bvp_important.drop(columns=['Timestamp'], inplace=True)
			bvp_important.rename(columns={'NewTimestamp':'Timestamp', 'BVP':'BVP'}, inplace=True)
			bvp_important_reordered = bvp_important.loc[:, ['Timestamp', 'BVP']]# [1, 0]] # ['Timestamp', 'BVP']]
			bvp_important_reordered.dropna(inplace=True)
			individual_bvp_data.append(bvp_important_reordered)

		for gsr_storage_path in GSR_STORAGE_PATH:
			print(gsr_storage_path)
			gsr = pd.read_csv(BASE_PATH + gsr_storage_path, names=['Timestamp', 'GSR'], dtype=str)
			gsr_important = gsr.loc[:, ['Timestamp', 'GSR']]
			mask = gsr_important.Timestamp.str.contains('.')
			gsr_important = gsr_important.assign(NewTimestamp=gsr_important.Timestamp[mask].str.split('.').str[0].values)
			gsr_important.drop(columns=['Timestamp'], inplace=True)
			gsr_important.rename(columns={'NewTimestamp':'Timestamp', 'GSR':'GSR'}, inplace=True)
			gsr_important_reordered = gsr_important.loc[:, ['Timestamp', 'GSR']]# [1, 0]] # ['Timestamp', 'GSR']]
			gsr_important_reordered.dropna(inplace=True)
			individual_gsr_data.append(gsr_important_reordered)

		for temp_storage_path in TEMP_STORAGE_PATH:
			print(temp_storage_path)
			temp = pd.read_csv(BASE_PATH + temp_storage_path, names=['Timestamp', 'TEMP'], dtype=str)
			temp_important = temp.loc[:, ['Timestamp', 'TEMP']]
			mask = temp_important.Timestamp.str.contains('.')
			temp_important = temp_important.assign(NewTimestamp=temp_important.Timestamp[mask].str.split('.').str[0].values)
			temp_important.drop(columns=['Timestamp'], inplace=True)
			temp_important.rename(columns={'NewTimestamp':'Timestamp', 'TEMP':'TEMP'}, inplace=True)
			temp_important_reordered = temp_important.loc[:, ['Timestamp', 'TEMP']]# [1, 0]] # ['Timestamp', 'TEMP']]
			temp_important_reordered.dropna(inplace=True)
			individual_temp_data.append(temp_important_reordered)

		# Load labels
		important_times = load_log_file_and_extract_relevant_times(BASE_PATH + LOG_STORAGE_PATH)
		experiment_start, experiment_end = (
			parse_relative_timestamp(important_times[0][0]),
			parse_relative_timestamp(important_times[-1][1])
		)
		print('EXPERIMENT_START AT: %s' % experiment_start)
		print('EXPERIMENT_END AT: %s' % experiment_end)

		# Trim data to handle only 'during the experiment recorded' data
		for idx, data in enumerate(individual_eeg_data):
			individual_eeg_data[idx] = trim_data_to_length_of_experiment(data, experiment_start, experiment_end) #, starting_times[idx]) # store properly trimmed data
			print('For this in individual_eeg_data[idx] has len %d' % len(individual_eeg_data[idx]))
		for idx, data in enumerate(individual_bvp_data):
			individual_bvp_data[idx] = trim_data_to_length_of_experiment(data, experiment_start, experiment_end) #, starting_times[idx]) # store properly trimmed data
			print('For this in individual_bvp_data[idx] has len %d' % len(individual_bvp_data[idx]))
		for idx, data in enumerate(individual_gsr_data):
			individual_gsr_data[idx] = trim_data_to_length_of_experiment(data, experiment_start, experiment_end) #, starting_times[idx]) # store properly trimmed data
			print('For this in individual_gsr_data[idx] has len %d' % len(individual_gsr_data[idx]))
		for idx, data in enumerate(individual_temp_data):
			individual_temp_data[idx] = trim_data_to_length_of_experiment(data, experiment_start, experiment_end) #, starting_times[idx]) # store properly trimmed data
			print('For this in individual_temp_data[idx] has len %d' % len(individual_temp_data[idx]))

		# Extract windows
		eeg_assumed_srate = 256
		bvp_assumed_srate = 64
		gsr_assumed_srate = 4
		temp_assumed_srate = 4

		eeg_windows = []
		bvp_windows = []
		gsr_windows = []
		temp_windows = []

		num_folders = len(EEG_STORAGE_PATH)

		common_starts = [
			min(individual_eeg_data[idx]['Timestamp'].iloc[0], individual_bvp_data[idx]['Timestamp'].iloc[0], individual_gsr_data[idx]['Timestamp'].iloc[0], individual_temp_data[idx]['Timestamp'].iloc[0])
			for idx in range(num_folders)
		]

		for idx, eeg in enumerate(individual_eeg_data):
			individual_eeg_data[idx] = eeg[eeg.Timestamp >= common_starts[idx]]
		for idx, bvp in enumerate(individual_bvp_data):
			individual_bvp_data[idx] = bvp[bvp.Timestamp >= common_starts[idx]]
		for idx, gsr in enumerate(individual_gsr_data):
			individual_gsr_data[idx] = gsr[gsr.Timestamp >= common_starts[idx]]
		for idx, temp in enumerate(individual_temp_data):
			individual_temp_data[idx] = temp[temp.Timestamp >= common_starts[idx]]

		for eeg in individual_eeg_data:
			for win in extract_valid_windows_for_duration_of_n_seconds(eeg, eeg_assumed_srate, EPOCH_DURATION_IN_SECONDS):
				eeg_windows.append(win) 
		for bvp in individual_bvp_data:	
			for win in extract_valid_windows_for_duration_of_n_seconds(bvp, bvp_assumed_srate, EPOCH_DURATION_IN_SECONDS):
				bvp_windows.append(win) 
		for gsr in individual_gsr_data:	
			for win in extract_valid_windows_for_duration_of_n_seconds(gsr, gsr_assumed_srate, EPOCH_DURATION_IN_SECONDS):
				gsr_windows.append(win) 
		for temp in individual_temp_data:	
			for win in extract_valid_windows_for_duration_of_n_seconds(temp, temp_assumed_srate, EPOCH_DURATION_IN_SECONDS):
				temp_windows.append(win) 

		# Assign labels to the windows, store as dict of windows for each label key
		pid = PARTICIPANT_DIR.split('_')[0]
		session_idx = '_S2' if '2nd_session' in PARTICIPANT_DIR else '_S1'
		labels_to_list_of_windows_dict, start_times_to_data_dict = combine_labels_and_data(eeg_windows, bvp_windows, gsr_windows, temp_windows, important_times, pid, session_idx)

		common_starts = [
			min(individual_eeg_data[idx]['Timestamp'].iloc[0], individual_bvp_data[idx]['Timestamp'].iloc[0], individual_gsr_data[idx]['Timestamp'].iloc[0], individual_temp_data[idx]['Timestamp'].iloc[0])
			for idx in range(num_folders)
		]

		for idx, eeg in enumerate(individual_eeg_data):
			individual_eeg_data[idx] = eeg[eeg.Timestamp >= common_starts[idx]]
		for idx, bvp in enumerate(individual_bvp_data):
			individual_bvp_data[idx] = bvp[bvp.Timestamp >= common_starts[idx]]
		for idx, gsr in enumerate(individual_gsr_data):
			individual_gsr_data[idx] = gsr[gsr.Timestamp >= common_starts[idx]]
		for idx, temp in enumerate(individual_temp_data):
			individual_temp_data[idx] = temp[temp.Timestamp >= common_starts[idx]]

		for eeg in individual_eeg_data:
			for win in extract_valid_windows_for_duration_of_n_seconds(eeg, eeg_assumed_srate, EPOCH_DURATION_IN_SECONDS):
				eeg_windows.append(win) 
		for bvp in individual_bvp_data:	
			for win in extract_valid_windows_for_duration_of_n_seconds(bvp, bvp_assumed_srate, EPOCH_DURATION_IN_SECONDS):
				bvp_windows.append(win) 
		for gsr in individual_gsr_data:	
			for win in extract_valid_windows_for_duration_of_n_seconds(gsr, gsr_assumed_srate, EPOCH_DURATION_IN_SECONDS):
				gsr_windows.append(win) 
		for temp in individual_temp_data:	
			for win in extract_valid_windows_for_duration_of_n_seconds(temp, temp_assumed_srate, EPOCH_DURATION_IN_SECONDS):
				temp_windows.append(win) 

		# Assign labels to the windows, store as dict of windows for each label key
		pid = PARTICIPANT_DIR.split('_')[0]
		session_idx = '_S2' if '2nd_session' in PARTICIPANT_DIR else '_S1'
		labels_to_list_of_windows_dict, start_times_to_data_dict = combine_labels_and_data(eeg_windows, bvp_windows, gsr_windows, temp_windows, important_times, pid, session_idx)

		# Extract all the features for each window
		labels_to_list_of_features_dict = extract_features_for_windows(labels_to_list_of_windows_dict)
		labels_to_list_of_features_with_all_labels_dict = extract_features_for_windows_with_all_data(start_times_to_data_dict)

		# Here there's the ML
		csv_results_header = ['CV_Run', 'Best_Estimator', 'Best_Params', 'Heldout_Test_Acc']

		X, y = combine_labels_and_data_to_same_length_collections(labels_to_list_of_features_dict)
		xy_list = [X, y]

		session_mod = '_S2' if '2nd_session' in PARTICIPANT_DIR else '_S1'

		FEATURES_STORAGE_PATH_THIS_PARTICIPANT = '/home/christoph/Desktop/Workflow/Work_2025/E-Learning_Paper_CAEAI/data_to_anonymize_for_upload/Anonymized_E_Learning_data/features_and_labels_pckls/' + PARTICIPANT_DIR.split('_')[0] + session_mod + '.pkl'

		with open(FEATURES_STORAGE_PATH_THIS_PARTICIPANT, 'wb') as handle:
			pickle.dump(xy_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

		X, y = combine_labels_and_data_to_same_length_collections_for_all_data(labels_to_list_of_features_with_all_labels_dict)
		xy_list = [X, y]

		FEATURES_STORAGE_PATH_THIS_PARTICIPANT_WITH_ALL_INFO = '/home/christoph/Desktop/Workflow/Work_2025/E-Learning_Paper_CAEAI/data_to_anonymize_for_upload/Anonymized_E_Learning_data/features_and_labels_pckls/' + PARTICIPANT_DIR.split('_')[0] + session_mod + '_with_all_info.pkl'

		with open(FEATURES_STORAGE_PATH_THIS_PARTICIPANT_WITH_ALL_INFO, 'wb') as handle:
			pickle.dump(xy_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

		regression_features_extracted = []
		regression_labels_extracted = []

		latest_full_ydat = None
		temp_x_dat_storage = []
		temp_y_dat = None

		for idx, y_data in enumerate(y):
			if (temp_y_dat == None) or (temp_y_dat == y_data.split('/')[0]):
				temp_y_dat = y_data.split('/')[0]
				temp_x_dat_storage.append(X[idx])
				latest_full_ydat = y_data
			else:
				temp_x_dat_storage_np = np.asarray(temp_x_dat_storage)
				features_reduced = []
				for i in range(temp_x_dat_storage_np.shape[1]):
					features_reduced.append(np.min(temp_x_dat_storage_np[:,i]))
					features_reduced.append(np.median(temp_x_dat_storage_np[:,i]))
					features_reduced.append(np.max(temp_x_dat_storage_np[:,i]))
					features_reduced.append(np.std(temp_x_dat_storage_np[:,i]))
				regression_features_extracted.append(features_reduced)
				regression_labels_extracted.append(latest_full_ydat)
				# now, start for the next one
				temp_y_dat = y_data.split('/')[0]
				temp_x_dat_storage = []
				temp_x_dat_storage.append(X[idx])
				latest_full_ydat = y_data
		# obviously, don't forget the final one:
		temp_x_dat_storage_np = np.asarray(temp_x_dat_storage)
		features_reduced = []
		for i in range(temp_x_dat_storage_np.shape[1]):
			features_reduced.append(np.min(temp_x_dat_storage_np[:,i]))
			features_reduced.append(np.median(temp_x_dat_storage_np[:,i]))
			features_reduced.append(np.max(temp_x_dat_storage_np[:,i]))
			features_reduced.append(np.std(temp_x_dat_storage_np[:,i]))
		regression_features_extracted.append(features_reduced)
		regression_labels_extracted.append(latest_full_ydat)

		FEATURES_STORAGE_PATH_THIS_PARTICIPANT_WITH_ALL_INFO_REDUCED_FEATURE_SETS = '/home/christoph/Desktop/Workflow/Work_2025/E-Learning_Paper_CAEAI/data_to_anonymize_for_upload/Anonymized_E_Learning_data/features_and_labels_pckls/' + PARTICIPANT_DIR.split('_')[0] + session_mod + '_with_all_info_reduced_feature_sets.pkl'

		with open(FEATURES_STORAGE_PATH_THIS_PARTICIPANT_WITH_ALL_INFO_REDUCED_FEATURE_SETS, 'wb') as handle:
			pickle.dump([regression_features_extracted, regression_labels_extracted], handle, protocol=pickle.HIGHEST_PROTOCOL)