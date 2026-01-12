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


# Just examples. Needs adjustment!
BASE_PATH = 'TODO_CHANGE_TO_YOUR_STORAGE_PATH'
LOG_STORAGE_PATH = 'EXAMPLE_LOG_FILE.LOG'
EEG_STORAGE_PATH = ['0/RawData/MuseS-4626_EEG_anonymized.csv', '1/RawData/MuseS-4626_EEG_anonymized.csv']
BVP_STORAGE_PATH = ['0/RawData/8413C6_BVP_anonymized.csv', '1/RawData/8413C6_BVP_anonymized.csv']
GSR_STORAGE_PATH = ['0/RawData/8413C6_GSR_anonymized.csv', '1/RawData/8413C6_GSR_anonymized.csv']
TEMP_STORAGE_PATH = ['0/RawData/8413C6_TEMP_anonymized.csv', '1/RawData/8413C6_TEMP_anonymized.csv']
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
		if datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') >= experiment_start: # .%f
			print('OVERALL, %d ROWS IN THE BEGINNING WERE WRITTEN BEFORE THE START OF THE FIRST TASK!' % start_rows_skipped_ctr)
			break
		else:
			start_rows_skipped_ctr += 1

	end_rows_skipped_ctr = 0
	for timestamp in data.iloc[::-1]['Timestamp']:
		if datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') <= experiment_end: # .%f
			print('OVERALL, %d ROWS AT THE END WERE WRITTEN AFTER FINISHING THE LAST TASK!' % end_rows_skipped_ctr)
			break
		else:
			end_rows_skipped_ctr += 1

	data['Timestamp'] = pd.to_datetime(data['Timestamp'])

	return data[data['Timestamp'].between(experiment_start, experiment_end)]


def roughly_same_starting_time(win_a, win_b):
	if (win_a is None) or (win_b is None) or win_a.empty or win_b.empty:
		return False
	win_a_start = win_a.iloc[0]
	win_a_minus_ten_seconds = win_a_start - datetime.timedelta(seconds=10)
	win_a_plus_ten_seconds = win_a_start + datetime.timedelta(seconds=10)
	win_b_time = win_b.iloc[0]
	return win_a_minus_ten_seconds <= win_b_time <= win_a_plus_ten_seconds


def get_label_for_timestamp(timestamp_str, labels):
	timestamp = timestamp_str.iloc[0]
	for idx, important_time in enumerate(labels):
		start_time_this_task = important_time[0]
		end_time_this_task = important_time[1]
		if start_time_this_task <= timestamp <= end_time_this_task:
			return LABEL_NUM_TO_LABEL_DICT[important_time[-1]]
	return 'UNKNOWN'


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


def combine_labels_and_data(eeg_data, bvp_data, gsr_data, temp_data, labels):
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
			labels_to_data_dict[get_label_for_timestamp(win['Timestamp'], labels)].append([eeg_data[idx], bvp_data[idx], gsr_data[idx], temp_data[idx]])
	else:
		min_len = min(len(eeg_data), len(bvp_data), len(gsr_data), len(temp_data))
		eeg_ctr = 0
		ctr = 0
		while (ctr < min_len) and (eeg_ctr < len(eeg_data)):
			if len(np.unique(eeg_data[eeg_ctr]['AF7'].to_numpy())) <= 3:
				eeg_ctr+=1
				continue
			labels_to_data_dict[get_label_for_timestamp(bvp_data[ctr]['Timestamp'], labels)].append([eeg_data[eeg_ctr], bvp_data[ctr], gsr_data[ctr], temp_data[ctr]])
			ctr += 1
	return labels_to_data_dict


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
	notchd_window = window 

	mean_channel = np.expand_dims(np.mean(np.asarray([notchd_window['AF7'], notchd_window['AF8'], notchd_window['TP9'], notchd_window['TP10']]), axis=0), axis=-1)
	prefrontal_mean_eeg = np.expand_dims(np.mean(np.asarray([notchd_window['AF7'], notchd_window['AF8']]), axis=0), axis=-1)
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


def merge_feature_tuples_to_1d_vector(eeg_features, bvp_features, gsr_features, temp_features):
	print('Still to implement')


def combine_labels_and_data_to_same_length_collections(labels_to_list_of_features_dict):
	label_list = []
	tuple_list = []
	# for each key, value run through to check the key and then assign respective label to it and store it in the label_list and tuple_list tmp_to_return
	for key, value in labels_to_list_of_features_dict.items():
		for dat in value:
			tuple_list.append(np.asarray(dat))
			label_list.append(LABEL_FOR_FLAGGING_DICT[key])
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
						current_task_end = datetime.datetime.strptime(log_line.split('INFO:logging_utilities:')[1].split(': Routine')[0], '%Y-%m-%d %H:%M:%S.%f')
						cl_level = round(float(log_line.split('with response ')[1]))
						# print('THIS TASK STARTED AT %s and ended at %s and the participant HAD A PERFORMANCE OF %.2f%%!' % (current_task_start, current_task_end, this_task_performance))
						# print('THIS TASKS COGNITIVE LOAD LEVEL WAS: %d' % cl_level)
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
					current_task_start = datetime.datetime.strptime(log_line.split('INFO:logging_utilities:')[1].split(': Routine')[0], '%Y-%m-%d %H:%M:%S.%f')

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


if __name__ == '__main__':
	# Load data
	individual_eeg_data = []
	individual_bvp_data = []
	individual_gsr_data = []
	individual_temp_data = []

	for eeg_storage_path in EEG_STORAGE_PATH:
		eeg = pd.read_csv(BASE_PATH + eeg_storage_path, names=['Timestamp', 'AF7', 'AF8', 'TP9', 'TP10', 'FpZ'], dtype=str)
		eeg_important = eeg.loc[:, ['Timestamp', 'AF7', 'AF8', 'TP9', 'TP10']]
		mask = eeg_important.Timestamp.str.contains('.')
		eeg_important = eeg_important.assign(NewTimestamp=eeg_important.Timestamp[mask].str.split('.', 0).str[0].values)
		eeg_important.drop(columns=['Timestamp'], inplace=True)
		eeg_important.rename(columns={'NewTimestamp':'Timestamp', 'AF7':'AF7', 'AF8':'AF8', 'TP9':'TP9', 'TP10':'TP10'}, inplace=True)
		eeg_important_reordered = eeg_important.loc[:, ['Timestamp', 'AF7', 'AF8', 'TP9', 'TP10']]# [5, 0, 1, 2, 3, 4]] #['Timestamp', 'AF7', 'AF8', 'TP9', 'TP10', 'FpZ']]
		eeg_important_reordered.dropna(inplace=True)
		individual_eeg_data.append(eeg_important_reordered)

	for bvp_storage_path in BVP_STORAGE_PATH:
		bvp = pd.read_csv(BASE_PATH + bvp_storage_path, names=['Timestamp', 'BVP'], dtype=str)
		bvp_important = bvp.loc[:, ['Timestamp', 'BVP']]
		mask = bvp_important.Timestamp.str.contains('.')
		bvp_important = bvp_important.assign(NewTimestamp=bvp_important.Timestamp[mask].str.split('.', 0).str[0].values)
		bvp_important.drop(columns=['Timestamp'], inplace=True)
		bvp_important.rename(columns={'NewTimestamp':'Timestamp', 'BVP':'BVP'}, inplace=True)
		bvp_important_reordered = bvp_important.loc[:, ['Timestamp', 'BVP']]# [1, 0]] # ['Timestamp', 'BVP']]
		bvp_important_reordered.dropna(inplace=True)
		individual_bvp_data.append(bvp_important_reordered)

	for gsr_storage_path in GSR_STORAGE_PATH:
		gsr = pd.read_csv(BASE_PATH + gsr_storage_path, names=['Timestamp', 'GSR'], dtype=str)
		gsr_important = gsr.loc[:, ['Timestamp', 'GSR']]
		mask = gsr_important.Timestamp.str.contains('.')
		gsr_important = gsr_important.assign(NewTimestamp=gsr_important.Timestamp[mask].str.split('.', 0).str[0].values)
		gsr_important.drop(columns=['Timestamp'], inplace=True)
		gsr_important.rename(columns={'NewTimestamp':'Timestamp', 'GSR':'GSR'}, inplace=True)
		gsr_important_reordered = gsr_important.loc[:, ['Timestamp', 'GSR']]# [1, 0]] # ['Timestamp', 'GSR']]
		gsr_important_reordered.dropna(inplace=True)
		individual_gsr_data.append(gsr_important_reordered)

	for temp_storage_path in TEMP_STORAGE_PATH:
		temp = pd.read_csv(BASE_PATH + temp_storage_path, names=['Timestamp', 'TEMP'], dtype=str)
		temp_important = temp.loc[:, ['Timestamp', 'TEMP']]
		mask = temp_important.Timestamp.str.contains('.')
		temp_important = temp_important.assign(NewTimestamp=temp_important.Timestamp[mask].str.split('.', 0).str[0].values)
		temp_important.drop(columns=['Timestamp'], inplace=True)
		temp_important.rename(columns={'NewTimestamp':'Timestamp', 'TEMP':'TEMP'}, inplace=True)
		temp_important_reordered = temp_important.loc[:, ['Timestamp', 'TEMP']]# [1, 0]] # ['Timestamp', 'TEMP']]
		temp_important_reordered.dropna(inplace=True)
		individual_temp_data.append(temp_important_reordered)

	# Load labels
	important_times = load_log_file_and_extract_relevant_times(BASE_PATH + LOG_STORAGE_PATH)
	experiment_start, experiment_end = important_times[0][0], important_times[-1][1]
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

	common_starts = [
		max(individual_eeg_data[0]['Timestamp'].iloc[0], individual_bvp_data[0]['Timestamp'].iloc[0], individual_gsr_data[0]['Timestamp'].iloc[0], individual_temp_data[0]['Timestamp'].iloc[0]),
		max(individual_eeg_data[1]['Timestamp'].iloc[0], individual_bvp_data[1]['Timestamp'].iloc[0], individual_gsr_data[1]['Timestamp'].iloc[0], individual_temp_data[1]['Timestamp'].iloc[0]),
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
	labels_to_list_of_windows_dict = combine_labels_and_data(eeg_windows, bvp_windows, gsr_windows, temp_windows, important_times)

	# Extract all the features for each window
	labels_to_list_of_features_dict = extract_features_for_windows(labels_to_list_of_windows_dict)

	# Here there's the ML
	csv_results_header = ['CV_Run', 'Best_Estimator', 'Best_Params', 'Heldout_Test_Acc']

	X, y = combine_labels_and_data_to_same_length_collections(labels_to_list_of_features_dict)

	np.random.seed(24)
	np.random.shuffle(X)
	np.random.shuffle(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	print('Len of X_train and X_test: %d to %d' % (len(X_train), len(X_test)))

	f1 = make_scorer(f1_score, average='weighted')

	parameter_grid_lr = [
		{'solver': ['lbfgs'], 'penalty': ['l2', None]},
		{'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
		{'solver': ['sag'], 'penalty': ['l2', None]},
		{'solver': ['saga'], 'penalty': ['l1', 'l2', None]}
	]

	parameter_grid_dt = [
		{'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': np.concatenate((np.arange(5, 305, 5), np.asarray([None])), axis=0)}
	]

	parameter_grid_svm = [
		{'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	]

	parameter_grid_nn = [
		{'leaf_size': list(range(1,50)), 'n_neighbors': list(range(1,30)), 'p': [1, 2]}
	]

	parameter_grid_mlp = [
		{'activation': ['logistic', 'tanh', 'relu'], 'hidden_layer_sizes': [(3,), (10,), (30,), (50,)]}
	]

	model_cv_definitions = [
		[parameter_grid_nn, 'NN-LABELS'],
		[parameter_grid_svm, 'SVM-LABELS'],
		[parameter_grid_dt, 'DT-LABELS'],
		[parameter_grid_lr, 'LR-LABELS'],
		[parameter_grid_mlp, 'MLP-LABELS'],
	]

	results_per_model = []
	n_jobs = 4

	for j, (grid, abbrev) in enumerate(model_cv_definitions):
		skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
		models_skf = []

		if abbrev == 'LR-LABELS':
			classifier = LogisticRegression(random_state=42, max_iter=10000)
		elif abbrev == 'DT-LABELS':
			classifier = DecisionTreeClassifier()
		elif abbrev == 'SVM-LABELS':
			classifier = LinearSVC(dual=False, max_iter=10000)
		elif abbrev == 'NN-LABELS':
			classifier = KNeighborsClassifier()
		elif abbrev == 'MLP-LABELS':
			classifier = MLPClassifier(random_state=42, max_iter=10000, early_stopping=True)

		gridsearch_cv = GridSearchCV(classifier, param_grid=grid, n_jobs=n_jobs, scoring=f1).fit(X=X_train, y=y_train)
		print('CV done! Best estimator: %s and best score: %f and best params: %s' % (gridsearch_cv.best_estimator_, gridsearch_cv.best_score_, gridsearch_cv.best_params_))
		optimized_clf_score = gridsearch_cv.best_estimator_.score(X=X_test, y=y_test)
		best_predictor_predictions = gridsearch_cv.best_estimator_.predict(X_test)
		best_predictor_f1_score = f1_score(y_test, best_predictor_predictions, average='weighted')
		print('Score: %f and F1: %f' % (optimized_clf_score, best_predictor_f1_score))

		results_per_model.append([j, [gridsearch_cv.best_estimator_, gridsearch_cv.best_params_, optimized_clf_score, best_predictor_f1_score, y_test, best_predictor_predictions], abbrev])

	for (run, cv_results, abbrev) in results_per_model:
		mean_f1 = 0
		mean_value = 0
		mean_value_ctr = 0
		best_model_score_over_runs = 0
		best_model_params = None
		best_model, best_params_of_best_model, best_model_score_this_run, f1, _, _ = cv_results
		mean_f1 += f1
		mean_value += best_model_score_this_run
		mean_value_ctr += 1
		if best_model_score_this_run >= best_model_score_over_runs:
			best_model_score_over_runs = best_model_score_this_run
			best_model_params = best_params_of_best_model
			# save best model
			# joblib.dump(best_model, "%s.pkl" % abbrev)
		print('%s: F1: %f, %s,' % (abbrev, f1, best_params_of_best_model))
		print('RESULTS: For %s the mean nested-cv score is: %f and mean nested-F1 is: %f' % (abbrev, mean_value / mean_value_ctr, mean_f1 / mean_value_ctr))
		print('RESULTS: Best Score of %f achieved for %s with params %s and saved as %s.pkl' % (best_model_score_over_runs, abbrev, best_model_params, abbrev))