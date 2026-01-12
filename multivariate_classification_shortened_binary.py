import os
import joblib
import pickle
import datetime
import itertools

import pandas as pd
import numpy as np
from numpy import inf

from scipy.signal import iirnotch, filtfilt
from scipy.signal import find_peaks

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, BayesianRidge, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, classification_report, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, LeaveOneGroupOut, StratifiedKFold


BASE_PATH = 'TODO_CHANGE_TO_YOUR_STORAGE_PATH'

participants = ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010']

multivariate_regression_data = {
	'P001_S1':[],
	'P001_S2':[],
	'P002_S1':[],
	'P002_S2':[],
	'P003_S1':[],
	'P003_S2':[],
	'P004_S1':[],
	'P004_S2':[],
	'P005_S1':[],
	'P005_S2':[],
	'P006_S1':[],
	'P006_S2':[],
	'P007_S1':[],
	'P007_S2':[],
	'P008_S1':[],
	'P008_S2':[],
	'P009_S1':[],
	'P009_S2':[],
	'P010_S1':[],
	'P010_S2':[],
}

multivariate_regression_labels = {
	'P001_S1':[],
	'P001_S2':[],
	'P002_S1':[],
	'P002_S2':[],
	'P003_S1':[],
	'P003_S2':[],
	'P004_S1':[],
	'P004_S2':[],
	'P005_S1':[],
	'P005_S2':[],
	'P006_S1':[],
	'P006_S2':[],
	'P007_S1':[],
	'P007_S2':[],
	'P008_S1':[],
	'P008_S2':[],
	'P009_S1':[],
	'P009_S2':[],
	'P010_S1':[],
	'P010_S2':[],	
}

reduced_feature_set_to_location_dir = {
	'EngagementIndex': [0, 1, 2, 3],
	'AsymmetryIndexPrefrontal': [16, 17, 18, 19],
	'HR': [64, 65, 66, 67],
	'GSR': [68, 69, 70, 71],
}

class_labels = ['low', 'high']

for file in os.listdir(BASE_PATH):
	if 'with_all_info_reduced_feature_sets' in file:
		with open(BASE_PATH + file, 'rb') as handle:
			labels_and_features = pickle.load(handle)
			multivariate_regression_data[file.split('_with')[0]] = labels_and_features[0]
			multivariate_regression_labels[file.split('_with')[0]] = labels_and_features[1]


def build_data(selected_participants, return_raw_scores=True, bin_edges=None, labels=None):
	X, y, groups = [], [], []
	for pid in selected_participants:
		for session in ['S1', 'S2']:
			key = f"{pid}_{session}"
			X.append(multivariate_regression_data[key])
			y_session = [float(label.split('/')[2]) for label in multivariate_regression_labels[key]]
			y.append(y_session)
			groups.extend([pid] * len(y_session))  # 4 tasks per session

	X = np.vstack(X)
	y = np.hstack(y)
	groups = np.array(groups)

	if not return_raw_scores:
		# Apply precomputed bin edges and labels for classification
		ys_to_return = []
		for y_to_return in y:
			if y_to_return <= 45:
				ys_to_return.append(class_labels[0])
			else:
				ys_to_return.append(class_labels[1])
		y = np.asarray(ys_to_return)

	return X, y, groups


rnd_state = 42

param_grid_mlp = {
	'model__hidden_layer_sizes': [(50,), (100,), (50, 25)],
	'model__alpha': [0.0001, 0.001, 0.01],
	'model__activation': ['relu', 'tanh'],
	'model__solver': ['adam'],
	'model__early_stopping': [True]
}

param_grid_nb = {}  # No hyperparameters to tune for GaussianNB

param_grid_gbc = {
	'model__n_estimators': [50, 100, 200],
	'model__learning_rate': [0.01, 0.1],
	'model__max_depth': [2, 3, 5],
	'model__subsample': [0.8, 1.0],
	'model__min_samples_split': [2, 5]
}

param_grid_rf = {
	'model__n_estimators': [50, 100, 200],
	'model__max_depth': [3, 5, None],
	'model__min_samples_split': [2, 5],
	'model__min_samples_leaf': [1, 2],
	'model__max_features': ['sqrt', 'log2', 0.5],
	'model__class_weight': [None, 'balanced']
}

param_grid_dt = {
	'model__max_depth': [2, 3, 5, None],
	'model__min_samples_split': [2, 5, 10],
	'model__min_samples_leaf': [1, 2, 5],
	'model__criterion': ['gini', 'entropy'],
	'model__class_weight': [None, 'balanced']
}

param_grid_knn = {
	'model__n_neighbors': [1, 3, 5, 7],
	'model__weights': ['uniform', 'distance'],
	'model__metric': ['euclidean', 'manhattan']
}

param_grid_svc = {
	'model__kernel': ['linear', 'rbf'],
	'model__C': [0.1, 1, 10, 100],
	'model__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Only used with RBF
	'model__class_weight': [None, 'balanced'],
	'model__decision_function_shape': ['ovr', 'ovo']
}

param_grid_logreg = {
	'model__penalty': ['l2'],  # 'l1' if using 'liblinear' solver
	'model__C': [0.01, 0.1, 1, 10],
	'model__solver': ['lbfgs'],  # 'liblinear' for small data or if using 'l1'
	'model__class_weight': [None, 'balanced']
}

results = []

# Loop: every participant becomes the test participant once
for test_pid in participants:
	remaining = [p for p in participants if p != test_pid]

	# Try all combinations of 2 validation participants from the remaining 9
	val_combinations = list(itertools.combinations(remaining, 2))

	for val_pids in val_combinations:
		train_pids = [p for p in remaining if p not in val_pids]

		# Build training data (raw scores first)
		X_train, y_train_raw, groups_train = build_data(train_pids, return_raw_scores=True)

		# Compute bin edges from y_train only
		bin_counts, bin_edges = 3, [0.0, ]

		# Build datasets
		X_train, y_train, groups_train = build_data(train_pids, return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)
		X_val, y_val, _ = build_data(val_pids, return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)
		X_test, y_test, _ = build_data([test_pid], return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)

		le = LabelEncoder()
		y_train = le.fit_transform(y_train)
		y_val = le.transform(y_val)
		y_test = le.transform(y_test)

		for model_name, model, param_grid in [
			('mlp', MLPClassifier(), param_grid_mlp),
			('nb', GaussianNB(), param_grid_nb),
			('gbc', GradientBoostingClassifier(), param_grid_gbc),
			('rf', RandomForestClassifier(), param_grid_rf),
			('dt', DecisionTreeClassifier(), param_grid_dt),
			('knn', KNeighborsClassifier(), param_grid_knn),
			('svc', SVC(), param_grid_svc),
			('logreg', LogisticRegression(), param_grid_logreg),
		]:
			print('Running Multivariate Regression LOO-CV for TRAIN-EVAL-TEST: [%s, %s, %s] and model: %s' % (train_pids, val_pids, test_pid, model_name))

			pipeline = Pipeline([
				('scaler', StandardScaler()),
				# ('pca', PCA(n_components=5)), Without PCA, hence shortened
				('model', model)
			])

			# Train model with GridSearchCV
			grid = GridSearchCV(
				estimator=pipeline,
				param_grid=param_grid,
				cv=LeaveOneGroupOut().split(X_train, y_train, groups_train),
				scoring='accuracy',
				n_jobs=-1,
				verbose=0
			)
			grid.fit(X_train, y_train)

			best_model = grid.best_estimator_

			val_preds = best_model.predict(X_val)
			val_acc = accuracy_score(y_val, val_preds)
			val_f1 = f1_score(y_val, val_preds, average='weighted')

			# Retrain on train + val for final test
			X_trainval = np.vstack([X_train, X_val])
			y_trainval = np.hstack([y_train, y_val])
			final_model = grid.best_estimator_
			final_model.fit(X_trainval, y_trainval)
			test_preds = final_model.predict(X_test)
			test_acc = accuracy_score(y_test, test_preds)
			test_f1 = f1_score(y_test, test_preds, average='weighted')

			results.append({
				'model': model_name,
				'test_participant': test_pid,
				'val_participants': val_pids,
				'train_participants': train_pids,
				'best_params': grid.best_params_,
				'val_acc': val_acc,
				'val_f1': val_f1,
				'test_acc': test_acc,
				'test_f1': test_f1,
			})

			print('Results: test_acc of %f and test_f1 of %f' % (test_acc, test_f1))


import pandas as pd
df = pd.DataFrame(results)

print("Average test ACC:", df['test_acc'].mean())
print("Average test F1:", df['test_f1'].mean())

# Group by test participant
print(df.groupby('test_participant')[['val_acc', 'val_f1', 'test_acc', 'test_f1']].mean())
print(df.groupby('model')[['val_acc', 'val_f1', 'test_acc', 'test_f1']].mean())

df.to_csv(BASE_PATH + 'standard_scaler_results_with_ridge_for_classification_without_pca_binary.csv')

results = []

# Loop: every participant becomes the test participant once
for test_pid in participants:
	remaining = [p for p in participants if p != test_pid]

	# Try all combinations of 2 validation participants from the remaining 9
	val_combinations = list(itertools.combinations(remaining, 2))

	for val_pids in val_combinations:
		train_pids = [p for p in remaining if p not in val_pids]

		# Build training data (raw scores first)
		X_train, y_train_raw, groups_train = build_data(train_pids, return_raw_scores=True)

		# Compute bin edges from y_train only
		bin_counts, bin_edges = 3, [0.0, ]
		class_labels = ['low', 'medium', 'high']

		# Build datasets
		X_train, y_train, groups_train = build_data(train_pids, return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)
		X_val, y_val, _ = build_data(val_pids, return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)
		X_test, y_test, _ = build_data([test_pid], return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)

		le = LabelEncoder()
		y_train = le.fit_transform(y_train)
		y_val = le.transform(y_val)
		y_test = le.transform(y_test)

		for model_name, model, param_grid in [
			('mlp', MLPClassifier(max_iter=1000), param_grid_mlp),
			('nb', GaussianNB(), param_grid_nb),
			('gbc', GradientBoostingClassifier(), param_grid_gbc),
			('rf', RandomForestClassifier(), param_grid_rf),
			('dt', DecisionTreeClassifier(), param_grid_dt),
			('knn', KNeighborsClassifier(), param_grid_knn),
			('svc', SVC(), param_grid_svc),
			('logreg', LogisticRegression(max_iter=1000), param_grid_logreg),
		]:
			print('Running Multivariate Regression LOO-CV for TRAIN-EVAL-TEST: [%s, %s, %s] and model: %s' % (train_pids, val_pids, test_pid, model_name))

			pipeline = Pipeline([
				('scaler', MinMaxScaler()),
				# ('pca', PCA(n_components=5)), Without PCA, hence shortened
				('model', model)
			])

			# Train model with GridSearchCV
			grid = GridSearchCV(
				estimator=pipeline,
				param_grid=param_grid,
				cv=LeaveOneGroupOut().split(X_train, y_train, groups_train),
				scoring='accuracy',
				n_jobs=-1,
				verbose=0
			)
			grid.fit(X_train, y_train)
			best_model = grid.best_estimator_

			val_preds = best_model.predict(X_val)
			val_acc = accuracy_score(y_val, val_preds)
			val_f1 = f1_score(y_val, val_preds, average='weighted')

			# Retrain on train + val for final test
			X_trainval = np.vstack([X_train, X_val])
			y_trainval = np.hstack([y_train, y_val])
			final_model = grid.best_estimator_
			final_model.fit(X_trainval, y_trainval)
			test_preds = final_model.predict(X_test)
			test_acc = accuracy_score(y_test, test_preds)
			test_f1 = f1_score(y_test, test_preds, average='weighted')

			results.append({
				'model': model_name,
				'test_participant': test_pid,
				'val_participants': val_pids,
				'train_participants': train_pids,
				'best_params': grid.best_params_,
				'val_acc': val_acc,
				'val_f1': val_f1,
				'test_acc': test_acc,
				'test_f1': test_f1,
			})

			print('Results: test_acc of %f and test_f1 of %f' % (test_acc, test_f1))


import pandas as pd
df = pd.DataFrame(results)

print("Average test ACC:", df['test_acc'].mean())
print("Average test F1:", df['test_f1'].mean())

# Group by test participant
print(df.groupby('test_participant')[['val_acc', 'val_f1', 'test_acc', 'test_f1']].mean())
print(df.groupby('model')[['val_acc', 'val_f1', 'test_acc', 'test_f1']].mean())

df.to_csv(BASE_PATH + 'min_max_scaler_results_with_ridge_for_classification_without_pca_binary.csv')

results = []

# Loop: every participant becomes the test participant once
for test_pid in participants:
	remaining = [p for p in participants if p != test_pid]

	# Try all combinations of 2 validation participants from the remaining 9
	val_combinations = list(itertools.combinations(remaining, 2))

	for val_pids in val_combinations:
		train_pids = [p for p in remaining if p not in val_pids]

		# Build training data (raw scores first)
		X_train, y_train_raw, groups_train = build_data(train_pids, return_raw_scores=True)

		# Compute bin edges from y_train only
		bin_counts, bin_edges = 3, [0.0, ]

		# Build datasets
		X_train, y_train, groups_train = build_data(train_pids, return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)
		X_val, y_val, _ = build_data(val_pids, return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)
		X_test, y_test, _ = build_data([test_pid], return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)

		le = LabelEncoder()
		y_train = le.fit_transform(y_train)
		y_val = le.transform(y_val)
		y_test = le.transform(y_test)

		for model_name, model, param_grid in [
			('mlp', MLPClassifier(), param_grid_mlp),
			('nb', GaussianNB(), param_grid_nb),
			('gbc', GradientBoostingClassifier(), param_grid_gbc),
			('rf', RandomForestClassifier(), param_grid_rf),
			('dt', DecisionTreeClassifier(), param_grid_dt),
			('knn', KNeighborsClassifier(), param_grid_knn),
			('svc', SVC(), param_grid_svc),
			('logreg', LogisticRegression(), param_grid_logreg),
		]:
			print('Running Multivariate Regression LOO-CV for TRAIN-EVAL-TEST: [%s, %s, %s] and model: %s' % (train_pids, val_pids, test_pid, model_name))

			pipeline = Pipeline([
				('scaler', StandardScaler()),
				('pca', PCA(n_components=5)),
				('model', model)
			])

			# Train model with GridSearchCV
			grid = GridSearchCV(
				estimator=pipeline,
				param_grid=param_grid,
				cv=LeaveOneGroupOut().split(X_train, y_train, groups_train),
				scoring='accuracy',
				n_jobs=-1,
				verbose=0
			)
			grid.fit(X_train, y_train)

			best_model = grid.best_estimator_

			val_preds = best_model.predict(X_val)
			val_acc = accuracy_score(y_val, val_preds)
			val_f1 = f1_score(y_val, val_preds, average='weighted')

			# Retrain on train + val for final test
			X_trainval = np.vstack([X_train, X_val])
			y_trainval = np.hstack([y_train, y_val])
			final_model = grid.best_estimator_
			final_model.fit(X_trainval, y_trainval)
			test_preds = final_model.predict(X_test)
			test_acc = accuracy_score(y_test, test_preds)
			test_f1 = f1_score(y_test, test_preds, average='weighted')

			results.append({
				'model': model_name,
				'test_participant': test_pid,
				'val_participants': val_pids,
				'train_participants': train_pids,
				'best_params': grid.best_params_,
				'val_acc': val_acc,
				'val_f1': val_f1,
				'test_acc': test_acc,
				'test_f1': test_f1,
			})

			print('Results: test_acc of %f and test_f1 of %f' % (test_acc, test_f1))


import pandas as pd
df = pd.DataFrame(results)

print("Average test ACC:", df['test_acc'].mean())
print("Average test F1:", df['test_f1'].mean())

# Group by test participant
print(df.groupby('test_participant')[['val_acc', 'val_f1', 'test_acc', 'test_f1']].mean())
print(df.groupby('model')[['val_acc', 'val_f1', 'test_acc', 'test_f1']].mean())

df.to_csv(BASE_PATH + 'standard_scaler_results_with_ridge_for_classification_with_pca_binary.csv')

results = []

# Loop: every participant becomes the test participant once
for test_pid in participants:
	remaining = [p for p in participants if p != test_pid]

	# Try all combinations of 2 validation participants from the remaining 9
	val_combinations = list(itertools.combinations(remaining, 2))

	for val_pids in val_combinations:
		train_pids = [p for p in remaining if p not in val_pids]

		# Build training data (raw scores first)
		X_train, y_train_raw, groups_train = build_data(train_pids, return_raw_scores=True)

		# Compute bin edges from y_train only
		bin_counts, bin_edges = 3, [0.0, ]
		class_labels = ['low', 'medium', 'high']

		# Build datasets
		X_train, y_train, groups_train = build_data(train_pids, return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)
		X_val, y_val, _ = build_data(val_pids, return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)
		X_test, y_test, _ = build_data([test_pid], return_raw_scores=False, bin_edges=bin_edges, labels=class_labels)

		le = LabelEncoder()
		y_train = le.fit_transform(y_train)
		y_val = le.transform(y_val)
		y_test = le.transform(y_test)

		for model_name, model, param_grid in [
			('mlp', MLPClassifier(max_iter=1000), param_grid_mlp),
			('nb', GaussianNB(), param_grid_nb),
			('gbc', GradientBoostingClassifier(), param_grid_gbc),
			('rf', RandomForestClassifier(), param_grid_rf),
			('dt', DecisionTreeClassifier(), param_grid_dt),
			('knn', KNeighborsClassifier(), param_grid_knn),
			('svc', SVC(), param_grid_svc),
			('logreg', LogisticRegression(max_iter=1000), param_grid_logreg),
		]:
			print('Running Multivariate Regression LOO-CV for TRAIN-EVAL-TEST: [%s, %s, %s] and model: %s' % (train_pids, val_pids, test_pid, model_name))

			pipeline = Pipeline([
				('scaler', MinMaxScaler()),
				('pca', PCA(n_components=5)),
				('model', model)
			])

			# Train model with GridSearchCV
			grid = GridSearchCV(
				estimator=pipeline,
				param_grid=param_grid,
				cv=LeaveOneGroupOut().split(X_train, y_train, groups_train),
				scoring='accuracy',
				n_jobs=-1,
				verbose=0
			)
			grid.fit(X_train, y_train)
			best_model = grid.best_estimator_

			val_preds = best_model.predict(X_val)
			val_acc = accuracy_score(y_val, val_preds)
			val_f1 = f1_score(y_val, val_preds, average='weighted')

			# Retrain on train + val for final test
			X_trainval = np.vstack([X_train, X_val])
			y_trainval = np.hstack([y_train, y_val])
			final_model = grid.best_estimator_
			final_model.fit(X_trainval, y_trainval)
			test_preds = final_model.predict(X_test)
			test_acc = accuracy_score(y_test, test_preds)
			test_f1 = f1_score(y_test, test_preds, average='weighted')

			results.append({
				'model': model_name,
				'test_participant': test_pid,
				'val_participants': val_pids,
				'train_participants': train_pids,
				'best_params': grid.best_params_,
				'val_acc': val_acc,
				'val_f1': val_f1,
				'test_acc': test_acc,
				'test_f1': test_f1,
			})

			print('Results: test_acc of %f and test_f1 of %f' % (test_acc, test_f1))


import pandas as pd
df = pd.DataFrame(results)

print("Average test ACC:", df['test_acc'].mean())
print("Average test F1:", df['test_f1'].mean())

# Group by test participant
print(df.groupby('test_participant')[['val_acc', 'val_f1', 'test_acc', 'test_f1']].mean())
print(df.groupby('model')[['val_acc', 'val_f1', 'test_acc', 'test_f1']].mean())

df.to_csv(BASE_PATH + 'min_max_scaler_results_with_ridge_for_classification_with_pca_binary.csv')