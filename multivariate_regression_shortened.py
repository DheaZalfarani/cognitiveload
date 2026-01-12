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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
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

for file in os.listdir(BASE_PATH):
	if 'with_all_info_reduced_feature_sets' in file:
		with open(BASE_PATH + file, 'rb') as handle:
			labels_and_features = pickle.load(handle)
			multivariate_regression_data[file.split('_with')[0]] = labels_and_features[0]
			multivariate_regression_labels[file.split('_with')[0]] = labels_and_features[1]


def build_data(selected_participants):
	X, y, groups = [], [], []
	for pid in selected_participants:
		for session in ['S1', 'S2']:
			key = f"{pid}_{session}"
			X.append(multivariate_regression_data[key])
			y_session = [float(label.split('/')[2]) for label in multivariate_regression_labels[key]]
			y.append(y_session)
			groups.extend([pid] * len(y_session))  # 4 tasks per session
	return np.vstack(X), np.hstack(y), np.array(groups)


from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

rnd_state = 42

param_grid_svr = {
	'model__C': [0.1, 1, 10, 100],
	'model__epsilon': [0.01, 0.1, 1],
	'model__kernel': ['linear', 'rbf'],
	'model__gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
}

param_grid_rf = {
	'model__n_estimators': [50, 100, 200, 300],
	'model__max_depth': [3, 5, None],
	'model__min_samples_split': [2, 5],
	'model__min_samples_leaf': [1, 2],
	'model__max_features': ['log2', 0.3, 0.5, 0.7],
}

param_grid_gbr = {
	'model__n_estimators': [50, 100, 200],
	'model__learning_rate': [0.01, 0.05, 0.1],
	'model__max_depth': [2, 3, 5],
	'model__subsample': [0.8, 1.0],
	'model__min_samples_split': [2, 5],
	'model__max_features': [0.3, 0.5, 'sqrt']
}

param_grid_lasso = {
	'model__alpha': [0.001, 0.01, 0.1, 0.5, 10.0, 100.0]
}

param_grid_ridge = {
	'model__alpha': [0.001, 0.01, 0.1, 0.5, 10.0, 100.0]
}

param_grid_enet = {
	'model__alpha': [0.001, 0.01, 0.1, 0.5, 10.0],
	'model__l1_ratio': [0.25, 0.5, 0.75]  # 0.1 ~ mostly Ridge, 0.9 ~ mostly Lasso
}

param_grid_svr = {
	'model__C': [0.1, 1, 10],
	'model__epsilon': [0.01, 0.1, 1],
	'model__kernel': ['linear', 'rbf'],
	'model__gamma': ['scale', 'auto']  # Only used with 'rbf'
}

param_grid_bayesian_ridge = {
	'model__alpha_1': [1e-6, 1e-5, 1e-4],
	'model__alpha_2': [1e-6, 1e-5, 1e-4],
	'model__lambda_1': [1e-6, 1e-5, 1e-4],
	'model__lambda_2': [1e-6, 1e-5, 1e-4],
	'model__fit_intercept': [True, False]
}

param_grid_dummy = {
	'model__strategy': ['mean', 'quantile'],
	'model__quantile': [0.3, 0.7],
}

results = []

# Loop: every participant becomes the test participant once
for test_pid in participants:
	remaining = [p for p in participants if p != test_pid]

	# Try all combinations of 2 validation participants from the remaining 9
	val_combinations = list(itertools.combinations(remaining, 2))

	for val_pids in val_combinations:
		train_pids = [p for p in remaining if p not in val_pids]

		# Build datasets
		X_train, y_train, groups_train = build_data(train_pids)
		X_val, y_val, _ = build_data(val_pids)
		X_test, y_test, _ = build_data([test_pid])

		models = [
			linear_model.Lasso(), 
			linear_model.Ridge(),
			linear_model.ElasticNet(),
			GradientBoostingRegressor(random_state=rnd_state),
			RandomForestRegressor(random_state=rnd_state),
			SVR(),
			BayesianRidge(),
			DummyRegressor(),
		]

		for model_name, model, param_grid in [
			# ('Lasso', models[0], param_grid_lasso),
			# ('Ridge', models[1], param_grid_ridge),
			# ('ElasticNet', models[2], param_grid_enet),
			('GradientBoostingRegressor', models[3], param_grid_gbr),
			('RandomForestRegressor', models[4], param_grid_rf),
			('SVR', models[5], param_grid_svr),
			('BayesianRidge', models[6], param_grid_bayesian_ridge),
			('Dummy', models[7], param_grid_dummy),
		]:
			print('Running Multivariate Regression LOO-CV for TRAIN-EVAL-TEST: [%s, %s, %s] and model: %s' % (train_pids, val_pids, test_pid, model_name))

			pipeline = Pipeline([
				('scaler', StandardScaler()),
				('pca', PCA(n_components=4)),
				('model', model)
			])

			# Train model with GridSearchCV
			grid = GridSearchCV(
				estimator=pipeline,
				param_grid=param_grid,
				cv=LeaveOneGroupOut().split(X_train, y_train, groups_train),
				scoring='r2',
				n_jobs=-1,
				verbose=0
			)
			grid.fit(X_train, y_train)

			# Evaluate on val set
			val_preds = grid.predict(X_val)
			val_r2 = r2_score(y_val, val_preds)
			val_mae = mean_absolute_error(y_val, val_preds)
			val_rmse = mean_squared_error(y_val, val_preds, squared=False)

			# Retrain on train + val for final test
			X_trainval = np.vstack([X_train, X_val])
			y_trainval = np.hstack([y_train, y_val])
			final_model = grid.best_estimator_
			final_model.fit(X_trainval, y_trainval)
			test_preds = final_model.predict(X_test)
			test_r2 = r2_score(y_test, test_preds)
			test_mae = mean_absolute_error(y_test, test_preds)
			test_rmse = mean_squared_error(y_test, test_preds, squared=False)

			results.append({
				'model': model_name,
				'test_participant': test_pid,
				'val_participants': val_pids,
				'train_participants': train_pids,
				'best_params': grid.best_params_,
				'val_r2': val_r2,
				'val_mae': val_mae,
				'val_rmse': val_rmse,
				'test_r2': test_r2,
				'test_mae': test_mae,
				'test_rmse': test_rmse
			})

			print('Results: val_r2 of %f and test_r2 of %f' % (val_r2, test_r2))


import pandas as pd
df = pd.DataFrame(results)

print("Average test R^2:", df['test_r2'].mean())
print("Average test MAE:", df['test_mae'].mean())
print("Average test RMSE:", df['test_rmse'].mean())

# Group by test participant
print(df.groupby('test_participant')[['test_r2', 'test_mae', 'test_rmse']].mean())
print(df.groupby('model')[['test_r2', 'test_mae', 'test_rmse']].mean())

df.to_csv(BASE_PATH + 'standard_scaler_results_with_ridge_and_PCA_shortened_modelamount.csv')

results = []

# Loop: every participant becomes the test participant once
for test_pid in participants:
	remaining = [p for p in participants if p != test_pid]

	# Try all combinations of 2 validation participants from the remaining 9
	val_combinations = list(itertools.combinations(remaining, 2))

	for val_pids in val_combinations:
		train_pids = [p for p in remaining if p not in val_pids]

		# Build datasets
		X_train, y_train, groups_train = build_data(train_pids)
		X_val, y_val, _ = build_data(val_pids)
		X_test, y_test, _ = build_data([test_pid])

		models = [
			linear_model.Lasso(), 
			linear_model.Ridge(),
			linear_model.ElasticNet(),
			GradientBoostingRegressor(random_state=rnd_state),
			RandomForestRegressor(random_state=rnd_state),
			SVR(),
			BayesianRidge(),
			DummyRegressor(),
		]

		for model_name, model, param_grid in [
			('Lasso', models[0], param_grid_lasso),
			('Ridge', models[1], param_grid_ridge),
			('ElasticNet', models[2], param_grid_enet),
			('GradientBoostingRegressor', models[3], param_grid_gbr),
			('RandomForestRegressor', models[4], param_grid_rf),
			('SVR', models[5], param_grid_svr),
			('BayesianRidge', models[6], param_grid_bayesian_ridge),
			('Dummy', models[7], param_grid_dummy),
		]:
			print('Running Multivariate Regression LOO-CV for TRAIN-EVAL-TEST: [%s, %s, %s] and model: %s' % (train_pids, val_pids, test_pid, model_name))

			pipeline = Pipeline([
				('scaler', StandardScaler()),
				('pca', PCA(n_components=4)),
				('model', model)
			])

			# Train model with GridSearchCV
			grid = GridSearchCV(
				estimator=pipeline,
				param_grid=param_grid,
				cv=LeaveOneGroupOut().split(X_train, y_train, groups_train),
				scoring='r2',
				n_jobs=-1,
				verbose=0
			)
			grid.fit(X_train, y_train)

			# Evaluate on val set
			val_preds = grid.predict(X_val)
			val_r2 = r2_score(y_val, val_preds)
			val_mae = mean_absolute_error(y_val, val_preds)
			val_rmse = mean_squared_error(y_val, val_preds, squared=False)

			# Retrain on train + val for final test
			X_trainval = np.vstack([X_train, X_val])
			y_trainval = np.hstack([y_train, y_val])
			final_model = grid.best_estimator_
			final_model.fit(X_trainval, y_trainval)
			test_preds = final_model.predict(X_test)
			test_r2 = r2_score(y_test, test_preds)
			test_mae = mean_absolute_error(y_test, test_preds)
			test_rmse = mean_squared_error(y_test, test_preds, squared=False)

			results.append({
				'model': model_name,
				'test_participant': test_pid,
				'val_participants': val_pids,
				'train_participants': train_pids,
				'best_params': grid.best_params_,
				'val_r2': val_r2,
				'val_mae': val_mae,
				'val_rmse': val_rmse,
				'test_r2': test_r2,
				'test_mae': test_mae,
				'test_rmse': test_rmse
			})

			print('Results: val_r2 of %f and test_r2 of %f' % (val_r2, test_r2))


import pandas as pd
df = pd.DataFrame(results)

print("Average test R^2:", df['test_r2'].mean())
print("Average test MAE:", df['test_mae'].mean())
print("Average test RMSE:", df['test_rmse'].mean())

# Group by test participant
print(df.groupby('test_participant')[['test_r2', 'test_mae', 'test_rmse']].mean())
print(df.groupby('model')[['test_r2', 'test_mae', 'test_rmse']].mean())

df.to_csv(BASE_PATH + 'min_max_scaler_results_with_ridge_and_PCA_shortened_modelamount.csv')