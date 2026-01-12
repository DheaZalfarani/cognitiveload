# cl_intervention_e_learning_2025
Source code to load, process, and extract features, as well as to perform statistical analyses and machine learning, for the cl_intervention_e_learning_2025 dataset

# Data Set:
Cognitive Load Classification and Real-Time Intervention for Enhanced Vocabulary Learning at Zenodo (https://doi.org/10.5281/zenodo.17350643)

 # Brief experimental description
The dataset (approximately 40 hours in total) consists of physiological signals from wearable electroencephalography (EEG), electrodermal activity (EDA), photoplethysmogram (PPG), acceleration, and temperature sensors, as well as log files from a computerized vocabulary E-Learning application. Data was recorded from 10 completely anonymized participants who performed computerized E-Learning vocabulary learning, designed to induce mental workload while learning from four different, unknown languages. Physiological signals were obtained from the Muse S EEG headband and Empatica E4 wristband.

Experimental Setup:

Session 1 (Baseline): Physiological data were recorded during tasks designed to induce labeled states of [overload, underload, high interest, and low interest]. This labeled data was used to develop a personalized machine learning model for classifying subjective cognitive load.

Session 2 (Intervention): The personalized model classified the participant's subjective cognitive load level in real-time. Based on these results, the E-Learning application was adjusted to steer the participant's cognitive load level and increase learning performance. As such, words were added to or removed from the vocabulary list, and the time each word was shown or the respective number of repetitions was adjusted. 

Labels: Self-reported labels were obtained using Likert scales (for subjective cognitive load and stress), NASA-TLX (for overall workload), and PANAS (for affective state), in addition to performance metrics extracted from the log files.

Vocabulary: Six languages were chosen: Esperanto, Hinglish, Nahuatl, Pinjin, Spanish, and Turkish. It was ensured that participants were unfamiliar with the respective language prior to enrollment. The study was performed in accordance with the local institute review board's ethical guidelines and the Declaration of Helsinki.

The completely anonymized dataset is publicly available and offers vast potential to the research community working on mental workload detection using consumer-grade wearable sensors. Among other applications, the data is suitable for developing real-time cognitive load detection methods, researching signal processing techniques, or investigating ML-adjusted E-Learning applications.

The link to the publication will be added here once the manuscript is accepted in the respective journal.

# Technical Info
The anonymized data is located in the top-level subfolder 'data'. Within this, the subfolders 'P001_1st_session' through 'P010_2nd_session' contain data from individual participants across their respective first and second sessions (i.e., suffixes '_1st_session' and '_2nd_session').

For each participant-session folder, multiple numerically named subfolders (0, 1, ...) exist, representing distinct recording runs in case an application had to be restarted. In these subfolders, a respective 'RawData' folder contains the sensor files. The main log file for a session (e.g., 'p009_2nd_session_anonymized.log'), located in the main session folder, holds the time-aligned labels for all runs.

Per recording, the following anonymized files exist with the suffix '_anonymized.csv':

Empatica E4: 'ACC.csv', 'BVP.csv', 'GSR.csv', and 'TEMP.csv'

Muse S: 'ACC.csv', 'EEG.csv', 'GYRO.csv', and 'PPG.csv'

Finally, the folder 'features_and_labels_pckls' contains pre-processed data, extracted features, and the respective labels for the extracted time-windows, all in .pkl format (e.g., 'P001_S1.pkl', 'P001_S1_with_all_info.pkl').

# Contact
Finally, please feel free to reach out should you encounter any issues or have any open questions regarding this data set, the experimental paradigm, the source code, or the publication. You can reach the authors via the contact information provided in the publication or via email to 'christoph.anders@hpi.de', 'christoph.anders@hpi.uni-potsdam.de', 'office-arnrich@hpi.uni-potsdam.de', or 'e_learning_2025@hpi.de'.
