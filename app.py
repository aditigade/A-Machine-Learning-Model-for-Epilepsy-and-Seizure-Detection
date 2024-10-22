import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import welch
import joblib

# Set page configuration
st.set_page_config(page_title="EEG Classification App", page_icon="🧠")

# Custom CSS for UI Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8;  /* Light blue background */
        color: #333;
    }
    h1 {
        color: #9CAF88;  /* Sage green color for title */
    }
    h2, h3 {
        color: #2E8B57;  /* Sea green for subtitles */
    }
    .stButton button {
        background-color: #808080 !important;  /* Indigo buttons */
        color: white !important;
        border-radius: 8px;
    }
    .stTextInput input, .stFileUploader div {
        background-color: #808080;  /* Light grey inputs */
        color: #000000;
        border: 2px solid #808080;
        border-radius: 5px;
    }
    .css-1d391kg {
        width: 500px;  /* Adjust the sidebar width */
    }
    .stSidebar {
        background-color: #001f3f;  /* Navy blue background */
        border-right: 3px solid #001f3f;  /* Navy blue border */
        color: white;  /* White text for contrast */
    }
    .stSidebar h2, .stSidebar h3, .stSidebar p {
        color: white;  /* Ensure all text inside sidebar is white */
        font-size: 1.1rem;  /* Increase font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("📊 EEG Dataset Information")
st.sidebar.write("""
The **Bonn University EEG Dataset** offers insight into epileptic and non-epileptic brain activities with five sets of recordings:

- **Set A (Label: 0)**: Non-epileptic (eyes open) – Demonstrates alert brain activity, serving as a control baseline.
- **Set B (Label: 1)**: Non-epileptic (eyes closed) – Captures relaxed brain states, aiding comparison between wakefulness and rest.
- **Set C (Label: 2)**: Epileptic (interictal, non-affected region) – Highlights stable, seizure-free brain activity from unaffected regions.
- **Set D (Label: 3)**: Epileptic (interictal, affected region) – Provides insights into pre-seizure neural activity from seizure-prone areas.
- **Set E (Label: 4)**: Epileptic (ictal) – Captures real-time seizure activity, crucial for studying seizure dynamics.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 About the App")
st.sidebar.write("""
This **EEG Classification App** leverages Machine Learning to analyze brain states from the Bonn University dataset.  
**Features:**
1. **Upload EEG files (.txt)** for visualization.
2. **Classify the brain state** using a RandomForest classifier.

The app aids research in epilepsy detection by offering real-time seizure predictions and insights into neural activities.
""")


# Define dataset folder path
DATASET_FOLDER = 'D:\ml-cp\dataset'

# Function to load data from a folder
def load_data_from_folder(folder_path, file_prefix, file_extension='txt'):
    data = []
    for i in range(1, 101):
        filename = f"{file_prefix}{i:03d}.{file_extension}"
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            data.append(np.loadtxt(file_path))
        else:
            st.write(f"File not found: {file_path}")
    return np.array(data)

# Load datasets from folders
def load_full_dataset():
    data_A = load_data_from_folder(os.path.join(DATASET_FOLDER, 'SET_A'), 'Z')
    data_B = load_data_from_folder(os.path.join(DATASET_FOLDER, 'SET_B'), 'O')
    data_C = load_data_from_folder(os.path.join(DATASET_FOLDER, 'SET_C'), 'N', 'TXT')
    data_D = load_data_from_folder(os.path.join(DATASET_FOLDER, 'SET_D'), 'F')
    data_E = load_data_from_folder(os.path.join(DATASET_FOLDER, 'SET_E'), 'S')

    labels_A = np.zeros(data_A.shape[0])
    labels_B = np.ones(data_B.shape[0])
    labels_C = np.full(data_C.shape[0], 2)
    labels_D = np.full(data_D.shape[0], 3)
    labels_E = np.full(data_E.shape[0], 4)

    data = np.vstack((data_A, data_B, data_C, data_D, data_E))
    labels = np.concatenate((labels_A, labels_B, labels_C, labels_D, labels_E))

    return data, labels

# Load and preprocess dataset
data, labels = load_full_dataset()
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data.reshape(data.shape[0], -1))

# Feature extraction using Welch's method
def extract_features(data):
    features = []
    for i in range(data.shape[0]):
        freqs, psd = welch(data[i], fs=173.61)
        features.append(psd)
    return np.array(features)

features = extract_features(data_normalized)

# Train RandomForest model
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Main app interface
st.title("🧠 EEG Seizure Detection")

# Function to load and preprocess uploaded files
def load_and_preprocess_file(file):
    data = np.loadtxt(file)
    data = data.reshape(1, -1)
    data_normalized = scaler.transform(data)
    features = extract_features(data_normalized)
    return features, data

# Class labels with detailed descriptions
class_names = [
    '0 - Set A: Non-epileptic (eyes open) - Demonstrates typical brain activity during alertness, useful as a control baseline.',
    '1 - Set B: Non-epileptic (eyes closed) - Captures relaxed brain states, helping compare wakeful vs. restful neural patterns.',
    '2 - Set C: Epileptic (interictal, non-affected region) - Records interictal brain signals from non-affected areas, highlighting stable, non-seizure periods in epileptic patients.',
    '3 - Set D: Epileptic (interictal, affected region) - Monitors interictal signals from seizure-prone regions, offering insights into pre-seizure neural activity.',
    '4 - Set E: Epileptic (ictal, during seizure) - Reflects real-time seizure (ictal) activity, essential for studying the dynamics of seizures for detection and prediction.'
]


# File uploader
uploaded_file = st.file_uploader("Upload an EEG file (in .txt format)", type="txt")

if uploaded_file:
    # Preprocess file and extract features
    features_for_prediction, eeg_data = load_and_preprocess_file(uploaded_file)

    # Plot EEG signal
    st.subheader("EEG Signal Visualization")
    plt.figure(figsize=(10, 4))
    plt.plot(eeg_data.flatten(), color='#2E8B57')
    plt.title('EEG Signal', color='#4B0082')
    plt.xlabel('Time', color='#333')
    plt.ylabel('Amplitude', color='#333')
    st.pyplot(plt)

    # Predict class
    predicted_label = model.predict(features_for_prediction)
    st.success(f"Predicted Class: **{class_names[int(predicted_label[0])]}**")


