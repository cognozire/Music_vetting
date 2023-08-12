import streamlit as st
import librosa
import numpy as np
import random
from scipy.spatial.distance import cosine
import os
def normalize_vector(vector, target_length):
    current_length = len(vector)
    if current_length > target_length:
        return vector[:target_length]
    else:
        return np.pad(vector, (0, target_length - current_length), mode='constant')

st.title("Music Vetting App")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

# Dropdown menu with musical notes
musical_notes = ['A',  'B', 'C', 'D',  'E', 'F', 'G']
selected_note = st.selectbox("Select a musical note", musical_notes)

if uploaded_file is not None:
    audio_data, sample_rate = librosa.load(uploaded_file, sr=None)
    mfccs1 = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
    mfccs1 = mfccs1.flatten()
    # st.audio(mfccs1, format='audio/wav', start_time=0, sample_rate=sample_rate)  # Specify sample_rate
    st.write(f"Uploaded audio file with sample rate: {sample_rate}")

    if st.button("Calculate Similarity"):
        note_folder = selected_note
        note_files = os.listdir(note_folder)
    
        similarity_scores = []
        for note_file in note_files:
            note_audio_data, note_sample_rate = librosa.load(os.path.join(note_folder, note_file), sr=None)
            mfccs2 = librosa.feature.mfcc(y=note_audio_data, sr=note_sample_rate)
            mfccs2 = mfccs2.flatten()
        # Ensure audio data has the same length
            max_len = max(len(mfccs1), len(mfccs2))
            mfccs1= normalize_vector(mfccs1, max_len)
            mfccs2 = normalize_vector(mfccs2, max_len)
        
            similarity = 1 - cosine(mfccs1, mfccs2)
            similarity_scores.append(similarity)
    
        similarity = np.max(similarity_scores)
        if(similarity>0.76):
            st.write(f"The uploaded audio is {selected_note} note.")
        else:
            st.write(f"The uploaded audio doesn't matches with {selected_note} note.")






