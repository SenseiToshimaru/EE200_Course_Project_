import streamlit as st
import librosa.display
import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

# Import database functions
from database import *

# 3. FRONTEND: STREAMLIT APP

st.set_page_config(page_title="Audio Fingerprinter", layout="wide")
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.title("Apollo 🪉 SoNg SeaRch")



# Connect to database
conn = init_db()

# Sidebar
# Sidebar
st.sidebar.header("App Modes")
mode = st.sidebar.radio("Select Mode:", ["Single-Clip Mode", "Batch Mode", "Database Management"])


st.sidebar.divider()
st.sidebar.header("🎵 Available Songs")

# Fetch current songs from database
existing_songs = get_db_stats(conn)

if existing_songs:
    # Initialize session state to track currently playing song
    if "current_playing" not in st.session_state:
        st.session_state.current_playing = None

    for song in sorted(existing_songs):
        # Create a button for each song. Clicking toggles or switches it.
        if st.sidebar.button(f"▶️ {song}", key=f"play_{song}", use_container_width=True):
            if st.session_state.current_playing == song:
                st.session_state.current_playing = None  # Turn off if clicked again
            else:
                st.session_state.current_playing = song  # Play this song / switch song
            st.rerun()

    # Audio player interface
    if st.session_state.current_playing:
        st.sidebar.markdown(f"**Now Playing:** {st.session_state.current_playing}")
        
        # Look for matching file types in local directory
        possible_extensions = ['.wav', '.mp3']
        audio_file_path = None
        for ext in possible_extensions:
            test_path = f"{st.session_state.current_playing}{ext}"
            if os.path.exists(test_path):
                audio_file_path = test_path
                break
        
        if audio_file_path:
            st.sidebar.audio(audio_file_path, autoplay=True)
        else:
            st.sidebar.error("Audio source file not found in directory.")
else:
    st.sidebar.write("No songs indexed yet.")

def save_uploaded_file(uploaded_file):
    """Saves Streamlit uploaded file to disk temporarily for Librosa to read."""
    _, ext = os.path.splitext(uploaded_file.name)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name    


# MODE 1: Database Management
if mode == "Database Management":
    st.header("Database Management")

    existing_songs = get_db_stats(conn)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Songs Indexed", value=len(existing_songs))
    with col2:
        with st.expander("👀 View Indexed Songs"):
            if existing_songs:
                for song in sorted(existing_songs):
                    st.write(f"🎵 {song}")
            else:
                st.write("Database is empty.")
    
    st.divider()

    uploaded_files = st.file_uploader("Upload database songs", accept_multiple_files=True, type=['wav', 'mp3'])
    if st.button("Build Database"):
        if uploaded_files:
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                tmp_path = save_uploaded_file(file)
                song_name = os.path.splitext(file.name)[0]
                
                index_song(tmp_path, song_name, conn)
                
                os.remove(tmp_path) # Cleanup disk
                progress_bar.progress((i + 1) / len(uploaded_files))
            st.success("Database built successfully!")
        else:
            st.warning("Please upload files first.")

# MODE 2: Single-Clip Mode
elif mode == "Single-Clip Mode":
    st.header("Please Upload Single File")
    query_file = st.file_uploader("Upload a short query clip", key="single", type=['wav', 'mp3'])
    
    if query_file and st.button("Identify Song"):
        with st.spinner("Processing audio and searching database..."):
            tmp_path = save_uploaded_file(query_file)
            
            # identification step
            best_match, stft_db, freqs, times, best_hist = process_query_clip(tmp_path, conn)
            os.remove(tmp_path)
            
            # Result
            if best_match != "No confident match found":
                st.success(f"**Matched Song:** {best_match}")
            else:
                st.error("No confident match found. The clip may be too noisy or not in the database.")
            
            # Visualizations
            st.subheader("Intermediate Steps")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**1. Spectrogram & Constellation**")
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                
                # Plotting Spectrogram
                img = librosa.display.specshow(stft_db, x_axis='time', y_axis='linear', 
                                               sr=11025, hop_length=512, ax=ax1, cmap='magma')
                
                # Mapping array indices back to time and frequency for scatter plot
                physical_times = librosa.frames_to_time(times, sr=11025, hop_length=512)
                fft_freqs = librosa.fft_frequencies(sr=11025, n_fft=1024)
                physical_freqs = fft_freqs[freqs]
                
                # Constellation Peaks
                ax1.scatter(physical_times, physical_freqs, color='cyan', s=10, marker='o', alpha=0.8, edgecolor='none')
                fig1.colorbar(img, ax=ax1, format="%+2.0f dB")
                st.pyplot(fig1)
                
            with col2:
                st.write("**2. Offset Histogram**")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                if best_hist:
                    ax2.hist(best_hist, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
                    ax2.set_title(f"Histogram for: {best_match}")
                    ax2.set_xlabel("Time Difference (DB Offset - Query Offset)")
                    ax2.set_ylabel("Number of Matches")
                else:
                    ax2.text(0.5, 0.5, 'No histogram data available', horizontalalignment='center', verticalalignment='center')
                st.pyplot(fig2)

# MODE 3: Batch Mode
elif mode == "Batch Mode":
    st.header("Batch Processing")
    st.write("Upload multiple query clips. The app will generate a `results.csv` file exactly as required.")
    
    batch_files = st.file_uploader("Upload batch queries", accept_multiple_files=True, key="batch", type=['wav', 'mp3'])
    
    if batch_files and st.button("Process Batch"):
        results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(batch_files):
            tmp_path = save_uploaded_file(file)
            
            best_match, _, _, _, _ = process_query_clip(tmp_path, conn)
            
            if best_match == "No confident match found":
                best_match = "Unknown"
                
            results.append({"Input File": file.name, "Matched Song Name": best_match})
            os.remove(tmp_path)
            
            progress_bar.progress((i + 1) / len(batch_files))
            
            # protecting RAM during batching
            gc.collect()
            
        # Creating Dataframe 
        df = pd.DataFrame(results)
        
        # Display the results as an interactive Streamlit table
        st.subheader("Batch Match Results")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.success("Batch processing complete!")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )