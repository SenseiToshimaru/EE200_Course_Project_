import streamlit as st
import librosa
import librosa.display
import numpy as np
import scipy.ndimage as ndimage
import sqlite3
import hashlib
import gc
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import tempfile
import os

# =========================================================
# 1. BACKEND: MEMORY-SAFE SIGNAL PROCESSING
# =========================================================

def init_db(db_name='fingerprints.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hashes (
            hash_val INTEGER,
            song_name TEXT,
            offset INTEGER
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON hashes(hash_val)')
    conn.commit()
    return conn

def get_spectrogram(file_path, sr=11025, n_fft=1024, hop_length=512):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    del y
    return stft_db

def get_constellation(stft_db, fps=21, max_peaks_per_sec=15):
    neighborhood = ndimage.generate_binary_structure(2, 1)
    neighborhood = ndimage.iterate_structure(neighborhood, 5)
    local_max = (ndimage.maximum_filter(stft_db, footprint=neighborhood) == stft_db)
    
    background = (stft_db == 0)
    detected_peaks = local_max ^ background
    freqs, times = np.where(detected_peaks)
    amplitudes = stft_db[freqs, times]
    
    capped_freqs = []
    capped_times = []
    max_time = stft_db.shape[1]
    
    for t_start in range(0, max_time, fps):
        t_end = t_start + fps
        mask = (times >= t_start) & (times < t_end)
        block_times = times[mask]
        block_freqs = freqs[mask]
        block_amps = amplitudes[mask]
        
        if len(block_amps) > 0:
            sort_idx = np.argsort(block_amps)[::-1][:max_peaks_per_sec]
            capped_freqs.extend(block_freqs[sort_idx])
            capped_times.extend(block_times[sort_idx])
            
    return np.array(capped_freqs), np.array(capped_times)

def generate_hashes(freqs, times, song_name, fan_out=5, target_zone_dt=50):
    sort_idx = np.argsort(times)
    freqs = freqs[sort_idx]
    times = times[sort_idx]
    
    hashes = []
    num_points = len(times)
    
    for i in range(num_points):
        # FORCE NATIVE PYTHON INTEGERS HERE
        anchor_freq = int(freqs[i])
        anchor_time = int(times[i])
        
        targets_found = 0
        
        for j in range(i + 1, num_points):
            target_freq = int(freqs[j])
            target_time = int(times[j])
            time_delta = target_time - anchor_time
            
            if time_delta > target_zone_dt:
                break 
            if time_delta > 0: 
                hash_str = f"{anchor_freq}|{target_freq}|{time_delta}"
                hash_int = int(hashlib.sha1(hash_str.encode('utf-8')).hexdigest()[:8], 16)
                
                # Append standard integers to the list so SQLite doesn't freak out
                hashes.append((hash_int, song_name, anchor_time))
                targets_found += 1
                if targets_found >= fan_out:
                    break
    return hashes

def index_song(file_path, song_name, db_conn):
    stft_db = get_spectrogram(file_path)
    freqs, times = get_constellation(stft_db)
    hashes = generate_hashes(freqs, times, song_name)
    
    cursor = db_conn.cursor()
    cursor.executemany('INSERT INTO hashes (hash_val, song_name, offset) VALUES (?, ?, ?)', hashes)
    db_conn.commit()
    
    del stft_db, freqs, times, hashes
    gc.collect()

def get_db_stats(db_conn):
    """Retrieves a list of all unique songs currently in the database."""
    cursor = db_conn.cursor()
    # DISTINCT ensures we only count each song name once, even if it has thousands of hashes
    cursor.execute("SELECT DISTINCT song_name FROM hashes")
    songs = [row[0] for row in cursor.fetchall()]
    return songs    

# =========================================================
# 2. BACKEND: MATCHING LOGIC
# =========================================================

def find_matches(query_hashes, db_conn):
    """Searches the SQLite database for matching hashes and calculates offset differences."""
    cursor = db_conn.cursor()
    hash_dict = {h[0]: h[2] for h in query_hashes}
    hash_vals = list(hash_dict.keys())
    
    if not hash_vals:
        return {}
        
    matches = []
    # SQLite limits IN clause variables; chunking handles large queries
    chunk_size = 900 
    for i in range(0, len(hash_vals), chunk_size):
        chunk = hash_vals[i:i+chunk_size]
        placeholders = ','.join('?' * len(chunk))
        query = f"SELECT hash_val, song_name, offset FROM hashes WHERE hash_val IN ({placeholders})"
        cursor.execute(query, chunk)
        matches.extend(cursor.fetchall())
        
    song_matches = {}
    for hash_val, song_name, db_offset in matches:
        query_offset = hash_dict[hash_val]
        # Core concept: True match aligns perfectly in time (db_offset - query_offset = constant)
        diff = db_offset - query_offset
        if song_name not in song_matches:
            song_matches[song_name] = []
        song_matches[song_name].append(diff)
        
    return song_matches

def determine_best_match(song_matches):
    """Finds the song with the highest coherent offset matches."""
    best_song = "No match found"
    max_score = 0
    best_histogram = []
    
    for song, diffs in song_matches.items():
        if not diffs: continue
        counts = Counter(diffs)
        # Find the most frequent time difference (the highest peak in the histogram)
        top_diff, count = counts.most_common(1)[0]
        
        if count > max_score:
            max_score = count
            best_song = song
            best_histogram = diffs
            
    # Set an arbitrary threshold to prevent false positives from noise
    if max_score < 10: 
        return "No confident match found", max_score, []
        
    return best_song, max_score, best_histogram

def process_query_clip(file_path, db_conn):
    """End-to-end processing for a single query clip."""
    stft_db = get_spectrogram(file_path)
    freqs, times = get_constellation(stft_db)
    query_hashes = generate_hashes(freqs, times, "query")
    
    song_matches = find_matches(query_hashes, db_conn)
    best_match, score, best_hist = determine_best_match(song_matches)
    
    return best_match, stft_db, freqs, times, best_hist

# =========================================================
# 3. FRONTEND: STREAMLIT APP
# =========================================================

st.set_page_config(page_title="Audio Fingerprinter", layout="wide")
st.title("🎵 Shazam-Style Audio Fingerprinting")

# Connect to database
conn = init_db()

# Sidebar
st.sidebar.header("App Modes")
mode = st.sidebar.radio("Select Mode:", ["Single-Clip Mode", "Batch Mode", "Database Management"])

def save_uploaded_file(uploaded_file):
    """Saves Streamlit uploaded file to disk temporarily for Librosa to read."""
    # Extract the original extension (.mp3 or .wav)
    _, ext = os.path.splitext(uploaded_file.name)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

# ---------------------------------------------------------
# MODE 1: Database Management
# ---------------------------------------------------------
if mode == "Database Management":
    st.header("🗄️ Database Management")

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
    # -----------------
    st.write("Upload the 50 provided songs here to build your local SQLite database.")
    
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

# ---------------------------------------------------------
# MODE 2: Single-Clip Mode
# ---------------------------------------------------------
elif mode == "Single-Clip Mode":
    st.header("🔍 Single-Clip Identifier")
    query_file = st.file_uploader("Upload a short query clip", key="single", type=['wav', 'mp3'])
    
    if query_file and st.button("Identify Song"):
        with st.spinner("Processing audio and searching database..."):
            tmp_path = save_uploaded_file(query_file)
            
            # Run identification pipeline
            best_match, stft_db, freqs, times, best_hist = process_query_clip(tmp_path, conn)
            os.remove(tmp_path)
            
            # Display Result
            if best_match != "No confident match found":
                st.success(f"**Matched Song:** {best_match}")
            else:
                st.error("No confident match found. The clip may be too noisy or not in the database.")
            
            # --- Visualizations ---
            st.subheader("Intermediate Steps")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**1. Spectrogram & Constellation**")
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                
                # Plot Spectrogram
                img = librosa.display.specshow(stft_db, x_axis='time', y_axis='linear', 
                                               sr=11025, hop_length=512, ax=ax1, cmap='magma')
                
                # Map array indices back to time and frequency for the scatter plot
                physical_times = librosa.frames_to_time(times, sr=11025, hop_length=512)
                fft_freqs = librosa.fft_frequencies(sr=11025, n_fft=1024)
                physical_freqs = fft_freqs[freqs]
                
                # Plot Constellation Peaks
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

# ---------------------------------------------------------
# MODE 3: Batch Mode
# ---------------------------------------------------------
elif mode == "Batch Mode":
    st.header("📂 Batch Processing")
    st.write("Upload multiple query clips. The app will generate a `results.csv` file exactly as required.")
    
    batch_files = st.file_uploader("Upload batch queries", accept_multiple_files=True, key="batch", type=['wav', 'mp3'])
    
    if batch_files and st.button("Process Batch"):
        results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(batch_files):
            tmp_path = save_uploaded_file(file)
            
            # We only need the best_match string for the CSV
            best_match, _, _, _, _ = process_query_clip(tmp_path, conn)
            
            # Fallback if no match is found (avoids CSV errors)
            if best_match == "No confident match found":
                best_match = "Unknown"
                
            results.append({"filename": file.name, "prediction": best_match})
            os.remove(tmp_path)
            
            progress_bar.progress((i + 1) / len(batch_files))
            
            # Explicit garbage collection to protect RAM during batching
            gc.collect()
            
        # Create Dataframe and write to strict format
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False)
        
        st.success("Batch processing complete!")
        st.download_button(
            label="Download results.csv",
            data=csv,
            file_name='results.csv',
            mime='text/csv',
        )