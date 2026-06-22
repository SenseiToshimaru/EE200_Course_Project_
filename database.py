import sqlite3
import hashlib
import gc
import numpy as np
import scipy.ndimage as ndimage
import librosa
from collections import Counter

# 1. BACKEND: MEMORY-SAFE SIGNAL PROCESSING

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
    # DISTINCT ensures we only count each song name once, even if there are thousands of hashes
    cursor.execute("SELECT DISTINCT song_name FROM hashes")
    songs = [row[0] for row in cursor.fetchall()]
    return songs    


# 2. BACKEND: MATCHING LOGIC

def find_matches(query_hashes, db_conn):
    """Searches the SQLite database for matching hashes and calculates offset differences."""
    cursor = db_conn.cursor()
    hash_dict = {h[0]: h[2] for h in query_hashes}
    hash_vals = list(hash_dict.keys())
    
    if not hash_vals:
        return {}
        
    matches = []
    
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
        top_diff, count = counts.most_common(1)[0]
        
        if count > max_score:
            max_score = count
            best_song = song
            best_histogram = diffs
            
    # Set an arbitrary threshold, prevent false positives from noise
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