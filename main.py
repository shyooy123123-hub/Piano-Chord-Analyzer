import subprocess
import sys

required_libraries = ['librosa', 'pyaudio', 'numpy', 'scipy']

def install_missing_libraries():
    for lib in required_libraries:
        try:
            if lib == 'pyaudio':
                import pyaudio
            elif lib == 'librosa':
                import librosa
            else:
                __import__(lib)
        except ImportError:
            print(f"[시스템] {lib} 라이브러리가 없습니다. 자동 설치를 시작합니다...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                print(f"[시스템] {lib} 설치 완료!")
            except Exception as e:
                print(f"[오류] {lib} 설치 중 문제가 발생했습니다: {e}")

install_missing_libraries()

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import pyaudio
import numpy as np
from scipy.fftpack import fft
import librosa
import threading
import json
import logging
import time
from collections import Counter

logging.basicConfig(filename='app.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

class PianoChordApp:
    def __init__(self, window):
        self.window = window
        self.window.title("화음 탐색기")
        self.window.geometry("520x800")
        self.window.configure(bg="#F8F9FA")
        
        self.RATE = 44100  
        self.CHUNK = 1024 * 16 
        self.is_running = False
        self.is_mic_on = False
        self.chord_history = [] 
        self.history_limit = 5  
        self.vote_threshold = 2 
        
        self.NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        self.raw_codes = self.load_code_list()
        self.chord_profiles = self.preprocess_chords()
        
        self.setup_ui(window)

    def setup_ui(self, window):
        header_frame = tk.Frame(window, bg="#343A40", height=80)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="화음 실시간 분석 시스템", font=("맑은 고딕", 18, "bold"), 
                 fg="white", bg="#343A40").pack(pady=20)


        file_info_frame = tk.LabelFrame(window, text=" 파일 재생 정보 ", bg="#F8F9FA", font=("맑은 고딕", 9))
        file_info_frame.pack(fill="x", padx=20, pady=15)

        self.time_label = tk.Label(file_info_frame, text="00:00 / 00:00", font=("Consolas", 11), bg="#F8F9FA")
        self.time_label.pack(pady=(5, 0))

        self.file_progress = ttk.Progressbar(file_info_frame, orient="horizontal", length=400, mode="determinate")
        self.file_progress.pack(pady=10, padx=20)

        vol_frame = tk.Frame(window, bg="#F8F9FA")
        vol_frame.pack(fill="x", padx=20)
        tk.Label(vol_frame, text="[ 입력 신호 강도 ]", font=("맑은 고딕", 9, "bold"), bg="#F8F9FA", fg="#495057").pack(anchor="w", padx=20)
        
        style = ttk.Style()
        style.configure("Vertical.TProgressbar", thickness=10)
        self.volume_bar = ttk.Progressbar(vol_frame, orient="horizontal", length=440, mode="determinate")
        self.volume_bar.pack(pady=5, padx=20)

        result_frame = tk.Frame(window, bg="white", highlightbackground="#DEE2E6", highlightthickness=1)
        result_frame.pack(fill="both", expand=True, padx=20, pady=10)

        tk.Label(result_frame, text="현재 감지된 음", font=("맑은 고딕", 10), bg="white", fg="#6C757D").pack(pady=(15, 0))
        self.notes_display = tk.Label(result_frame, text="-", font=("Arial", 55, "bold"), fg="#FD7E14", bg="white")
        self.notes_display.pack()

        tk.Label(result_frame, text="유력한 화음", font=("맑은 고딕", 10), bg="white", fg="#6C757D").pack(pady=(10, 0))
        self.chord_display = tk.Label(result_frame, text="대기 중", font=("Arial", 35, "bold"), fg="#212529", bg="white")
        self.chord_display.pack(pady=5)

        self.info_display = tk.Label(result_frame, text="분석 대기 중...", font=("맑은 고딕", 9), bg="white", fg="#ADB5BD")
        self.info_display.pack(pady=15)

        btn_frame = tk.Frame(window, bg="#F8F9FA")
        btn_frame.pack(fill="x", pady=20)

        self.btn_mic = tk.Button(btn_frame, text="🎙 마이크 입력", command=self.toggle_mic, 
                                 bg="#6C757D", fg="white", font=("맑은 고딕", 10, "bold"), width=15, relief="flat")
        self.btn_mic.grid(row=0, column=0, padx=20, sticky="e")
        
        self.btn_file = tk.Button(btn_frame, text="📂 파일 불러오기", command=self.open_file_dialog, 
                                  bg="#007BFF", fg="white", font=("맑은 고딕", 10, "bold"), width=15, relief="flat")
        self.btn_file.grid(row=0, column=1, padx=20, sticky="w")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self.btn_stop = tk.Button(window, text="⏹ 시스템 중지", command=self.stop_analysis, 
                                  bg="#DC3545", fg="white", font=("맑은 고딕", 11, "bold"), width=40, height=2, relief="flat")
        self.btn_stop.pack(pady=(0, 30))

    def load_code_list(self):
        try:
            with open('codes.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return {}

    def freq_to_note_index(self, freq):
        if freq < 60 or freq > 5000: return None
        h = 12 * np.log2(freq / 440.0) + 69
        return int(round(h)) % 12

    def preprocess_chords(self):
        profiles = {}
        for name, freqs in self.raw_codes.items():
            indices = set()
            for f in freqs:
                idx = self.freq_to_note_index(f)
                if idx is not None: indices.add(idx)
            if indices: profiles[name] = indices
        return profiles

    def get_stable_chord(self, new_chord):
        if new_chord != "...":
            self.chord_history.append(new_chord)
        if len(self.chord_history) > self.history_limit:
            self.chord_history.pop(0)
        if not self.chord_history: return "..."
        counts = Counter(self.chord_history)
        most_common = counts.most_common(1)[0]
        return most_common[0] if most_common[1] >= self.vote_threshold else "분석 중"

    def process_core(self, chunk):
        yf = fft(chunk)
        xf = np.linspace(0.0, self.RATE / 2.0, self.CHUNK // 2)
        mags = 2.0 / self.CHUNK * np.abs(yf[:self.CHUNK // 2])
        max_m = np.max(mags)
        chroma = np.zeros(12)
        if max_m > 0.004: 
            indices = np.argpartition(mags, -120)[-120:]
            for idx in indices:
                if mags[idx] > max_m * 0.08:
                    n_idx = self.freq_to_note_index(xf[idx])
                    if n_idx is not None: chroma[n_idx] += mags[idx]

        detected_indices = [i for i in range(12) if chroma[i] > np.mean(chroma) * 1.02]
        sorted_indices = sorted(detected_indices, key=lambda i: chroma[i], reverse=True)
        top_notes = [self.NOTE_NAMES[i] for i in sorted_indices[:3]]
        
        current_best, max_score = "...", 0
        detected_set = set(detected_indices)
        if detected_set:
            for name, target_indices in self.chord_profiles.items():
                matches = detected_set.intersection(target_indices)
                score = len(matches) * 25 
                if list(target_indices)[0] in detected_set: score += 15
                if score > max_score: max_score, current_best = score, name
        
        return top_notes, self.get_stable_chord(current_best), max_score

    def open_file_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav")])
        if path:
            self.chord_history = []
            self.is_running = True
            threading.Thread(target=self.file_play_and_analyze, args=(path,), daemon=True).start()

    def file_play_and_analyze(self, path):
        p = pyaudio.PyAudio()
        try:
            y, sr = librosa.load(path, sr=self.RATE)
            duration = len(y) / self.RATE
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=self.RATE, output=True)
            for i in range(0, len(y) - self.CHUNK, self.CHUNK):
                if not self.is_running: break
                start_t = time.time()
                chunk = y[i : i + self.CHUNK]
                stream.write(chunk.astype(np.float32).tobytes())
                notes, chord, score = self.process_core(chunk)
                current_t = i / self.RATE
                vol = min(100, np.sqrt(np.mean(chunk**2)) * 800) 
                time_str = f"{int(current_t//60):02}:{int(current_t%60):02} / {int(duration//60):02}:{int(duration%60):02}"
                progress = (current_t / duration) * 100
                self.window.after(0, self.update_ui, notes, chord, score, time.time()-start_t, vol, time_str, progress)
            stream.stop_stream()
            stream.close()
        except: pass
        finally:
            p.terminate()
            self.is_running = False

    def toggle_mic(self):
        if not self.is_mic_on:
            self.chord_history = []
            self.is_mic_on = True
            self.btn_mic.config(text="🎙 마이크 ON", bg="#28A745")
            threading.Thread(target=self.mic_thread_logic, daemon=True).start()
        else:
            self.is_mic_on = False
            self.btn_mic.config(text="🎙 마이크 입력", bg="#6C757D")

    def mic_thread_logic(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
            while self.is_mic_on:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if not self.is_running:
                    notes, chord, score = self.process_core(chunk)
                    vol = min(100, np.sqrt(np.mean(chunk**2)) * 800)
                    self.window.after(0, self.update_ui, notes, chord, score, 0, vol, "실시간 마이크 분석", 0)
            stream.stop_stream()
            stream.close()
        except: pass
        finally: p.terminate()

    def update_ui(self, notes, chord, score, p_time, vol, time_info, progress):
        self.volume_bar['value'] = vol
        self.file_progress['value'] = progress
        self.time_label.config(text=time_info)
        
        self.notes_display.config(text=", ".join(notes) if notes else "-")
        if chord not in ["...", "분석 중"]:
            self.chord_display.config(text=chord.replace("_", " "))
            self.info_display.config(text=f"신뢰 지수: {score} | 분석 시간: {p_time:.3f}s")
        else:
            self.chord_display.config(text=chord)

    def stop_analysis(self):
        self.is_running = False
        self.is_mic_on = False
        self.chord_history = []
        self.file_progress['value'] = 0
        self.volume_bar['value'] = 0
        self.chord_display.config(text="대기 중")
        self.btn_mic.config(text="🎙 마이크 입력", bg="#6C757D")

if __name__ == "__main__":
    root = tk.Tk()
    app = PianoChordApp(root)
    root.mainloop()