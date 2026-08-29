import io
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.34, 3.17, 3.18]
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_stream = io.BytesIO(contents)

        # Load pre-sliced 15-second snippet instantly
        y, sr = librosa.load(audio_stream, sr=22050)

        # 1. Fast BPM Detection
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm_val = float(np.mean(tempo))
        bpm_str = f"{round(bpm_val)} BPM"

        # 2. Fast Key Detection using Chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_vals = np.sum(chroma, axis=1)

        if np.sum(chroma_vals) > 0:
            chroma_vals = chroma_vals / np.sum(chroma_vals)

        best_corr = -1
        detected_key = "Unknown"

        for i in range(12):
            maj_shift = np.roll(MAJOR_PROFILE, i)
            min_shift = np.roll(MINOR_PROFILE, i)

            maj_corr = np.corrcoef(chroma_vals, maj_shift)[0, 1]
            min_corr = np.corrcoef(chroma_vals, min_shift)[0, 1]

            if maj_corr > best_corr:
                best_corr = maj_corr
                detected_key = f"{PITCHES[i]} Major"
            if min_corr > best_corr:
                best_corr = min_corr
                detected_key = f"{PITCHES[i]} Minor"

        return {"bpm": bpm_str, "key": detected_key}

    except Exception as e:
        print("Error processing audio:", e)
        return {"bpm": "Error processing file", "key": ""}