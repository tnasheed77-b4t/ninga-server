import io
import os
import subprocess
import tempfile
import asyncio
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

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

# Initialize the Gemini client (It will automatically find your GEMINI_API_KEY environment variable)
client = genai.Client()

def cleanup_temp_files(*filepaths):
    """Deletes temporary files from the server after the download is sent."""
    for path in filepaths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

@app.get("/")
def health_check():
    return {"status": "ok"}

# --- 1. GEMINI SEARCH ENDPOINT (Replaces DuckDuckGo) ---
@app.post("/ask")
async def ask_endpoint(query: str = Form(...)):
    """Sends a user's search query to Gemini grounded with live Google Search."""
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=query,
            config=types.GenerateContentConfig(
                # Built-in tool that allows Gemini to search the live web
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        
        if not response.text:
            return {"answer": "Search returned an empty response. Try a different query."}
            
        return {"answer": response.text}
    except Exception as e:
        print(f"Google Search Error: {e}")
        return {"answer": f"Search Error: {str(e)}"}

# --- 2. BPM & KEY ANALYZER ENDPOINT ---
@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_stream = io.BytesIO(contents)

        y, sr = librosa.load(audio_stream, sr=22050, offset=0, duration=15)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm_val = float(np.mean(tempo))
        bpm_str = f"{round(bpm_val)} BPM"

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

# --- 3. MP4 TO MP3 CONVERTER ENDPOINT ---
@app.post("/convert")
async def convert_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    target_format: str = Form("mp3") 
):
    temp_in = None
    temp_out = None
    try:
        contents = await file.read()
        in_suffix = os.path.splitext(file.filename)[1] or ".tmp"
        
        fd_in, temp_in = tempfile.mkstemp(suffix=in_suffix)
        fd_out, temp_out = tempfile.mkstemp(suffix=".mp3")
        
        os.close(fd_in)
        os.close(fd_out)
        
        with open(temp_in, "wb") as f:
            f.write(contents)
            
        command = [
            "ffmpeg", "-y", 
            "-i", temp_in, 
            "-vn", 
            "-b:a", "320k", 
            temp_out
        ]
        
        await asyncio.to_thread(subprocess.run, command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
        background_tasks.add_task(cleanup_temp_files, temp_in, temp_out)
        
        out_filename = f"{os.path.splitext(file.filename)[0]}_HQ.mp3"
        return FileResponse(temp_out, media_type="audio/mpeg", filename=out_filename)
        
    except Exception as e:
        print("Conversion error:", e)
        cleanup_temp_files(temp_in, temp_out)
        return {"error": "Failed to convert file"}