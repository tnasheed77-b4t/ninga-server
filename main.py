import io
import os
import shutil
import subprocess
import tempfile
import asyncio
import librosa
import numpy as np
import yt_dlp  # type: ignore
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI()

# Expose custom headers (X-BPM, X-Key, X-Title) so the Web App & Chrome Extension can read them
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-BPM", "X-Key", "X-Title", "Content-Disposition"],
)

MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.34, 3.17, 3.18]
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Initialize the Gemini client (automatically finds GEMINI_API_KEY environment variable)
client = genai.Client()

def cleanup_temp_files(*paths):
    """Deletes temporary files or folders from the server after the download is sent."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
            except OSError:
                pass

def detect_bpm_and_key(audio_source):
    """Runs your 15-second BPM & Key detection on a file path or byte stream."""
    y, sr = librosa.load(audio_source, sr=22050, offset=0, duration=15)

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

    return bpm_str, detected_key

@app.get("/")
def health_check():
    return {"status": "ok"}

# --- 1. GEMINI SEARCH ENDPOINT ---
@app.post("/ask")
async def ask_endpoint(query: str = Form(...)):
    """Sends a user's search query to Gemini grounded with live Google Search."""
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        
        if not response.text:
            return {"answer": "Search returned an empty response. Try a different query."}
            
        return {"answer": response.text}
    except Exception as e:
        print(f"Google Search Error: {e}")
        return {"answer": f"Search Error: {str(e)}"}

# --- 2. YT-DLP + FFMPEG LINK DOWNLOADER & AUTO BPM/KEY ENDPOINT ---
@app.post("/download_url")
async def download_url_endpoint(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    target_format: str = Form("wav")
):
    temp_dir = tempfile.mkdtemp()
    try:
        fmt = "wav" if target_format.lower().strip() == "wav" else "mp3"
        out_tmpl = os.path.join(temp_dir, "audio_download.%(ext)s")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': out_tmpl,
            'noplaylist': True,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': fmt,
                'preferredquality': '320',
            }],
        }

        def download_and_extract():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return info.get('title', 'ninga_audio')

        title = await asyncio.to_thread(download_and_extract)
        final_file = os.path.join(temp_dir, f"audio_download.{fmt}")

        # Automatically detect BPM & Key from the converted audio file
        bpm_str, detected_key = await asyncio.to_thread(detect_bpm_and_key, final_file)

        # Build clean ASCII filename for download headers
        safe_title = "".join(c for c in title if c.isascii() and (c.isalnum() or c in (' ', '-', '_'))).strip() or "ninga_track"
        out_filename = f"{safe_title}.{fmt}"
        media_type = "audio/wav" if fmt == "wav" else "audio/mpeg"

        background_tasks.add_task(cleanup_temp_files, temp_dir)

        headers = {
            "X-BPM": bpm_str,
            "X-Key": detected_key,
            "X-Title": safe_title
        }
        return FileResponse(final_file, media_type=media_type, filename=out_filename, headers=headers)

    except Exception as e:
        print("yt-dlp download error:", e)
        cleanup_temp_files(temp_dir)
        return JSONResponse(status_code=400, content={"error": f"Failed to download link: {str(e)}"})

# --- 3. BPM & KEY ANALYZER ENDPOINT (For Drag & Drop Audio Files) ---
@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_stream = io.BytesIO(contents)
        bpm_str, detected_key = await asyncio.to_thread(detect_bpm_and_key, audio_stream)
        return {"bpm": bpm_str, "key": detected_key}

    except Exception as e:
        print("Error processing audio:", e)
        return {"bpm": "Error processing file", "key": ""}

# --- 4. LOCAL FILE CONVERTER ENDPOINT (Supports both WAV and MP3) ---
@app.post("/convert")
async def convert_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    target_format: str = Form("mp3") 
):
    temp_in = None
    temp_out = None
    try:
        fmt = "wav" if target_format.lower().strip() == "wav" else "mp3"
        contents = await file.read()
        in_suffix = os.path.splitext(file.filename)[1] or ".tmp"
        
        fd_in, temp_in = tempfile.mkstemp(suffix=in_suffix)
        fd_out, temp_out = tempfile.mkstemp(suffix=f".{fmt}")
        
        os.close(fd_in)
        os.close(fd_out)
        
        with open(temp_in, "wb") as f:
            f.write(contents)
            
        command = ["ffmpeg", "-y", "-i", temp_in, "-vn"]
        if fmt == "mp3":
            command.extend(["-b:a", "320k"])
        command.append(temp_out)
        
        await asyncio.to_thread(subprocess.run, command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
        background_tasks.add_task(cleanup_temp_files, temp_in, temp_out)
        
        out_filename = f"{os.path.splitext(file.filename)[0]}_HQ.{fmt}"
        media_type = "audio/wav" if fmt == "wav" else "audio/mpeg"
        return FileResponse(temp_out, media_type=media_type, filename=out_filename)
        
    except Exception as e:
        print("Conversion error:", e)
        cleanup_temp_files(temp_in, temp_out)
        return {"error": "Failed to convert file"}