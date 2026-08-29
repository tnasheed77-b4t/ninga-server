import io
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BPM & Key Detection API")

# Enable CORS so your web frontend (Cloudflare Pages / Canva Embed)
# can freely send requests to this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Krumhansl-Schmuckler profiles for accurate key estimation
MAJOR_PROFILE = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
]
MINOR_PROFILE = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 2.69, 3.34, 3.17, 3.18,
]
PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def analyze_audio_buffer(file_bytes: bytes):
    """Core audio analysis algorithm using Librosa and Krumhansl-Schmuckler profiles."""
    # Load audio bytes directly into librosa using io.BytesIO
    audio_stream = io.BytesIO(file_bytes)
    
    # PERFORMANCE TWEAK: Downsample to 22050Hz to drastically speed up 
    # librosa.beat.beat_track and chroma feature extraction on large files.
    y, sr = librosa.load(audio_stream, sr=22050)

    # 1. Detect BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm_val = float(np.mean(tempo))
    bpm_str = f'{round(bpm_val)} BPM'

    # 2. Detect Key (Major/Minor)
    y_harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_vals = np.sum(chroma, axis=1)

    # Prevent division by zero if audio is silent
    sum_chroma = np.sum(chroma_vals)
    if sum_chroma > 0:
        chroma_vals = chroma_vals / sum_chroma
    else:
        chroma_vals = np.zeros_like(chroma_vals)

    best_corr = -1
    detected_key = 'Unknown'

    for i in range(12):
        maj_shift = np.roll(MAJOR_PROFILE, i)
        min_shift = np.roll(MINOR_PROFILE, i)

        # Use safe correlation calculation in case of flat arrays
        maj_corr = np.corrcoef(chroma_vals, maj_shift)[0, 1]
        min_corr = np.corrcoef(chroma_vals, min_shift)[0, 1]

        # Handle potential NaN values resulting from flat/silent signal correlation
        if not np.isnan(maj_corr) and maj_corr > best_corr:
            best_corr = maj_corr
            detected_key = f'{PITCHES[i]} Major'
        if not np.isnan(min_corr) and min_corr > best_corr:
            best_corr = min_corr
            detected_key = f'{PITCHES[i]} Minor'

    return bpm_str, detected_key


@app.get("/")
def health_check():
    return {"status": "API is online"}


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    # Validate extension
    valid_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload .wav, .mp3, .flac, .ogg, or .m4a"
        )

    try:
        # Read uploaded file into memory
        contents = await file.read()
        
        # Run your analysis
        bpm, key = analyze_audio_buffer(contents)
        
        return {
            "success": True,
            "filename": file.filename,
            "bpm": bpm,
            "key": key
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")