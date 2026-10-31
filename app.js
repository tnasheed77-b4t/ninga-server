const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const urlInput = document.getElementById('urlInput');
const resultLabel = document.getElementById('resultLabel');
const logoImg = document.getElementById('logoImg');
const fallback = document.getElementById('fallback');

// Handle logo fallback cleanly without inline HTML onerror (required by Chrome Extension CSP)
logoImg.addEventListener('error', () => {
  logoImg.style.display = 'none';
  fallback.style.display = 'block';
});

// Toggle Elements (WAV vs MP3)
const modeWav = document.getElementById('modeWav');
const modeMp3 = document.getElementById('modeMp3');
let selectedFormat = 'wav';

modeWav.addEventListener('click', (e) => {
  e.stopPropagation();
  selectedFormat = 'wav';
  modeWav.classList.add('active');
  modeMp3.classList.remove('active');
});

modeMp3.addEventListener('click', (e) => {
  e.stopPropagation();
  selectedFormat = 'mp3';
  modeMp3.classList.add('active');
  modeWav.classList.remove('active');
});

// Prevent clicking inside the "Paste Link Here" box from opening the local file dialog
urlInput.addEventListener('click', (e) => e.stopPropagation());

// Helper to switch between the "Paste Link Here" input and the BPM/Key status display
function showStatus(text, allowClickToReset = false) {
  urlInput.style.display = 'none';
  resultLabel.style.display = 'block';
  resultLabel.innerText = text;
  if (allowClickToReset) {
    resultLabel.onclick = (e) => {
      e.stopPropagation();
      resetToPasteInput();
    };
  } else {
    resultLabel.onclick = null;
  }
}

function resetToPasteInput() {
  urlInput.value = '';
  resultLabel.style.display = 'none';
  urlInput.style.display = 'block';
}

// --- 1. YT-DLP + FFMPEG LINK DOWNLOAD & AUTO BPM/KEY ---
urlInput.addEventListener('paste', () => {
  setTimeout(() => {
    const url = urlInput.value.trim();
    if (url) processLink(url);
  }, 50);
});

urlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    const url = urlInput.value.trim();
    if (url) processLink(url);
  }
});

async function processLink(url) {
  showStatus(`Downloading ${selectedFormat.toUpperCase()} & Analyzing... ⏳`);

  const formData = new FormData();
  formData.append('url', url);
  formData.append('target_format', selectedFormat);

  try {
    const response = await fetch('https://ninga-audio-api.onrender.com/download_url', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Could not download link');

    const bpm = response.headers.get('X-BPM') || '';
    const key = response.headers.get('X-Key') || '';
    const title = response.headers.get('X-Title') || 'ninga_audio';

    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = downloadUrl;
    a.download = `${title}.${selectedFormat}`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(downloadUrl);

    const displayResult = (bpm && key) ? `${bpm}  |  ${key}` : (bpm || '✅ Download Complete!');
    showStatus(displayResult, true);

  } catch (err) {
    console.error(err);
    showStatus('❌ Error downloading link (Click to retry)', true);
  }
}

// --- 2. LOCAL FILE DRAG & DROP / CLICK SUPPORT (STILL WORKS ON WEB APP) ---
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

['dragleave', 'dragend'].forEach(eventType => {
  dropZone.addEventListener(eventType, () => dropZone.classList.remove('drag-over'));
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
  const fileName = file.name.toLowerCase();
  if (fileName.endsWith('.mp4') || fileName.endsWith('.mov')) {
    await runConverter(file);
  } else {
    await runAnalyzer(file);
  }
  fileInput.value = '';
}

async function runAnalyzer(file) {
  const validExtensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'];
  const fileName = file.name.toLowerCase();
  const isValid = validExtensions.some(ext => fileName.endsWith(ext));

  if (!isValid) {
    showStatus('Unsupported audio format!', true);
    return;
  }

  showStatus('Extracting first 15s...');

  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const sampleRate = audioBuffer.sampleRate;
    const durationSamples = Math.min(15 * sampleRate, audioBuffer.length);
    
    const offlineCtx = new OfflineAudioContext(1, durationSamples, 22050);
    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start(0, 0, durationSamples / sampleRate);
    
    const renderedBuffer = await offlineCtx.startRendering();
    const wavBlob = audioBufferToWavBlob(renderedBuffer);

    showStatus('Analyzing audio...');
    
    const formData = new FormData();
    formData.append('file', wavBlob, 'snippet.wav');

    const response = await fetch('https://ninga-audio-api.onrender.com/analyze', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Server returned an error');

    const data = await response.json();
    const displayResult = data.key ? `${data.bpm}  |  ${data.key}` : `${data.bpm}`;
    showStatus(displayResult, true);
    
  } catch (err) {
    console.error(err);
    showStatus('Error processing file (Click to reset)', true);
  }
}

async function runConverter(file) {
  const maxSize = 95 * 1024 * 1024; 
  if (file.size > maxSize) {
    showStatus('❌ File too large (Max 95MB)', true);
    return;
  }

  showStatus('Preparing file... ⏳');

  try {
    const fileBuffer = await file.arrayBuffer();
    const safeBlob = new Blob([fileBuffer], { type: file.type || 'application/octet-stream' });
    const safeName = file.name || 'upload.mp4';

    showStatus(`Uploading & Converting to ${selectedFormat.toUpperCase()}... ⏳`);

    const formData = new FormData();
    formData.append('file', safeBlob, safeName);
    formData.append('target_format', selectedFormat); 

    const response = await fetch('https://ninga-audio-api.onrender.com/convert', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Server rejected the conversion.');

    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = downloadUrl;
    
    const baseName = safeName.substring(0, safeName.lastIndexOf('.')) || safeName;
    a.download = `${baseName}_HQ.${selectedFormat}`;
    
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(downloadUrl);
    
    showStatus('✅ Download Complete! (Click to reset)', true);

  } catch (error) {
    console.error(error);
    showStatus('❌ ' + (error.message || 'Error converting file'), true);
  }
}

// --- 3. ASK NINGA SEARCH LOGIC ---
const searchContainer = document.getElementById('searchContainer');
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchResult = document.getElementById('searchResult');

searchContainer.addEventListener('click', (e) => e.stopPropagation());

async function executeSearch() {
  const query = searchInput.value.trim();
  if (!query) return;

  searchResult.innerText = "Asking Ninga... ⏳";
  searchBtn.disabled = true;

  const formData = new FormData();
  formData.append('query', query);

  try {
    const res = await fetch('https://ninga-audio-api.onrender.com/ask', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Server returned an error');

    const data = await res.json();
    searchResult.innerText = data.answer || "No response received.";
  } catch (err) {
    console.error(err);
    searchResult.innerText = "❌ Search request failed. Check server connection.";
  } finally {
    searchBtn.disabled = false;
  }
}

searchBtn.addEventListener('click', executeSearch);

searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    executeSearch();
  }
});

// WAV Blob helper
function audioBufferToWavBlob(buffer) {
  const numOfChan = buffer.numberOfChannels;
  const length = buffer.length * numOfChan * 2 + 44;
  const out = new DataView(new ArrayBuffer(length));
  let channels = [], sample, offset = 0, pos = 0;

  function setUint16(data) { out.setUint16(pos, data, true); pos += 2; }
  function setUint32(data) { out.setUint32(pos, data, true); pos += 4; }

  setUint32(0x46464952); setUint32(length - 8); setUint32(0x45564157);
  setUint32(0x20746d66); setUint32(16); setUint16(1); setUint16(numOfChan);
  setUint32(buffer.sampleRate); setUint32(buffer.sampleRate * 2 * numOfChan);
  setUint16(numOfChan * 2); setUint16(16); setUint32(0x61746164); setUint32(length - pos - 4);

  for (let i = 0; i < buffer.numberOfChannels; i++) channels.push(buffer.getChannelData(i));

  while (offset < buffer.length) {
    for (let i = 0; i < numOfChan; i++) {
      sample = Math.max(-1, Math.min(1, channels[i][offset]));
      sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
      out.setInt16(pos, sample, true); pos += 2;
    }
    offset++;
  }
  return new Blob([out.buffer], { type: 'audio/wav' });
}

// Register Service Worker only on HTTP/HTTPS web app (not inside Chrome Extension)
if ('serviceWorker' in navigator && window.location.protocol.startsWith('http')) {
  navigator.serviceWorker.register('sw.js');
}