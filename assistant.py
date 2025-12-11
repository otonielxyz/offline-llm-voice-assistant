"""
assistant.py
Author: Otoniel Torres Bernal

Local voice assistant that runs entirely on-device using:
  - Picovoice Porcupine for wake-word detection
  - Vosk for speech-to-text (STT)
  - A local LLM server (OpenAI-compatible API, e.g. LM Studio) for responses
  - Coqui TTS (XTTS v2) for text-to-speech

Configuration is loaded from config.json (see config.example.json).
"""

import json
import random
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import scipy.io.wavfile as wavfile
import torch
import pvporcupine
from pvrecorder import PvRecorder
from vosk import KaldiRecognizer, Model
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play


# ───────────────── CONFIG LOADING ────────────────────────────────

CONFIG_PATH = Path("config.json")


def load_config() -> dict:
    """
    Load configuration from config.json.

    Copy config.example.json → config.json and fill in your own values.
    """
    if not CONFIG_PATH.is_file():
        raise SystemExit(
            "Missing config.json. Copy config.example.json to config.json "
            "and fill in your Picovoice key, model paths, and LLM settings."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


config = load_config()

BASE_URL = config.get("llm_url", "http://127.0.0.1:1234").rstrip("/")
MODEL_NAME = config.get("llm_model", "mythomax-12-13b")

EXIT_PHRASE = config.get("exit_phrase", "bye sol").lower().strip()
WAKE_PHRASE = config.get("wake_phrase", "hey sol").lower().strip()
FOLLOWUP_TIMEOUT = float(config.get("followup_timeout", 8.0))

ACCESS_KEY = config.get("picovoice_access_key", "").strip()
WAKE_WORD_PATH = Path(config.get("wake_word_path", "wake_words/hey_sol.ppn"))
VOSK_MODEL_PATH = Path(
    config.get("vosk_model_path", "models/vosk-model-small-en-us-0.15")
)
SPEAKER_WAV_PATH = Path(
    config.get("speaker_wav", "voice_data/clips/example_voice.wav")
)

if not ACCESS_KEY:
    print("[ERROR] picovoice_access_key not set in config.json")
    sys.exit(1)

if not WAKE_WORD_PATH.is_file():
    print(f"[ERROR] Wake-word file not found: {WAKE_WORD_PATH}")
    sys.exit(1)

if not VOSK_MODEL_PATH.exists():
    print(f"[ERROR] Vosk model missing at {VOSK_MODEL_PATH}")
    sys.exit(1)

# ───────────────── ENGINE INITIALIZATION ─────────────────────────

# Porcupine wake-word
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=[str(WAKE_WORD_PATH)],
    sensitivities=[0.5],
)
recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
recorder.start()

# Vosk STT model
vosk_model = Model(str(VOSK_MODEL_PATH))

# Coqui TTS
tts_engine = TTS("xtts_v2", gpu=torch.cuda.is_available())

# ───────────────── GRACEFUL SHUTDOWN ─────────────────────────────


def cleanup_and_exit(signum=None, frame=None):
    """Stop audio resources and exit cleanly."""
    try:
        recorder.stop()
        recorder.delete()
        porcupine.delete()
    except Exception:
        pass
    print("\nExiting.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

# ───────────────── PRE-ROLL AUDIO BUFFER ─────────────────────────

FS = 16000
PRE_ROLL_SECONDS = 0.5
PRE_ROLL_CHUNK_DURATION = 0.1  # seconds
PRE_ROLL_CHUNK_SIZE = int(FS * PRE_ROLL_CHUNK_DURATION)
pre_roll = deque(maxlen=int(PRE_ROLL_SECONDS / PRE_ROLL_CHUNK_DURATION))


def fill_pre_roll():
    """Continuously capture short audio chunks into a rolling buffer."""
    stream = sd.InputStream(samplerate=FS, channels=1, dtype="int16")
    with stream:
        while True:
            data, _ = stream.read(PRE_ROLL_CHUNK_SIZE)
            pre_roll.append(data)


threading.Thread(target=fill_pre_roll, daemon=True).start()

# ───────────────── RECORDING ─────────────────────────────────────


def record_audio(
    fs=16000,
    filename="speech.wav",
    silence_threshold=500,
    trailing_silence=2.0,
    chunk_duration=0.2,
    min_speech=0.5,
    max_utterance=60.0,
    prepend=None,
):
    """
    Record microphone input until sustained silence is detected.

    - Starts from scratch or with optional `prepend` audio (pre-roll).
    - Stops after `trailing_silence` seconds of low volume (after min_speech).
    - Caps recording at `max_utterance` seconds.
    """
    chunk_size = int(fs * chunk_duration)
    buffer = []
    if prepend:
        buffer.extend(prepend)
    stream = sd.InputStream(samplerate=fs, channels=1, dtype="int16")
    silence_chunks = 0
    total_time = 0.0

    with stream:
        while True:
            data, _ = stream.read(chunk_size)
            buffer.append(data)
            total_time += chunk_duration
            level = np.abs(data).mean()

            if total_time >= min_speech:
                if level < silence_threshold:
                    silence_chunks += 1
                else:
                    silence_chunks = 0

                if silence_chunks * chunk_duration >= trailing_silence:
                    # small tail padding to avoid clipping last word
                    extra, _ = stream.read(chunk_size)
                    buffer.append(extra)
                    break

            if total_time >= max_utterance:
                break

    audio = np.concatenate(buffer, axis=0)
    wavfile.write(filename, fs, audio)
    return filename


# ───────────────── SPEECH-TO-TEXT (VOSK) ─────────────────────────


def transcribe_audio(filename="speech.wav"):
    rec = KaldiRecognizer(vosk_model, 16000)
    rec.SetWords(False)
    try:
        fs, data = wavfile.read(filename)
    except Exception as e:
        print(f"[STT] failed to read {filename}: {e}")
        return ""
    if fs != 16000:
        raise RuntimeError(f"[STT] Expected 16 kHz WAV, got {fs} Hz")

    text = ""
    step = 4000
    for i in range(0, len(data), step):
        chunk = data[i : i + step].tobytes()
        if rec.AcceptWaveform(chunk):
            res = json.loads(rec.Result())
            text += " " + res.get("text", "")
    final = json.loads(rec.FinalResult())
    text += " " + final.get("text", "")
    return text.strip()


# ───────────────── LLM BACKEND ───────────────────────────────────

conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]


def query_llm(prompt, retries=2):
    """Send conversation + new user prompt to the local LLM server."""
    conversation_history.append({"role": "user", "content": prompt})
    endpoint = f"{BASE_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": conversation_history}

    result = ""
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            content = data.get("choices", [])[0].get("message", {}).get("content", "")
            if content:
                conversation_history.append({"role": "assistant", "content": content})
                result = content
                break
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error to LLM at {BASE_URL}. Is it running?")
        except requests.exceptions.HTTPError as e:
            print(f"[LLM] HTTP error (attempt {attempt}): {e}")
        except Exception as e:
            print(f"[LLM] unexpected error: {e}")
        time.sleep(1)

    return result


# ───────────────── TEXT-TO-SPEECH ────────────────────────────────


def speak(text):
    """Convert text to speech using XTTS v2 and play the audio."""
    if not text:
        return
    try:
        tts_engine.tts_to_file(
            text=text,
            file_path="response.wav",
            speaker_wav=str(SPEAKER_WAV_PATH),
            language="en",
        )
        audio = AudioSegment.from_file("response.wav", format="wav")
        play(audio)
    except Exception as e:
        print(f"[TTS] playback failed: {e}")


# ───────────────── FOLLOW-UP HANDLING ────────────────────────────

CANCEL_PHRASES = ["never mind", "cancel that", "forget it", "stop", "scratch that"]


def listen_for_followup():
    """
    Listen once for a follow-up utterance (shorter silence threshold).
    Returns the transcribed text, or "" if nothing useful captured.
    """
    fname = record_audio(
        silence_threshold=500,
        trailing_silence=2.0,
        chunk_duration=0.2,
        min_speech=0.3,
        max_utterance=30.0,
        prepend=None,
    )
    user_text = transcribe_audio(fname)
    if not user_text:
        return ""
    lower = user_text.lower().strip()
    if any(phrase in lower for phrase in CANCEL_PHRASES):
        print("[Info] Cancel phrase detected during follow-up; dropping.")
        return ""
    return user_text


# ───────────────── MAIN LOOP ─────────────────────────────────────


def main():
    print(
        f"Assistant ready. Say '{WAKE_PHRASE}' to begin, and '{EXIT_PHRASE}' to exit.\n"
        f"Using LLM at {BASE_URL} (model {MODEL_NAME})."
    )

    while True:
        # Wait for wake word
        pcm = recorder.read()
        if porcupine.process(pcm) < 0:
            continue

        ACKS = [
            "Yes?",
            "How can I help?",
            "What do you need?",
            "Listening...",
            "I'm here.",
            "Ready for your command.",
        ]

        def acknowledge():
            speak(random.choice(ACKS))
            time.sleep(0.2)
            print("Listening...")

        print("✅ Wake-word detected.")
        acknowledge()

        # Pre-roll copy so buffer isn't mutated mid-record
        pre = list(pre_roll)

        # Record user utterance
        fname = record_audio(
            silence_threshold=500,
            trailing_silence=2.5,
            chunk_duration=0.2,
            min_speech=0.4,
            max_utterance=60.0,
            prepend=pre,
        )
        user_text = transcribe_audio(fname)
        if not user_text:
            print(f"No speech detected. Say '{WAKE_PHRASE}' to start again.")
            continue
        print(f"You said: {user_text}")

        lower = user_text.lower().strip()
        if lower == EXIT_PHRASE:
            print("Goodbye!")
            cleanup_and_exit()

        if any(phrase in lower for phrase in CANCEL_PHRASES):
            print("[Info] Cancel phrase received; returning to idle.")
            continue

        # Query LLM
        response = query_llm(user_text)
        if not response:
            print("[LLM] No response received. Try again.")
            continue
        print(f"Assistant: {response}")
        speak(response)

        # Follow-up window (no wake word required)
        print(
            f"(You can follow up within the next {FOLLOWUP_TIMEOUT:.1f}s "
            f"without saying '{WAKE_PHRASE}')"
        )
        start = time.time()
        while time.time() - start < FOLLOWUP_TIMEOUT:
            time.sleep(0.1)
            print("Listening for follow-up (or say exit)...")
            followup = listen_for_followup()
            if not followup:
                break

            print(f"Follow-up heard: {followup}")
            lower_f = followup.lower().strip()

            if lower_f == EXIT_PHRASE:
                print("Goodbye!")
                cleanup_and_exit()

            if any(phrase in lower_f for phrase in CANCEL_PHRASES):
                print("[Info] Cancel phrase in follow-up; returning to idle.")
                break

            response = query_llm(followup)
            if response:
                print(f"Assistant: {response}")
                speak(response)
                start = time.time()  # extend follow-up window
            else:
                print("[LLM] No response to follow-up; returning to idle.")
                break

        print("Returning to idle. Say the wake phrase to interact again.")


if __name__ == "__main__":
    main()
