import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
import tempfile
tempfile.tempdir = TEMP_DIR

import gradio as gr
import subprocess
import sys
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import soundfile as sf
import noisereduce as nr
import csv
import re
import difflib
import json
import zipfile

SAVED_VOICES_DIR = os.path.join(BASE_DIR, "saved_voices")
os.makedirs(SAVED_VOICES_DIR, exist_ok=True)

# ─── Language Detection: Hindi/Urdu (native script + Roman/Latin) ───
# Unicode-range checks only catch text already typed in Devanagari/Arabic
# script. Most users type Hindi/Urdu phonetically in Latin letters
# ("kya haal hai"), which is plain ASCII and invisible to a script check —
# that's "Roman Hindi/Urdu" and needs its own detector.
ROMAN_HINDI_URDU_WORDS = {
    "kya", "kyu", "kyun", "kyon", "hai", "hain", "nahi", "nahin", "haan", "han",
    "tum", "tumhara", "tumhari", "aap", "mai", "main", "mera", "meri", "mujhe",
    "hum", "humein", "unka", "uska", "iska", "yeh", "woh", "wo", "kaise", "kaisa",
    "kaisi", "kahan", "kaha", "kab", "kaun", "kitna", "kitni", "acha", "accha",
    "theek", "thik", "bhai", "yaar", "dost", "pyar", "pyaar", "dil", "zindagi",
    "shukriya", "dhanyavaad", "namaste", "salaam", "bahut", "bohot", "bohat",
    "kuch", "kuchh", "abhi", "phir", "fir", "wala", "waala", "waali", "gaya",
    "gayi", "raha", "rahi", "rahe", "hoga", "hogi", "tha", "thi", "the", "aur",
    "lekin", "magar", "kyunki", "matlab", "samajh", "dekh", "dekho", "suno",
    "bolo", "bol", "jao", "aao", "aana", "jana", "chal", "chalo", "karo",
    "karna", "karte", "kar", "diya", "diyo", "de", "do", "liya", "loge",
    "acha", "sahi", "galat", "paisa", "paise", "ghar", "khana", "paani",
    "zindagi", "mohabbat", "ishq", "dosti", "bacha", "bache", "log", "logo",
    "khud", "apna", "apne", "apni", "sab", "sabhi", "koi", "kisi", "kuchhbhi",
}

def detect_hindi_urdu(text):
    """
    Detect whether `text` is Hindi/Urdu, and how it's written.

    Returns (is_hindi_or_urdu, mode) where mode is one of:
      "devanagari" - already in Hindi script (\\u0900-\\u097F)
      "arabic"     - already in Urdu/Arabic script (\\u0600-\\u06FF)
      "roman"      - phonetic Hindi/Urdu typed in Latin letters
      None         - not detected as Hindi/Urdu
    """
    if not text or not text.strip():
        return False, None

    if any('\u0900' <= c <= '\u097F' for c in text):
        return True, "devanagari"
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return True, "arabic"

    # Roman/Latin heuristic: tokenize and score against the common-word list.
    # Require BOTH a minimum hit count and a minimum ratio so a single
    # ambiguous word (e.g. "ka" as an abbreviation) doesn't false-positive
    # on ordinary English text.
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return False, None

    hits = sum(1 for w in words if w in ROMAN_HINDI_URDU_WORDS)
    ratio = hits / len(words)

    if hits >= 2 and ratio >= 0.15:
        return True, "roman"
    return False, None

def transliterate_roman_to_devanagari(text):
    """
    Best-effort Roman -> Devanagari transliteration so Roman Hindi/Urdu gets
    routed through the same native-script pronunciation pipeline as text
    already typed in Devanagari. Falls back to the original text if the
    transliteration library isn't available or fails on this input — the
    pipeline still runs, just without the pronunciation boost.
    """
    try:
        from transliterate import translit
        return translit(text, 'hi')
    except Exception:
        try:
            from transliterate import roman_to_devanagari
            return roman_to_devanagari(text)
        except Exception:
            return text

def _find_exe(name, windows_venv_relpath=None):
    """
    Locate a console-script executable. Inside Docker (Linux) these tools are
    installed straight onto PATH, so shutil.which() finds them. On a local
    Windows dev checkout using the old venv\\Scripts layout, fall back to that
    relative path so nothing breaks pre-Docker.
    """
    found = shutil.which(name)
    if found:
        return found
    if windows_venv_relpath:
        candidate = os.path.join(BASE_DIR, *windows_venv_relpath)
        if os.path.exists(candidate):
            return candidate
    # Last resort: return the bare command name so subprocess raises a clear,
    # readable FileNotFoundError instead of silently pointing at a dead path.
    return name

EDGE_TTS_EXE = _find_exe("edge-tts", ["venv", "Scripts", "edge-tts.exe"])
F5_TTS_EXE = _find_exe("f5-tts_infer-cli", ["venv", "Scripts", "f5-tts_infer-cli.exe"])
RVC_MODELS_DIR = os.path.join(BASE_DIR, "rvc_models")

_rvc_venv_python_nix = os.path.join(BASE_DIR, "rvc_venv", "bin", "python")
_rvc_venv_python_win = os.path.join(BASE_DIR, "rvc_venv", "Scripts", "python.exe")
if os.path.exists(_rvc_venv_python_nix):
    RVC_PYTHON_EXE = _rvc_venv_python_nix
elif os.path.exists(_rvc_venv_python_win):
    RVC_PYTHON_EXE = _rvc_venv_python_win
else:
    # No dedicated RVC venv found (e.g. single-environment Docker image) —
    # fall back to the interpreter running this app.
    RVC_PYTHON_EXE = sys.executable

RVC_INFER_SCRIPT = os.path.join(BASE_DIR, "rvc_infer.py")
os.makedirs(RVC_MODELS_DIR, exist_ok=True)

# ─── Utility: Run edge-tts via subprocess (avoids asyncio conflicts with Gradio) ───
def run_edge_tts(text, voice, output_path, rate=None, pitch=None):
    """
    Generate Microsoft Edge-TTS audio.
    Always return a clean WAV file.
    """

    mp3_path = output_path.replace(".wav", ".mp3")

    cmd = [
        EDGE_TTS_EXE,
        "--voice", voice,
        "--text", text,
        "--write-media", mp3_path
    ]

    if rate:
        cmd += ["--rate", rate]

    if pitch:
        cmd += ["--pitch", pitch]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        return False, result.stderr

    audio = AudioSegment.from_file(mp3_path)
    audio.export(output_path, format="wav")

    return True, None

# ─── Voice Library ───
def get_saved_voices():
    voices = []
    if os.path.exists(SAVED_VOICES_DIR):
        for d in sorted(os.listdir(SAVED_VOICES_DIR)):
            if os.path.isdir(os.path.join(SAVED_VOICES_DIR, d)):
                voices.append(d)
    return voices

def load_voice(name):
    if not name:
        return None, ""
    audio_path = os.path.join(SAVED_VOICES_DIR, name, "audio.wav")
    text_path = os.path.join(SAVED_VOICES_DIR, name, "text.txt")
    text = ""
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    if not os.path.exists(audio_path):
        return None, text
    return audio_path, text

def save_voice(name, audio_path, text):
    if not name or not audio_path:
        return "❌ Provide a name AND audio file.", gr.update(), gr.update(), gr.update(), gr.update()
    name = name.strip().replace(" ", "_")
    voice_dir = os.path.join(SAVED_VOICES_DIR, name)
    os.makedirs(voice_dir, exist_ok=True)
    shutil.copy(audio_path, os.path.join(voice_dir, "audio.wav"))
    with open(os.path.join(voice_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(text or "")
    choices = get_saved_voices()
    return f"✅ Voice '{name}' saved!", gr.update(choices=choices, value=name), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

def delete_voice(name):
    if not name:
        return "Select a voice first.", gr.update(), gr.update(), gr.update(), gr.update()
    voice_dir = os.path.join(SAVED_VOICES_DIR, name)
    if os.path.exists(voice_dir):
        shutil.rmtree(voice_dir)
    choices = get_saved_voices()
    return f"🗑️ Deleted '{name}'", gr.update(choices=choices, value=None), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

# ─── RVC Backend ───
def get_rvc_models():
    models = []
    if os.path.exists(RVC_MODELS_DIR):
        for f in os.listdir(RVC_MODELS_DIR):
            if f.endswith(".pth"):
                models.append(f)
    return models

def run_rvc_conversion(input_audio, model_name, pitch, output_name="rvc_output.wav"):
    """
    Run the RVC inference script to convert input_audio using the specified model.
    Returns (output_path, log) where log is None on success or an error string on failure.
    """
    if not input_audio:
        return None, "Please upload a reference audio."
    if not model_name:
        return None, "Please select an RVC model (.pth)."

    model_path = os.path.join(RVC_MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        return None, f"❌ RVC model not found: {model_name}"

    output_path = os.path.join(BASE_DIR, output_name)

    cmd = [
        RVC_PYTHON_EXE, RVC_INFER_SCRIPT,
        "--model", model_path,
        "--input", input_audio,
        "--output", output_path,
        "--pitch", str(int(pitch)),
        "--method", "rmvpe",
        "--device", "cpu"
    ]

    # Optionally provide an index file if present
    index_path = model_path.replace(".pth", ".index")
    if os.path.exists(index_path):
        cmd += ["--index", index_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    except FileNotFoundError as e:
        return None, f"❌ RVC backend not available: {e}"

    if result.returncode == 0 and os.path.exists(output_path):
        return output_path, "✅ Voice converted successfully!"
    else:
        # include stdout/stderr to help debugging
        return None, f"❌ RVC Error:\n{result.stdout}\n{result.stderr}"

# ─── F5-TTS Core Engine ───
def run_f5tts(text, ref_audio_path, ref_text, output_name="output_cloned.wav", base_audio=None):
    """
    Generate cloned speech using F5-TTS CLI.
    """
    output_path = os.path.join(BASE_DIR, output_name)
    trimmed = os.path.join(TEMP_DIR, "trimmed_ref_gen.wav")

    # Use Microsoft-generated audio for Hindi/Urdu if available
    source_audio = base_audio if base_audio else ref_audio_path

    audio = AudioSegment.from_file(source_audio)
    if len(audio) > 8000:
        audio = audio[:8000]
    audio.export(trimmed, format="wav")

    # Remove previous output
    if os.path.exists(output_path):
        os.remove(output_path)

    # If no transcript was provided, transcribe once
    reference_text = ref_text

    if not reference_text or not reference_text.strip():

        class DummyProgress:
            def __call__(self, *args, **kwargs):
                pass

        reference_text = extract_text_fn(trimmed, progress=DummyProgress())

        if reference_text.startswith("Error"):
            return None, f"Failed to transcribe reference audio:\n{reference_text}"

    cmd = [
        F5_TTS_EXE,
        "--model", "F5TTS_Base",
        "--ref_audio", trimmed,
        "--ref_text", reference_text,
        "--gen_text", text,
        "--output_dir", BASE_DIR,
        "--output_file", output_name,
        "--device", "cpu",
        "--nfe_step", "16"
    ]

    print("\n==============================")
    print("Running F5-TTS")
    print("==============================")
    print("Command:")
    print(" ".join(cmd))
    print("==============================\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    print("F5-TTS finished")
    print("Return code:", result.returncode)

    if result.stdout:
        print("\nSTDOUT:\n")
        print(result.stdout)

    if result.stderr:
        print("\nSTDERR:\n")
        print(result.stderr)

    if result.returncode != 0:
        return None, result.stderr

    if os.path.exists(output_path):
        print("Generated:", output_path)
        return output_path, None

    return None, "F5-TTS completed but no output file was created."

# ─── Tab 1: Standard Clone ───
def clone_voice_tab1(text, ref_text, audio_ref, progress=gr.Progress()):
    if not text: return None, "Enter text to generate."
    if not audio_ref: return None, "Upload a reference audio."
    progress(0.2, desc="Processing reference...")
    progress(0.4, desc="Running F5-TTS (1-3 min)...")
    path, log = run_f5tts(text, audio_ref, ref_text)
    progress(1.0)
    return path, log

# ─── Tab 2: Dramatic Story Mode ───
NARRATOR_VOICES = {
    "Guy (Passionate Male)": "en-US-GuyNeural",
    "Christopher (Authority Male)": "en-US-ChristopherNeural",
    "Andrew (Confident Male)": "en-US-AndrewNeural",
    "Eric (Rational Male)": "en-US-EricNeural",
    "Brian (Casual Male)": "en-US-BrianNeural",
    "Jenny (Friendly Female)": "en-US-JennyNeural",
    "Aria (Confident Female)": "en-US-AriaNeural",
    "Ava (Expressive Female)": "en-US-AvaNeural",
    "Ryan (British Male)": "en-GB-RyanNeural",
    "Sonia (British Female)": "en-GB-SoniaNeural",
}

def dramatic_clone(text, saved_voice_name, narrator_style, progress=gr.Progress()):
    if not text:
        return None, None, "Enter a story script."
    if not saved_voice_name:
        return None, None, "Select a saved voice from your library first."

    log_lines = []

    # Step 1: Generate emotional narration via edge-tts
    progress(0.1, desc="Step 1: Generating dramatic narration...")
    voice_id = NARRATOR_VOICES.get(narrator_style, "en-US-GuyNeural")
    emotion_path = os.path.join(TEMP_DIR, "emotion_base.mp3")
    ok, err = run_edge_tts(text, voice_id, emotion_path)
    if not ok:
        return None, None, f"❌ Edge-TTS failed: {err}"
    log_lines.append(f"Step 1: ✅ Emotional narration generated ({narrator_style})")

    # Step 2: Clone into anime voice using F5-TTS
    progress(0.4, desc="Step 2: Cloning into anime voice (1-3 min)...")
    voice_audio, voice_text = load_voice(saved_voice_name)
    if not voice_audio:
        log_lines.append(f"Step 2: ⚠️ Voice '{saved_voice_name}' audio not found. Showing emotion base only.")
        return emotion_path, None, "\n".join(log_lines)

    clone_path, clone_log = run_f5tts(text, voice_audio, voice_text, "dramatic_clone.wav")
    log_lines.append(f"Step 2: {clone_log}")
    progress(1.0)
    return emotion_path, clone_path, "\n".join(log_lines)

# ─── Tab 3: Hindi/Urdu ───
def generate_hindi(text, voice_id, use_transliteration, speed, pitch, progress=gr.Progress()):
    if not text:
        return None, "Enter some text."

    status = []
    final_text = text

    if use_transliteration:
        is_hin_urd, mode = detect_hindi_urdu(text)
        if mode == "roman":
            final_text = transliterate_roman_to_devanagari(text)
            status.append(f"🔄 Roman Hindi/Urdu detected → transliterated to: {final_text}")
        elif mode in ("devanagari", "arabic"):
            status.append("Text already in native script — no transliteration needed.")
        else:
            status.append("⚠️ Could not confidently detect Roman Hindi/Urdu — sending text as-is.")

    output_path = os.path.join(TEMP_DIR, "hindi_output.mp3")

    rate_arg = f"{speed:+d}%" if speed != 0 else None
    pitch_arg = f"{pitch:+d}Hz" if pitch != 0 else None

    progress(0.5, desc="Generating voice...")
    ok, err = run_edge_tts(final_text, voice_id, output_path, rate=rate_arg, pitch=pitch_arg)
    if not ok:
        return None, f"❌ Error: {err}"

    status.append("✅ Generated successfully!")
    progress(1.0)
    return output_path, "\n".join(status)

# ─── Extract Text (Whisper) ───
def extract_text_fn(audio_path, progress=gr.Progress()):
    if not audio_path: return "Upload an audio file first!"
    try:
        trimmed = os.path.join(TEMP_DIR, "extract_temp.wav")
        audio = AudioSegment.from_file(audio_path)
        if len(audio) > 8000: audio = audio[:8000]
        audio.export(trimmed, format="wav")
        progress(0.4, desc="Loading Whisper...")
        import torch
        from transformers import pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base",
                        device=device, torch_dtype=torch.float16)
        progress(0.7, desc="Transcribing...")
        result = pipe(trimmed, chunk_length_s=30, generate_kwargs={"task": "transcribe"})
        text = result['text'].strip()
        del pipe
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# ─── Tab 4: Multi-Voice Podcast ───

def parse_podcast_script(script, available_voices):
    """
    Parses a multi-speaker podcast script.

    Accepts formats like:
        Naruto: Hello
        Naruto : Hello
        Naruto     :    Hello
        naruto: Hello

    Returns:
        parsed_lines -> [(speaker, text), ...]
        errors -> list of warning/error messages
    """

    parsed_lines = []
    errors = []

    # Normalize voice dictionary to lowercase
    voice_lookup = {
        name.strip().lower(): name
        for name in available_voices.keys()
    }

    for line_number, raw_line in enumerate(script.splitlines(), start=1):

        line = raw_line.strip()

        # Skip blank lines
        if not line:
            continue

        # Match: Speaker : Text
        match = re.match(r'^\s*(.*?)\s*:\s*(.+?)\s*$', line)

        if not match:
            errors.append(
                f"Line {line_number}: Invalid format → '{raw_line}'"
            )
            continue

        speaker = match.group(1).strip()
        text = match.group(2).strip()

        if not speaker:
            errors.append(
                f"Line {line_number}: Missing speaker name."
            )
            continue

        if not text:
            errors.append(
                f"Line {line_number}: Empty dialogue."
            )
            continue

        speaker_lower = speaker.lower()

        if speaker_lower not in voice_lookup:

            suggestions = difflib.get_close_matches(
                speaker_lower,
                voice_lookup.keys(),
                n=1,
                cutoff=0.6
            )

            if suggestions:
                errors.append(
                    f"Unknown speaker '{speaker}'. Did you mean '{voice_lookup[suggestions[0]]}'?"
                )
            else:
                errors.append(
                    f"Unknown speaker '{speaker}'."
                )

            continue

        parsed_lines.append(
            (
                voice_lookup[speaker_lower],
                text
            )
        )

    return parsed_lines, errors

# ─── Perfect Pronunciation Pipeline: Microsoft Neural → RVC ───
def microsoft_to_rvc_pronunciation_clone(dialogue, voice_name, voice_audio, voice_text, out_name):
    """
    The core Task 2 hybrid pipeline for Hindi/Urdu (native script OR Roman):
      1. Detect Hindi/Urdu (script-based or Roman-word heuristic).
      2. Transliterate Roman input to Devanagari so the neural voice reads
         it natively instead of trying to sound it out as English.
      3. Generate a perfect-pronunciation Microsoft Neural base clip.
      4. Route that base through RVC voice conversion for this character,
         which preserves the neural clip's exact phoneme timing while
         swapping in the character's timbre — this is what actually keeps
         the pronunciation "perfect" after cloning.
      5. If no RVC model exists for this character yet, fall back to
         F5-TTS zero-shot cloning (clearly logged as lower-fidelity).

    Returns (output_path_or_None, log_lines: list[str], used_rvc: bool)
    """
    log_lines = []
    is_hin_urd, mode = detect_hindi_urdu(dialogue)

    if not is_hin_urd:
        return None, ["Not Hindi/Urdu — this pipeline doesn't apply to this line."], False

    synthesis_text = dialogue
    if mode == "roman":
        synthesis_text = transliterate_roman_to_devanagari(dialogue)
        log_lines.append(f"🔤 Roman Hindi/Urdu detected → transliterated: \"{synthesis_text}\"")
    else:
        log_lines.append(f"🌐 {mode.title()} script Hindi/Urdu detected")

    # Devanagari and Arabic-script Urdu both get an accurate read from the
    # Hindi neural voice for spoken pronunciation purposes; a dedicated Urdu
    # voice is used only when the line is unambiguously Arabic-script Urdu.
    base_voice_id = "ur-PK-AsadNeural" if mode == "arabic" else "hi-IN-MadhurNeural"

    base_path = os.path.join(TEMP_DIR, f"neural_base_{abs(hash(dialogue)) % 100000}.wav")
    ok, err = run_edge_tts(synthesis_text, base_voice_id, base_path)
    if not ok:
        log_lines.append(f"❌ Neural base generation failed:\n{err}")
        return None, log_lines, False

    log_lines.append(f"✅ Native-pronunciation neural base generated ({base_voice_id})")

    rvc_model_name = f"{voice_name}.pth"
    if os.path.exists(os.path.join(RVC_MODELS_DIR, rvc_model_name)):
        path, gen_log = run_rvc_conversion(base_path, rvc_model_name, 0, output_name=out_name)
        if path:
            log_lines.append(f"🎯 Microsoft Neural → RVC: cloned onto '{voice_name}' (pronunciation preserved exactly)")
            return path, log_lines, True
        log_lines.append(f"❌ RVC conversion failed, falling back to F5-TTS: {gen_log}")
    else:
        log_lines.append(
            f"⚠️ No RVC model '{rvc_model_name}' found in rvc_models/ — falling back to F5-TTS cloning. "
            f"Add an RVC model for '{voice_name}' for the full Microsoft→RVC pronunciation pipeline."
        )

    path, gen_log = run_f5tts(synthesis_text, voice_audio, voice_text, output_name=out_name, base_audio=base_path)
    if path:
        log_lines.append("🔁 Fallback: cloned via F5-TTS using the neural base as reference audio")
    else:
        log_lines.append(f"❌ F5-TTS fallback also failed: {gen_log}")
    return path, log_lines, False

def generate_podcast(script_text, pause_ms, progress=gr.Progress()):
    if not script_text.strip():
        return None, "Write a script first."

    saved = get_saved_voices()
    available_voices = {name: name for name in saved}

    parsed, errors = parse_podcast_script(script_text, available_voices)

    # Crash-proof parsing: surface warnings for bad lines/unknown speakers
    # instead of ever raising, but keep going with whatever DID parse.
    if not parsed:
        msg = "❌ Could not parse any lines from the script.\n\nUse format:\nNARUTO: Hey Luffy!\nLUFFY: Hey Naruto!"
        if errors:
            msg += "\n\nDetails:\n" + "\n".join(errors)
        return None, msg

    # Collect unique character names
    characters = list(dict.fromkeys([name for name, _ in parsed]))
    saved_lower = {v.lower(): v for v in saved}

    # Match characters to saved voices (case-insensitive)
    voice_map = {}
    missing = []
    for char in characters:
        if char.lower() in saved_lower:
            voice_map[char] = saved_lower[char.lower()]
        else:
            missing.append(char)

    if missing:
        return None, (
            f"❌ These characters have no matching saved voice:\n"
            f"  {', '.join(missing)}\n\n"
            f"Your saved voices: {', '.join(saved)}\n\n"
            f"Character names in your script must match saved voice names.\n"
            f"Go to the Voice Cloner tab to save voices first."
        )

    log_lines = []
    if errors:
        log_lines.append("⚠️ Some lines were skipped (polite warnings, not crashes):")
        log_lines.extend(f"  {e}" for e in errors)
        log_lines.append("")

    log_lines.append(f"📋 Parsed {len(parsed)} lines from {len(characters)} characters")
    for char in characters:
        log_lines.append(f"  {char} → voice '{voice_map[char]}'")

    # Generate each line
    audio_segments = []

    for i, (char, dialogue) in enumerate(parsed):
        progress((i + 1) / len(parsed), desc=f"Generating line {i+1}/{len(parsed)}: {char}...")
        log_lines.append(f"\n🎙️ [{i+1}/{len(parsed)}] {char}: \"{dialogue[:50]}...\"")

        voice_name = voice_map[char]
        voice_audio, voice_text = load_voice(voice_name)
        if not voice_audio:
            log_lines.append(f"  ⚠️ Audio file missing for '{voice_name}', skipping.")
            continue

        out_name = f"podcast_line_{i}.wav"

        is_hin_urd, _mode = detect_hindi_urdu(dialogue)

        if is_hin_urd:
            path, hybrid_log, used_rvc = microsoft_to_rvc_pronunciation_clone(
                dialogue, voice_name, voice_audio, voice_text, out_name
            )
            log_lines.extend(f"  {line}" for line in hybrid_log)
            gen_log = None if path else "Microsoft→RVC pronunciation pipeline failed."
        else:
            path, gen_log = run_f5tts(dialogue, voice_audio, voice_text, output_name=out_name)

        if path and os.path.exists(path):
            seg = AudioSegment.from_file(path)
            audio_segments.append(seg)
            log_lines.append(f"  ✅ {len(seg)/1000:.1f}s generated")
        else:
            log_lines.append(f"  ❌ Failed: {gen_log}")

    if not audio_segments:
        return None, "\n".join(log_lines) + "\n\n❌ No audio was generated."

    # Stitch together with fades + crossfade so cuts between characters
    # sound like a natural conversation instead of robotic hard silences.
    log_lines.append(f"\n🔗 Stitching {len(audio_segments)} segments...")
    pause = AudioSegment.silent(duration=int(pause_ms))

    final = audio_segments[0].fade_in(30).fade_out(40)
    crossfade_ms = min(60, len(pause) // 2) if len(pause) > 0 else 0

    for seg in audio_segments[1:]:
        seg = seg.fade_in(30).fade_out(40)
        if crossfade_ms > 0:
            final = final.append(pause, crossfade=crossfade_ms).append(seg, crossfade=crossfade_ms)
        else:
            final = final + pause + seg

    output_path = os.path.join(BASE_DIR, "podcast_output.wav")
    final.export(output_path, format="wav")
    log_lines.append(f"✅ Final podcast: {len(final)/1000:.1f}s total")

    return output_path, "\n".join(log_lines)

# ─── Audio Editor Functions ───
def edit_audio_trim(audio_path, start_s, end_s):
    if not audio_path: return None, "Upload audio first."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        trimmed = audio[start_ms:end_ms]
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        trimmed.export(out, format="wav")
        return out, f"✅ Trimmed: Kept {start_s}s to {end_s}s"
    except Exception as e:
        return None, f"❌ Error: {e}"

def edit_audio_cut(audio_path, start_s, end_s):
    if not audio_path: return None, "Upload audio first."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        cut = audio[:start_ms] + audio[end_ms:]
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        cut.export(out, format="wav")
        return out, f"✅ Cut: Removed {start_s}s to {end_s}s"
    except Exception as e:
        return None, f"❌ Error: {e}"

def edit_audio_replace(audio_path, start_s, end_s, text, voice_name, progress=gr.Progress()):
    if not audio_path: return None, "Upload audio first."
    if not text: return None, "Enter text to generate."
    if not voice_name: return None, "Select a voice."
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)
        
        voice_audio, voice_text = load_voice(voice_name)
        if not voice_audio:
            return None, f"❌ Audio file missing for '{voice_name}'"
            
        progress(0.3, desc="Generating new segment...")
        new_path, gen_log = run_f5tts(text, voice_audio, voice_text, output_name="replacement.wav")
        if not new_path or not os.path.exists(new_path):
            return None, f"❌ Generation failed: {gen_log}"
            
        new_seg = AudioSegment.from_file(new_path)
        final = audio[:start_ms] + new_seg + audio[end_ms:]
        
        out = os.path.join(BASE_DIR, "edited_audio.wav")
        final.export(out, format="wav")
        return out, f"✅ Replaced {start_s}s to {end_s}s with new generated audio."
    except Exception as e:
        return None, f"❌ Error: {e}"

# ─── ML FEATURE: Audio Dataset Preprocessing ───
TRAINING_DIR = os.path.join(BASE_DIR, "training_data")
os.makedirs(TRAINING_DIR, exist_ok=True)


def remove_background_noise(audio: AudioSegment) -> AudioSegment:
    """
    Noise removal stage. Uses non-stationary spectral noise reduction, which
    (unlike a fixed noise print) adapts to background static/music that
    varies over time — the messy-internet-audio case this pipeline exists for.
    Never raises: a pathological clip just passes through un-denoised.
    """
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    try:
        reduced = nr.reduce_noise(y=samples, sr=audio.frame_rate, stationary=False, prop_decrease=0.75)
    except Exception:
        reduced = samples
    reduced = np.clip(reduced, -32768, 32767).astype(np.int16)
    temp_clean = os.path.join(TEMP_DIR, "clean.wav")
    sf.write(temp_clean, reduced, audio.frame_rate)
    return AudioSegment.from_wav(temp_clean)


def remove_silence(audio: AudioSegment, min_silence_len=800, keep_silence=200, thresh_offset=18):
    """
    Silence removal stage. Cuts dead-air gaps so training time isn't wasted
    on nothing. Guards against dBFS == -inf (fully silent/corrupted clip),
    which would otherwise blow up the threshold math.
    """
    reference_dbfs = audio.dBFS if audio.dBFS != float("-inf") else -60.0
    pieces = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=reference_dbfs - thresh_offset,
        keep_silence=keep_silence
    )
    trimmed = AudioSegment.empty()
    for p in pieces:
        trimmed += p
    return trimmed, len(pieces)


def normalize_loudness(audio: AudioSegment, target_dbfs: float) -> AudioSegment:
    """Loudness normalization stage. Gains the clip to a target dBFS so every
    sample the model trains on has consistent volume. Silent/near-silent
    audio (dBFS == -inf) is left untouched rather than blown up by gain math."""
    if audio.dBFS == float("-inf") or len(audio) == 0:
        return audio
    change = target_dbfs - audio.dBFS
    return audio.apply_gain(change)


def is_chunk_acceptable(chunk: AudioSegment, min_ms=2000, min_dbfs=-45.0, clip_ratio_limit=0.98):
    """
    Quality-filter / reject-bad-chunks stage. Returns (True, None) if the
    chunk is good training data, else (False, reason).
    Rejects: too short, too quiet (near-silent), and clipped/distorted audio.
    """
    if len(chunk) < min_ms:
        return False, "too_short"
    if chunk.dBFS == float("-inf") or chunk.dBFS < min_dbfs:
        return False, "too_quiet"
    # Clipping check: how close the loudest sample is to the format's max
    # possible amplitude. Near 1.0 means the recording was distorted.
    if chunk.max_possible_amplitude > 0:
        peak_ratio = chunk.max / chunk.max_possible_amplitude
        if peak_ratio >= clip_ratio_limit:
            return False, "clipped"
    return True, None


def compute_dataset_stats(accepted_chunks, rejected_reasons, original_duration, processed_duration):
    """Dataset statistics stage: numbers a training run needs at a glance."""
    durations = [len(c) / 1000 for c in accepted_chunks]
    loudness = [c.dBFS for c in accepted_chunks if c.dBFS != float("-inf")]
    return {
        "original_duration_sec": round(original_duration, 2),
        "processed_duration_sec": round(processed_duration, 2),
        "silence_removed_sec": round(original_duration - processed_duration, 2),
        "accepted_chunks": len(accepted_chunks),
        "rejected_chunks": sum(rejected_reasons.values()),
        "rejection_breakdown": dict(rejected_reasons),
        "total_training_duration_sec": round(sum(durations), 2),
        "avg_chunk_duration_sec": round(sum(durations) / len(durations), 2) if durations else 0,
        "min_chunk_duration_sec": round(min(durations), 2) if durations else 0,
        "max_chunk_duration_sec": round(max(durations), 2) if durations else 0,
        "avg_chunk_loudness_dbfs": round(sum(loudness) / len(loudness), 2) if loudness else None,
    }


def export_clean_dataset(session_dir, chunks, stats):
    """
    Clean dataset export stage: writes the audio chunks + metadata.csv +
    stats.json into session_dir, then packages the whole thing into a single
    downloadable .zip so the dataset is easy to hand off to a training job.
    Returns (metadata_path, stats_path, zip_path).
    """
    os.makedirs(session_dir, exist_ok=True)

    metadata_path = os.path.join(session_dir, "metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "duration_seconds", "loudness_dbfs"])
        for idx, chunk in enumerate(chunks):
            filename = f"chunk_{idx:04d}.wav"
            chunk.export(os.path.join(session_dir, filename), format="wav")
            loudness = round(chunk.dBFS, 2) if chunk.dBFS != float("-inf") else None
            writer.writerow([filename, round(len(chunk) / 1000, 2), loudness])

    stats_path = os.path.join(session_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    zip_path = session_dir.rstrip(os.sep) + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(session_dir):
            zf.write(os.path.join(session_dir, fname), arcname=fname)

    return metadata_path, stats_path, zip_path


def preprocess_training_audio(
    audio_path,
    chunk_seconds=10,
    normalize_db=-20.0,
    progress=gr.Progress()
):
    """
    Professional Voice Dataset Preprocessing Pipeline

    Stages (Task 3):
    ✔ Noise Removal        — remove_background_noise()
    ✔ Silence Removal      — remove_silence()
    ✔ Loudness Normalization — normalize_loudness() (raw pass + per-chunk)
    ✔ Reject Bad Chunks    — is_chunk_acceptable()
    ✔ Dataset Statistics   — compute_dataset_stats()
    ✔ Clean Dataset Export — export_clean_dataset() (metadata.csv + stats.json + .zip)
    """

    if not audio_path:
        return None, None, "Please upload an audio file."

    try:
        progress(0.05, desc="Loading audio...")
        audio = AudioSegment.from_file(audio_path)
        original_duration = len(audio) / 1000

        if original_duration < 60:
            return None, None, (
                "❌ Training audio is too short.\n\n"
                f"Current Length : {original_duration:.1f} sec\n"
                "Minimum Required : 60 sec (1 minute)"
            )

        progress(0.15, desc="Normalizing loudness (raw pass)...")
        audio = normalize_loudness(audio, normalize_db)

        progress(0.30, desc="Converting to ML format (mono, 16kHz)...")
        audio = audio.set_channels(1).set_frame_rate(16000)

        progress(0.45, desc="Removing background noise...")
        clean_audio = remove_background_noise(audio)

        progress(0.60, desc="Cutting silent dead air...")
        processed, num_speech_segments = remove_silence(clean_audio)

        if num_speech_segments == 0 or len(processed) == 0:
            return None, None, "❌ No speech detected (audio may be silent, all noise, or too quiet)."

        progress(0.72, desc="Chunking + filtering...")
        chunk_ms = int(chunk_seconds * 1000)

        accepted_chunks = []
        rejected_reasons = {"too_short": 0, "too_quiet": 0, "clipped": 0}

        for i in range(0, len(processed), chunk_ms):
            c = processed[i:i + chunk_ms]

            ok, reason = is_chunk_acceptable(c)
            if not ok:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                continue

            # Per-chunk loudness normalization: the raw-pass normalization
            # can drift after noise reduction / silence trimming, so each
            # final chunk gets normalized again for consistent training volume.
            c = normalize_loudness(c, normalize_db)
            c = c.fade_in(50).fade_out(50)
            accepted_chunks.append(c)

        if len(accepted_chunks) == 0:
            return None, None, (
                "❌ No usable training chunks survived quality filtering.\n\n"
                f"Rejected — too short: {rejected_reasons['too_short']}, "
                f"too quiet: {rejected_reasons['too_quiet']}, "
                f"clipped: {rejected_reasons['clipped']}.\n"
                "Try a longer or cleaner source recording."
            )

        progress(0.85, desc="Computing dataset statistics...")
        final_duration = len(processed) / 1000
        stats = compute_dataset_stats(accepted_chunks, rejected_reasons, original_duration, final_duration)

        progress(0.92, desc="Exporting clean dataset...")
        session_name = f"session_{len(os.listdir(TRAINING_DIR)) + 1}"
        session_dir = os.path.join(TRAINING_DIR, session_name)
        metadata_path, stats_path, zip_path = export_clean_dataset(session_dir, accepted_chunks, stats)

        report = f"""✅ DATASET PREPROCESSING COMPLETE

Original Duration : {stats['original_duration_sec']} sec
Processed Duration : {stats['processed_duration_sec']} sec
Silence Removed : {stats['silence_removed_sec']} sec

Noise Reduction : Applied (non-stationary, background music/static)
Loudness Normalization : {normalize_db} dBFS (raw pass + per-chunk)
Sample Rate : 16000 Hz  |  Channels : Mono

── Chunk Quality Filtering ──
Accepted Chunks : {stats['accepted_chunks']}
Rejected Chunks : {stats['rejected_chunks']}  (too short: {rejected_reasons['too_short']}, too quiet: {rejected_reasons['too_quiet']}, clipped: {rejected_reasons['clipped']})

── Dataset Statistics ──
Total Training Duration : {stats['total_training_duration_sec']} sec
Avg / Min / Max Chunk Duration : {stats['avg_chunk_duration_sec']} / {stats['min_chunk_duration_sec']} / {stats['max_chunk_duration_sec']} sec
Avg Chunk Loudness : {stats['avg_chunk_loudness_dbfs']} dBFS

── Clean Dataset Export ──
metadata.csv : {metadata_path}
stats.json   : {stats_path}
Zipped dataset : {zip_path}
Output Folder : {session_dir}
"""

        progress(1.0)
        return session_dir, zip_path, report

    except Exception as e:
        return None, None, f"❌ {str(e)}"


def analyze_voice_similarity(audio_a, audio_b, progress=gr.Progress()):
    """Real ML: Compare two audio files using Whisper encoder embeddings + cosine similarity."""
    if not audio_a or not audio_b:
        return "Upload both audio files to compare."
    try:
        progress(0.2, desc="Loading Whisper encoder...")
        import torch
        import numpy as np
        from transformers import WhisperProcessor, WhisperModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperModel.from_pretrained("openai/whisper-base").to(device).to(dtype)

        def get_embedding(path):
            audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
            if len(audio) > 15000:
                audio = audio[:15000]
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device).to(dtype)
            with torch.no_grad():
                encoder_out = model.encoder(input_features)
                embedding = encoder_out.last_hidden_state.mean(dim=1).squeeze()
            return embedding

        progress(0.5, desc="Extracting voice embeddings...")
        emb_a = get_embedding(audio_a)
        progress(0.7, desc="Comparing voice signatures...")
        emb_b = get_embedding(audio_b)

        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        similarity_pct = max(0, min(100, cos_sim * 100))

        # Cleanup GPU
        del model, processor, emb_a, emb_b
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grade = "🟢 Excellent" if similarity_pct > 85 else "🟡 Good" if similarity_pct > 70 else "🔴 Poor"
        progress(1.0)
        return (
            f"🧠 Voice Similarity Analysis\n"
            f"{'='*40}\n"
            f"Cosine Similarity Score: {similarity_pct:.1f}%\n"
            f"Quality Grade: {grade}\n\n"
            f"{'='*40}\n"
            f"If the score is below 70%, consider:\n"
            f"  • Using a longer/cleaner reference audio\n"
            f"  • Fine-tuning the model with more training data\n"
            f"  • Adjusting the pitch shift parameter"
        )
    except Exception as e:
        return f"❌ Analysis Error: {e}"

# ═══════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════
import base64
logo_path = os.path.join(BASE_DIR, "LOGO.jpg")
logo_b64 = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")

custom_css = """
footer {display: none !important;}
.zenvyro-header {text-align: center; padding: 20px 0; border-bottom: 2px solid #eee; margin-bottom: 20px;}
.zenvyro-logo {font-size: 2.5em; font-weight: 800; color: #2563eb; letter-spacing: 2px;}
.zenvyro-subtitle {font-size: 1.1em; color: #64748b; margin-top: 5px;}
"""

header_html = f"""
    <div class="zenvyro-header">
        <img src="data:image/jpeg;base64,{logo_b64}" alt="Zenvyrolabs Logo" style="height: 80px; margin-bottom: 10px; display: inline-block;">
        <div class="zenvyro-logo">ZENVYROLABS</div>
        <div class="zenvyro-subtitle">Internal Advanced Voice Studio • Clone anime voices • Dramatic storytelling • Multi-voice podcasts</div>
    </div>
"""

with gr.Blocks(
    title="🎙️ Zenvyrolabs Voice Studio",
    css=custom_css
) as interface:
    gr.HTML(header_html)

    with gr.Tabs():
        # ─── TAB 1: Voice Cloner ───
        with gr.TabItem("🎭 Voice Cloner"):
            gr.Markdown("Upload any voice clip → the AI clones it and speaks your text in that voice.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📂 Voice Library")
                    saved_dd = gr.Dropdown(choices=get_saved_voices(), label="Saved Voices", interactive=True)
                    with gr.Row():
                        load_btn = gr.Button("📂 Load", size="sm")
                        del_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    gr.Markdown("---")
                    gr.Markdown("### 💾 Save Voice")
                    voice_name = gr.Textbox(label="Name", placeholder="e.g. Gojo_Dramatic")
                    save_btn = gr.Button("💾 Save to Library", variant="primary")
                    lib_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=2):
                    gen_text1 = gr.Textbox(label="Script to Speak", lines=6, placeholder="Type your story here...")
                    ref_audio1 = gr.Audio(type="filepath", label="Reference Voice (auto-trims to 8s)")
                    with gr.Row():
                        ref_text1 = gr.Textbox(label="Reference Text", lines=2, scale=4,
                            placeholder="Type exact words from the reference audio...")
                        extract_btn1 = gr.Button("🔍 Auto-Extract", variant="secondary", scale=1)
                    clone_btn1 = gr.Button("🎙️ Generate Clone", variant="primary", size="lg")

            with gr.Row():
                out_audio1 = gr.Audio(label="Generated Audio")
                out_log1 = gr.Textbox(label="Log")

            load_btn.click(fn=load_voice, inputs=[saved_dd], outputs=[ref_audio1, ref_text1])
            extract_btn1.click(fn=extract_text_fn, inputs=[ref_audio1], outputs=[ref_text1])
            clone_btn1.click(fn=clone_voice_tab1, inputs=[gen_text1, ref_text1, ref_audio1], outputs=[out_audio1, out_log1])

        # ─── TAB 2: Dramatic Story Mode ───
        with gr.TabItem("🎬 Dramatic Story Mode", visible=False):
            gr.Markdown("""### How it works:
1. **Step 1:** Microsoft Neural AI creates a dramatic, emotional narration (perfect pronunciation & emotions).
2. **Step 2:** F5-TTS re-generates the same script using your saved anime voice (Gojo, Naruto, etc).
3. You get **two outputs** — pick whichever sounds better!

**Pro tip:** The emotion base alone sounds incredible for YouTube. The anime clone adds character flavor.""")

            with gr.Row():
                with gr.Column():
                    saved_dd2 = gr.Dropdown(choices=get_saved_voices(), label="Select Saved Anime Voice", interactive=True)
                    narrator_style = gr.Dropdown(
                        choices=list(NARRATOR_VOICES.keys()),
                        label="Emotion Narrator Style", value="Guy (Passionate Male)"
                    )
                    story_text = gr.Textbox(label="Your Story Script", lines=10,
                        placeholder="My daughter went missing five years ago...")
                    dramatic_btn = gr.Button("🎬 Generate Dramatic Voiceover", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### Step 1: Emotional Narration (Microsoft Neural)")
                    emotion_audio = gr.Audio(label="Emotion Base")
                    gr.Markdown("### Step 2: Anime Voice Clone (F5-TTS)")
                    clone_audio = gr.Audio(label="Anime Voice Version")
                    dramatic_log = gr.Textbox(label="Generation Log")

            dramatic_btn.click(fn=dramatic_clone,
                inputs=[story_text, saved_dd2, narrator_style],
                outputs=[emotion_audio, clone_audio, dramatic_log])

        # ─── TAB 3: Multi-Voice Podcast ───
        with gr.TabItem("🎙️ Multi-Voice Podcast"):
            gr.Markdown("""### Create Podcasts with Multiple Anime Voices
Write a script with character names that **match your saved voices**. Each line is generated with the correct voice and stitched into one seamless audio.

**Script Format:**
```
NARUTO: Hey Luffy, what's up man!
LUFFY: Yo Naruto! Just finished eating, I'm pumped!
NARUTO: Wanna go train together?
LUFFY: Let's gooo!
```
⚠️ Character names must **exactly match** your saved voice names (case-insensitive).""")

            with gr.Row():
                with gr.Column():
                    podcast_voices_dd = gr.Dropdown(choices=get_saved_voices(), multiselect=True, label="Your Saved Voices", info="Select the characters you want to use in your podcast script", interactive=True)
                    podcast_script = gr.Textbox(label="Podcast Script", lines=14,
                        placeholder="NARUTO: Hey Luffy, what's going on?\nLUFFY: Hey Naruto! Just had the best meat ever!\nNARUTO: That sounds awesome, want to spar?\nLUFFY: You're on!")
                    pause_slider = gr.Slider(100, 2000, value=500, step=50,
                        label="Pause Between Lines (ms)", info="How long to pause between each character's line")
                    podcast_btn = gr.Button("🎙️ Generate Full Podcast", variant="primary", size="lg")

                with gr.Column():
                    podcast_audio = gr.Audio(label="Final Podcast Audio")
                    podcast_log = gr.Textbox(label="Generation Log", lines=15)

            podcast_btn.click(fn=generate_podcast,
                inputs=[podcast_script, pause_slider],
                outputs=[podcast_audio, podcast_log])

        # ─── TAB 4: Hindi / Urdu ───
        with gr.TabItem("🌏 Hindi / Urdu"):
            gr.Markdown("""### Perfect Hindi & Urdu Pronunciation
**Fix:** Auto-converts Roman Hindi/Urdu → Devanagari script before generating, so pronunciation is accurate.
- Type **Roman** (kya haal hai) → auto-converts to **Devanagari** (क्या हाल है)
- Or type directly in **Devanagari** for best quality""")

            with gr.Row():
                with gr.Column():
                    hindi_text = gr.Textbox(label="Hindi / Urdu Text", lines=6,
                        placeholder="Hello bhai, kya haal hai? Aaj hum ek bahut hi dilchasp kahani sunenge...")
                    transliterate_toggle = gr.Checkbox(label="🔄 Auto-convert Roman → Devanagari (Recommended!)", value=True)
                    hindi_voice = gr.Dropdown(
                        choices=["hi-IN-MadhurNeural", "hi-IN-SwaraNeural",
                                 "ur-PK-AsadNeural", "ur-PK-UzmaNeural",
                                 "ur-IN-SalmanNeural", "ur-IN-GulNeural"],
                        label="Voice", value="hi-IN-MadhurNeural",
                        info="Madhur=Hindi Male, Swara=Hindi Female, Asad=Urdu Male, Uzma=Urdu Female"
                    )
                    with gr.Row():
                        hindi_speed = gr.Slider(-30, 30, value=0, step=5, label="Speed (%)")
                        hindi_pitch = gr.Slider(-20, 20, value=0, step=2, label="Pitch (Hz)")
                    hindi_btn = gr.Button("🎙️ Generate Hindi/Urdu Voice", variant="primary", size="lg")

                with gr.Column():
                    hindi_audio = gr.Audio(label="Generated Audio")
                    hindi_log = gr.Textbox(label="Status")

            hindi_btn.click(fn=generate_hindi,
                inputs=[hindi_text, hindi_voice, transliterate_toggle, hindi_speed, hindi_pitch],
                outputs=[hindi_audio, hindi_log])

        # ─── TAB 5: Audio Editor ───
        with gr.TabItem("✂️ Audio Editor"):
            gr.Markdown("Upload an audio file (or download a generated one and upload here) to trim, cut, or completely replace a bad segment with a newly generated voice!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    edit_audio_in = gr.Audio(type="filepath", label="Source Audio", interactive=True)
                    start_s = gr.Number(label="Start Time (seconds)", value=0.0)
                    end_s = gr.Number(label="End Time (seconds)", value=5.0)
                    
                    with gr.Row():
                        trim_btn = gr.Button("✂️ Trim (Keep Only Selection)", variant="secondary")
                        cut_btn = gr.Button("🗑️ Cut (Remove Selection)", variant="secondary")
                        
                    gr.Markdown("### Replace Segment")
                    replace_text = gr.Textbox(label="New Text for Segment", lines=2)
                    replace_voice = gr.Dropdown(choices=get_saved_voices(), label="Select Voice for New Segment", interactive=True)
                    replace_btn = gr.Button("🔄 Replace Segment", variant="primary")
                
                with gr.Column(scale=1):
                    edit_audio_out = gr.Audio(label="Edited Audio")
                    edit_log = gr.Textbox(label="Status Log")
                    
            trim_btn.click(fn=edit_audio_trim, inputs=[edit_audio_in, start_s, end_s], outputs=[edit_audio_out, edit_log])
            cut_btn.click(fn=edit_audio_cut, inputs=[edit_audio_in, start_s, end_s], outputs=[edit_audio_out, edit_log])
            replace_btn.click(fn=edit_audio_replace, inputs=[edit_audio_in, start_s, end_s, replace_text, replace_voice], outputs=[edit_audio_out, edit_log])

        # ─── TAB 6: Voice-to-Voice (RVC) ───
        with gr.TabItem("🎤 Voice-to-Voice (RVC)", visible=False):
            gr.Markdown("""### True Emotional Voice Cloning (Speech-to-Speech)
Upload an audio of **you acting out a line**, select a downloaded `.pth` anime character model, and the AI will convert your voice while preserving exactly the timing, emotion, and breath.
*(Models must be placed in the `rvc_models/` folder next to this app)*""")
            with gr.Row():
                with gr.Column():
                    rvc_in = gr.Audio(type="filepath", label="Input Audio (Your acting/reference)")
                    rvc_model = gr.Dropdown(choices=get_rvc_models(), label="RVC Model (.pth)", interactive=True)
                    rvc_refresh = gr.Button("🔄 Refresh Models List", size="sm")
                    rvc_pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift (Semitones)", info="Use +12 for Male->Female, -12 for Female->Male. Leave 0 if same gender.")
                    rvc_btn = gr.Button("🎤 Convert Voice", variant="primary", size="lg")
                with gr.Column():
                    rvc_out = gr.Audio(label="Converted Audio")
                    rvc_log = gr.Textbox(label="Status Log", lines=10)
                    
            rvc_btn.click(fn=run_rvc_conversion, inputs=[rvc_in, rvc_model, rvc_pitch], outputs=[rvc_out, rvc_log])
            rvc_refresh.click(fn=lambda: gr.update(choices=get_rvc_models()), outputs=[rvc_model])

        # ─── TAB 7: Perfect Pronunciation Clone ───
        with gr.TabItem("🌟 Perfect Pronunciation Clone", visible=False):
            gr.Markdown("""### Get Anime Voices with PERFECT Pronunciation
F5-TTS sometimes struggles with pronunciation. This tab fixes that! 
It uses **Edge-TTS (Eric, Guy, etc.)** to generate perfect, native pronunciation, and then uses **RVC** to seamlessly morph that audio into your Anime character's voice.
*(Requires an RVC `.pth` model in `rvc_models/`)*""")
            with gr.Row():
                with gr.Column():
                    perf_text = gr.Textbox(label="Script", lines=6, placeholder="Type perfectly pronounced English here...")
                    perf_neural = gr.Dropdown(choices=list(NARRATOR_VOICES.keys()), label="Base Neural Voice (for acting/pronunciation)", value="Eric (Rational Male)")
                    perf_rvc = gr.Dropdown(choices=get_rvc_models(), label="Target Anime Voice (RVC Model)", interactive=True)
                    perf_pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift", info="Match Neural gender to Anime gender. e.g. Male to Female: +12")
                    perf_btn = gr.Button("🌟 Generate Perfect Clone", variant="primary", size="lg")
                with gr.Column():
                    perf_audio = gr.Audio(label="Final Perfect Audio")
                    perf_log = gr.Textbox(label="Status Log")

            def run_perfect_clone(text, neural_voice, rvc_model, pitch, progress=gr.Progress()):
                if not text: return None, "Please enter text."
                if not rvc_model: return None, "Please select an RVC model."
                
                progress(0.2, desc="Generating perfect pronunciation...")
                voice_id = NARRATOR_VOICES.get(neural_voice, "en-US-EricNeural")
                temp_audio = os.path.join(TEMP_DIR, "perf_base.mp3")
                ok, err = run_edge_tts(text, voice_id, temp_audio)
                if not ok:
                    return None, f"❌ Edge-TTS failed: {err}"
                
                progress(0.6, desc="Morphing into Anime Voice (RVC)...")
                final_path, log = run_rvc_conversion(temp_audio, rvc_model, pitch)
                progress(1.0)
                return final_path, log
            
            perf_btn.click(fn=run_perfect_clone, inputs=[perf_text, perf_neural, perf_rvc, perf_pitch], outputs=[perf_audio, perf_log])

        # ─── TAB 8: Voice Training Studio (Real ML) ───
        with gr.TabItem("🧠 Voice Training Studio"):
            gr.Markdown("""### 🧠 AI Model Training Pipeline
This is the **core Machine Learning** feature of the application. Instead of relying on zero-shot cloning (which can sound robotic), you can **train a custom voice model** by feeding it high-quality audio data.

**How it works (Real ML Pipeline):**
1. **Upload** a long audio recording of your target voice (5-10 minutes recommended).
2. **Preprocess** — Our pipeline will automatically normalize volume levels, resample to 16kHz mono (the standard for speech ML models), remove silence, and chunk the audio into clean 10-second training segments.
3. **Analyze** — Use the Voice Quality Analyzer to compare your cloned output vs the original and get a real ML similarity score using Whisper neural embeddings.

*This is the exact same data preprocessing pipeline used in production ML systems at companies like ElevenLabs and OpenAI.*""")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Upload Raw Training Audio")
                    train_audio = gr.Audio(type="filepath", label="Raw Training Audio (5-10 min recommended)")
                    chunk_size = gr.Slider(5, 30, value=10, step=1, label="Chunk Size (seconds)", info="Each chunk becomes one training sample")
                    norm_db = gr.Slider(-30, -10, value=-20, step=1, label="Target Volume (dBFS)", info="Normalizes all chunks to this volume level for consistent training")
                    preprocess_btn = gr.Button("⚙️ Preprocess Dataset", variant="primary", size="lg")

                with gr.Column():
                    gr.Markdown("### Preprocessing Results")
                    train_output_dir = gr.Textbox(label="Output Directory", interactive=False)
                    train_dataset_zip = gr.File(label="📦 Download Clean Dataset (.zip)")
                    train_log = gr.Textbox(label="Pipeline Log (stats, rejected chunks, etc.)", lines=16)

            preprocess_btn.click(fn=preprocess_training_audio,
                inputs=[train_audio, chunk_size, norm_db],
                outputs=[train_output_dir, train_dataset_zip, train_log])

            gr.Markdown("---")
            gr.Markdown("""### Step 2: Voice Quality Analyzer (Cosine Similarity)
Upload the **original voice** and your **cloned output** to measure how accurate the clone is using real ML metrics.
The system uses **OpenAI Whisper's neural encoder** to extract voice embeddings and computes **cosine similarity** — the same technique used in speaker verification systems.""")

            with gr.Row():
                with gr.Column():
                    sim_audio_a = gr.Audio(type="filepath", label="Audio A: Original Voice")
                    sim_audio_b = gr.Audio(type="filepath", label="Audio B: Cloned Voice")
                    sim_btn = gr.Button("🧠 Analyze Similarity", variant="primary", size="lg")
                with gr.Column():
                    sim_result = gr.Textbox(label="ML Analysis Results", lines=12)

            sim_btn.click(fn=analyze_voice_similarity,
                inputs=[sim_audio_a, sim_audio_b],
                outputs=[sim_result])

    # Global event bindings
    save_btn.click(fn=save_voice, inputs=[voice_name, ref_audio1, ref_text1], outputs=[lib_status, saved_dd, saved_dd2, podcast_voices_dd, replace_voice])
    del_btn.click(fn=delete_voice, inputs=[saved_dd], outputs=[lib_status, saved_dd, saved_dd2, podcast_voices_dd, replace_voice])

if __name__ == "__main__":
    print("Launching Advanced Voice Studio...")
    print(f"Saved Voices: {get_saved_voices()}")

    # Running in Docker sets RUNNING_IN_DOCKER=1 (see docker-compose.yml):
    # bind to all interfaces and skip inbrowser (there's no browser in the
    # container). Local/dev runs keep the old "open my browser" behavior.
    in_docker = os.environ.get("RUNNING_IN_DOCKER", "0") == "1"

    interface.launch(
        server_name="0.0.0.0" if in_docker else "127.0.0.1",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        inbrowser=not in_docker
    )
