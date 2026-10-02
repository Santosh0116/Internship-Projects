!pip install SpeechRecognition pydub

#!/usr/bin/env python3
"""
Speech-to-Text Transcription Tool
=================================
AI Internship Project - Codec Technologies

Converts speech to text using the `SpeechRecognition` library and the Google Web Speech API.

Input modes
-----------
  * Audio file  : .wav works directly. With `pydub` + `ffmpeg` installed, other formats
                  (mp3, m4a, flac, ogg ...) are converted automatically, and every file is
                  normalised (mono, 16 kHz, volume-levelled) for better accuracy.
  * Microphone  : live dictation using `pyaudio` (works on your own computer, NOT on Google Colab).

Pipeline (one function per stage)
---------------------------------
  prepare_audio_file()  -> convert / normalise the audio            (processing)
  load_audio()          -> read the file and split it into chunks    (loading)
  transcribe_chunk()    -> send one chunk to Google, with retries    (recognition + error handling)
  tidy_text()           -> join chunks, capitalise, add full stop    (post-processing)
  save_transcript()     -> write the text to a .txt file             (saving)

Usage
-----
    pip install SpeechRecognition pydub          # + pyaudio for microphone mode
    python speech_to_text.py -i recording.wav
    python speech_to_text.py -i interview.mp3 -o interview.txt --language en-GB
    python speech_to_text.py --mic

Error handling
--------------
  * sr.UnknownValueError -> audio was silent / unintelligible: chunk is skipped, a warning is shown.
  * sr.RequestError      -> network or API problem: retried with increasing delay; if it keeps
                            failing, the text transcribed so far is still saved.
  * Missing/corrupt files, missing FLAC converter, missing microphone/PyAudio -> clear messages.

Note: the free Google Web Speech API is meant for light use. It limits clip length and request
volume and needs an internet connection; for production use a paid, keyed service.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import speech_recognition as sr

try:  # pydub is optional: it enables non-WAV input and audio normalisation
    from pydub import AudioSegment
    from pydub.effects import normalize

    HAS_PYDUB = True
except ImportError:
    AudioSegment = None
    normalize = None
    HAS_PYDUB = False

TARGET_SAMPLE_RATE = 16000        # Hz - speech recognisers work best around 16 kHz
DEFAULT_CHUNK_SECONDS = 30        # the free Google API works best with clips under ~1 minute
STOP_PHRASES = {"stop recording", "stop listening", "end recording"}


# =============================================================================
# Custom exceptions and result container
# =============================================================================
class AudioLoadError(Exception):
    """The audio file or microphone could not be opened or read."""


class TranscriptionError(Exception):
    """The recognition service failed (network / API / missing FLAC converter)."""


@dataclass
class TranscriptionResult:
    text: str = ""
    chunks_total: int = 0
    chunks_unintelligible: int = 0
    audio_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# =============================================================================
# 1. PROCESSING AUDIO: convert + normalise
# =============================================================================
def prepare_audio_file(audio_path: Path, workdir: Path) -> Path:
    """Return a WAV file that SpeechRecognition can read.

    With pydub: decode any format, convert to mono / 16 kHz / 16-bit and normalise volume.
    Without pydub: only plain PCM .wav files are accepted.
    """
    if not audio_path.is_file():
        raise AudioLoadError(f"File not found: {audio_path}")

    if not HAS_PYDUB:
        if audio_path.suffix.lower() == ".wav":
            return audio_path
        raise AudioLoadError(
            f"'{audio_path.suffix}' files need pydub and ffmpeg. Install them "
            "(pip install pydub, plus ffmpeg) or convert the file to .wav first."
        )

    try:
        segment = AudioSegment.from_file(str(audio_path))
    except Exception as exc:  # pydub raises several types (CouldntDecodeError, missing ffmpeg, ...)
        raise AudioLoadError(f"Could not decode '{audio_path.name}' (is ffmpeg installed?): {exc}") from exc

    segment = segment.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE).set_sample_width(2)
    segment = normalize(segment)  # raises quiet recordings to a consistent loudness
    prepared = workdir / f"{audio_path.stem}_prepared.wav"
    segment.export(str(prepared), format="wav")
    return prepared


# =============================================================================
# 2. LOADING AUDIO: read file -> list of chunks
# =============================================================================
def load_audio(recognizer: sr.Recognizer, wav_path: Path,
               chunk_seconds: int = DEFAULT_CHUNK_SECONDS) -> Tuple[List["sr.AudioData"], float]:
    """Read the audio and split it into `chunk_seconds` pieces. Returns (chunks, total_seconds)."""
    chunks: List["sr.AudioData"] = []
    try:
        with sr.AudioFile(str(wav_path)) as source:
            total_seconds = float(getattr(source, "DURATION", 0.0) or 0.0)
            while True:
                chunk = recognizer.record(source, duration=chunk_seconds)
                if not chunk.frame_data:      # end of file reached
                    break
                chunks.append(chunk)
    except ValueError as exc:                 # SpeechRecognition raises ValueError for unreadable formats
        raise AudioLoadError(f"'{wav_path.name}' is not a valid PCM WAV / AIFF / FLAC file: {exc}") from exc
    except (OSError, EOFError) as exc:
        raise AudioLoadError(f"Could not read '{wav_path.name}': {exc}") from exc

    if not chunks:
        raise AudioLoadError("The audio file contains no audio data.")
    return chunks, total_seconds


# =============================================================================
# 3. RECOGNITION with error handling
# =============================================================================
def transcribe_chunk(recognizer: sr.Recognizer, audio: "sr.AudioData", language: str,
                     retries: int = 3, retry_delay: float = 2.0) -> Optional[str]:
    """Transcribe one chunk with Google Speech Recognition.

    Returns the text, or None if no intelligible speech was found.
    Raises TranscriptionError if the service cannot be reached after `retries` attempts.
    """
    for attempt in range(1, retries + 1):
        try:
            return recognizer.recognize_google(audio, language=language)
        except sr.UnknownValueError:
            return None                        # silence / unintelligible: not an error, don't retry
        except sr.RequestError as exc:         # no internet, quota exceeded, API unreachable ...
            if attempt == retries:
                raise TranscriptionError(
                    f"Speech service unreachable after {retries} attempts "
                    f"(check your internet connection): {exc}") from exc
            wait = retry_delay * attempt
            print(f"  network/API problem ({exc}); retrying in {wait:.0f}s "
                  f"[attempt {attempt}/{retries}]")
            time.sleep(wait)
        except OSError as exc:                 # e.g. the FLAC converter used internally is missing
            raise TranscriptionError(
                f"Audio conversion failed ({exc}). Install the FLAC tool: "
                "'sudo apt install flac' (Linux) or 'brew install flac' (macOS).") from exc
    return None  # pragma: no cover (loop always returns or raises)


def tidy_text(text: str) -> str:
    """Light post-processing: normalise spaces, capitalise the start, end with a full stop."""
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


# =============================================================================
# 4. FILE TRANSCRIPTION (orchestrates the stages above)
# =============================================================================
def transcribe_file(audio_path: Path, language: str = "en-US",
                    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
                    retries: int = 3, retry_delay: float = 2.0) -> TranscriptionResult:
    recognizer = sr.Recognizer()
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = prepare_audio_file(audio_path, Path(tmp))
        chunks, total_seconds = load_audio(recognizer, wav_path, chunk_seconds)  # chunks live in memory

    result = TranscriptionResult(chunks_total=len(chunks), audio_seconds=total_seconds)
    print(f"Loaded {total_seconds:.1f}s of audio in {len(chunks)} chunk(s). Language: {language}\n")

    pieces: List[str] = []
    for i, chunk in enumerate(chunks):
        start = i * chunk_seconds
        end = min(start + chunk_seconds, total_seconds) if total_seconds else start + chunk_seconds
        label = f"[{i + 1}/{len(chunks)}] {start:.0f}s-{end:.0f}s"
        try:
            text = transcribe_chunk(recognizer, chunk, language, retries, retry_delay)
        except TranscriptionError as exc:
            result.error = str(exc)
            print(f"{label} FAILED: {exc}")
            break                                # keep what we have so far
        if text is None:
            result.chunks_unintelligible += 1
            print(f"{label} no intelligible speech (skipped)")
            continue
        pieces.append(text)
        print(f"{label} {text[:70]}{'...' if len(text) > 70 else ''}")

    result.text = tidy_text(" ".join(pieces))
    return result


# =============================================================================
# 5. MICROPHONE TRANSCRIPTION
# =============================================================================
def transcribe_microphone(language: str = "en-US", phrase_time_limit: int = 15,
                          retries: int = 3, retry_delay: float = 2.0) -> TranscriptionResult:
    """Live dictation. Stops on 'stop recording', Ctrl+C, or after 3 silent timeouts in a row."""
    recognizer = sr.Recognizer()
    try:
        microphone = sr.Microphone()
    except (AttributeError, OSError) as exc:     # PyAudio not installed / no input device
        raise AudioLoadError(
            f"Microphone not available ({exc}). Install PyAudio (see the setup guide) and check "
            "that a microphone is connected. Microphone mode does not work on Google Colab.") from exc

    result = TranscriptionResult()
    pieces: List[str] = []
    idle = 0
    try:
        with microphone as source:
            print("Calibrating for background noise (1 second)...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Speak now. Say 'stop recording' or press Ctrl+C to finish.\n")
            while idle < 3:
                try:
                    audio = recognizer.listen(source, timeout=8, phrase_time_limit=phrase_time_limit)
                except sr.WaitTimeoutError:
                    idle += 1
                    print(f"  (no speech heard - {idle}/3)")
                    continue
                idle = 0
                result.chunks_total += 1
                try:
                    text = transcribe_chunk(recognizer, audio, language, retries, retry_delay)
                except TranscriptionError as exc:
                    result.error = str(exc)
                    print(f"  FAILED: {exc}")
                    break
                if text is None:
                    result.chunks_unintelligible += 1
                    print("  (could not understand that - please repeat)")
                    continue
                print(f"  heard: {text}")
                if text.lower().strip() in STOP_PHRASES:
                    break
                pieces.append(text)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except OSError as exc:                       # e.g. "No Default Input Device Available"
        raise AudioLoadError(f"Could not open the microphone: {exc}") from exc

    result.text = tidy_text(" ".join(pieces))
    return result


# =============================================================================
# 6. SAVING
# =============================================================================
def save_transcript(text: str, output_path: Path) -> Path:
    """Write the transcript to a UTF-8 .txt file (creating folders if needed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    return output_path


# =============================================================================
# 7. COMMAND-LINE INTERFACE
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Speech-to-text transcription (Google Speech Recognition)",
                                allow_abbrev=False)
    p.add_argument("-i", "--input", help="Audio file to transcribe (.wav; other formats need pydub + ffmpeg)")
    p.add_argument("--mic", action="store_true", help="Transcribe live from the microphone instead")
    p.add_argument("-o", "--output", help="Transcript .txt path (default: same name as the audio file)")
    p.add_argument("--language", default="en-US", help="Language code, e.g. en-US, en-GB, en-IN, hi-IN (default: en-US)")
    p.add_argument("--chunk_seconds", type=int, default=DEFAULT_CHUNK_SECONDS,
                   help="Split long audio into pieces of this length (default: 30)")
    p.add_argument("--retries", type=int, default=3, help="Attempts per chunk on network errors (default: 3)")
    p.add_argument("--retry_delay", type=float, default=2.0, help="Base delay between retries in seconds")
    return p


def main(args_list: list[str] | None = None) -> int:
    parser = build_parser()
    # Passing args_list explicitly to avoid parsing sys.argv inside Colab notebooks
    if args_list is None:
        args_list = []
    args, _ = parser.parse_known_args(args_list)

    if not args.input and not args.mic:
        parser.print_help()
        print("\nGive an audio file with -i FILE, or use --mic for live dictation.")
        return 2

    try:
        if args.mic:
            result = transcribe_microphone(args.language, retries=args.retries, retry_delay=args.retry_delay)
            output = Path(args.output or "transcript.txt")
        else:
            audio_path = Path(args.input)
            result = transcribe_file(audio_path, args.language, args.chunk_seconds,
                                     args.retries, args.retry_delay)
            output = Path(args.output) if args.output else audio_path.with_suffix(".txt")
    except AudioLoadError as exc:
        print(f"\nERROR (audio): {exc}")
        return 1

    print("\n" + "=" * 60)
    if result.text:
        print("TRANSCRIPT")
        print("=" * 60)
        print(result.text)
        try:
            print(f"\nSaved to: {save_transcript(result.text, output)}")
        except OSError as exc:
            print(f"\nERROR (saving): could not write '{output}': {exc}")
            return 1
    else:
        print("No speech could be transcribed, so no file was written.")

    print(f"Chunks: {result.chunks_total} | unintelligible: {result.chunks_unintelligible}"
          + (f" | audio length: {result.audio_seconds:.1f}s" if result.audio_seconds else ""))
    if result.error:
        print(f"\nStopped early because of an error:\n  {result.error}")
        return 1
    return 0


if __name__ == "__main__":
    # By default, pass an empty list inside interactive notebooks to avoid SystemExit
    import ipykernel
    args_to_pass = [] if "ipykernel" in sys.modules else None
    exit_code = main(args_to_pass)
    if exit_code and "ipykernel" not in sys.modules:
        sys.exit(exit_code)
