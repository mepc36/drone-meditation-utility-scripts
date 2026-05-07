# file: detect_note.py
import sys, os, math, argparse, csv
from pathlib import Path
import numpy as np
import soundfile as sf

SUPPORTED_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc"}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def freq_to_note(f_hz, a4=440.0):
    if f_hz <= 0 or not np.isfinite(f_hz):
        return None
    midi = 69 + 12 * math.log2(f_hz / a4)
    midi_rounded = int(round(midi))
    cents = int(round(1200 * (midi - midi_rounded)))
    name = NOTE_NAMES[midi_rounded % 12]
    octave = midi_rounded // 12 - 1
    return {"name": name, "octave": octave, "midi": midi_rounded, "cents": cents}

def dominant_freq(y, sr):
    # mono + normalize
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # analyze up to 5s
    N = min(len(y), int(sr * 5.0))
    y = y[:N]

    # window + zero pad for finer bin resolution
    win = np.hanning(N)
    y_win = y * win
    pad = 8  # zero-padding factor
    nfft = int(2 ** math.ceil(math.log2(max(16, N * pad))))
    spec = np.fft.rfft(y_win, n=nfft)
    mag = np.abs(spec)

    # ignore DC and very low bins (< 50 Hz)
    low_bin = max(1, int(50.0 * nfft / sr))
    mag[:low_bin] = 0.0

    # find peak bin
    k = int(np.argmax(mag))
    if k <= 0 or k >= len(mag) - 1:
        return k * sr / nfft

    # quadratic interpolation around the peak
    alpha = mag[k - 1]
    beta  = mag[k]
    gamma = mag[k + 1]
    denom = (alpha - 2 * beta + gamma)
    delta = 0.0 if denom == 0 else 0.5 * (alpha - gamma) / denom
    peak_bin = k + delta
    freq = peak_bin * sr / nfft
    return float(freq)

def analyze(path):
    try:
        y, sr = sf.read(path, always_2d=False)
    except Exception as e:
        return {"path": str(path), "error": f"read_error: {e}"}
    try:
        f = dominant_freq(y, sr)
        note = freq_to_note(f)
        if note is None:
            return {"path": str(path), "frequency_hz": f, "note": None, "display": "no stable pitch detected"}
        name = f"{note['name']}{note['octave']}"
        disp = f"{name}  ({f:.2f} Hz, {note['cents']} cents)"
        return {"path": str(path), "frequency_hz": float(f), "note": name, "cents": int(note["cents"]), "midi": int(note["midi"]), "display": disp}
    except Exception as e:
        return {"path": str(path), "error": f"analyze_error: {e}"}

def collect_files(inputs, recursive=False):
    files = []
    def add_dir(d: Path):
        if recursive:
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                    files.append(p)
        else:
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                    files.append(p)

    if not inputs:
        inputs = [Path("./input")]

    for item in inputs:
        p = Path(item)
        if p.is_dir():
            add_dir(p)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
        else:
            # ignore unsupported entries silently
            pass

    # de-dup + sort
    uniq = sorted(set(map(lambda x: x.resolve(), files)))
    return uniq

def main():
    ap = argparse.ArgumentParser(description="Detect dominant note of audio files.")
    ap.add_argument("paths", nargs="*", help="Files or folders to analyze. Defaults to ./input if none supplied.")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into folders.")
    ap.add_argument("--csv", type=str, help="Optional CSV output path.")
    args = ap.parse_args()

    files = collect_files(args.paths, recursive=args.recursive)
    if not files:
        print("No audio files found. Supported: " + ", ".join(sorted(SUPPORTED_EXTS)))
        sys.exit(1)

    results = []
    for fpath in files:
        res = analyze(fpath)
        results.append(res)
        if "error" in res:
            print(f"{res['path']}: {res['error']}")
        else:
            print(f"{res['path']}: {res['display']}")

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path","frequency_hz","note","cents","midi","error"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "path": r.get("path"),
                    "frequency_hz": r.get("frequency_hz"),
                    "note": r.get("note"),
                    "cents": r.get("cents"),
                    "midi": r.get("midi"),
                    "error": r.get("error")
                })
        print(f"Wrote CSV: {csv_path}")

if __name__ == "__main__":
    main()
