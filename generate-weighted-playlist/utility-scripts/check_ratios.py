import os, json, re

cfg_path = "/Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/input/config/config.json"
d = "/Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/output/audio"

with open(cfg_path) as f:
    cfg = json.load(f)

bpms = [int(x) for x in str(cfg["bpms"]).split(":")]
bpm_percents = [int(x) for x in str(cfg.get("slow_to_fast_bpm_percents", "100")).split(":")]
vol_values = [float(x) for x in str(cfg.get("loud_medium_soft_values", "0")).split(":")]
vol_percents = [int(x) for x in str(cfg.get("loud_medium_soft_percents", "100")).split(":")]
pan_percents = [int(x) for x in str(cfg.get("center_diagonal_dualpan_leftorright_percents", "25:25:25:25")).split(":")]


files = [f for f in os.listdir(d) if f.endswith('.wav')]
total = len(files)
strings = [f for f in files if '_strings' in f]
non_strings = [f for f in files if '_strings' not in f]
ns = len(non_strings)

# BPM counts from filenames
bpm_counts = {bpm: 0 for bpm in bpms}
for f in non_strings:
    m = re.search(r'_bpm-(\d+)_', f)
    if m:
        b = int(m.group(1))
        if b in bpm_counts:
            bpm_counts[b] += 1

# Volume counts
vol_counts = {}
for v in vol_values:
    tag = f"_vol-{abs(int(v))}_"
    vol_counts[v] = sum(1 for f in non_strings if tag in f)

# Panning
center    = sum(1 for f in files if f.endswith('_center.wav'))
diagonal  = sum(1 for f in files if f.endswith('_diagonal.wav'))
dualpan   = sum(1 for f in files if f.endswith('_dualpan.wav'))
leftright = sum(1 for f in files if f.endswith('_leftorright.wav'))

print(f"Total files: {total}  |  non-strings: {ns}  |  strings: {len(strings)}")
print()

print(f"BPM  (config target {':'.join(f'{p}%' for p in bpm_percents)} for BPMs {':'.join(str(b) for b in bpms)})")
for bpm, pct in zip(bpms, bpm_percents):
    count = bpm_counts.get(bpm, 0)
    actual = count / ns * 100 if ns else 0
    label = "slow" if bpm == min(bpms) else "fast"
    print(f"  {label}({bpm}): {count}/{ns} = {actual:.1f}%  (target {pct}%)")

print()
print(f"VOLUME (config target {':'.join(f'{p}%' for p in vol_percents)})")
for v, pct in zip(vol_values, vol_percents):
    count = vol_counts.get(v, 0)
    actual = count / ns * 100 if ns else 0
    label = "loud" if v == max(vol_values) else "quiet"
    print(f"  {label}({int(v)}dB): {count}/{ns} = {actual:.1f}%  (target {pct}%)")

print()
print(f"PANNING (config target center:{pan_percents[0]}% diagonal:{pan_percents[1]}% dualpan:{pan_percents[2]}% leftorright:{pan_percents[3]}%)")
for label, count in [("center", center), ("diagonal", diagonal), ("dualpan", dualpan), ("leftorright", leftright)]:
    print(f"  {label}: {count}/{total} = {count/total*100:.1f}%")

print()
print(f"STRINGS (all strings added exactly once, no duplicates)")
print(f"  non-strings: {ns}/{total} = {ns/total*100:.1f}%  |  strings: {len(strings)}/{total} = {len(strings)/total*100:.1f}%")
