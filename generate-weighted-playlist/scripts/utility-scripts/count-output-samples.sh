#!/usr/bin/env zsh
ls /Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/output/audio/*.wav | sed 's|.*/||' | awk '
{
  name = $0
  sub(/\.wav$/, "", name)

  # Detect group from sample type in filename (stab checked first to avoid matching _kick in kickstab)
  if (name ~ /kickstab|snarestab/) grp = "stab"
  else if (name ~ /_kick|_snare/)  grp = "ks"
  else if (name ~ /acappella/)     grp = "acap"
  else if (name ~ /strings/)       grp = "str"
  else if (name ~ /^silence/)      grp = "sil"
  else                             grp = "other"

  # Beat count: last underscore-separated component is the rhythm name (e.g. "quarter-eighth-eighth").
  # All beat-name tokens are single hyphenless words, so hyphen count + 1 = beat count.
  n = split(name, parts, "_")
  n_beats = split(parts[n], b, "-")

  if      (grp == "ks")   { ks_f++;   ks_b   += n_beats }
  else if (grp == "stab") { stab_f++; stab_b += n_beats }
  else if (grp == "acap") { acap_f++; acap_b += n_beats }
  else if (grp == "str")  { str_f++ }
  else if (grp == "sil")  { sil_f++ }
}
END {
  non_str_b = ks_b + stab_b + acap_b
  non_str_f = ks_f + stab_f + acap_f
  printf "%-10s %6s  %6s\n", "group", "files", "beats"
  printf "%-10s %6d  %6d\n", "kicksnare", ks_f, ks_b
  printf "%-10s %6d  %6d\n", "stab",      stab_f, stab_b
  printf "%-10s %6d  %6d\n", "acappella", acap_f, acap_b
  printf "%-10s %6d\n",      "strings",   str_f
  printf "%-10s %6d\n\n",    "silence",   sil_f

  printf "beat ratio (target 40:40:20)\n"
  printf "  kicksnare : %.1f%%\n", (non_str_b > 0 ? ks_b/non_str_b*100 : 0)
  printf "  stab      : %.1f%%\n", (non_str_b > 0 ? stab_b/non_str_b*100 : 0)
  printf "  acappella : %.1f%%\n\n", (non_str_b > 0 ? acap_b/non_str_b*100 : 0)

  printf "file ratio\n"
  printf "  kicksnare : %.1f%%\n", (non_str_f > 0 ? ks_f/non_str_f*100 : 0)
  printf "  stab      : %.1f%%\n", (non_str_f > 0 ? stab_f/non_str_f*100 : 0)
  printf "  acappella : %.1f%%\n\n", (non_str_f > 0 ? acap_f/non_str_f*100 : 0)

  printf "all-samples total : %d\n", ks_f + stab_f + acap_f + str_f + sil_f
}
'
