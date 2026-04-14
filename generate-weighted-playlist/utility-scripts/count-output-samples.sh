#!/usr/bin/env zsh
ls /Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/output/audio/*.wav | sed 's|.*/||; s/_vol-.*//' | awk '
  /kickstab|snarestab/ { stab++; next }
  /_kick|_snare/       { ks++;   next }
  /acappella/          { acap++; next }
  /strings/            { str++;  next }
  /^silence/           { sil++;  next }
  END {
    non_str = ks + stab + acap
    printf "kicksnare : %d\nstab      : %d\nacappella : %d\nstrings   : %d\nsilence   : %d\n\nnon-strings total : %d\n  kicksnare : %.1f%%\n  stab      : %.1f%%\n  acappella : %.1f%%\n\nall-samples total : %d\n",
      ks, stab, acap, str, sil, non_str,
      (non_str > 0 ? ks/non_str*100 : 0),
      (non_str > 0 ? stab/non_str*100 : 0),
      (non_str > 0 ? acap/non_str*100 : 0),
      ks+stab+acap+str+sil
  }
'
