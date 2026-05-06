#!/usr/bin/env bash
echo "kick/snare:" $(find /Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/input/audio -type f -name "*.wav" -exec basename {} \; | awk -F'[_.]' '$3=="kick" || $3=="snare"' | wc -l) && \
echo "kickstab/snarestab:" $(find /Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/input/audio -type f -name "*.wav" -exec basename {} \; | awk -F'[_.]' '$3=="kickstab" || $3=="snarestab"' | wc -l) && \
echo "acappella:" $(find /Users/martinconnor/Desktop/x-drone-meditation-xvii/7-python/generate-weighted-playlist/input/audio -type f -name "*.wav" -exec basename {} \; | awk -F'[_.]' '$3=="acappella"' | wc -l)
