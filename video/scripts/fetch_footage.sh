#!/bin/bash
# Downloads the Mixkit (Free License) clips listed in ASSETS.md into public/footage/src
set -euo pipefail
cd "$(dirname "$0")/../public/footage" && mkdir -p src && cd src
get() { [ -s "$1.mp4" ] || curl -fsSL -A "Mozilla/5.0" -o "$1.mp4" "$2"; }
get 171 https://assets.mixkit.co/videos/171/171-720.mp4
get 100917 https://assets.mixkit.co/slmqd3rqqid7lnj7kq9bdmhbonjw
get 100899 https://assets.mixkit.co/ra8udcqr0qh570sq2b7m9zte5q20
get 100898 https://assets.mixkit.co/cmvjew45m5hsp7p8vo7quw4j2v09
get 52304 https://assets.mixkit.co/videos/52304/52304-1080.mp4
get 52312 https://assets.mixkit.co/videos/52312/52312-1080.mp4
get 3759 https://assets.mixkit.co/videos/3759/3759-1080.mp4
get 50951 https://assets.mixkit.co/videos/50951/50951-1080.mp4
get 41999 https://assets.mixkit.co/videos/41999/41999-1080.mp4
get 3465 https://assets.mixkit.co/videos/3465/3465-1080.mp4
get 4426 https://assets.mixkit.co/videos/4426/4426-1080.mp4
get 3463 https://assets.mixkit.co/videos/3463/3463-1080.mp4
get 1038 https://assets.mixkit.co/videos/1038/1038-1080.mp4
get 40938 https://assets.mixkit.co/videos/40938/40938-1080.mp4
get 33899 https://assets.mixkit.co/videos/33899/33899-1080.mp4
echo "footage ready: $(ls | wc -l) files"
