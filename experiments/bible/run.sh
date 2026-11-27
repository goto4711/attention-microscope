#!/usr/bin/env bash
# Bible parallel pair, steps 1-3. Run from the repository root:
#   bash experiments/bible/run.sh            # full sample (1000 verse pairs)
#   bash experiments/bible/run.sh --limit 50 # quick trial
# Needs data/patois/jnt.txt; downloads WEB (ebible.org) and Pythia on first run.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${MODEL:-EleutherAI/pythia-410m}

if [ ! -f data/web/eng-web_vpl.txt ]; then
  mkdir -p data/web
  curl -sSL -o data/web/eng-web_vpl.zip https://ebible.org/Scriptures/eng-web_vpl.zip
  unzip -o -q data/web/eng-web_vpl.zip -d data/web
fi
python experiments/bible/parse_jnt.py
python experiments/bible/align.py
python experiments/bible/features.py
python experiments/bible/extract.py --model "$MODEL" "$@"
python experiments/bible/probe.py --model "${MODEL##*/}"
