#!/usr/bin/env bash
# Fetch everything the experiments need. Idempotent; safe to re-run.
#
#   ./scripts/fetch_data.sh cabride    # 4.5 h test video (4.1 GB)   -- needed for the main result
#   ./scripts/fetch_data.sh videomme   # Video-MME long split (95 GB) -- needed for the benchmark
#   ./scripts/fetch_data.sh momentseeker  # MomentSeeker (94 GB) -- the task-matched one
#   ./scripts/fetch_data.sh lvb        # LongVideoBench (162 GB, gated)
#   ./scripts/fetch_data.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
PY=${PY:-.venv/bin/python}
HF=${HF:-.venv/bin/hf}

cabride() {
  # Glasgow - Fort William - Mallaig cab ride (2012), Internet Archive.
  # 4 h 31 m, 960x720, h264, 25 fps. Chosen because it is long, freely
  # redistributable, has a stable identifier, and contains rare brief events
  # (tunnels, level crossings, passing trains) in hours of similar-looking track.
  local out=data/glasgow_mallaig.mp4
  if [ -s "$out" ]; then echo "have $out"; return; fi
  echo "downloading cab ride (4.1 GB)..."
  curl -sL --retry 8 --retry-all-errors -o "$out" \
    "https://archive.org/download/Glasgow_-_Fort_William_-_Mallaig_-_Cab_Ride_2012/Glasgow%20-%20Fort%20William%20-%20Mallaig%20-%20Cab%20Ride%20%282012%29.mp4"
  echo "wrote $out"
}

demo_clip() {
  # the 400 s slice the quick-start and the frame-access benchmarks use
  local out=data/demo_clip.mp4
  [ -s "$out" ] && { echo "have $out"; return; }
  cabride
  ffmpeg -v error -y -ss 14900 -i data/glasgow_mallaig.mp4 -t 400 -c copy "$out"
  echo "wrote $out"
}

videomme() {
  # ungated. The long split is 300 videos / 205.5 h / 900 questions.
  [ -d data/vmme_long ] && [ "$(ls data/vmme_long | wc -l)" -ge 300 ] && \
    { echo "have data/vmme_long"; return; }
  echo "downloading Video-MME (95 GB)..."
  $HF download lmms-lab/Video-MME --repo-type dataset --local-dir data/vmme_raw --max-workers 8
  echo "extracting the long split..."
  $PY - <<'EOF'
import pyarrow.parquet as pq, zipfile, os, glob
from huggingface_hub import hf_hub_download
p = hf_hub_download("lmms-lab/Video-MME", "videomme/test-00000-of-00001.parquet",
                    repo_type="dataset")
want = set(pq.read_table(p).to_pandas().query("duration=='long'").videoID.unique())
os.makedirs("data/vmme_long", exist_ok=True)
have = {os.path.splitext(f)[0] for f in os.listdir("data/vmme_long")}
for z in sorted(glob.glob("data/vmme_raw/videos_chunked_*.zip")):
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            vid = os.path.splitext(os.path.basename(n))[0]
            if vid in want and vid not in have:
                with zf.open(n) as s, open(f"data/vmme_long/{vid}.mp4", "wb") as d:
                    while (b := s.read(1 << 22)):
                        d.write(b)
print(len(os.listdir("data/vmme_long")), "videos extracted")
EOF
}

momentseeker() {
  # Long-video moment retrieval: 1,000 text queries over 265 videos averaging
  # 500 s+. This is the benchmark whose task matches what framesieve does.
  [ -d data/ms_videos ] && [ "$(ls data/ms_videos | wc -l)" -ge 265 ] && \
    { echo "have data/ms_videos"; return; }
  echo "downloading MomentSeeker (94 GB)..."
  $HF download avery00/MomentSeeker --repo-type dataset --local-dir data/ms_raw --max-workers 8
  echo "reassembling and extracting the referenced videos..."
  ( cd data/ms_raw && cat videos.tar.gz.part_* > videos.tar.gz )
  $PY - <<'EOF'
import json, os
d = json.load(open("data/ms_raw/t2v.json"))
want = sorted({os.path.basename(x["src_video_path"]) for x in d})
open("/tmp/ms_paths.txt", "w").write("\n".join("videos/" + v for v in want))
print(len(want), "videos referenced")
EOF
  mkdir -p data/ms_videos
  tar xzf data/ms_raw/videos.tar.gz -C data/ms_videos --strip-components=1 -T /tmp/ms_paths.txt
  rm -f data/ms_raw/videos.tar.gz data/ms_raw/videos.tar.gz.part_*
  echo "extracted $(ls data/ms_videos | wc -l) videos"
}

lvb() {
  # GATED: accept the terms at https://huggingface.co/datasets/longvideobench/LongVideoBench
  # while logged in, then `hf auth login`.
  [ -d data/lvb_raw ] && { echo "have data/lvb_raw"; return; }
  echo "downloading LongVideoBench (162 GB, gated)..."
  $HF download longvideobench/LongVideoBench --repo-type dataset \
    --local-dir data/lvb_raw --max-workers 8
}

case "${1:-all}" in
  cabride)  cabride; demo_clip ;;
  demo)     demo_clip ;;
  videomme) videomme ;;
  momentseeker|ms) momentseeker ;;
  lvb)      lvb ;;
  all)      cabride; demo_clip; videomme; momentseeker; lvb ;;
  *) echo "usage: $0 {cabride|demo|videomme|momentseeker|lvb|all}"; exit 1 ;;
esac
