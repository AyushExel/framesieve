#!/usr/bin/env python3
"""Find a moment by what was said, what was shown, or both.

    python examples/05_search_speech_and_frames.py talk.mp4 "the part about pricing"

Indexing with audio=True transcribes the video with Whisper and indexes the
timed segments beside the frames. A frame similarity and a sentence similarity
are different quantities, so the two are never ranked against each other: each
is ranked within itself, and when both point at the same second the result comes
back once marked "both" -- which is the strongest signal either can give you.
"""
import sys

import framesieve as fs

video_path = sys.argv[1] if len(sys.argv) > 1 else "data/demo_clip.mp4"
query = sys.argv[2] if len(sys.argv) > 2 else "a gadget"

video = fs.open(video_path, audio=True)
print(video)
if not video.has_speech:
    sys.exit("no transcript — the file has no audio track")
print(f"{len(video.speech)} transcript segments\n")

for source, label in ((None, "both, merged"), ("visual", "frames only"),
                      ("speech", "transcript only")):
    print(f"--- {label}")
    for hit in video.search(query, k=5, source=source):
        said = f"   {hit.text[:56]}" if hit.text else ""
        print(f"    {hit.timecode:>9}  {hit.source:<7} {hit.score:.3f}{said}")
    print()

# Agreement is the interesting case: raise merge_gap_s to pair hits that are
# near each other rather than exactly aligned.
agreed = [h for h in video.search(query, k=8, merge_gap_s=20.0)
          if h.source == "both"]
if agreed:
    print("where the picture and the audio agree:")
    for h in agreed:
        print(f"    {h.timecode}  {h.text[:60]}")
else:
    print("no moment where both modalities agreed on this query")
