#!/usr/bin/env python3
"""Find a moment by what was shown, what was said, or what was written on screen.

    python examples/05_search_speech_and_frames.py talk.mp4 "the part about pricing"

audio=True transcribes with Whisper; ocr=True reads the text in the frames. Both
index timed spans beside the frame embeddings.

A frame similarity, a spoken sentence and a line of on-screen text are three
different quantities, so they are never ranked against each other: each is
ranked within itself, and when several point at the same second the result comes
back once naming all of them -- which is the strongest signal any of them gives.
"""
import sys

import framesieve as fs

video_path = sys.argv[1] if len(sys.argv) > 1 else "data/demo_clip.mp4"
query = sys.argv[2] if len(sys.argv) > 2 else "a gadget"

video = fs.open(video_path, audio=True, ocr=True)
print(video)
print(f"searchable by: {', '.join(video.sources)}")
if video.has_speech:
    print(f"  {len(video.speech)} transcript spans")
if video.has_text:
    print(f"  {len(video.text)} frames carried legible text")
print()

sources = [(None, "everything, merged")] + [(s, f"{s} only") for s in video.sources]
for source, label in sources:
    print(f"--- {label}")
    for hit in video.search(query, k=5, source=source):
        said = f"   {hit.text[:56]}" if hit.text else ""
        print(f"    {hit.timecode:>9}  {hit.source:<7} {hit.score:.3f}{said}")
    print()

# Agreement is the interesting case: raise merge_gap_s to pair hits that are
# near each other rather than exactly aligned.
agreed = [h for h in video.search(query, k=8, merge_gap_s=20.0)
          if "+" in h.source]
if agreed:
    print("where more than one signal agreed:")
    for h in agreed:
        print(f"    {h.timecode}  {h.source:<20} {(h.text or '')[:52]}")
else:
    print("no moment where two signals agreed on this query")
