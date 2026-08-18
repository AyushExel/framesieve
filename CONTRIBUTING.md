# Contributing

Bug reports, benchmarks on your own footage, and encoder or VLM backends are all
welcome. A few things about how this repo works that will save you time.

## Setup

```bash
git clone <repo> && cd framesieve
python3 -m venv .venv

# torch first, from the index for YOUR platform. Installing it as a transitive
# dependency is the single most common way to end up on a CPU-only wheel, which
# is silent and about 30x slower.
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128

.venv/bin/pip install -e ".[dev,vlm]"
.venv/bin/pytest -q
.venv/bin/ruff check .
```

You also need `ffmpeg` with h.264 support on your `PATH`.

## What the tests cover, and what they do not

`tests/test_api.py` is the public contract: if you change anything in
`framesieve/api.py`, those tests should have to change too, and that is the
signal to think about whether it is a breaking change.

`tests/test_core.py` is the algorithms — event recall, the selection strategies,
score pooling. These run in about a second and touch no model.

Anything needing a GPU, a model download or a video file **skips itself** rather
than failing. That is deliberate: CI runs on CPU, and a red build should mean a
real bug rather than a missing dependency. If you add a test that needs one of
those, guard it the same way.

## Measurements

Numbers in the README and in `docs/` are generated, not typed. If you change
something that moves one, regenerate the artefact under `runs/`:

```bash
.venv/bin/python scripts/verify.py       # 13 checks on env, decode and determinism
```

(`scripts/build_post.py` is author tooling for the write-ups: it rebuilds
whichever `docs/*.template.html` are present from `runs/` and skips the ones
that are not. The templates live with the blog, not in this repo, so from a
clone it has little to do.)

`scripts/verify.py` is worth running before you send a change. It checks the
things that, if wrong, corrupt a result silently rather than crashing —
including that the index and the refine stage return **the same image** for the
same timestamp. That was false for the first day of this project.

## Style

`ruff check .` is the whole style gate, and `pyproject.toml` says which rules are
relaxed where. Two conventions it does not enforce:

- **Comments say why, not what.** The code says what it does. A comment earns its
  place by recording a decision, a measurement, or a trap.
- **A number in a comment or a docstring should be one you measured.** If you
  cannot point at the run that produced it, leave it out.

## Adding an encoder

`framesieve/encoders.py` holds a registry and one class per family. A new encoder
needs `encode_frames`, `encode_text`, and a pinned revision — revisions are
pinned because an index built against a silently-updated checkpoint is not
comparable with one built last week, and nothing will tell you.
