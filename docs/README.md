# framesieve docs

- **[Quickstart](quickstart.md)** — install, two commands, the Python API, and
  where it will not help
- **[API reference](api.md)** — `VideoIndex`, `SearchResults`, `Hit`, the
  selection strategies, and `framesieve.pooling`
- **[How it works](how-it-works.md)** — the cascade, the cost model, why the
  index is small, and what this is not
- **[Scaling to a library](scaling.md)** — when to move from a per-video index
  in memory to LanceDB on disk, and how
- **[METHOD.md](METHOD.md)** — measurement protocol and the traps that produced
  silently wrong results before they were caught
- **[frame-access.md](frame-access.md)** — the three ways to get a frame back by
  timestamp, and what each costs

## The write-ups

- **[Search a day of video in a second](https://batchnorm.com)** — the measurements behind
  the design, including the parts that failed
- **[The pooling function nobody tunes](https://batchnorm.com)** — the finding that fell
  out of building it, which turned out not to be about video

Both live on the blog rather than in this repo. Every number in them comes from
an artefact under `runs/`, which is here, so they are checkable against the code
that produced them.
