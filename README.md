# stat139_final

What makes listeners skip? A study of Spotify track skips by audio characteristics.

## Setup

The raw data are gitignored. Download `log_mini.csv` and `tf_mini.csv` from
[rachaelmmccue/Spotify-Sequential-Skip-Prediction](https://github.com/rachaelmmccue/Spotify-Sequential-Skip-Prediction)
(`data/training_set/` and `data/track_features/`) into `data/`, then:

```sh
uv sync
uv run python main.py eda     # cleans, aggregates, writes figures to assets/
uv run python main.py model   # fits the baseline WLS, writes assets/ and reports/eda/tables/
```

## Reports

Both reports are written in [Typst](https://typst.app). The proposal is self-contained;
the EDA report needs the figures and tables above and must be compiled from the repo root:

```sh
typst compile proposal/main.typ
typst compile --root . reports/eda/main.typ
```
