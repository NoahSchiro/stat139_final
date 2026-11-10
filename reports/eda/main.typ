#show link: set text(fill: blue)

#let title = "Exploratory data analysis: skip behavior in Spotify listening sessions"
#let authors = "Gabe Gonzalez, Debra Ritter, Noah Schiro, Franco Valencia, Elaine Zhang"

#set page(paper: "us-letter", margin: (x: 0.85in, top: 0.6in, bottom: 0.75in), numbering: "1")
#set document(title: title, author: authors)
#set text(size: 10.5pt)
#set par(spacing: 0.7em)
#set enum(spacing: 0.4em)
#set list(spacing: 0.4em)
#show heading: set text(size: 11pt)
#show heading: set block(above: 1em, below: 0.35em)
#show figure: set block(breakable: true)

// A figure from assets/ with an adjustable width.
#let fig(name, caption, width: 100%) = figure(
  image("../../assets/" + name, width: width),
  caption: caption,
)

// Build tables from row tuples.
#let rows-of(rows) = rows.map(
  row => row.map(cell => [#cell])
).flatten()

#align(center, text(size: 14pt, weight: "bold")[#title])
#align(center, authors)
#align(center, text(size: 9.5pt, fill: luma(90))[November 10, 2026])

= Overview

This report documents the exploratory data analysis of the Spotify music-streaming
session logs. The raw data consists of 10,000 listening sessions (`data/log_mini.csv`,
167,880 play events) joined to per-track audio features (`data/tf_mini.csv`, 50,704
unique tracks). Cleaning (`src/clean.py`) parses dates, coerces behavioral columns to
booleans, masks impossible feature values (for example, tempo = 0, loudness <= -60 dB,
time signature = 0) as missing, and derives the outcome variable `skip_class`, the *most
severe* skip flag of each play ordered.

Throughout, *skip rate* refers to the share of plays with `skip_2 = True` (51.7%
overall) unless stated otherwise. Findings:

- *Skipping is the norm*: 51.7% of plays end in a short-listen skip, and only 33.3% of
  tracks are played all the way through.
- *Skip behavior is strongly autocorrelated within a session*: P(skip $|$ previous play
  skipped) = 74.9% versus 28.0% after a completed play.
- *Session structure matters*: the first track of a session is skipped far less often
  (37.7%) than later positions (52.7% on average).
- *Listening context separates cleanly*: personalized playlists and catalog browsing
  skip least (about 45%), while radio, charts, and the user's own collection skip most
  (about 55%).
- *Audio features are individually weak predictors of skipping*: The strongest correlation
  is $|r| = 0.037$
- *Data quality is high after cleaning*: at most 39 missing values in any play-level
  column, 3,145 plays (1.9%) with no outcome label, and 167 plays with durations above
  10 minutes.

The following is a list of questions, broken down into a few categories:
- Summary
- Outcome structure
- Session dynamics
- Listening context
- Audio features

Each question is accompanied with a table, figure, or bit of text which answers that question.

= Descriptive summary <sec-summary>

#block(width: 100%)[
  _Analysis 0: What does each variable look like: missingness, central tendency,
  spread, and level counts?_
]

#figure(
  text(size: 8pt, table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    inset: (y: 2.5pt),
    table.header([Variable], [Non-missing], [Missing], [Mean], [Median], [SD], [IQR]),
    ..rows-of((
      ("session_position", "167,880", "0", "9.326", "9.000", "5.458", "9.000"),
      ("session_length", "167,880", "0", "17.652", "20.000", "3.422", "5.000"),
      ("hist_user_behavior_n_seekfwd", "167,880", "0", "0.039", "0.000", "0.367", "0.000"),
      ("hist_user_behavior_n_seekback", "167,880", "0", "0.046", "0.000", "0.607", "0.000"),
      ("hour_of_day", "167,880", "0", "14.193", "15.000", "5.996", "8.000"),
      ("duration", "167,880", "0", "215.893", "212.027", "60.224", "57.297"),
      ("release_year", "167,880", "0", "2013.369", "2017.000", "9.740", "3.000"),
      ("us_popularity_estimate", "167,880", "0", "99.745", "99.989", "0.893", "0.085"),
      ("acousticness", "167,880", "0", "0.218", "0.115", "0.246", "0.306"),
      ("beat_strength", "167,880", "0", "0.548", "0.551", "0.159", "0.232"),
      ("bounciness", "167,880", "0", "0.580", "0.594", "0.179", "0.273"),
      ("danceability", "167,880", "0", "0.669", "0.687", "0.159", "0.224"),
      ("dyn_range_mean", "167,880", "0", "9.098", "8.966", "2.540", "3.540"),
      ("energy", "167,880", "0", "0.627", "0.631", "0.182", "0.247"),
      ("flatness", "167,880", "0", "1.008", "1.014", "0.039", "0.045"),
      ("instrumentalness", "167,880", "0", "0.032", "0.000", "0.143", "0.000"),
      ("liveness", "167,880", "0", "0.190", "0.125", "0.151", "0.136"),
      ("loudness", "167,878", "2", "-7.049", "-6.506", "3.159", "3.149"),
      ("mechanism", "167,880", "0", "0.597", "0.639", "0.208", "0.305"),
      ("organism", "167,880", "0", "0.348", "0.306", "0.189", "0.265"),
      ("speechiness", "167,880", "0", "0.142", "0.086", "0.133", "0.161"),
      ("tempo", "167,845", "35", "122.700", "123.943", "29.772", "47.070"),
      ("valence", "167,880", "0", "0.459", "0.442", "0.229", "0.346"),
      ("acoustic_vector_0", "167,880", "0", "-0.516", "-0.659", "0.386", "0.638"),
      ("acoustic_vector_1", "167,880", "0", "0.269", "0.312", "0.194", "0.149"),
      ("acoustic_vector_2", "167,880", "0", "0.206", "0.235", "0.165", "0.123"),
      ("acoustic_vector_3", "167,880", "0", "-0.016", "0.044", "0.266", "0.305"),
      ("acoustic_vector_4", "167,880", "0", "-0.140", "-0.273", "0.314", "0.312"),
      ("acoustic_vector_5", "167,880", "0", "0.048", "0.038", "0.116", "0.170"),
      ("acoustic_vector_6", "167,880", "0", "-0.283", "-0.376", "0.301", "0.309"),
      ("acoustic_vector_7", "167,880", "0", "0.060", "0.144", "0.262", "0.215"),
    )),
  )),
  caption: [Summary of the 31 numeric variables.],
)

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    inset: (y: 2.5pt),
    table.header([Binary variable], [True], [False]),
    ..rows-of((
      ("skip_1", "70,762 (42.2%)", "97,118 (57.8%)"),
      ("skip_2", "86,824 (51.7%)", "81,056 (48.3%)"),
      ("skip_3", "108,850 (64.8%)", "59,030 (35.2%)"),
      ("not_skipped", "55,884 (33.3%)", "111,996 (66.7%)"),
      ("context_switch", "6,867 (4.1%)", "161,013 (95.9%)"),
      ("no_pause_before_play", "128,865 (76.8%)", "39,015 (23.2%)"),
      ("short_pause_before_play", "24,617 (14.7%)", "143,263 (85.3%)"),
      ("long_pause_before_play", "29,015 (17.3%)", "138,865 (82.7%)"),
      ("hist_user_behavior_is_shuffle", "51,830 (30.9%)", "116,050 (69.1%)"),
      ("premium", "136,053 (81.0%)", "31,827 (19.0%)"),
    )),
  ),
  caption: [Counts of the ten binary play-level flags.],
)

#figure(
  text(size: 8.5pt, table(
    columns: 4,
    align: (left, left, right, right),
    inset: (y: 2.5pt),
    table.header([Variable], [Level], [n], [Share]),
    ..rows-of((
      ("skip_class", "completed", "55,884", "33.3%"),
      ("skip_class", "skip_3", "22,027", "13.1%"),
      ("skip_class", "skip_2", "16,062", "9.6%"),
      ("skip_class", "skip_1", "70,762", "42.2%"),
      ("skip_class", "unknown", "3,145", "1.9%"),
      ("context_type", "catalog", "38,263", "22.8%"),
      ("context_type", "charts", "2,441", "1.5%"),
      ("context_type", "editorial_playlist", "35,663", "21.2%"),
      ("context_type", "personalized_playlist", "2,976", "1.8%"),
      ("context_type", "radio", "21,107", "12.6%"),
      ("context_type", "user_collection", "67,430", "40.2%"),
      ("reason_start", "appload", "1,942", "1.2%"),
      ("reason_start", "backbtn", "13,034", "7.8%"),
      ("reason_start", "clickrow", "17,514", "10.4%"),
      ("reason_start", "endplay", "20", "0.0%"),
      ("reason_start", "fwdbtn", "79,912", "47.6%"),
      ("reason_start", "playbtn", "155", "0.1%"),
      ("reason_start", "remote", "136", "0.1%"),
      ("reason_start", "trackdone", "55,075", "32.8%"),
      ("reason_start", "trackerror", "92", "0.1%"),
      ("reason_end", "backbtn", "13,129", "7.8%"),
      ("reason_end", "clickrow", "8", "0.0%"),
      ("reason_end", "endplay", "14,823", "8.8%"),
      ("reason_end", "fwdbtn", "81,389", "48.5%"),
      ("reason_end", "logout", "563", "0.3%"),
      ("reason_end", "remote", "267", "0.2%"),
      ("reason_end", "trackdone", "57,701", "34.4%"),
      ("key", "0", "18,085", "10.8%"),
      ("key", "1", "25,946", "15.5%"),
      ("key", "2", "14,577", "8.7%"),
      ("key", "3", "4,017", "2.4%"),
      ("key", "4", "10,217", "6.1%"),
      ("key", "5", "12,958", "7.7%"),
      ("key", "6", "11,653", "6.9%"),
      ("key", "7", "15,758", "9.4%"),
      ("key", "8", "12,715", "7.6%"),
      ("key", "9", "14,724", "8.8%"),
      ("key", "10", "13,371", "8.0%"),
      ("key", "11", "13,859", "8.3%"),
      ("mode", "major", "103,063", "61.4%"),
      ("mode", "minor", "64,817", "38.6%"),
      ("time_signature", "1.0", "1,017", "0.6%"),
      ("time_signature", "3.0", "7,546", "4.5%"),
      ("time_signature", "4.0", "157,101", "93.6%"),
      ("time_signature", "5.0", "2,177", "1.3%"),
    )),
  )),
  caption: [Level counts for categorical variables with more than two levels
  (`reason_start` = `hist_user_behavior_reason_start`, `reason_end` =
  `hist_user_behavior_reason_end`).],
)

= Outcome structure

#block[
  _Analysis 1: How are the four skip flags structured? Are skips nested (as in
  skip_1 $<=$ skip_2 $<=$ skip_3)?_
]

#figure(
  table(
    columns: 6,
    align: (center, center, center, center, right, left),
    inset: (y: 2.5pt),
    table.header([skip_1], [skip_2], [skip_3], [not_skipped], [Plays], [Reading]),
    ..rows-of((
      ("✓", "✓", "✓", "—", "70,762", "briefest-listen skip"),
      ("—", "✓", "✓", "—", "16,061", "skip_2"),
      ("—", "—", "✓", "—", "22,027", "skip_3 only"),
      ("—", "✓", "—", "—", "1", "nesting violation"),
      ("—", "—", "—", "✓", "55,884", "completed"),
      ("—", "—", "—", "—", "3,145", "unlabeled"),
    )),
  ),
  caption: [Observed flag patterns.],
)

#fig(
  "01_skip_flag_rates.png",
  [Skip-flag rates. The flags are nested: skip_3 is nearly twice as common as skip_1.],
)

#block[
  _Analysis 2: How deep into a track do listeners get, i.e. what is the distribution
  of skip_class?_
]

#fig(
  "02_skip_class_distribution.png",
  [Distribution of skip_class: the most severe skip flag of each play.],
)

= Session dynamics

#block[
  _Analysis 3: Does skip probability depend on position within the session?_
]

#fig(
  "03_skip_rate_by_position.png",
  [skip_2 rate by session position. Position 1 stands out as notably low.],
)

#block[
  _Analysis 4: Is skipping autocorrelated within a session, i.e. P(skip at t+1 |
  skip at t)?_
]

#fig(
  "04_skip_autocorrelation.png",
  [Conditional skip rates given the previous outcome within the same session.],
)

#block[
  _Analysis 5: How long do sessions run, is the length capped, and does length relate
  to skipping?_
]

#fig(
  "05_session_length.png",
  [Session-length distribution and its relation to the per-session skip rate.],
)

#block[
  _Analysis 6: Do people replay tracks within a session, and are replays skipped
  less?_
]

#fig(
  "06_within_session_repeats.png",
  [Within-session replays: frequency and skip rates of first vs repeat plays.],
)

#block[
  _Analysis 7: Do shuffle-mode sessions skip differently than sequentially ordered
  listening?_
]

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    inset: (y: 2.5pt),
    table.header([Context type], [Sequential], [Shuffle]),
    ..rows-of((
      ("catalog", "44.8%", "48.1%"),
      ("charts", "55.1%", "56.1%"),
      ("editorial_playlist", "50.9%", "50.7%"),
      ("personalized_playlist", "44.1%", "46.7%"),
      ("radio", "55.4%", "— (never shuffled)"),
      ("user_collection", "50.7%", "60.6%"),
    )),
  ),
  caption: [skip_2 rate by context type and shuffle mode.],
)

#fig(
  "07_shuffle_effect.png",
  [Shuffle vs sequential skip rates by context type.],
)

= Listening context

#block[
  _Analysis 8: How does skip rate vary by context_type? Why do algorithmic contexts
  skip more?_
]

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    inset: (y: 2.5pt),
    table.header([Context type], [Skip rate], [Plays]),
    ..rows-of((
      ("personalized_playlist", "45.2%", "2,976"),
      ("catalog", "45.7%", "38,263"),
      ("editorial_playlist", "50.8%", "35,663"),
      ("user_collection", "54.6%", "67,430"),
      ("radio", "55.4%", "21,107"),
      ("charts", "55.5%", "2,441"),
    )),
  ),
  caption: [skip_2 rate by context type, sorted ascending.],
)

#fig(
  "08_skip_rate_by_context.png",
  [Skip rate by context type against the overall rate (dashed line).],
)

#block[
  _Analysis 9: How long do consecutive forward-press chains run within a session?_
]

#fig(
  "09_skip_chain_lengths.png",
  [Distribution of consecutive forward-press chain lengths.],
)

#block[
  _Analysis 10: What is the relationship between pre-play pauses, context switches,
  and skipping?_
]

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    inset: (y: 2.5pt),
    table.header([Pause combination], [Skip rate], [Plays]),
    ..rows-of((
      ("no_pause (only)", "54.6%", "128,865"),
      ("short_pause + long_pause", "47.7%", "24,617"),
      ("none (session start)", "37.7%", "10,000"),
      ("long_pause (only)", "21.3%", "4,398"),
    )),
  ),
  caption: [skip_2 rate by pre-play pause combination. The `none` row is exactly the
  set of session-opening plays.],
)

#fig(
  "10_skip_rate_by_pause.png",
  [Skip rate by pre-play pause combination.],
)

#block[
  _Analysis 11: Do premium and free users skip differently, and does their context
  mix explain it?_
]

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    inset: (y: 2.5pt),
    table.header([Context type], [Free (% of plays)], [Premium (% of plays)]),
    ..rows-of((
      ("catalog", "20.9", "23.2"),
      ("charts", "1.0", "1.6"),
      ("editorial_playlist", "22.9", "20.9"),
      ("personalized_playlist", "0.9", "2.0"),
      ("radio", "8.4", "13.6"),
      ("user_collection", "45.9", "38.8"),
    )),
  ),
  caption: [Context mix of free vs premium plays (column shares sum to 100).],
)

#fig(
  "11_premium_vs_free.png",
  [Skip rates and context mix for premium vs free users.],
)

#block[
  _Analysis 12: Is there an hour-of-day effect on skipping? Does play volume confound
  it?_
]

#fig(
  "12_hour_of_day.png",
  [Skip rate and play volume by hour of day.],
)

= Audio features

#block[
  _Analysis 13: What do the marginal distributions of the audio features look like
  (skew, degenerate mass)?_
]

#fig(
  "13_feature_distributions.png",
  [Marginal histograms of the audio features.],
)

#block[
  _Analysis 14: Which numeric features correlate with skipping?_
]

#fig(
  "14_feature_correlations.png",
  [Correlation of each numeric feature with skip_2.],
  width: 62%,
)

#block[
  _Analysis 15: Does skip probability rise monotonically with track duration?_
]

#fig(
  "15_skip_rate_by_duration.png",
  [skip_2 rate across duration deciles.],
)

#block[
  _Analysis 16: Does the duration-skip relationship persist within each context
  type?_
]

#fig(
  "16_duration_by_context.png",
  [skip_2 rate by duration quintile within each context type.],
)

#block[
  _Analysis 17: How correlated are the audio features with each other?_
]

#fig(
  "17_feature_collinearity.png",
  [Correlation matrix of the audio features.],
  width: 80%,
)
