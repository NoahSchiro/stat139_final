#show link: set text(fill: rgb("#1a5fb4"))

#let title = "What Makes Listeners Skip? EDA and Baseline Model"
#let authors = "Elaine Zhang, Franco Valencia, Gabriel Gonzalez, Noah Schiro, Debra Ritter"

// Page setup is identical to proposal/main.typ.
#set page(paper: "us-letter", margin: (x: 1in, top: 0.8in, bottom: 0.8in),
  numbering: "1", number-align: center)
#set document(title: title, author: authors)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.62em, spacing: 1em)
#set enum(spacing: 0.7em, indent: 0.5em)
#set list(spacing: 0.7em, indent: 0.5em)
#set heading(numbering: "1.")
#show heading: set text(size: 12.5pt, weight: "bold")
#show heading: set block(above: 1.6em, below: 0.7em)
#show raw.where(block: false): set text(size: 9.5pt)

// Figure and table rules; the proposal has neither.
#show figure: set block(breakable: true, above: 1.2em, below: 1.2em)
#show figure.caption: set text(size: 9.5pt)
#show table: set text(size: 9pt)
#set table(inset: (x: 6pt, y: 3pt), stroke: (x: none, y: 0.4pt + luma(170)))

// A figure from assets/ with an adjustable width.
#let fig(name, caption, width: 100%) = figure(
  image("../../assets/" + name, width: width),
  caption: caption,
)

// Build tables from row tuples.
#let rows-of(rows) = rows.map(
  row => row.map(cell => [#cell])
).flatten()

#align(center)[
  #block(width: 85%, text(size: 16pt, weight: "bold")[#title])
  #v(0.6em)
  #text(size: 11pt)[#authors]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[Stat 139 Final Project --- EDA and Baseline Model (Milestone 3)]
]
#v(0.5em)
#line(length: 100%, stroke: 0.5pt + gray)
#v(0.5em)

= Overview

This report documents the exploratory data analysis (EDA) for our study of Spotify
music-streaming session logs. We are interested in whether, and which, audio
characteristics of a track are associated with how often listeners skip it, after
adjusting for popularity and listening context. Our main hypothesis is that, holding
popularity and context mix fixed, the audio features of a track are jointly associated
with its skip rate.

Throughout, a _play_ is one row of the log (one track played once in one session), a
_track_ is a song, and the _listening context_ is where the play came from: the user's
own collection, catalog browsing, an editorial playlist, a personalized playlist, radio,
or charts. Sections 2--4 describe the raw play-level data. Sections 5--7 move to the
units of analysis from the proposal: tracks with at least five plays for the regression,
and sessions within a listening context for the ANOVA. Section 7 fits the baseline
model. Additional EDA is in the appendix.

Main findings:

- 51.7% of plays result in the track being listened to only briefly before being skipped
  (`skip_2`). Only 33.3% of plays are played all the way through.
- Track position in a listening session matters. Skips are strongly autocorrelated, with
  a skip rate of 74.9% after a skip vs. 28.0% after a complete play. The first track of
  a session is skipped far less often (37.7%) than tracks later in a session (52.7%,
  averaged over positions 2 to 20).
- Personalized playlists and catalog browsing have the lowest skip rates among listening
  contexts, at approximately 45%. The user's own collection, radio, and charts have the
  highest, at approximately 55%.
- No single audio feature correlates strongly with the track-level skip rate (all
  $|r| < 0.09$), but the sixteen audio features together add significant explanatory
  power over the controls (nested $F_(16, 5370) = 9.96$, $p < 10^(-24)$), raising $R^2$
  from 0.063 to 0.090.
- Several audio features are highly collinear. For example, `bounciness`, `organism`,
  `beat_strength` and `mechanism` all have variance inflation factors above 30, so their
  individual coefficients cannot be meaningfully interpreted until redundant features are
  removed or combined.

= Data and source <sec-data>

The data is the public mini sample of the Spotify Music Streaming Sessions Dataset
(Brost, Mehrotra & Jehan, 2019, arXiv:1901.09851), obtained from
#link("https://github.com/rachaelmmccue/Spotify-Sequential-Skip-Prediction")[github.com/rachaelmmccue/Spotify-Sequential-Skip-Prediction].
It consists of two files: `log_mini.csv`, with 167,880 plays from 10,000 listening
sessions collected over 18 days, and `tf_mini.csv`, with audio features for the 50,704
unique tracks in the log. The join on track ID matched all 167,880 rows.

Cleaning (`src/clean.py`) parses dates, coerces the behavioral flags to booleans, and
masks physically impossible feature values as missing. This includes: a tempo of 0 (26
tracks), loudness at or below $-60$ dB (1 track), and a time signature of 0 (28 tracks).
No rows are dropped. After cleaning, at most 39 plays are missing any one feature, 3,145
plays (1.9%) have all four skip flags set to false, so we cannot tell how long they were
played, and 167 plays are of tracks longer than ten minutes.

@tab-numeric summarizes the numeric variables we use (the eight latent `acoustic_vector`
dimensions are omitted because the proposal excludes them from the model), @tab-binary
the behavioral flags, and @tab-categorical the categorical variables. Two variables have
unusual distributions. `us_popularity_estimate` is concentrated at the top of its range
(median 99.99, IQR 0.085), so it carries little information at the play level.
`instrumentalness` has a median of exactly zero: most tracks are vocal, with a small tail
of instrumental ones.

#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    table.header([Variable], [Non-missing], [Missing], [Mean], [Median], [SD], [IQR]),
    ..rows-of((
      ("session_position", "167,880", "0", "9.33", "9.00", "5.46", "9.00"),
      ("session_length", "167,880", "0", "17.65", "20.00", "3.42", "5.00"),
      ("hour_of_day", "167,880", "0", "14.19", "15.00", "6.00", "8.00"),
      ("duration (s)", "167,880", "0", "215.9", "212.0", "60.2", "57.3"),
      ("release_year", "167,880", "0", "2013.4", "2017", "9.74", "3.00"),
      ("us_popularity_estimate", "167,880", "0", "99.74", "99.99", "0.89", "0.085"),
      ("acousticness", "167,880", "0", "0.218", "0.115", "0.246", "0.306"),
      ("beat_strength", "167,880", "0", "0.548", "0.551", "0.159", "0.232"),
      ("bounciness", "167,880", "0", "0.580", "0.594", "0.179", "0.273"),
      ("danceability", "167,880", "0", "0.669", "0.687", "0.159", "0.224"),
      ("dyn_range_mean", "167,880", "0", "9.10", "8.97", "2.54", "3.54"),
      ("energy", "167,880", "0", "0.627", "0.631", "0.182", "0.247"),
      ("flatness", "167,880", "0", "1.008", "1.014", "0.039", "0.045"),
      ("instrumentalness", "167,880", "0", "0.032", "0.000", "0.143", "0.000"),
      ("liveness", "167,880", "0", "0.190", "0.125", "0.151", "0.136"),
      ("loudness (dB)", "167,878", "2", "-7.05", "-6.51", "3.16", "3.15"),
      ("mechanism", "167,880", "0", "0.597", "0.639", "0.208", "0.305"),
      ("organism", "167,880", "0", "0.348", "0.306", "0.189", "0.265"),
      ("speechiness", "167,880", "0", "0.142", "0.086", "0.133", "0.161"),
      ("tempo (BPM)", "167,845", "35", "122.7", "123.9", "29.8", "47.1"),
      ("valence", "167,880", "0", "0.459", "0.442", "0.229", "0.346"),
    )),
  ),
  caption: [Play-level summary of the numeric variables used in the analysis.],
) <tab-numeric>

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    table.header([Binary variable], [True], [False]),
    ..rows-of((
      ("skip_1 (very brief listen)", "70,762 (42.2%)", "97,118 (57.8%)"),
      ("skip_2 (brief listen; primary outcome)", "86,824 (51.7%)", "81,056 (48.3%)"),
      ("skip_3 (most of track played)", "108,850 (64.8%)", "59,030 (35.2%)"),
      ("not_skipped", "55,884 (33.3%)", "111,996 (66.7%)"),
      ("hist_user_behavior_is_shuffle", "51,830 (30.9%)", "116,050 (69.1%)"),
      ("premium", "136,053 (81.0%)", "31,827 (19.0%)"),
      ("context_switch", "6,867 (4.1%)", "161,013 (95.9%)"),
    )),
  ),
  caption: [Counts of the play-level flags that enter the analysis.],
) <tab-binary>

#figure(
  table(
    columns: 4,
    align: (left, left, right, right),
    table.header([Variable], [Level], [n], [Share]),
    ..rows-of((
      ("context_type", "user_collection", "67,430", "40.2%"),
      ("", "catalog", "38,263", "22.8%"),
      ("", "editorial_playlist", "35,663", "21.2%"),
      ("", "radio", "21,107", "12.6%"),
      ("", "personalized_playlist", "2,976", "1.8%"),
      ("", "charts", "2,441", "1.5%"),
      ("mode", "major", "103,063", "61.4%"),
      ("", "minor", "64,817", "38.6%"),
      ("time_signature", "4", "157,101", "93.6%"),
      ("", "3", "7,546", "4.5%"),
      ("", "5", "2,177", "1.3%"),
      ("", "1", "1,017", "0.6%"),
    )),
  ),
  caption: [Level counts for the categorical variables. Two contexts (personalized
  playlists and charts) are rare, which will limit the precision of any
  context-specific estimates.],
) <tab-categorical>

= Outcome definition <sec-outcome>

The log carries three nested skip flags plus `not_skipped`. @tab-flags shows that they
are nested in practice as well as in name: every `skip_1` play is also `skip_2` and
`skip_3`, with a single exception (one play flagged `skip_2` but not `skip_3`), and
3,145 plays have no flag set. We keep the single violating row and treat the unflagged
plays as not `skip_2`, which is how the raw flag records them.

#figure(
  table(
    columns: 6,
    align: (center, center, center, center, right, left),
    table.header([skip_1], [skip_2], [skip_3], [not_skipped], [Plays], [Reading]),
    ..rows-of((
      ("✓", "✓", "✓", "—", "70,762", "very brief listen"),
      ("—", "✓", "✓", "—", "16,061", "brief listen"),
      ("—", "—", "✓", "—", "22,027", "most of track played"),
      ("—", "✓", "—", "—", "1", "nesting violation"),
      ("—", "—", "—", "✓", "55,884", "completed"),
      ("—", "—", "—", "—", "3,145", "no flag set"),
    )),
  ),
  caption: [Observed combinations of the skip flags.],
) <tab-flags>

As stated in the proposal, our primary outcome is `skip_2`: a play counts as a skip if
the track was played only briefly. This is the definition of a skip used by the Spotify
Sequential Skip Prediction Challenge, the competition for which this dataset was
released, and it splits the plays almost in half (51.7%). `skip_1` and `skip_3` are kept
for sensitivity analyses. For description only, we introduce `skip_class`, the most
severe flag set on each play (@fig-skip-class). Of the plays that are skipped at all
(any of the three flags true), 65% are `skip_1`, i.e. skipped within the first few
seconds.

#fig(
  "02_skip_class_distribution.png",
  [Distribution of `skip_class`, the most severe skip flag of each play.],
  width: 75%,
) <fig-skip-class>

= Play-level patterns <sec-play>

== Session dynamics

Plays are not independent. Within a session, skipping is strongly autocorrelated
(@fig-autocorr): the skip rate is 74.9% when the previous track was skipped and 28.0%
when it was played through. Session position matters too (@fig-position): the opening
track of a session is skipped 37.7% of the time, against 52.7% averaged over positions 2
to 20. This is consistent with listeners actively choosing their first track and then
browsing. Both facts motivate aggregating plays to tracks and sessions rather than
modeling plays directly, and the effect of session position leads us to add mean session
position to the baseline model as a covariate.

#fig(
  "04_skip_autocorrelation.png",
  [Skip rate conditional on the previous play's outcome within the same session.],
  width: 70%,
) <fig-autocorr>

#fig(
  "03_skip_rate_by_position.png",
  [`skip_2` rate by session position. Position 1 is the outlier; from position 2 on the
  rate is flat at about 53%.],
  width: 85%,
) <fig-position>

== Listening context

@fig-context and @tab-context show the context effect that motivates our second research
question. Plays from contexts the listener chose deliberately (personalized playlists,
catalog) have the lowest skip rates, about 45%. Plays from the user's own collection,
radio, and charts have the highest, about 55%. It is surprising that plays from the
user's own collection are skipped as often as plays from radio. @tab-shuffle suggests
part of the reason: within the user's own collection, the skip rate is 50.7% when
playing in order and 60.6% on shuffle, the largest shuffle effect of any context. Radio
is never shuffled, so there is no comparison to make there. The proportion of a track's
plays on shuffle is therefore a covariate in the baseline model.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
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
  caption: [`skip_2` rate by context type, sorted ascending.],
) <tab-context>

#fig(
  "08_skip_rate_by_context.png",
  [Skip rate by context type against the overall rate (dashed line).],
  width: 80%,
) <fig-context>

#figure(
  table(
    columns: 3,
    align: (left, right, right),
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
  caption: [`skip_2` rate by context type and shuffle mode.],
) <tab-shuffle>

= Track-level data <sec-tracks>

The regression unit in the proposal is the track. @fig-plays-per-track shows that 50,704
distinct tracks appear in the log, but the distribution of plays per track is extremely
heavy-tailed. We thus add a minimum-play threshold. Only 5,397 tracks (10.6%) are played
five or more times; they nevertheless account for 101,875 plays, 60.7% of the data.
Raising the threshold to 10 or 20 plays keeps 2,233 and 920 tracks (48.6% and 38.3% of
plays), which we will use as sensitivity checks.

#fig(
  "23_plays_per_track.png",
  [Plays per track on log-log axes with the candidate thresholds. Most tracks appear once
  or twice; a few hundred appear more than fifty times.],
  width: 85%,
) <fig-plays-per-track>

The aggregation (`src/aggregate.py`) collapses each qualifying track to one row with its
`skip_2` rate (the outcome), play count (the regression weight), its audio features, and
covariates aggregated from its plays: the proportion of plays in each context type, on
shuffle, and by premium users, and the mean session position at which it was played.

@fig-track-rate shows the outcome. Across the 5,397 tracks the skip rate averages 0.526
(median 0.524, SD 0.190) and is roughly symmetric, with small spikes at the fractions
that a handful of plays can produce (0.4, 0.5, 0.6). Only 1.4% of tracks are never
skipped and 2.0% always are. The right panel shows why we weight by play count: the
spread of skip rates narrows as the play count grows (SD 0.218 for tracks with 5--9
plays, 0.160 for 10--19, 0.108 for 20 or more), as expected for a proportion with
variance $p(1-p)\/n$.

#fig(
  "24_track_skip_rate.png",
  [Left: distribution of track-level skip rate for tracks with at least five plays.
  Right: skip rate against play count. The variance shrinks with $n$, motivating weights
  proportional to the play count.],
) <fig-track-rate>

== Audio features and skip rate

@fig-track-features plots the track skip rate against each of the sixteen audio
features, with decile means and 95% confidence intervals over the raw scatter. The
gradients are small but several are monotone: skip rate rises with `duration` (from 0.50
in the shortest decile to 0.56 in the longest), `speechiness`, `bounciness`,
`beat_strength`, `dyn_range_mean` and `danceability`, and falls slightly with
`acousticness` and `organism`. `energy`, `tempo`, `valence`, `loudness` and `liveness`
show no trend. Two features have degenerate distributions that will matter for modeling:
`instrumentalness` is essentially zero for almost every track, and `flatness` and
`loudness` are concentrated in narrow bands.

#fig(
  "25_track_skip_vs_features.png",
  [Track skip rate by audio feature. Gray points are tracks; red points are decile means
  with 95% confidence intervals. The y-axis is restricted to 0.3--0.75 to make the
  gradients visible.],
) <fig-track-features>

@fig-track-corr quantifies this. The strongest univariate correlations with track skip
rate are `dyn_range_mean` ($r = +0.087$), `bounciness` and `speechiness` ($+0.081$),
`duration` ($+0.068$), `beat_strength` ($+0.066$) and `acousticness` ($-0.054$). These
are four to ten times larger than the play-level correlations (where the largest was
0.037), because aggregating averages out the play-level noise, but they remain small: no
feature on its own explains more than one percent of the variance in skip rate.

The right panel of @fig-track-corr shows the multicollinearity among the audio features.
`bounciness`, `beat_strength`, `dyn_range_mean` and `danceability` have pairwise
correlations between 0.79 and 0.96; `mechanism` and `organism` are nearly mirror images
($r = -0.88$); `organism` correlates with `acousticness` ($0.75$); and `energy`
correlates with `loudness` ($0.69$) and against `flatness` ($-0.63$). The baseline model
includes all sixteen features so the nested F-test answers the joint question. We return
to this in @sec-model-diag.

#fig(
  "26_track_feature_correlations.png",
  [Left: Pearson correlation of each audio feature (and the two track-level controls)
  with the track skip rate. Right: correlation matrix of the audio features at the track
  level.],
) <fig-track-corr>

= Session-level view for the ANOVA <sec-sessions>

The ANOVA in the proposal compares skip rates across the six listening contexts, with
each session's skip rate within a context as one observation. @fig-session-context shows
those distributions for session--context pairs with at least three plays (11,556 rows).
The ordering of medians matches @tab-context, and the group standard deviations are
similar (0.26 to 0.30). Group sizes are very unbalanced: 4,329 observations for the
user's collection against 163 for charts.

#fig(
  "27_session_skip_by_context.png",
  [Per-session skip rate by listening context. Boxes show quartiles, triangles the mean,
  and the dashed line the overall play-level rate.],
) <fig-session-context>

= Baseline model <sec-model>

== Specification

The baseline model is the weighted least-squares regression from the proposal, fitted to
the 5,397 tracks with at least five plays:

$ "skip rate"_i = beta_0 + bold(x)_i^top bold(beta)_"audio" + bold(z)_i^top bold(gamma) + epsilon_i, quad "Var"(epsilon_i) prop 1 \/ n_i, $

where $bold(x)_i$ holds the sixteen standardized audio features, $bold(z)_i$ the
standardized covariates (`us_popularity_estimate`, `release_year`, the share of the
track's plays in each context other than the user's collection, the shares on shuffle
and by premium users, and mean session position), and $n_i$ is the number of plays. The
user's collection is the omitted context so the six context shares, which sum to one,
are not collinear. The hypothesis is tested with a nested F-test comparing this model to
one with only the covariates.

== Fit and hypothesis test

#figure(
  table(
    columns: 2,
    align: (left, right),
    ..rows-of((
      ("Tracks (n)", "5,397"),
      ("Plays covered", "101,875"),
      ("R² (covariates only)", "0.063"),
      ("R² (covariates + audio)", "0.090"),
      ("Adjusted R² (full)", "0.086"),
      ("Nested F (audio features)", "9.96 on 16 and 5370 df"),
      ("Nested F p-value", "2.9e-25"),
    )),
  ),
  caption: [Baseline model fit. The nested F-test compares the full model to the
  covariates-only model.],
) <tab-fit>

@tab-fit tests the main hypothesis. The audio features add significant explanatory power
over popularity and context ($F_(16, 5370) = 9.96$, $p approx 3 times 10^(-25)$), so we
reject the null that none of them is associated with skip rate. The effect is small:
$R^2$ rises from 0.063 to 0.090, so the audio features account for just under three
percent of the variance in track skip rate beyond the controls, and the whole model for
about nine percent. Most of the variation in how often a track is skipped is not
explained by the track's features.

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.header([Predictor], [Coef.], [SE], [$t$], [$p$]),
    ..rows-of((
      ("us_popularity_estimate", "-0.0015", "0.0024", "-0.61", "0.544"),
      ("release_year", "+0.0050", "0.0032", "+1.56", "0.119"),
      ("prop_catalog", "-0.0073", "0.0031", "-2.38", "0.017"),
      ("prop_editorial_playlist", "+0.0016", "0.0025", "+0.63", "0.531"),
      ("prop_personalized_playlist", "-0.0003", "0.0028", "-0.09", "0.927"),
      ("prop_radio", "+0.0165", "0.0030", "+5.47", "<0.001"),
      ("prop_charts", "-0.0039", "0.0007", "-5.29", "<0.001"),
      ("prop_shuffle", "+0.0301", "0.0027", "+11.24", "<0.001"),
      ("prop_premium", "-0.0074", "0.0025", "-2.94", "0.003"),
      ("mean_position", "+0.0185", "0.0025", "+7.52", "<0.001"),
      ("acousticness", "-0.0209", "0.0076", "-2.76", "0.006"),
      ("beat_strength", "-0.0416", "0.0108", "-3.86", "<0.001"),
      ("bounciness", "+0.0556", "0.0144", "+3.85", "<0.001"),
      ("danceability", "+0.0061", "0.0049", "+1.25", "0.212"),
      ("dyn_range_mean", "-0.0085", "0.0067", "-1.26", "0.209"),
      ("energy", "+0.0077", "0.0039", "+1.99", "0.047"),
      ("flatness", "+0.0003", "0.0035", "+0.09", "0.927"),
      ("instrumentalness", "-0.0000", "0.0028", "-0.01", "0.991"),
      ("liveness", "+0.0011", "0.0020", "+0.54", "0.587"),
      ("loudness", "-0.0108", "0.0031", "-3.53", "<0.001"),
      ("mechanism", "+0.0100", "0.0105", "+0.95", "0.344"),
      ("organism", "+0.0253", "0.0139", "+1.81", "0.070"),
      ("speechiness", "+0.0047", "0.0027", "+1.73", "0.083"),
      ("tempo", "-0.0033", "0.0028", "-1.18", "0.238"),
      ("valence", "-0.0030", "0.0026", "-1.18", "0.240"),
      ("duration", "+0.0104", "0.0021", "+4.88", "<0.001"),
    )),
  ),
  caption: [Standardized WLS coefficients (change in skip rate per 1 SD of the
  predictor). The first ten rows are covariates, the rest audio features.],
) <tab-coef>

#fig(
  "28_baseline_coefficients.png",
  [Coefficients from @tab-coef with 95% confidence intervals, sorted by size. Red: audio
  features; blue: covariates.],
  width: 85%,
) <fig-coef>

@tab-coef and @fig-coef show which terms carry the effect. Among the covariates, the
share of plays on shuffle has the largest coefficient of any predictor ($+0.030$ per SD),
followed by mean session position ($+0.019$) and the share of plays on radio ($+0.017$);
the share of premium plays and the catalog share lower the skip rate. Popularity and
release year are not significant once context is controlled for, which is expected given
how compressed the popularity scale is. Among the audio features, `duration` ($+0.010$,
$p < 0.001$), `loudness` ($-0.011$), `acousticness` ($-0.021$) and `energy` ($+0.008$)
are individually significant with coefficients in the direction the EDA suggested.
`beat_strength` ($-0.042$) and `bounciness` ($+0.056$) have the largest audio
coefficients, but with opposite signs and standard errors three times those of the other
features. That is a symptom of the multicollinearity identified above, not evidence of
two opposing effects.

== Diagnostics <sec-model-diag>

#fig(
  "29_baseline_diagnostics.png",
  [Diagnostics for the baseline model. Left: weighted residuals against fitted values.
  Center: normal QQ plot of weighted residuals. Right: Cook's distance by track.],
) <fig-diag>

@fig-diag shows the regression assumptions are in reasonable shape. The weighted
residuals are centered on zero with constant spread across the fitted range, the QQ plot
is close to a straight line with mild heavy tails, and the largest Cook's distance is
below 0.001, so no single track drives the fit. Fitted values lie between 0.35 and 0.70,
well inside the unit interval, so the bounded outcome does not cause problems for a
linear model here. The faint diagonal streaks in the residual plot are tracks with the
same small play count and the same number of skips, an artifact of the discrete outcome.

The one assumption that fails is the absence of multicollinearity. @tab-vif lists
variance inflation factors above 5: `bounciness` (78), `organism` (60), `beat_strength`
(37), `mechanism` (33), `dyn_range_mean` (22), `acousticness` (17) and `danceability`
(7). Factors this large mean the individual coefficients for these features are
unstable, even though the joint test is unaffected. Before the interaction analysis we
will reduce each group of correlated features to one representative (for instance
`danceability` for the rhythm features and `acousticness` for the acoustic features,
dropping the Spotify-internal `mechanism` and `organism`) and confirm the nested F-test
result holds.

#figure(
  table(
    columns: 2,
    align: (left, right),
    table.header([Predictor], [VIF]),
    ..rows-of((
      ("bounciness", "78.2"),
      ("organism", "60.2"),
      ("beat_strength", "36.8"),
      ("mechanism", "33.1"),
      ("dyn_range_mean", "21.5"),
      ("acousticness", "16.5"),
      ("danceability", "7.2"),
      ("energy", "4.7"),
      ("flatness", "3.2"),
      ("loudness", "2.7"),
    )),
  ),
  caption: [Variance inflation factors above 2.5 in the full baseline model.],
) <tab-vif>

== Next steps

The baseline supports the main hypothesis. The next steps are to reduce the audio
features to a non-collinear subset, re-fit, and then add the feature-by-context
interactions for the second research question. The sensitivity analyses from the
proposal (the `skip_1` and `skip_3` outcomes and the 10- and 20-play thresholds) will be
run on the reduced model. Because the QQ plot shows mild heavy tails, we will also
report bootstrap confidence intervals alongside the model-based ones.

#pagebreak()

= Appendix: additional EDA

The figures below are exploratory results that are not used in the baseline model.

#fig(
  "01_skip_flag_rates.png",
  [Rates of the four skip flags. The flags are nested: `skip_3` is nearly twice as common
  as `skip_1`.],
  width: 70%,
)

#fig(
  "05_session_length.png",
  [Session length is capped at 20 plays (50.7% of sessions hit the cap); per-session
  skip rate correlates weakly with length ($r = 0.17$).],
)

#fig(
  "06_within_session_repeats.png",
  [Within-session replays (13.0% of plays) are skipped slightly less often than first
  plays (49.6% vs. 52.0%).],
)

#fig(
  "07_shuffle_effect.png",
  [Shuffle versus sequential skip rates by context type (the data behind @tab-shuffle).],
)

#fig(
  "09_skip_chain_lengths.png",
  [Length of consecutive forward-press chains within a session, a second view of the
  autocorrelation in @fig-autocorr.],
)

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    table.header([Pause combination], [Skip rate], [Plays]),
    ..rows-of((
      ("no_pause (only)", "54.6%", "128,865"),
      ("short_pause + long_pause", "47.7%", "24,617"),
      ("none (session start)", "37.7%", "10,000"),
      ("long_pause (only)", "21.3%", "4,398"),
    )),
  ),
  caption: [`skip_2` rate by pre-play pause combination. The `none` row is exactly the set
  of session-opening plays.],
)

#fig(
  "10_skip_rate_by_pause.png",
  [Skip rate by pre-play pause combination. A long pause before a play is associated with
  a much lower skip rate.],
  width: 80%,
)

#fig(
  "11_premium_vs_free.png",
  [Skip rates and context mix for premium versus free users. Free users are
  over-represented in their own collection and under-represented on radio.],
)

#fig(
  "12_hour_of_day.png",
  [Skip rate and play volume by hour of day. The hour effect (46.7% at 6:00 to 54.9%
  at 2:00) is small relative to the context and position effects.],
)

#fig(
  "13_feature_distributions.png",
  [Play-level marginal histograms of the audio features, showing the skew in
  `acousticness`, `speechiness`, `liveness` and `instrumentalness`.],
)

#fig(
  "15_skip_rate_by_duration.png",
  [Play-level `skip_2` rate across duration deciles; the gradient reappears at the track
  level in @fig-track-features.],
  width: 85%,
)

#fig(
  "16_duration_by_context.png",
  [Play-level `skip_2` rate by duration quintile within each context type. The duration
  gradient is present in every context, a first look at the interaction question.],
  width: 85%,
)

#fig(
  "18_popularity.png",
  [`us_popularity_estimate` is compressed into the top of its range; the bottom decile
  skips at 52.1% against 46.5% for the top decile.],
)

#fig(
  "19_release_year.png",
  [Release year: 86% of plays are of tracks released in the 2010s, and skip rate varies
  little by decade.],
)

#fig(
  "22_track_coverage.png",
  [Track reuse across sessions and skip rate by reuse quartile, the session-level
  counterpart of @fig-plays-per-track.],
)
