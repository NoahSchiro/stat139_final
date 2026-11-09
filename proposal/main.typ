#show link: set text(fill: rgb("#1a5fb4"))

#let title = "What Makes Listeners Skip? A Study on Spotify Track Skip by Audio Characteristics"
#let authors = "Elaine Zhang, Franco Valencia, Gabriel Gonzalez, Noah Schiro, Debra Ritter"

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

#align(center)[
  #block(width: 85%, text(size: 16pt, weight: "bold")[#title])
  #v(0.6em)
  #text(size: 11pt)[#authors]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[Stat 139 Final Project --- Proposal (Milestone 2)]
]
#v(0.5em)
#line(length: 100%, stroke: 0.5pt + gray)
#v(0.5em)

= Introduction
Every day, millions of Spotify users decide in just a few seconds whether a song is worth
their attention. In the Spotify listening data we analyze, about half of all plays end
with the listener skipping the track almost immediately, which says a lot about listener
satisfaction. Yet, we know surprisingly little about why some tracks get skipped, and
others hold our attention. Is it something about the music itself---such as its energy,
tempo, mood, or length---or is it more so about simply where the song is heard? A listener
who chose the album, for example, is in a different position from one who is getting
whatever the radio serves next. Using 167,880 listening events from Spotify's public Music
Streaming Sessions Dataset, we ask:

+ After accounting for popularity and listening context, which audio characteristics of a
  track are associated with how often listeners skip it?
+ Does the role of those characteristics differ across listening contexts?

= Data source
Our data source is the Spotify Music Streaming Sessions Dataset (Brost, Mehrotra & Jehan,
2019, _The Music Streaming Sessions Dataset_, arXiv:1901.09851). A public mini sample is
available at
#link("https://github.com/rachaelmmccue/Spotify-Sequential-Skip-Prediction")[github.com/rachaelmmccue/Spotify-Sequential-Skip-Prediction]
and is freely accessible to the teaching staff.

- `log_mini.csv` contains 167,880 rows, each representing a play in one track. This table
  contains data regarding a single listening session. Its variables are: session ID,
  position in session, skip indicators, listening context, shuffle, premium, hour of day,
  date, reason the track started/ended.
- `tf_mini.csv` contains 50,704 rows, each representing a unique song. This table contains
  metadata regarding the characteristics of a song. Its variables are: duration, release
  year, US popularity estimate, 17 audio features, and 8 latent "acoustic vector"
  dimensions.

= Variables
*Outcome.* We will use track-level skip rate: the proportion of a track's plays with
`skip_2 = TRUE`. `skip_2` means the track was played only briefly and is the challenge's
main skip definition. `skip_1` (very brief) and `skip_3` (most of the track played) will
be used in sensitivity analyses.

*Main predictors (track-level audio features).* `acousticness`, `beat_strength`,
`bounciness`, `danceability`, `dyn_range_mean`, `energy`, `flatness`, `instrumentalness`,
`liveness`, `loudness`, `mechanism`, `organism`, `speechiness`, `tempo`, `valence`, `key`,
`mode` (major/minor), `time_signature`, `duration`.

*Controls / adjustment variables.* At the track level, `us_popularity_estimate` and
`release_year` will be controlled for, as these capture the "trendiness" of a track and
allow us to isolate the relationship of an audio's characteristics purely to its skip
rate. Aggregated from the plays of each track, we will also introduce the proportion of
plays in each `context_type`, proportion on shuffle, proportion by premium users, and mean
`session_position` as covariates, as these are context-related variables that can also
affect skip rate.

*Excluded.* We will leave out the 8 `acoustic_vector` dimensions from the main model
because they cannot be interpreted. They may appear in an appendix. Outcome-like variables
(`not_skipped`, `hist_user_behavior_reason_end`, seek counts within the play) are also
excluded because they leak the outcome.

= Planned analysis
+ *Data preparation:* Join the two files, collapse to one row per track (containing skip
  rate and the number of plays), and remove tracks that don't have many plays (e.g., less
  than 5 plays).
+ *EDA:* Plot the distribution of skip rates, skip rate against each audio feature, and a
  feature correlation matrix to check for multicollinearity.
+ *ANOVA by context:* Use a one-way ANOVA to compare skip rates across the six listening
  contexts (i.e., own collection, album/catalog, editorial playlists, personalized
  playlists, radio, charts). Each session's skip rate within a context is one observation.
  We will also perform a permutation test as a nonparametric check.
+ *Main regression:* Use weighted least squares (using the number of plays as the weights,
  since songs with more plays will have smaller variance for skip rate) to regress skip
  rate on standardized audio features. Control for the aforementioned covariates. Use
  nested F-tests to check whether groups of features (e.g., mood: valence, energy,
  danceability) add additional explanatory power.
+ *Interactions:* Add feature $times$ context terms (e.g., energy $times$ radio) to test
  whether features matter more in specific contexts, especially ones that the user didn't
  choose.
+ *Diagnostics:* Residual and QQ plots, checks for unequal variance and nonlinearity,
  Cook's distance, and VIF.
+ *Robustness:* If time permits, we can conduct additional analyses and checks using
  bootstrap confidence intervals, the alternative skip definitions, different minimum-play
  thresholds of 10 and 20, and LASSO regularization as a secondary check on which features
  contain little-to-no information.

= Challenges and limitations
Our main challenge is sparsity. Many tracks appear only once or twice in the sample, so we
must restrict to frequently played tracks. That restriction makes the sample lean more
toward popular music. Additionally, plays are grouped within listening sessions, and there
are no user IDs. Aggregating to tracks and sessions reduces this dependence, but it will
not remove it entirely.

Our results will also be observational. Context and popularity are confounded with audio
features, so we can describe associations but not make causal claims. The sample covers
only 18 days, which rules out seasonal questions, and tracks are anonymized, which limits
song-level interpretation. Some features (e.g., `mechanism`, `organism`) appear to be
internal Spotify measures, making them difficult or pointless to interpret.

