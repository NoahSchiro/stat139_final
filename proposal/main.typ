#show link: set text(fill: blue)

#let title = "Pending"
#let authors = "Gabe Gonzalez, Debra Ritter, Noah Schiro, Franco Valencia, Elaine Zhang"

#set page(paper: "us-letter", margin: (x: 0.85in, top: 0.5in, bottom: 0.7in))
#set document(title: title, author: authors)
#set text(size: 10.5pt)
#set par(spacing: 0.7em)
#set enum(spacing: 0.4em)
#set list(spacing: 0.4em)
#show heading: set text(size: 11pt)
#show heading: set block(above: 0.5em, below: 0.25em)

#align(center, text(size: 14pt, weight: "bold")[#title])
#align(center, authors)

= Data source
Requirement from the milestone text: "describe the data source"

= Measurements
Requirement from the milestone text: "variables you want to measure"

= Analyses
Requirement from the milestone text: "a brief explanation of what analyses you expect to perform"

= Challenges
Requirement from the milestone text: "a description of any challenges you have faced (for example, difficulties obtaining data)."
