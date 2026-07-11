# Automated review log

Marker file for the scheduled "review recent changes" routine. Records the commit
SHA up to which the last run reviewed. The next run picks up from `last_reviewed`
and only reads commits after that point.

Do not edit by hand unless you know what you're doing. The routine rewrites
`last_reviewed` after each successful review.

## last_reviewed

bde4f9be00d1c59b648a4f3c8e59d63c9121d99c

## history

- 2026-07-11: reviewed 7b53e5d..bde4f9b (elo-promotion feature, 10 commits). 5 findings surfaced (2 med, 3 low). See notification transcript.
