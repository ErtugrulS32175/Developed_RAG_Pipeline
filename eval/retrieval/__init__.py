"""Does the right passage come back, and at what rank.

Needs no generator, so it runs entirely on local services -- which is why
retrieval quality can be tuned without renting a GPU. Scored at page level for
ranking, but the metric that actually predicts answer quality is whether the
expected answer STRING reached the assembled context.
"""
