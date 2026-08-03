"""Checks on a generated answer against the passages it was given.

Answer-side checking is one-sided: it can show a figure was not taken from the
evidence, never that the right figure was chosen. Which is why the answer is
asked to quote the line it rests on -- a quote can be matched against its
passage, a paraphrase cannot. The checked contract returns answered, abstained
or review_required; review-required results deliberately expose no answer text.
"""
