"""Is the answer right, and if not, whose fault was it.

The fault split is the point: an answer can only be wrong two ways, and they
need different fixes. The key was absent from the context, so no generator
could have answered -- or it was there and was not used.

Scoring has three outcomes, not two. Treating "the scorer could not confirm it"
as "wrong" once reported a set at 0.588 that was nearer 0.99.
"""
