"""Finding the passages an answer could be built from.

Ends at the assembled context. What a model then does with that context is
generation's job, and the split matters because the two fail differently: a
retrieval failure means the answer was never possible, a generation failure
means it was there and was not used. Every metric in eval/ turns on telling
those apart.
"""
