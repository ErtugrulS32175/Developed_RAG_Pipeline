"""Turning an assembled context into an answer.

Holds the model client and the prompts -- including the structured prompt that
asks an answer to quote the line it rests on, which is what makes the answer
checkable afterwards. Separate from retrieval because the prompts change often
and independently, and because the context that feeds this can come from more
than one retrieval engine.
"""
