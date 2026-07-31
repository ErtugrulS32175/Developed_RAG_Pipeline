"""The searchable index: what goes into it, how it is stored, how it is vectorised.

Ingest, the Postgres/pgvector store and the embedding client sit together
because they are used together and share one contract -- the chunk shape and
the vector dimensions. Splitting them would make both the writing side and the
reading side import across two packages to do one thing.
"""
