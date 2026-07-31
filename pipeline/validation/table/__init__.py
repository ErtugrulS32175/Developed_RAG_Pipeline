"""Checks on an extracted table against what the image actually showed.

Two models reading the same table, a number re-read from the cell it came from,
and a serial column used as a row-count checksum. Each catches a different
failure: disagreement, a misread digit, and silently dropped rows -- the last
being invisible to the other two, since every value it keeps is correct.
"""
