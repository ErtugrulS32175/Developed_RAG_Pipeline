"""Checks that run on live output, in production, with no ground truth.

That last part is what separates this package from `eval/`: measurement needs a
known right answer and runs offline, while everything here has to decide
something about output nobody has graded. The two were once mixed and the
result was a production module importing the test tooling.

One discipline holds across both products, and it was learned by measuring:

  * a check FLAGS, it does not block. Refusing an answer on a check whose
    false-positive rate is unknown trades a rare wrong answer for a steady
    stream of wrongly-refused right ones.
  * a check earns promotion to a gate only after that rate is measured against
    output already known to be correct.
  * a check that fires on almost nothing is not thereby safe. One measured here
    flagged 0.5% of correct answers and caught none of the wrong ones, which
    makes it quiet, not useful.

Split by product because the material differs -- table cells against their OCR,
answers against the passages they cite -- while the discipline does not.
"""
