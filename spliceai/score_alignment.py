"""Line up SpliceAI's REF and ALT scores so that they can be compared position by position.

Kept out of utils.py, which loads Keras on import, so that it can be tested without TensorFlow:
    python3 -m unittest tests.score_alignment_tests -v
"""

import numpy as np


def trim_shared_bases(ref, alt):
    """Trim the bases REF and ALT share, leaving the bases the variant actually changes.

    Bases shared at the end are dropped first, then bases shared at the start, and each allele always
    keeps at least one base, so an insertion or deletion keeps its anchor base the way VCF writes it.

    Args:
        ref (str): REF allele
        alt (str): ALT allele

    Returns:
        tuple: (number of bases dropped from the start, trimmed REF, trimmed ALT)
    """
    bases_dropped_from_start = 0
    while len(ref) > 1 and len(alt) > 1 and ref[-1].upper() == alt[-1].upper():
        ref, alt = ref[:-1], alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0].upper() == alt[0].upper():
        ref, alt = ref[1:], alt[1:]
        bases_dropped_from_start += 1
    return bases_dropped_from_start, ref, alt


def span_fits_in_output_window(ref, alt, cov):
    """Whether the bases a variant changes lie inside the window the scores cover.

    The scores cover cov positions starting cov//2 before the variant. Alleles of the same length need
    no more than that, since align_ref_and_alt_scores hands both tracks back untouched. When the lengths
    differ, the bases the variant changes have to be collapsed onto the REF positions, and a variant that
    deletes far more than it inserts, or one written with a long run of shared leading bases, puts those
    bases past the end of the window, where the ALT scores stop short and there is nothing to collapse.
    Callers skip those records rather than report scores for positions the model never covered.

    Args:
        ref (str): REF allele
        alt (str): ALT allele
        cov (int): number of REF positions in the output window

    Returns:
        bool: True when align_ref_and_alt_scores can line the two up
    """
    bases_dropped_from_start, trimmed_ref, trimmed_alt = trim_shared_bases(ref, alt)
    if len(trimmed_ref) == len(trimmed_alt):
        return True
    return cov//2 + bases_dropped_from_start + len(trimmed_ref) <= cov


def align_ref_and_alt_scores(y_ref, y_alt, ref, alt, cov):
    """Line up the model's scores for the ALT sequence with the REF positions they are compared against.

    When REF and ALT differ in length, the ALT sequence has more or fewer positions than the REF one, so
    its scores are collapsed onto the REF positions before the two are subtracted. The bases the alleles
    share are trimmed off first, so the result doesn't depend on how many unchanged bases the variant was
    written with. Variants with REF and ALT both longer than one base used to have their ALT scores
    collapsed onto the first base and the rest of the REF bases set to zero, even when the alleles were
    the same length and nothing needed lining up
    (https://github.com/broadinstitute/SpliceAI-lookup/issues/137).

    After trimming:
    - same length (an SNV or MNV): the positions already line up
    - one-base ALT (a deletion): the anchor base keeps its score and the deleted bases get zero
    - one-base REF (an insertion): the anchor base gets the highest score among itself and the inserted bases
    - otherwise (a deletion-insertion): the whole span is reported once, at the position holding the
      highest REF score in it, where the ALT track carries the highest score among the bases put in the
      span's place, so the difference between the two is what the variant changed. Every other position
      of the span is given its REF score on both tracks, which reports no change there. Reporting at the
      strongest REF position rather than at the span's first base keeps a splice site the variant
      replaces at its own coordinate, which is what masking needs, since it keeps a loss only where a
      splice site is annotated. Acceptor and donor are scored on separate channels and reported
      separately, so each picks its own position.

    The two one-base cases keep SpliceAI's original handling, since an insertion or deletion written with a
    single anchor base was never part of the multi-base support this module replaces.

    Args:
        y_ref (numpy.ndarray): REF scores in genomic order, of shape (1, cov, 3)
        y_alt (numpy.ndarray): ALT scores in genomic order, of shape (1, cov + len(alt) - len(ref), 3), with
            the variant's first base at index cov//2
        ref (str): REF allele
        alt (str): ALT allele
        cov (int): number of REF positions in the output window

    Returns:
        tuple: (y_ref, y_alt), each of shape (1, cov, 3), one score per REF position
    """
    bases_dropped_from_start, ref, alt = trim_shared_bases(ref, alt)
    start = cov//2 + bases_dropped_from_start
    if len(ref) == len(alt):
        return y_ref, y_alt

    if len(alt) == 1:
        return y_ref, np.concatenate([
            y_alt[:, :start+1],
            np.zeros((1, len(ref)-1, 3)),
            y_alt[:, start+1:]],
            axis=1)

    if len(ref) == 1:
        return y_ref, np.concatenate([
            y_alt[:, :start],
            np.max(y_alt[:, start:start+len(alt)], axis=1)[:, None, :],
            y_alt[:, start+len(alt):]],
            axis=1)

    # A deletion-insertion replaces every base of the span at once, so no base inside it has a counterpart
    # to be compared against. The comparison for the whole span is reported at the position holding the
    # strongest REF score, so that a splice site the variant replaces keeps its score at its own
    # coordinate; np.argmax takes the first maximum, which is the earliest genomic position when several
    # tie. Acceptor and donor are separate channels, so each picks its own position. Every other position
    # of the span holds its REF score on both tracks, so the difference between the tracks, which is what
    # the caller reports as a change, is zero there. The REF track itself needs no rewriting, since the
    # strongest REF score already sits at that position.
    span_ref = y_ref[:, start:start+len(ref)]
    y_alt_span = span_ref.copy()
    np.put_along_axis(
        y_alt_span,
        np.argmax(span_ref, axis=1)[:, None, :],
        np.max(y_alt[:, start:start+len(alt)], axis=1)[:, None, :],
        axis=1)
    return y_ref, np.concatenate([y_alt[:, :start], y_alt_span, y_alt[:, start+len(alt):]], axis=1)
