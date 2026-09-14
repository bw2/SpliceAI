"""Tests for spliceai/score_alignment.py.

Unlike test_delta_score.py, these need only numpy, not the models.

Run from the repo root with:  python3 -m unittest tests.score_alignment_tests -v
"""

import unittest

import numpy as np

from spliceai.score_alignment import align_ref_and_alt_scores, span_fits_in_output_window, trim_shared_bases

# A distance of 5, so the output window has 11 positions and the variant's first base is at index 5.
COV = 11
CENTER = COV // 2


def scores(length, first):
    """Distinct increasing scores, so a moved, collapsed or copied value shows up in the result."""
    return np.arange(first, first + length * 3, dtype=float).reshape(1, length, 3)


def zeros(length):
    return np.zeros((1, length, 3))


class TrimSharedBasesTest(unittest.TestCase):

    def test_snv_is_unchanged(self):
        self.assertEqual(trim_shared_bases("G", "A"), (0, "G", "A"))

    def test_unchanged_bases_before_the_change_are_trimmed(self):
        self.assertEqual(trim_shared_bases("TG", "TA"), (1, "G", "A"))
        self.assertEqual(trim_shared_bases("GCTG", "GCTA"), (3, "G", "A"))

    def test_unchanged_bases_after_the_change_are_trimmed(self):
        self.assertEqual(trim_shared_bases("ACT", "CCT"), (0, "A", "C"))
        self.assertEqual(trim_shared_bases("CGA", "TGA"), (0, "C", "T"))

    def test_genuine_mnv_is_unchanged(self):
        self.assertEqual(trim_shared_bases("TG", "CA"), (0, "TG", "CA"))

    def test_indels_with_one_anchor_base_are_unchanged(self):
        self.assertEqual(trim_shared_bases("GGGC", "G"), (0, "GGGC", "G"))
        self.assertEqual(trim_shared_bases("A", "AGAGAG"), (0, "A", "AGAGAG"))

    def test_indels_with_extra_shared_bases_keep_one_anchor_base(self):
        self.assertEqual(trim_shared_bases("CTG", "CT"), (1, "TG", "T"))
        self.assertEqual(trim_shared_bases("CA", "CAT"), (1, "A", "AT"))

    def test_shared_bases_at_the_end_are_trimmed_before_those_at_the_start(self):
        # CAA>CA could become CA>C or, one base later, AA>A; trimming the end first gives CA>C
        self.assertEqual(trim_shared_bases("CAA", "CA"), (0, "CA", "C"))

    def test_deletion_insertion_trims_to_the_bases_it_changes(self):
        self.assertEqual(trim_shared_bases("CAT", "CGGT"), (1, "A", "GG"))

    def test_comparison_ignores_case(self):
        self.assertEqual(trim_shared_bases("TG", "tA"), (1, "G", "A"))


class SpanFitsInOutputWindowTest(unittest.TestCase):

    def test_ordinary_variants_fit(self):
        for ref, alt in (("G", "A"), ("TG", "TA"), ("GGGC", "G"), ("A", "AGAGAG"), ("CTG", "CT"),
                         ("CA", "CAT"), ("AT", "GCC"), ("ATG", "GC"), ("GAT", "GGCC")):
            with self.subTest(ref=ref, alt=alt):
                self.assertTrue(span_fits_in_output_window(ref, alt, COV))

    def test_alleles_of_the_same_length_always_fit(self):
        # align_ref_and_alt_scores hands both tracks back untouched for these, so nothing can run short
        for ref, alt in (("G", "A"), ("TG", "CA"), ("A" * 600, "C" * 600)):
            with self.subTest(ref=ref, alt=alt):
                self.assertTrue(span_fits_in_output_window(ref, alt, COV))
                self.assertTrue(span_fits_in_output_window(ref, alt, 1001))

    def test_a_ref_reaching_the_end_of_the_window_is_the_last_one_that_fits(self):
        # the span starts halfway through the window, so the REF can reach its end but not pass it
        self.assertTrue(span_fits_in_output_window("A" * (COV - CENTER), "GC", COV))
        self.assertFalse(span_fits_in_output_window("A" * (COV - CENTER + 1), "GC", COV))

    def test_a_deletion_insertion_that_replaces_past_the_window_does_not_fit(self):
        # the bases it changes end past the last position the scores cover
        self.assertFalse(span_fits_in_output_window("A" * 600, "GC", 1001))

    def test_shared_leading_bases_can_push_the_span_past_the_window(self):
        # trimming moves the span to where the alleles differ, which here is past the end of the window
        self.assertFalse(span_fits_in_output_window("A" * 600, "A" * 597 + "GC", 1001))

    def test_a_deletion_longer_than_the_window_does_not_fit(self):
        # its deleted bases run past the end of the window, so there is nothing to compare there
        self.assertFalse(span_fits_in_output_window("A" * 600, "A", 1001))
        self.assertTrue(span_fits_in_output_window("A" * 400, "A", 1001))


class AlignRefAndAltScoresTest(unittest.TestCase):

    def align(self, ref, alt):
        """Return (y_ref, y_alt, aligned_ref, aligned_alt). REF scores start at 1, ALT scores at 1000, so
        every value shows which track it came from."""
        y_ref = scores(COV, first=1)
        y_alt = scores(COV + len(alt) - len(ref), first=1000)
        aligned_ref, aligned_alt = align_ref_and_alt_scores(y_ref, y_alt, ref, alt, COV)
        return y_ref, y_alt, aligned_ref, aligned_alt

    # --- alleles of the same length line up already, and indels keep SpliceAI's original handling ---

    def test_same_length_alleles_are_not_realigned(self):
        for ref, alt in (("G", "A"), ("TG", "TA"), ("GCTG", "GCTA"), ("TG", "CA"), ("ACT", "CCT")):
            with self.subTest(ref=ref, alt=alt):
                y_ref, y_alt, aligned_ref, aligned_alt = self.align(ref, alt)
                np.testing.assert_array_equal(aligned_ref, y_ref)
                np.testing.assert_array_equal(aligned_alt, y_alt)

    def test_one_anchor_deletion_matches_the_original_realignment(self):
        y_ref, y_alt, aligned_ref, aligned_alt = self.align("GGGC", "G")
        np.testing.assert_array_equal(aligned_ref, y_ref)
        np.testing.assert_array_equal(aligned_alt, np.concatenate(
            [y_alt[:, :CENTER+1], zeros(3), y_alt[:, CENTER+1:]], axis=1))

    def test_one_anchor_insertion_matches_the_original_realignment(self):
        y_ref, y_alt, aligned_ref, aligned_alt = self.align("A", "AGA")
        np.testing.assert_array_equal(aligned_ref, y_ref)
        np.testing.assert_array_equal(aligned_alt, np.concatenate(
            [y_alt[:, :CENTER], np.max(y_alt[:, CENTER:CENTER+3], axis=1)[:, None, :], y_alt[:, CENTER+3:]],
            axis=1))

    def test_deletion_with_an_extra_shared_base_is_realigned_like_its_one_anchor_spelling(self):
        # CTG>CT is TG>T one base later: the T keeps its score and only the deleted G gets zero
        y_ref, y_alt, aligned_ref, aligned_alt = self.align("CTG", "CT")
        np.testing.assert_array_equal(aligned_ref, y_ref)
        np.testing.assert_array_equal(aligned_alt, np.concatenate(
            [y_alt[:, :CENTER+2], zeros(1), y_alt[:, CENTER+2:]], axis=1))

    def test_insertion_with_an_extra_shared_base_is_realigned_like_its_one_anchor_spelling(self):
        # CA>CAT is A>AT one base later: the C keeps its own score
        y_ref, y_alt, aligned_ref, aligned_alt = self.align("CA", "CAT")
        np.testing.assert_array_equal(aligned_ref, y_ref)
        np.testing.assert_array_equal(aligned_alt, np.concatenate(
            [y_alt[:, :CENTER+1], np.max(y_alt[:, CENTER+1:CENTER+3], axis=1)[:, None, :], y_alt[:, CENTER+3:]],
            axis=1))

    # --- a deletion-insertion reports its whole span once, where the REF signal is strongest ---

    def peaked(self, ref, alt, peak_offset):
        """Align with one strong REF score `peak_offset` bases into the window, the rest flat."""
        y_ref = np.full((1, COV, 3), 0.05)
        y_alt = np.full((1, COV + len(alt) - len(ref), 3), 0.05)
        y_ref[0, CENTER + peak_offset, :] = 0.9
        aligned_ref, aligned_alt = align_ref_and_alt_scores(y_ref, y_alt, ref, alt, COV)
        return y_ref, y_alt, aligned_ref, aligned_alt

    def test_deletion_insertion_is_reported_at_the_strongest_ref_position(self):
        # Wherever the site the variant replaces sits inside the span, the comparison is reported at that
        # position. Masking keeps a loss only where a splice site is annotated, so reporting it at the
        # span's first base instead would have thrown the loss away whenever the site was not that base.
        for peak_offset in (0, 1, 2):
            with self.subTest(peak_offset=peak_offset):
                y_ref, y_alt, _, aligned_alt = self.peaked("ATG", "CC", peak_offset)
                anchor = CENTER + peak_offset
                np.testing.assert_array_equal(
                    aligned_alt[:, anchor], np.max(y_alt[:, CENTER:CENTER+2], axis=1))
                for other in sorted({CENTER, CENTER+1, CENTER+2} - {anchor}):
                    np.testing.assert_array_equal(aligned_alt[:, other], y_ref[:, other])

    def test_each_channel_is_reported_at_its_own_strongest_position(self):
        # acceptor and donor are scored separately, so a span holding an acceptor at one base and a donor
        # at another reports each at the base where its own signal is
        y_ref = np.full((1, COV, 3), 0.05)
        y_alt = np.full((1, COV - 1, 3), 0.05)
        y_ref[0, CENTER, 1] = 0.9      # acceptor peaks at the span's first base
        y_ref[0, CENTER+2, 2] = 0.8    # donor peaks at its last
        _, aligned_alt = align_ref_and_alt_scores(y_ref, y_alt, "ATG", "CC", COV)
        self.assertEqual(aligned_alt[0, CENTER, 1], np.max(y_alt[0, CENTER:CENTER+2, 1]))
        self.assertEqual(aligned_alt[0, CENTER+2, 2], np.max(y_alt[0, CENTER:CENTER+2, 2]))
        self.assertEqual(aligned_alt[0, CENTER+2, 1], y_ref[0, CENTER+2, 1])
        self.assertEqual(aligned_alt[0, CENTER, 2], y_ref[0, CENTER, 2])

    def test_tied_ref_scores_are_reported_at_the_earliest_position(self):
        y_ref = np.full((1, COV, 3), 0.05)
        y_alt = np.full((1, COV - 1, 3), 0.05)
        y_ref[0, CENTER, :] = y_ref[0, CENTER+2, :] = 0.7
        _, aligned_alt = align_ref_and_alt_scores(y_ref, y_alt, "ATG", "CC", COV)
        np.testing.assert_array_equal(aligned_alt[:, CENTER], np.max(y_alt[:, CENTER:CENTER+2], axis=1))
        np.testing.assert_array_equal(aligned_alt[:, CENTER+2], y_ref[:, CENTER+2])

    def test_deletion_insertion_leaves_the_ref_track_untouched(self):
        # the strongest REF score already sits at the position the span is reported at
        for ref, alt in (("AT", "GCC"), ("ATG", "GC"), ("GAT", "GGCC")):
            with self.subTest(ref=ref, alt=alt):
                y_ref, _, aligned_ref, _ = self.align(ref, alt)
                np.testing.assert_array_equal(aligned_ref, y_ref)

    def test_deletion_insertion_leaves_the_rest_of_the_span_showing_no_change(self):
        y_ref, _, _, aligned_alt = self.peaked("ATG", "GC", 0)
        np.testing.assert_array_equal(aligned_alt[:, CENTER+1:CENTER+3], y_ref[:, CENTER+1:CENTER+3])

    def test_deletion_insertion_leaves_positions_outside_the_span_alone(self):
        y_ref, y_alt, aligned_ref, aligned_alt = self.align("AT", "GCC")
        np.testing.assert_array_equal(aligned_ref, y_ref)
        np.testing.assert_array_equal(aligned_alt[:, :CENTER], y_alt[:, :CENTER])
        np.testing.assert_array_equal(aligned_alt[:, CENTER+2:], y_alt[:, CENTER+3:])

    def test_deletion_insertion_with_shared_bases_is_reported_inside_the_bases_it_changes(self):
        # GAT>GGCC changes AT>GCC one base later, so the span covers CENTER+1 and CENTER+2
        y_ref, y_alt, _, aligned_alt = self.peaked("GAT", "GGCC", 2)
        np.testing.assert_array_equal(aligned_alt[:, :CENTER+1], y_alt[:, :CENTER+1])
        np.testing.assert_array_equal(aligned_alt[:, CENTER+1], y_ref[:, CENTER+1])
        np.testing.assert_array_equal(
            aligned_alt[:, CENTER+2], np.max(y_alt[:, CENTER+1:CENTER+4], axis=1))

    def test_output_has_one_score_per_ref_position(self):
        for ref, alt in (("G", "A"), ("TG", "TA"), ("GGGC", "G"), ("A", "AGAGAG"), ("CTG", "CT"),
                         ("CA", "CAT"), ("AT", "GCC"), ("ATG", "GC"), ("GAT", "GGCC")):
            with self.subTest(ref=ref, alt=alt):
                _, _, aligned_ref, aligned_alt = self.align(ref, alt)
                self.assertEqual(aligned_ref.shape, (1, COV, 3))
                self.assertEqual(aligned_alt.shape, (1, COV, 3))


if __name__ == "__main__":
    unittest.main()
