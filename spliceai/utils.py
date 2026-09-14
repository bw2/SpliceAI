from pkg_resources import resource_filename
import pandas as pd
import numpy as np
from pyfastx import Fasta
from keras.models import load_model
import logging
from sys import exit
from spliceai.score_alignment import align_ref_and_alt_scores, span_fits_in_output_window, trim_shared_bases

INFO_FIELD_KEYS = [
    'ALLELE',
    'NAME',
    'STRAND',
    'DS_AG',
    'DS_AL',
    'DS_DG',
    'DS_DL',
    'DP_AG',
    'DP_AL',
    'DP_DG',
    'DP_DL',
    'DS_AG_REF',
    'DS_AG_ALT',
    'DS_AL_REF',
    'DS_AL_ALT',
    'DS_DG_REF',
    'DS_DG_ALT',
    'DS_DL_REF',
    'DS_DL_ALT',
]

FLOAT_FORMAT = "0.3f"

# the minimum raw score for a position to be included in the ALL_NON_ZERO_SCORES array
MIN_SCORE_THRESHOLD = 0.01

INSERTED_BASES_CONTEXT = 5

class Annotator:

    def __init__(self, ref_fasta, annotations):

        if annotations == 'grch37':
            annotations = resource_filename(__name__, 'annotations/grch37.txt')
        elif annotations == 'grch38':
            annotations = resource_filename(__name__, 'annotations/grch38.txt')

        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            self.genes = df['#NAME'].to_numpy()
            self.chroms = df['CHROM'].to_numpy()
            self.strands = df['STRAND'].to_numpy()
            self.tx_starts = df['TX_START'].to_numpy()+1
            self.tx_ends = df['TX_END'].to_numpy()
            self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i])+1
                                for c in df['EXON_START'].to_numpy()]
            self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                              for c in df['EXON_END'].to_numpy()]
        except IOError as e:
            logging.error('{}'.format(e))
            exit()
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file {} not formatted properly: {}'.format(annotations, e))
            exit()

        try:
            self.ref_fasta = Fasta(ref_fasta)
        except IOError as e:
            logging.error('{}'.format(e))
            exit()

        paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
        self.models = [load_model(resource_filename(__name__, x)) for x in paths]

    def get_name_and_strand(self, chrom, pos):

        chrom = normalise_chrom(chrom, list(self.chroms)[0])
        idxs = np.intersect1d(np.nonzero(self.chroms == chrom)[0],
                              np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                              np.nonzero(pos <= self.tx_ends)[0]))

        if len(idxs) >= 1:
            return self.genes[idxs], self.strands[idxs], idxs
        else:
            return [], [], []

    def get_pos_data(self, idx, pos):

        dist_tx_start = self.tx_starts[idx]-pos
        dist_tx_end = self.tx_ends[idx]-pos
        dist_exon_bdry = min(np.union1d(self.exon_starts[idx], self.exon_ends[idx])-pos, key=abs)
        dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)

        return dist_ann


def one_hot_encode(seq):

    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    return map[np.fromstring(seq, np.int8) % 5]


def normalise_chrom(source, target):

    def has_prefix(x):
        return x.startswith('chr')

    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')
    elif not has_prefix(source) and has_prefix(target):
        return 'chr'+source

    return source


def get_delta_scores_for_transcript(x_ref, x_alt, ref, alt, strand, cov, ann):
    x_ref = one_hot_encode(x_ref)[None, :]
    x_alt = one_hot_encode(x_alt)[None, :]

    if strand == '-':
        x_ref = x_ref[:, ::-1, ::-1]
        x_alt = x_alt[:, ::-1, ::-1]

    y_ref = np.mean([ann.models[m].predict(x_ref, verbose=0) for m in range(5)], axis=0)
    y_alt = np.mean([ann.models[m].predict(x_alt, verbose=0) for m in range(5)], axis=0)

    if strand == '-':
        y_ref = y_ref[:, ::-1]
        y_alt = y_alt[:, ::-1]

    # save the original scores for inserted bases. The trimmed alleles decide this, so an insertion
    # written with shared leading bases keeps the same scores as its shortest spelling.
    _, trimmed_ref, trimmed_alt = trim_shared_bases(ref, alt)
    y_alt_with_inserted_bases = y_alt if len(trimmed_ref) == 1 and len(trimmed_alt) > 1 else None
    y_ref, y_alt = align_ref_and_alt_scores(y_ref, y_alt, ref, alt, cov)

    return y_ref, y_alt, y_alt_with_inserted_bases


def get_ref_scores_for_transcript(x_ref, strand, ann):
    x_ref = one_hot_encode(x_ref)[None, :]

    if strand == '-':
        x_ref = x_ref[:, ::-1, ::-1]

    y_ref = np.mean([ann.models[m].predict(x_ref, verbose=0) for m in range(5)], axis=0)

    if strand == '-':
        y_ref = y_ref[:, ::-1]

    return y_ref


def get_reference_scores(chrom, pos, ann, dist_var):
    """REF-only counterpart of get_delta_scores: no ALT allele needed.

    For each transcript overlapping `pos`, reports the single highest predicted
    acceptor probability and highest predicted donor probability anywhere in
    the +/-dist_var window around `pos` (whether or not that position is an
    annotated splice site), plus the full above-threshold curve for
    visualization.
    """
    cov = 2*dist_var+1
    wid = 10000+cov
    scores = []

    (genes, strands, idxs) = ann.get_name_and_strand(chrom, pos)
    if len(idxs) == 0:
        return scores

    fasta_chrom = normalise_chrom(chrom, list(ann.ref_fasta.keys())[0])
    try:
        # The start index is clamped because pyfastx does not raise on a negative one: it
        # segfaults, taking the whole process with it. A window that overruns the *end* of a
        # contig is already answered with a short read, so clamping makes the two edges behave
        # the same way and the length check below rejects both.
        seq = ann.ref_fasta[fasta_chrom][max(pos-wid//2-1, 0):pos+wid//2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping position (fasta issue): {}-{}'.format(chrom, pos))
        return scores

    if len(seq) != wid:
        logging.warning('Skipping position (too close to a chromosome end): {}-{}'.format(chrom, pos))
        return scores

    genomic_coords = np.arange(pos - cov//2, pos + cov//2 + 1)

    # transcripts sharing the same tx start/stop & strand share the same REF scores
    ref_scores_transcript_cache = {}
    for i in range(len(idxs)):
        dist_ann = ann.get_pos_data(idxs[i], pos)
        pad_size = [max(wid//2+dist_ann[0], 0), max(wid//2-dist_ann[1], 0)]
        x_ref = 'N'*pad_size[0]+seq[pad_size[0]:wid-pad_size[1]]+'N'*pad_size[1]

        strand = strands[i]
        cache_key = (x_ref, strand)
        if cache_key not in ref_scores_transcript_cache:
            ref_scores_transcript_cache[cache_key] = get_ref_scores_for_transcript(x_ref, strand, ann)
        y_ref = ref_scores_transcript_cache[cache_key]

        if len(genomic_coords) != y_ref.shape[1]:
            raise ValueError(f"SpliceAI internal error: len(genomic_coords) != y_ref.shape[1]: "
                              f"{len(genomic_coords)} != {y_ref.shape[1]}")

        idx_a = int(y_ref[0, :, 1].argmax())
        idx_d = int(y_ref[0, :, 2].argmax())

        scores.append({
            "NAME": genes[i],
            "STRAND": strand,
            "RA_MAX": f"{y_ref[0, idx_a, 1]:{FLOAT_FORMAT}}",
            "RA_MAX_POS": idx_a - cov//2,  # offset from the queried position, matching DP_AG/DP_AL/DP_DG/DP_DL
            "RD_MAX": f"{y_ref[0, idx_d, 2]:{FLOAT_FORMAT}}",
            "RD_MAX_POS": idx_d - cov//2,
            "ALL_NON_ZERO_SCORES": [
                {
                    "pos": int(genomic_coord),
                    "RA": f"{ra_score:{FLOAT_FORMAT}}",
                    "RD": f"{rd_score:{FLOAT_FORMAT}}",
                } for i2, (genomic_coord, ra_score, rd_score) in enumerate(zip(
                    genomic_coords, y_ref[0, :, 1], y_ref[0, :, 2])
                ) if ra_score >= MIN_SCORE_THRESHOLD or rd_score >= MIN_SCORE_THRESHOLD or i2 in (idx_a, idx_d)
            ],
        })

    return scores


def get_delta_scores(record, ann, dist_var, mask):

    cov = 2*dist_var+1
    wid = 10000+cov
    scores = []

    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Skipping record (bad input): {}'.format(record))
        return scores

    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
    if len(idxs) == 0:
        return scores

    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    try:
        # See the matching comment in get_reference_scores: a negative start index segfaults
        # pyfastx rather than raising, so clamp it and let the length check below reject the
        # window the same way it already rejects one that overruns the end of a contig.
        seq = ann.ref_fasta[chrom][max(record.pos-wid//2-1, 0):record.pos+wid//2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(record))
        return scores

    # Checked before the REF comparison below, which reads seq at a fixed offset of wid//2 and
    # so is only looking at the variant's own base when the window came back whole.
    if len(seq) != wid:
        logging.warning('Skipping record (too close to a chromosome end): {}'.format(record))
        return scores

    if seq[wid//2:wid//2+len(record.ref)].upper() != record.ref:
        logging.warning('Skipping record (ref issue): {}'.format(record))
        return scores

    if len(record.ref) > 2*dist_var:
        logging.warning('Skipping record (ref too long): {}'.format(record))
        return scores

    genomic_coords = np.arange(record.pos - cov//2, record.pos + cov//2 + 1)

    # many of the transcripts in a gene can have the same tx start & stop positions, so their results can be cached
    # since SpliceAI scores (prior to masking) depend only on transcript start & stop coordinates and strand.
    delta_scores_transcript_cache = {}
    model_prediction_count = 0
    total_count = 0
    for j in range(len(record.alts)):
        # An allele that deletes far more than it inserts, or one written with a long run of shared
        # leading bases, changes bases the output window doesn't reach, leaving nothing there to score.
        # Skipping it here is what keeps align_ref_and_alt_scores from reducing over an empty slice.
        if not span_fits_in_output_window(record.ref, str(record.alts[j]), cov):
            logging.warning('Skipping record (the bases it changes fall outside the scored window): {}'.format(record))
            continue

        for i in range(len(idxs)):

            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue

            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            pad_size = [max(wid//2+dist_ann[0], 0), max(wid//2-dist_ann[1], 0)]
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])

            # An allele written with shared leading bases changes where the alleles first differ rather
            # than at the record's own position. Everything below is measured from there, so that a
            # padded spelling is reported like its shortest one. A deletion-insertion's comparison is
            # collapsed onto the strongest REF base of the span, which can be a later one, and the
            # acceptor and donor channels choose independently.
            bases_dropped_from_start, trimmed_ref, trimmed_alt = trim_shared_bases(record.ref, str(record.alts[j]))
            changed_span_start = record.pos + bases_dropped_from_start
            anchor_index = cov//2 + bases_dropped_from_start
            anchor_seq_index = wid//2 + bases_dropped_from_start

            x_ref = 'N'*pad_size[0]+seq[pad_size[0]:wid-pad_size[1]]+'N'*pad_size[1]
            x_alt = x_ref[:wid//2]+str(record.alts[j])+x_ref[wid//2+ref_len:]

            total_count += 1
            strand = strands[i]
            args = (x_ref, x_alt, record.ref, str(record.alts[j]), strand, cov)
            if args not in delta_scores_transcript_cache:
                model_prediction_count += 1
                delta_scores_transcript_cache[args] = get_delta_scores_for_transcript(*args, ann=ann)

            y_ref, y_alt, y_alt_with_inserted_bases = delta_scores_transcript_cache[args]

            y = np.concatenate([y_ref, y_alt])

            idx_pa = (y[1, :, 1]-y[0, :, 1]).argmax()
            idx_na = (y[0, :, 1]-y[1, :, 1]).argmax()
            idx_pd = (y[1, :, 2]-y[0, :, 2]).argmax()
            idx_nd = (y[0, :, 2]-y[1, :, 2]).argmax()

            mask_pa = np.logical_and((idx_pa-cov//2 == dist_ann[2]), mask)
            mask_na = np.logical_and((idx_na-cov//2 != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd-cov//2 == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd-cov//2 != dist_ann[2]), mask)

            if len(genomic_coords) != y_ref.shape[1]:
                raise ValueError(f"SpliceAI internal error: len(genomic_coords) != y_ref.shape[1]: "
                                 f"{len(genomic_coords)} != {y_ref.shape[1]}")

            if len(genomic_coords) != y_alt.shape[1]:
                raise ValueError(f"SpliceAI internal error: len(genomic_coords) != y_alt.shape[1]: "
                                 f"{len(genomic_coords)} != {y_alt.shape[1]}")

            DS_AG = (y[1, idx_pa, 1]-y[0, idx_pa, 1])*(1-mask_pa)
            DS_AL = (y[0, idx_na, 1]-y[1, idx_na, 1])*(1-mask_na)
            DS_DG = (y[1, idx_pd, 2]-y[0, idx_pd, 2])*(1-mask_pd)
            DS_DL = (y[0, idx_nd, 2]-y[1, idx_nd, 2])*(1-mask_nd)

            DP_AG =  int(idx_pa-cov//2)
            DP_AL =  int(idx_na-cov//2)
            DP_DG =  int(idx_pd-cov//2)
            DP_DL =  int(idx_nd-cov//2)

            # if the variant is an insertion and the model predicts a change in splicing within the inserted bases,
            # retrieve scores for each inserted base to address https://github.com/broadinstitute/SpliceAI-lookup/issues/84

            if (len(trimmed_ref) == 1 and len(trimmed_alt) > 1
                    and ((DS_AG >= 0.01 and DP_AG == bases_dropped_from_start)
                         or (DS_DG >= 0.01 and DP_DG == bases_dropped_from_start))):

                inserted_bases_genomic_coords = np.concatenate([
                    np.arange(changed_span_start - INSERTED_BASES_CONTEXT + 1, changed_span_start + 1),
                    [f"+{offset}" for offset in np.arange(1, len(trimmed_alt))],
                    np.arange(changed_span_start + 1, changed_span_start + INSERTED_BASES_CONTEXT + 1),
                ])

                y_ref_inserted_bases = np.concatenate([
                    y_ref[:, 1 + anchor_index - INSERTED_BASES_CONTEXT : 1 + anchor_index],
                    np.zeros((1, len(trimmed_alt) - 1, 3)),
                    y_ref[:, 1 + anchor_index : 1 + anchor_index + INSERTED_BASES_CONTEXT],
                ], axis=1)

                y_alt_inserted_bases = y_alt_with_inserted_bases[
                    :, 1 + anchor_index - INSERTED_BASES_CONTEXT: 1 + anchor_index + (len(trimmed_alt) - 1) + INSERTED_BASES_CONTEXT]

                assert y_ref_inserted_bases.shape == y_alt_inserted_bases.shape

                ref_seq = (
                    seq[anchor_seq_index - INSERTED_BASES_CONTEXT + 1: anchor_seq_index + 1] +
                    " " * (len(trimmed_alt) - 1) +
                    seq[anchor_seq_index + 1 : anchor_seq_index + 1 + INSERTED_BASES_CONTEXT]
                )
                alt_seq = (
                    seq[anchor_seq_index - INSERTED_BASES_CONTEXT + 1: anchor_seq_index + 1] +
                    trimmed_alt[1:] +
                    seq[anchor_seq_index + len(trimmed_ref) : anchor_seq_index + len(trimmed_ref) + INSERTED_BASES_CONTEXT]
                )

                assert len(ref_seq) == len(alt_seq), f"len(ref_seq) != len(alt_seq): {len(ref_seq)} != {len(alt_seq)}"

            else:
                inserted_bases_genomic_coords = ref_seq = alt_seq = y_ref_inserted_bases = y_alt_inserted_bases = None

            # Report every position that clears the threshold and the positions where the delta
            # scores are maximized, as before, plus the variant's own position -- that one even
            # when nothing there clears the threshold, since it is what the caller asked about.
            all_non_zero_scores = []
            for window_i in sorted(
                    {int(i) for i in np.flatnonzero(
                        np.max(np.stack([y_ref[0, :, 1], y_alt[0, :, 1], y_ref[0, :, 2], y_alt[0, :, 2]]), axis=0) >= MIN_SCORE_THRESHOLD)}
                    | {int(idx_pa), int(idx_na), int(idx_pd), int(idx_nd), cov//2}):
                genomic_coord = int(genomic_coords[window_i])
                reference_base = seq[genomic_coord - record.pos + wid//2].upper()
                if genomic_coord == changed_span_start and ref_len != alt_len:
                    # insertion or deletion: name the alleles where they first differ. They are the
                    # trimmed ones; for a variant already written with no shared bases they are the
                    # whole alleles. A deletion-insertion's comparison is reported at the strongest REF
                    # base of the span instead, which can be a later one, and acceptor and donor can
                    # pick different ones, so no single row carries "its" scores.
                    ref_base, alt_base = trimmed_ref, trimmed_alt
                elif changed_span_start <= genomic_coord < changed_span_start + len(trimmed_ref):
                    # one of the bases the variant replaces: for a change of the same length each position
                    # has its own ALT base, otherwise the base is deleted by the variant
                    ref_base = reference_base
                    alt_base = trimmed_alt[genomic_coord - changed_span_start] if len(trimmed_ref) == len(trimmed_alt) else "-"
                else:
                    ref_base, alt_base = reference_base, reference_base

                all_non_zero_scores.append({
                    "pos": genomic_coord,
                    "ref": ref_base,
                    "alt": alt_base,
                    "RA": f"{y_ref[0, window_i, 1]:{FLOAT_FORMAT}}",
                    "AA": f"{y_alt[0, window_i, 1]:{FLOAT_FORMAT}}",
                    "RD": f"{y_ref[0, window_i, 2]:{FLOAT_FORMAT}}",
                    "AD": f"{y_alt[0, window_i, 2]:{FLOAT_FORMAT}}",
                })

            scores.append({
                "ALLELE": record.alts[j],
                "NAME": genes[i],
                "STRAND": strands[i],
                "DS_AG": f"{DS_AG:{FLOAT_FORMAT}}",
                "DS_AL": f"{DS_AL:{FLOAT_FORMAT}}",
                "DS_DG": f"{DS_DG:{FLOAT_FORMAT}}",
                "DS_DL": f"{DS_DL:{FLOAT_FORMAT}}",
                "DP_AG": DP_AG,
                "DP_AL": DP_AL,
                "DP_DG": DP_DG,
                "DP_DL": DP_DL,
                "DS_AG_REF": f"{y[0, idx_pa, 1]:{FLOAT_FORMAT}}",
                "DS_AL_REF": f"{y[0, idx_na, 1]:{FLOAT_FORMAT}}",
                "DS_DG_REF": f"{y[0, idx_pd, 2]:{FLOAT_FORMAT}}",
                "DS_DL_REF": f"{y[0, idx_nd, 2]:{FLOAT_FORMAT}}",
                "DS_AG_ALT": f"{y[1, idx_pa, 1]:{FLOAT_FORMAT}}",
                "DS_AL_ALT": f"{y[1, idx_na, 1]:{FLOAT_FORMAT}}",
                "DS_DG_ALT": f"{y[1, idx_pd, 2]:{FLOAT_FORMAT}}",
                "DS_DL_ALT": f"{y[1, idx_nd, 2]:{FLOAT_FORMAT}}",
                "ALL_NON_ZERO_SCORES": all_non_zero_scores,
                "SCORES_FOR_INSERTED_BASES": [] if y_alt_inserted_bases is None else [
                    {
                        "chrom": chrom,
                        "pos": genomic_coord,
                        "ref": ref_base,
                        "alt": alt_base,
                        "RA": f"{ref_acceptor_score:{FLOAT_FORMAT}}",  # REF acceptor score
                        "RD": f"{ref_donor_score:{FLOAT_FORMAT}}",     # REF donor score
                        "AA": f"{alt_acceptor_score:{FLOAT_FORMAT}}",  # ALT acceptor score
                        "AD": f"{alt_donor_score:{FLOAT_FORMAT}}",     # ALT donor score
                    } for i, (genomic_coord, ref_base, alt_base, ref_acceptor_score, alt_acceptor_score, ref_donor_score, alt_donor_score) in enumerate(zip(
                        inserted_bases_genomic_coords, ref_seq, alt_seq, y_ref_inserted_bases[0, :, 1], y_alt_inserted_bases[0, :, 1], y_ref_inserted_bases[0, :, 2], y_alt_inserted_bases[0, :, 2]))
                ],
            })

            # print SCORES_FOR_INSERTED_BASES
            #if scores[-1]["SCORES_FOR_INSERTED_BASES"]:
            #    from pprint import pprint
            #    import pandas as pd
            #    import tabulate
            #    print("="*100)
            #    print(f"Variant: {record.chrom}-{record.pos}-{record.ref}-{record.alts[j]}, strand: {strands[i]}")
            #    pprint("-".join([scores[-1]["SCORES_FOR_INSERTED_BASES"][0][key] for key in ("chrom", "pos", "ref", "alt")]))
            #    print(tabulate.tabulate(pd.DataFrame(scores[-1]["SCORES_FOR_INSERTED_BASES"]), headers="keys", tablefmt="pretty"))

    return scores

