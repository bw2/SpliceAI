from pkg_resources import resource_filename
import pandas as pd
import numpy as np
from pyfaidx import Fasta
from keras.models import load_model
import logging


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
            self.ref_fasta = Fasta(ref_fasta, rebuild=False)
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


def get_delta_scores(record, ann, dist_var, mask):

    cov = 2*dist_var+1
    wid = 10000+cov
    delta_scores = []

    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Skipping record (bad input): {}'.format(record))
        return delta_scores

    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
    if len(idxs) == 0:
        return delta_scores

    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    try:
        seq = ann.ref_fasta[chrom][record.pos-wid//2-1:record.pos+wid//2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(record))
        return delta_scores

    if seq[wid//2:wid//2+len(record.ref)].upper() != record.ref:
        logging.warning('Skipping record (ref issue): {}'.format(record))
        return delta_scores

    if len(seq) != wid:
        logging.warning('Skipping record (near chromosome end): {}'.format(record))
        return delta_scores

    if len(record.ref) > 2*dist_var:
        logging.warning('Skipping record (ref too long): {}'.format(record))
        return delta_scores

    all_scores = []
    for j in range(len(record.alts)):
        for i in range(len(idxs)):

            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue

            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            pad_size = [max(wid//2+dist_ann[0], 0), max(wid//2-dist_ann[1], 0)]
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])
            del_len = max(ref_len-alt_len, 0)

            x_ref = 'N'*pad_size[0]+seq[pad_size[0]:wid-pad_size[1]]+'N'*pad_size[1]
            x_alt = x_ref[:wid//2]+str(record.alts[j])+x_ref[wid//2+ref_len:]

            print("x_ref:", genes[i], ", strand: ", strands[i], x_ref)
            print("x_alt:", genes[i], ", strand: ", strands[i], x_alt)

            x_ref = one_hot_encode(x_ref)[None, :]
            x_alt = one_hot_encode(x_alt)[None, :]

            if strands[i] == '-':
                x_ref = x_ref[:, ::-1, ::-1]
                x_alt = x_alt[:, ::-1, ::-1]

            y_ref = np.mean([ann.models[m].predict(x_ref) for m in range(5)], axis=0)
            y_alt = np.mean([ann.models[m].predict(x_alt) for m in range(5)], axis=0)

            if strands[i] == '-':
                y_ref = y_ref[:, ::-1]
                y_alt = y_alt[:, ::-1]

            if ref_len > 1 and alt_len == 1:
                y_alt = np.concatenate([
                    y_alt[:, :cov//2+alt_len],
                    np.zeros((1, del_len, 3)),
                    y_alt[:, cov//2+alt_len:]],
                    axis=1)
            elif ref_len == 1 and alt_len > 1:
                y_alt = np.concatenate([
                    y_alt[:, :cov//2],
                    np.max(y_alt[:, cov//2:cov//2+alt_len], axis=1)[:, None, :],
                    y_alt[:, cov//2+alt_len:]],
                    axis=1)
            #MNP handling
            elif ref_len > 1 and alt_len > 1:
                zblock = np.zeros((1,ref_len-1,3))
                y_alt = np.concatenate([
                    y_alt[:, :cov//2],
                    np.max(y_alt[:, cov//2:cov//2+alt_len], axis=1)[:, None, :],
                    zblock,
                    y_alt[:, cov//2+alt_len:]],
                    axis=1)

            y = np.concatenate([y_ref, y_alt])
            np.save(f'y-{record.chrom}-{record.pos}-{record.ref}-{record.alts[0]}.npy', y)

            # y shape: (2, 10000, 3)
            #   axis 0: 0=ref, 1=alt
            #   axis 1: base position in window
            #   axis 2: 0=neither, 1=acceptor, 2=donor
            # for any position, the total of the 3 ref probabilities = 1, and the total of the 3 alt probabilites = 1
            # (based on: https://www.youtube.com/watch?v=oJvhj-tYbBI&start=651)
            ref_acceptor_probabilities = y[0, :, 1]
            alt_acceptor_probabilities = y[1, :, 1]
            ref_donor_probabilities = y[0, :, 2]
            alt_donor_probabilities = y[1, :, 2]

            idx_pa = (alt_acceptor_probabilities - ref_acceptor_probabilities).argmax()  # delta score: AG
            idx_na = (ref_acceptor_probabilities - alt_acceptor_probabilities).argmax()  # delta score: AL
            idx_pd = (alt_donor_probabilities - ref_donor_probabilities).argmax()  # delta score: DG
            idx_nd = (ref_donor_probabilities - alt_donor_probabilities).argmax()  # delta score: DL

            mask_pa = np.logical_and((idx_pa-cov//2 == dist_ann[2]), mask)
            mask_na = np.logical_and((idx_na-cov//2 != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd-cov//2 == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd-cov//2 != dist_ann[2]), mask)

            delta_scores.append("{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}".format(
                                record.alts[j],
                                genes[i],
                                (y[1, idx_pa, 1]-y[0, idx_pa, 1])*(1-mask_pa),  # delta score: AG
                                (y[0, idx_na, 1]-y[1, idx_na, 1])*(1-mask_na),  # delta score: AL
                                (y[1, idx_pd, 2]-y[0, idx_pd, 2])*(1-mask_pd),  # delta score: DG
                                (y[0, idx_nd, 2]-y[1, idx_nd, 2])*(1-mask_nd),  # delta score: DL
                                idx_pa-cov//2,    # DP_AG (position: acceptor gain)
                                idx_na-cov//2,    # DP_AL (position: acceptor loss)
                                idx_pd-cov//2,    # DP_DG (position: donor gain)
                                idx_nd-cov//2,    # DP_DL (position: donor loss)
                                y[0, idx_pa, 1],  # REF acceptor probability @ acceptor gain position
                                y[1, idx_pa, 1],  # ALT acceptor probability @ acceptor gain position
                                y[0, idx_na, 1],  # REF acceptor probability @ acceptor loss position
                                y[1, idx_na, 1],  # ALT acceptor probability @ acceptor loss position
                                y[0, idx_pd, 2],  # REF donor probability @ donor gain position
                                y[1, idx_pd, 2],  # ALT donor probability @ donor gain position
                                y[0, idx_nd, 2],  # REF donor probability @ donor loss position
                                y[1, idx_nd, 2])) # ALT donor probability @ donor loss position

            all_scores.append({
                # TODO add position(s)
                'ref_acceptor_probabilities': ref_acceptor_probabilities,
                'alt_acceptor_probabilities': alt_acceptor_probabilities,
                'ref_donor_probabilities': ref_donor_probabilities,
                'alt_donor_probabilities': alt_donor_probabilities,
            })

    return delta_scores, all_scores

