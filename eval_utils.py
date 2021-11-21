# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re
from data_utils import aspect_cate_list

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def extract_spans_para(task, seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    if task == 'aste':
        for s in sents:
            # It is bad because editing is problem.
            try:
                c, ab = s.split(' because ')
                c = opinion2word.get(c[6:], 'nope')    # 'good' -> 'positive'
                a, b = ab.split(' is ')
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                a, b, c = '', '', ''
            quads.append((a, b, c))
    elif task == 'tasd':
        for s in sents:
            # food quality is bad because pizza is bad.
            try:
                ac_sp, at_sp = s.split(' because ')
                
                ac, sp = ac_sp.split(' is ')
                at, sp2 = at_sp.split(' is ')
                
                sp = opinion2word.get(sp, 'nope')
                sp2 = opinion2word.get(sp2, 'nope')
                if sp != sp2:
                    print(f'Sentiment polairty of AC({sp}) and AT({sp2}) is inconsistent!')
                
                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                ac, at, sp = '', '', ''
            
            quads.append((ac, at, sp))
    elif task == 'asqp':
        for s in sents:
            # food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(' because ')
                ac, sp = ac_sp.split(' is ')
                at, ot = at_ot.split(' is ')

                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, at, sp, ot = '', '', '', ''

            quads.append((ac, at, sp, ot))
    else:
        raise NotImplementedError
    return quads


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para('asqp', gold_seqs[i], 'gold')
        pred_list = extract_spans_para('asqp', pred_seqs[i], 'pred')

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels)
    print(scores)

    return scores, all_labels, all_preds
