import math
from statistics import mean

import sklearn
import sklearn.metrics
import torch

from process_data import _make_char_vector, _make_word_vector, gen_cmntrank_batches, gen_editanch_batches
from process_data import to_var, make_vector
from wiki_util import tokenizeText


## general evaluation
def predict(pred_cmnt, pred_ctx, w2i, c2i, model, max_ctx_length):
    print("prediction on single sample ...")
    model.eval()

    _, pred_cmnt_words = tokenizeText(pred_cmnt)
    pred_cmnt_chars = [list(i) for i in pred_cmnt_words]
    _, pred_ctx_words = tokenizeText(pred_ctx)
    pred_ctx_chars = [list(i) for i in pred_ctx_words]

    cmnt_sent_len = len(pred_cmnt_words)
    cmnt_word_len = int(mean([len(w) for w in pred_cmnt_chars]))
    ctx_sent_len = max_ctx_length
    ctx_word_len = int(mean([len(w) for w in pred_ctx_chars]))

    cmnt_words, cmnt_chars, ctx_words, ctx_chars = [], [], [], []
    # c, cc, q, cq, a in batch

    cmnt_words.append(_make_word_vector(pred_cmnt_words, w2i, cmnt_sent_len))
    cmnt_chars.append(_make_char_vector(pred_cmnt_chars, c2i, cmnt_sent_len, cmnt_word_len))
    ctx_words.append(_make_word_vector(pred_ctx_words, w2i, ctx_sent_len))
    ctx_chars.append(_make_char_vector(pred_ctx_chars, c2i, ctx_sent_len, ctx_word_len))

    cmnt_words = to_var(torch.LongTensor(cmnt_words))
    cmnt_chars = to_var(torch.stack(cmnt_chars, 0))
    ctx_words = to_var(torch.LongTensor(ctx_words))
    ctx_chars = to_var(torch.stack(ctx_chars, 0))

    logit, _ = model(ctx_words, ctx_chars, cmnt_words, cmnt_chars)
    a = torch.max(logit.cpu(), -1)
    print(logit)
    print(a)
    y_pred = a[1].data[0]
    y_prob = a[0].data[0]
    y_pred_2 = [int(i) for i in (torch.max(logit, -1)[1].view(1).data).tolist()][0]
    print(y_pred_2)
    # y_pred = y_pred[0]
    print(y_pred, y_prob)
    return y_pred


def compute_rank_score(pos_score, neg_scores):
    p1, p3, p5 = 0, 0, 0

    pos_list = [0 if pos_score > neg_score else 1 for neg_score in neg_scores]
    pos = sum(pos_list)
    # precision @K
    if pos == 0: p1 = 1
    if pos < 3: p3 = 1
    if pos < 5: p5 = 1
    # MRR
    mrr = 1 / (pos + 1)
    # NDCG: DCG/IDCG (In our case, we set the rel=1 if relevent, otherwise rel=0; Then IDCG=1)
    ndcg = 1 / math.log2(pos + 2)

    return p1, p3, p5, mrr, ndcg


def get_rank(pos_score, neg_scores):
    pos_list = [0 if pos_score > neg_score else 1 for neg_score in neg_scores]
    pos = sum(pos_list)
    return pos + 1


def isEditPredCorrect(pred, truth):
    for i in len(pred):
        if pred[i] != truth[i]:
            return False

    return True


def eval_rank(score_pos, score_neg, cand_num):
    score_pos_list = score_pos.data.cpu().squeeze(1).numpy().tolist()
    score_neg_list = score_neg.data.cpu().squeeze(1).numpy().tolist()

    correct_p1, correct_p3, correct_p5, total_mrr, total_ndcg = 0, 0, 0, 0, 0
    neg_num = cand_num - 1
    batch_num = int(len(score_neg) / neg_num)
    rank_list = []
    for i in range(batch_num):
        score_pos_i = score_pos_list[i * neg_num: (i + 1) * neg_num]
        score_neg_i = score_neg_list[i * neg_num: (i + 1) * neg_num]

        p1, p3, p5, mrr, ndcg = compute_rank_score(score_pos_i[0], score_neg_i)
        rank = get_rank(score_pos_i[0], score_neg_i)
        rank_list.append(rank)
        correct_p1 += p1
        correct_p3 += p3
        correct_p5 += p5
        total_mrr += mrr
        total_ndcg += ndcg

    return correct_p1, correct_p3, correct_p5, total_mrr, total_ndcg, rank_list


# def eval_rank_orig(score_pos, score_neg, batch_size):
#     score_pos_list = score_pos.data.cpu().squeeze(1).numpy().tolist()
#     score_neg_list = score_neg.data.cpu().squeeze(1).numpy().tolist()
#
#     total_p1, total_p3, total_p5, total_mrr, total_ndcg = 0, 0, 0, 0, 0
#     sample_num = int(len(score_neg) / batch_size)
#     for i in range(batch_size):
#         score_pos_i = score_pos_list[i * sample_num: (i+1) * sample_num]
#         score_neg_i = score_neg_list[i * sample_num: (i+1) * sample_num]
#         pos_list = [0 if score_pos_i[i] >= score_neg_i[i] else 1 for i in range(sample_num)]
#         pos = sum(pos_list)
#         sorted_neg = ["%.4f" % i for i in sorted(score_neg_i, reverse=True)]
#         #print(pos, "%.4f" % score_pos_i[0], "\t".join(sorted_neg), sep='\t')
#         if pos == 0:
#             total_p1 += 1
#
#         if pos < 3:
#             total_p3 += 1
#
#         if pos < 5:
#             total_p5 += 1
#
#         # MRR
#         total_mrr += 1 / (pos + 1)
#
#         # NDCG: DCG/IDCG (In our case, we set the rel=1 if relevent, otherwise rel=0; Then IDCG=1)
#         total_ndcg += 1 / math.log2(pos + 2)
#
#     return total_p1, total_p3, total_p5, total_mrr, total_ndcg

## general evaluation
def eval(dataset, val_df, w2i, model, args):
    # print(" evaluation on", val_df.shape[0], " samples ...")
    model.eval()
    corrects, avg_loss = 0, 0

    cmnt_rank_p1, cmnt_rank_p3, cmnt_rank_p5, cmnt_rank_mrr, cmnt_rank_ndcg = 0, 0, 0, 0, 0
    ea_pred, ea_truth = [], []
    cr_total, ea_total = 0, 0
    pred_cr_list, pred_ea_list = [], []
    for batch in dataset.iterate_minibatches(val_df, args.batch_size):
        cmnt_sent_len = args.max_cmnt_length
        ctx_sent_len = args.max_ctx_length
        diff_sent_len = args.max_diff_length

        ###########################################################
        # Comment Ranking Task
        ###########################################################
        # generate positive and negative batches
        pos_batch, neg_batch = gen_cmntrank_batches(batch, w2i, cmnt_sent_len, diff_sent_len, ctx_sent_len,
                                                    args.rank_num)

        pos_cmnt, pos_src_token, pos_src_action, pos_tgt_token, pos_tgt_action = \
            make_vector(pos_batch, w2i, cmnt_sent_len, ctx_sent_len)
        neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action = \
            make_vector(neg_batch, w2i, cmnt_sent_len, ctx_sent_len)
        score_pos, _ = model(pos_cmnt, pos_src_token, pos_src_action, pos_tgt_token, pos_tgt_action, cr_mode=True)
        score_neg, _ = model(neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action, cr_mode=True)

        cr_p1_corr, cr_p3_corr, cr_p5_corr, cr_mrr, cr_ndcg, pred_rank = eval_rank(score_pos, score_neg, args.rank_num)
        cmnt_rank_p1 += cr_p1_corr
        cmnt_rank_p3 += cr_p3_corr
        cmnt_rank_p5 += cr_p5_corr
        cmnt_rank_mrr += cr_mrr
        cmnt_rank_ndcg += cr_ndcg
        cr_total += int(len(score_pos) / (args.rank_num - 1))
        pred_cr_list += pred_rank

        ###########################################################
        # Edits Anchoring
        ###########################################################
        # generate positive and negative batches
        ea_batch, ea_truth_cur = gen_editanch_batches(batch, w2i, cmnt_sent_len, diff_sent_len, ctx_sent_len,
                                                      args.anchor_num)
        if len(pos_batch[0]) > 0:
            cmnt, src_token, src_action, tgt_token, tgt_action = \
                make_vector(ea_batch, w2i, cmnt_sent_len, ctx_sent_len)
            # neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action = \
            #                         make_vector(neg_batch, w2i, cmnt_sent_len, ctx_sent_len)
            logit, _ = model(cmnt, src_token, src_action, tgt_token, tgt_action, cr_mode=False)
            # logit_neg, _ = model(neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action, cr_mode=False)

            ea_pred_cur = (torch.max(logit, 1)[1].view(logit.size(0)).data).tolist()
            # ea_truth_cur = [1] * logit_pos.size(0) + [0] * logit_neg.size(0)

            ea_pred += ea_pred_cur
            ea_truth += ea_truth_cur

            ea_total += int(len(score_pos) / (args.anchor_num - 1))

    # # output the prediction results
    # with open(args.checkpoint_path + 'test_out.txt', 'w') as f:
    #     for i in range(len(y_truth)):
    #         line = cmnt_readable_all[i] + '\t' + ctx_readable_all[i] + '\t' + str(y_pred[i]) + '\t' + str(y_truth[i])
    #         f.write(line + '\n')

    # if args.test:
    #     print(total_rank)
    # print("\t".join([str(i) for i in pred_cr_list]))
    # print("\t".join([str(i) for i in ea_pred]))

    cr_p1_acc = cmnt_rank_p1 / cr_total
    cr_p3_acc = cmnt_rank_p3 / cr_total
    cr_p5_acc = cmnt_rank_p5 / cr_total
    cr_mrr = cmnt_rank_mrr / cr_total
    cr_ndcg = cmnt_rank_ndcg / cr_total

    ea_acc = (sklearn.metrics.accuracy_score(ea_truth, ea_pred))
    ea_f1 = (sklearn.metrics.f1_score(ea_truth, ea_pred, pos_label=1))
    ea_prec = (sklearn.metrics.precision_score(ea_truth, ea_pred, pos_label=1))
    ea_recall = (sklearn.metrics.recall_score(ea_truth, ea_pred, pos_label=1))

    print("\n*** Validation Results *** ")
    # print("[Task-CR] P@1:", "%.3f" % cr_p1_acc, "% P@3:", "%.3f" % cr_p3_acc, "% P@5:", "%.3f" % cr_p5_acc,\
    #         '%', ' (', cmnt_rank_p1, '/', cr_total, ',', cmnt_rank_p3, '/', cr_total, ',', cmnt_rank_p5, '/', cr_total,')', sep='')
    # print("[Task-RA] P@1:", "%.3f" % ea_p1_acc, "% P@3:", "%.3f" % ea_p3_acc, "% P@5:", "%.3f" % ea_p5_acc,\
    #         '%', ' (', edit_anch_p1, '/', ea_total, ',', edit_anch_p3, '/', ea_total, ',', edit_anch_p5, '/', ea_total,')', sep='')
    print("[Task-CR] P@1:", "%.3f" % cr_p1_acc, " P@3:", "%.3f" % cr_p3_acc, " P@5:", "%.3f" % cr_p5_acc, " MRR:",
          "%.3f" % cr_mrr, " NDCG:", "%.3f" % cr_ndcg, sep='')
    print("[Task-EA] ACC:", "%.3f" % ea_acc, " F1:", "%.3f" % ea_f1, " Precision:", "%.3f" % ea_prec, " Recall:",
          "%.3f" % ea_recall, sep='')
    return cr_p1_acc, ea_f1


def dump_cmntrank_case(pos_batch, neg_batch, idx, rank_num, diff_url, rank, pos_score, neg_scores):
    neg_num = rank_num - 1
    pos_cmnt = pos_batch[0][idx * neg_num]
    neg_cmnts = neg_batch[0][idx * neg_num: (idx + 1) * neg_num]

    before_edit = pos_batch[1][idx * neg_num]
    after_edit = pos_batch[3][idx * neg_num]

    match = False
    for token in pos_cmnt:
        if token in before_edit + after_edit:
            match = True
            break

    neg_match_words = []
    neg_match = False
    for neg_cmnt in neg_cmnts:
        for token in neg_cmnt:
            if len(token) <= 3:
                continue
            if token in before_edit + after_edit:
                neg_match = True
                neg_match_words.append(token)

    if not match and neg_match:
        print("\n ====== cmntrank case (Not Matched) ======")
        print("Rank", rank)
        print(diff_url)
        print("pos_cmnt (", "{0:.3f}".format(pos_score), "): ", " ".join(pos_cmnt), sep='')

        for i, neg_cmnt in enumerate(neg_cmnts):
            print("neg_cmnt ", i, " (", "{0:.3f}".format(neg_scores[i]), "): ", " ".join(neg_cmnt), sep='')
        pass
        print("neg_match_words:", " ".join(neg_match_words))


def dump_editanch_case(comment, edit, pred, truth):
    print("\n ====== editanch case ======")
    print("pred/truth: ", pred, "/", truth)
    print("comment:", " ".join(comment))
    print("edit:", " ".join(edit))


def case_study(dataset, val_df, w2i, model, args):
    model.eval()
    print("Start the case study")
    # for batch in dataset.iterate_minibatches(val_df[:500], args.batch_size):
    for batch in dataset.iterate_minibatches(val_df, args.batch_size):
        cmnt_sent_len = args.max_cmnt_length
        ctx_sent_len = args.max_ctx_length
        diff_sent_len = args.max_diff_length

        ###########################################################
        # Comment Ranking Task
        ###########################################################
        # generate positive and negative batches

        if args.cr_train:
            pos_batch, neg_batch = gen_cmntrank_batches(batch, w2i, cmnt_sent_len, diff_sent_len, ctx_sent_len,
                                                        args.rank_num)

            pos_cmnt, pos_src_token, pos_src_action, pos_tgt_token, pos_tgt_action = \
                make_vector(pos_batch, w2i, cmnt_sent_len, ctx_sent_len)
            neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action = \
                make_vector(neg_batch, w2i, cmnt_sent_len, ctx_sent_len)
            score_pos, _ = model(pos_cmnt, pos_src_token, pos_src_action, pos_tgt_token, pos_tgt_action, cr_mode=True)
            score_neg, _ = model(neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action, cr_mode=True)

            score_pos_list = score_pos.data.cpu().squeeze(1).numpy().tolist()
            score_neg_list = score_neg.data.cpu().squeeze(1).numpy().tolist()

            neg_num = args.rank_num - 1
            batch_num = int(len(score_neg) / neg_num)
            for i in range(batch_num):
                score_pos_i = score_pos_list[i * neg_num: (i + 1) * neg_num]
                score_neg_i = score_neg_list[i * neg_num: (i + 1) * neg_num]

                rank = get_rank(score_pos_i[0], score_neg_i)
                dump_cmntrank_case(pos_batch, neg_batch, i, args.rank_num, batch[8][i], rank, score_pos_i[0],
                                   score_neg_i)

        ###########################################################
        # Edits Anchoring
        ###########################################################
        # generate positive and negative batches

        if args.ea_train:
            ea_batch, ea_truth_cur = gen_editanch_batches(batch, w2i, cmnt_sent_len, diff_sent_len, ctx_sent_len,
                                                          args.anchor_num)
            cmnt, src_token, src_action, tgt_token, tgt_action = \
                make_vector(ea_batch, w2i, cmnt_sent_len, ctx_sent_len)
            # neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action = \
            #                         make_vector(neg_batch, w2i, cmnt_sent_len, ctx_sent_len)
            logit, _ = model(cmnt, src_token, src_action, tgt_token, tgt_action, cr_mode=False)
            # logit_neg, _ = model(neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action, cr_mode=False)
            ea_pred_cur = (torch.max(logit, 1)[1].view(logit.size(0)).data).tolist()

            for i in range(len(ea_truth_cur)):
                # if ea_pred_cur[i] == ea_truth_cur[i]:
                dump_editanch_case(ea_batch[0][i], ea_batch[3][i], ea_pred_cur[i], ea_truth_cur[i])

            pass
            # ea_truth_cur = [1] * logit_pos.size(0) + [0] * logit_neg.size(0)
