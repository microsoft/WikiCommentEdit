import datetime
import glob
import os
import sys

import torch
import torch.nn.functional as F

from eval import eval
from process_data import to_var, make_vector, gen_cmntrank_batches, gen_editanch_batches


def rank_loss(score_pos, score_neg):
    # normalize context attention
    # ctx_att_norm = F.normalize(ctx_att, p=2, dim=1)
    batch_size = len(score_pos)
    # y = Variable(torch.FloatTensor([1] * batch_size))
    margin = to_var(torch.FloatTensor(score_pos.size()).fill_(1))
    # loss = F.margin_ranking_loss(score_pos, -score_neg, y, margin=10.0)

    loss_list = margin - score_pos + score_neg
    loss_list = loss_list.clamp(min=0)
    loss = loss_list.sum()

    return loss, loss_list


def cal_batch_loss(loss_list, batch_size, index_list):
    loss_list = loss_list.data.squeeze(1).numpy().tolist()
    loss_step = int(len(loss_list) / batch_size)
    # return [sum(loss_list[i * loss_step: (i + 1) * loss_step]) / loss_step for i in range(batch_size)]
    start = 0
    batch_loss_list = []
    for i in range(batch_size):
        cur_bs = index_list[i]
        end = start + cur_bs
        if cur_bs == 0:
            batch_loss_list.append(0)
        else:
            batch_loss_list.append(sum(loss_list[start:end]) / cur_bs)
        start = end
    return batch_loss_list


def train(args, model, dataset, train_df, val_df, optimizer, w2i, n_epoch, start_epoch, batch_size):
    print('----Train---')
    label = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model.train()

    cmnt_sent_len = args.max_cmnt_length
    diff_sent_len = args.max_diff_length
    ctx_sent_len = args.max_ctx_length
    train_size = len(train_df)
    # if args.use_cl:
    #     sample_size = int(train_size * 0.5)
    # else:
    #     sample_size = -1
    sample_size = int(train_size)
    cr_best_acc, ea_best_acc = 0, 0

    for epoch in range(start_epoch, n_epoch + 1):
        print('============================== Epoch ', epoch, ' ==============================')

        for step, batch in enumerate(dataset.iterate_minibatches(train_df, batch_size, epoch, n_epoch), start=1):
            batch_sample_weights = None
            total_loss = 0
            if args.cr_train:
                cr_pos_batch, cr_neg_batch = gen_cmntrank_batches(batch, w2i, cmnt_sent_len, diff_sent_len,
                                                                  ctx_sent_len, args.rank_num)
                if len(cr_pos_batch[0]) > 0:
                    # TODO: tuning the code for more effective way: combine the positive sample and negative samples
                    pos_cmnt, pos_src_token, pos_src_action, pos_tgt_token, pos_tgt_action = \
                        make_vector(cr_pos_batch, w2i, cmnt_sent_len, ctx_sent_len)
                    neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action = \
                        make_vector(cr_neg_batch, w2i, cmnt_sent_len, ctx_sent_len)
                    score_pos, _ = model(pos_cmnt, pos_src_token, pos_src_action, pos_tgt_token, pos_tgt_action,
                                         cr_mode=True)
                    score_neg, _ = model(neg_cmnt, neg_src_token, neg_src_action, neg_tgt_token, neg_tgt_action,
                                         cr_mode=True)

                    loss, _ = rank_loss(score_pos, score_neg)
                    # batch_sample_weights = cal_batch_loss(loss_list, batch_size, index_list)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # train revision anchoring
            if args.ea_train:
                ea_batch, ea_truth = gen_editanch_batches(batch, w2i, cmnt_sent_len, diff_sent_len, ctx_sent_len,
                                                          args.anchor_num)
                if len(ea_batch[0]) > 0:
                    cmnt, src_token, src_action, tgt_token, tgt_action = \
                        make_vector(ea_batch, w2i, cmnt_sent_len, ctx_sent_len)
                    logit, _ = model(cmnt, src_token, src_action, tgt_token, tgt_action, cr_mode=False)
                    target = to_var(torch.tensor(ea_truth))

                    loss = F.cross_entropy(logit, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # update the loss for each data sample
            # new_sample_weights += batch_sample_weights

            if step % args.log_interval == 0:
                # corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                # p1_corr, p3_corr, p5_corr, mrr, ndcg = eval_rank(score_pos, score_neg, args.batch_size)
                # p1_acc = p1_corr / args.batch_size * 100
                # p3_acc = p3_corr / args.batch_size * 100
                # p5_acc = p5_corr / args.batch_size * 100
                try:
                    sys.stdout.write(
                        '\rEpoch[{}] Batch[{}] - loss: {:.6f}\n'.format(epoch, step, loss.data.item()))
                    sys.stdout.flush()
                except:
                    print("Unexpected error:", sys.exc_info()[0])

            if step % args.test_interval == 0:
                if args.val_size > 0:
                    val_df = val_df[:args.val_size]
                cr_acc, ea_acc = eval(dataset, val_df, w2i, model, args)
                model.train()  # change model back to training mode
                if args.cr_train and cr_acc > cr_best_acc:
                    cr_best_acc = cr_acc
                    if args.save_best:
                        save(model, args.save_dir, 'best_cr', epoch, step, cr_best_acc, args.no_action)
                if args.ea_train and ea_acc > ea_best_acc:
                    ea_best_acc = ea_acc
                    if args.save_best:
                        save(model, args.save_dir, 'best_ea', epoch, step, ea_best_acc, args.no_action)
        # sample_weights = new_sample_weights


def save(model, save_dir, save_prefix, epoch, steps, best_result, no_action=False):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)

    # delete previously saved checkpoints
    exist_files = sorted(glob.glob(save_prefix + '*'))
    for file_name in exist_files:
        if os.path.exists(file_name):
            os.remove(file_name)

    result_str = '%.3f' % best_result
    save_path = '{}_steps_{:02}_{:06}_{}.pt'.format(save_prefix, epoch, steps, result_str)
    print("Save best model", save_path)
    torch.save(model.state_dict(), save_path)
