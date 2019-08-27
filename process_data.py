import json
import os
import pickle
import random

import numpy as np
import spacy
import torch
from torch.autograd import Variable

# TODO global
NULL = "-NULL-"
UNK = "-UNK-"
ENT = "-ENT-"

# initialize the spacy
nlp = spacy.load('en')


def word_tokenize(text):
    doc = nlp(text)
    tokens = [token.string.strip() for token in doc]
    return tokens


def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)


def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)


def lower_list(str_list):
    return [str_var.lower() for str_var in str_list]


def load_processed_json(fpath_data, fpath_shared):
    data = json.load(open(fpath_data))
    shared = json.load(open(fpath_shared))
    return data, shared


def load_glove_weights(glove_dir, embd_dim, vocab_size, word_index):
    embeddings_index = {}
    if embd_dim < 300:
        glove_version = 'glove.6B.'
    else:
        glove_version = 'glove.840B.'

    with open(os.path.join(glove_dir, glove_version + str(embd_dim) + 'd.txt'), encoding='utf-8') as f:
        for line in f:
            try:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings_index[word] = vector
            except:
                continue

    print('Found %s word vectors in glove.' % len(embeddings_index))
    embedding_matrix = np.zeros((vocab_size, embd_dim))
    print('embed_matrix.shape', embedding_matrix.shape)
    found_ct = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        # words not found in embedding index will be all-zeros.
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_ct += 1
    print(found_ct, 'words are found in glove')

    return embedding_matrix


def to_var(x):
    # if torch.cuda.is_available():
    #     x = x.cuda()
    # return Variable(x)
    x = Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_np(x):
    return x.data.cpu().numpy()


def _make_action_vector(actions, seq_len):
    index_vec = [action for action in actions]
    pad_len = max(0, seq_len - len(index_vec))
    index_vec += [-1] * pad_len
    index_vec = index_vec[:seq_len]
    return index_vec


def _make_word_vector(sentence, w2i, seq_len):
    index_vec = [w2i[w] if w in w2i else w2i[UNK] for w in sentence]
    pad_len = max(0, seq_len - len(index_vec))
    index_vec += [w2i[NULL]] * pad_len
    index_vec = index_vec[:seq_len]
    return index_vec


def _make_char_vector(data, c2i, sent_len, word_len):
    tmp = torch.ones(sent_len, word_len).type(torch.LongTensor)  # TODO use fills
    for i, word in enumerate(data):
        for j, ch in enumerate(word):
            tmp[i][j] = c2i[ch] if ch in c2i else c2i[UNK]
            if j == word_len - 1:
                break
        if i == sent_len - 1:
            break
    return tmp


def make_diff(diffs_raw):
    return diffs_raw


def make_vector_one_sample(pred_cmnt, pred_ctx, w2i, c2i, ctx_sent_len, ctx_word_len, query_sent_len, query_word_len):
    cmnt_words, cmnt_chars, ctx_words, ctx_chars, ans, diffs = [], [], [], [], [], []
    # c, cc, q, cq, a in batch

    cmnt_words.append(_make_word_vector(batch[0][i], w2i, ctx_sent_len))
    cmnt_chars.append(_make_char_vector(batch[1][i], c2i, ctx_sent_len, ctx_word_len))
    ctx_words.append(_make_word_vector(batch[2][i], w2i, query_sent_len))
    ctx_chars.append(_make_char_vector(batch[3][i], c2i, query_sent_len, query_word_len))
    ans.append(batch[4][i])
    # append the diffs
    diffs_raw = batch[5][i]
    diffs_ex = [-1] * query_sent_len
    for diff_idx in diffs_raw:
        diffs_ex[diff_idx - 1] = 1
    diffs.append(diffs_ex)

    cmnt_words = to_var(torch.LongTensor(cmnt_words))
    cmnt_chars = to_var(torch.stack(cmnt_chars, 0))
    ctx_words = to_var(torch.LongTensor(ctx_words))
    ctx_chars = to_var(torch.stack(ctx_chars, 0))
    ans = to_var(torch.LongTensor(ans))
    diffs = to_var(torch.FloatTensor(diffs))
    return cmnt_words, cmnt_chars, ctx_words, ctx_chars


'''
Generate the word vector for each batch
'''


def make_vector(batch, w2i, cmnt_sent_len, ctx_sent_len):
    cmnt_words, src_token, src_action, tgt_token, tgt_action = [], [], [], [], []
    # batch_cmnt, batch_neg_cmnt, batch_origin, batch_target

    for i in range(len(batch[0])):
        cmnt_words.append(_make_word_vector(batch[0][i], w2i, cmnt_sent_len))

        src_token.append(_make_word_vector(batch[1][i], w2i, ctx_sent_len))
        src_action.append(_make_action_vector(batch[2][i], ctx_sent_len))

        tgt_token.append(_make_word_vector(batch[3][i], w2i, ctx_sent_len))
        tgt_action.append(_make_action_vector(batch[4][i], ctx_sent_len))

    cmnt_words = to_var(torch.LongTensor(cmnt_words))
    # neg_cmnt_words = to_var(torch.LongTensor(neg_cmnt_words))

    src_token = to_var(torch.LongTensor(src_token))
    src_action = to_var(torch.LongTensor(src_action))

    tgt_token = to_var(torch.LongTensor(tgt_token))
    tgt_action = to_var(torch.LongTensor(tgt_action))

    return cmnt_words, src_token, src_action, tgt_token, tgt_action


'''
generate the batches for training and evaluation
type definition:
type 1: for the comment rank task
type 2: for the diff anchoring task
type 3: use the target diff only
'''


def gen_cmntrank_batches(batch, w2i, cmnt_sent_len, diff_sent_len, ctx_sent_len, rank_num):
    '''
    Batch Content:
    0,1. batch comment, batch neg_cmnt
    2,3. batch src_tokens, batch src_actions
    4,5. batch tgt_tokens, batch tgt_actions
    6,7. batch pos_edits, batch neg_edits
    '''

    pos_cmnts, pos_src_tokens, pos_src_actions, pos_tgt_tokens, pos_tgt_actions = [], [], [], [], []
    neg_cmnts, neg_src_tokens, neg_src_actions, neg_tgt_tokens, neg_tgt_actions = [], [], [], [], []
    sample_index_list = []
    for i in range(len(batch[0])):

        sample_size = 0

        cmnt = batch[0][i]
        neg_cmnt = batch[1][i]

        src_tokens = batch[2][i]
        src_actions = batch[3][i]

        tgt_tokens = batch[4][i]
        tgt_actions = batch[5][i]

        if rank_num - 1 > len(neg_cmnt):
            continue

        neg_sample_indices = random.sample(range(len(neg_cmnt)), rank_num - 1)

        for neg_idx in neg_sample_indices:
            pos_cmnts.append(cmnt)
            pos_src_tokens.append(src_tokens)
            pos_src_actions.append(src_actions)
            pos_tgt_tokens.append(tgt_tokens)
            pos_tgt_actions.append(tgt_actions)

            neg_cmnts.append(neg_cmnt[neg_idx])
            neg_src_tokens.append(src_tokens)
            neg_src_actions.append(src_actions)
            neg_tgt_tokens.append(tgt_tokens)
            neg_tgt_actions.append(tgt_actions)
            sample_size += 1

        sample_index_list.append(sample_size)
    return (pos_cmnts, pos_src_tokens, pos_src_actions, pos_tgt_tokens, pos_tgt_actions), \
           (neg_cmnts, neg_src_tokens, neg_src_actions, neg_tgt_tokens, neg_tgt_actions)


def gen_editanch_batches(batch, w2i, cmnt_sent_len, diff_sent_len, ctx_sent_len, anchor_num):
    '''
    Batch Content:
    0,1. batch comment, batch neg_cmnt
    2,3. batch src_tokens, batch src_actions
    4,5. batch tgt_tokens, batch tgt_actions
    6,7. batch pos_edits, batch neg_edits
    '''

    cmnts, src_tokens, src_actions, tgt_tokens, tgt_actions, ea_truth = [], [], [], [], [], []

    for i in range(len(batch[0])):

        cmnt = batch[0][i]

        pos_edits = batch[6][i]
        neg_edits = batch[7][i]

        if len(pos_edits) > anchor_num:
            pos_edits = pos_edits[:anchor_num]

        if anchor_num - len(pos_edits) < 0:
            neg_sample_indices = []
        elif anchor_num - len(pos_edits) > len(neg_edits):
            neg_sample_indices = range(len(neg_edits))
        else:
            neg_sample_indices = random.sample(range(len(neg_edits)), anchor_num - len(pos_edits))

        for pos_edit in pos_edits:
            cmnts.append(cmnt)
            src_tokens.append([])
            src_actions.append([])
            tgt_tokens.append(pos_edit)
            tgt_actions.append([1] * len(pos_edit))
            ea_truth.append(1)

        for neg_idx in neg_sample_indices:
            cmnts.append(cmnt)
            src_tokens.append([])
            src_actions.append([])
            tgt_tokens.append(neg_edits[neg_idx])
            tgt_actions.append([1] * len(neg_edits[neg_idx]))
            ea_truth.append(0)

    return (cmnts, src_tokens, src_actions, tgt_tokens, tgt_actions), ea_truth


def find_cont_diffs(tokens, token_diff):
    # split the token_diff into the consecutive parts
    token_cont_list = []

    if len(token_diff) == 0:
        return token_cont_list

    if len(token_diff) == 1:
        token_cont_list.append(token_diff)
        return token_cont_list

    start_idx, cur_idx = 0, 1
    while cur_idx < len(token_diff):
        # if cur_idx == len(token_diff) - 1:
        #     token_list.append(list(range(start_idx, cur_idx + 1)))
        #     cur_idx += 1
        if token_diff[cur_idx] != token_diff[cur_idx - 1] + 1:
            token_cont_list.append(list(range(token_diff[start_idx], token_diff[cur_idx - 1] + 1)))
            start_idx = cur_idx
            cur_idx += 1
        else:
            cur_idx += 1

    # handle the last list
    token_cont_list.append(list(range(token_diff[start_idx], token_diff[cur_idx - 1] + 1)))

    return token_cont_list


def find_diff_context(tokens, token_diff, context_length=50):
    cont_difflist = find_cont_diffs(tokens, token_diff)
    diff_context = set()
    for cont_diff in cont_difflist:
        # avoid the case when only one markup or space included in the context
        if len(cont_diff) == 1 and len(tokens[cont_diff[0]]) <= 1:
            continue
        diff_context_cur = find_diff_context_int(tokens, cont_diff, context_length)
        diff_context.update(diff_context_cur)

    diff_context = [x for x in diff_context if x not in token_diff]
    return sorted(list(diff_context))


'''
Find context difference
The function requires token_diff is consecutive.
'''


def find_diff_context_int(tokens, token_diff, context_length):
    if len(token_diff) == 0:
        return []

    # if len(token_diff) == 1:
    #     diff_context = [token_diff[0]]
    # else:
    #     diff_context = range(token_diff[0], token_diff[-1] + 1)

    # diff_context = [x for x in diff_context if x not in token_diff]

    start_idx = token_diff[0]
    end_idx = token_diff[-1]

    context_start = start_idx - context_length
    context_start = context_start if context_start > 0 else 0
    context_end = end_idx + context_length + 1
    context_end = context_end if context_end < len(tokens) else len(tokens)

    diff_context = list(range(context_start, start_idx)) + list(range(end_idx + 1, context_end))
    diff_words = [tokens[i] for i in diff_context]

    # if len(diff_context) > context_length:
    #     diff_context = diff_context[:int(context_length/2)] + diff_context[-int(context_length/2):]        
    # else:
    #     remain_length = context_length - len(diff_context)
    return diff_context
