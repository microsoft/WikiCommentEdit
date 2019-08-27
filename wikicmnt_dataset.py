import itertools
import json
import os
import os.path
from collections import Counter

import numpy as np
# from torchtext import data
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
from torch.autograd import Variable
from tqdm import tqdm

from wiki_util import tokenizeText


class Wiki_DataSet:

    def __init__(self, args):
        """Create an Wiki dataset instance. """

        # self.word_embed_file = self.data_folder + 'embedding/wiki.ar.vec'
        # word_embed_file = data_folder + "embedding/Wiki-CBOW"
        self.data_dir = args.data_path

        self.data_file = self.data_dir + "wikicmnt.json"

        self.vocab_file = self.data_dir + 'vocabulary.json'
        self.train_df_file = self.data_dir + 'train_df.pkl'
        self.val_df_file = self.data_dir + 'val_df.pkl'
        self.test_df_file = self.data_dir + 'test_df.pkl'
        self.tf_file = self.data_dir + 'term_freq.json'
        self.weight_file = self.data_dir + 'train_weights.json'

        self.glove_dir = args.glove_path
        self.glove_vec_size = args.word_embd_size
        self.lemmatizer = WordNetLemmatizer()
        self.class_num = -1
        self.rank_num = args.rank_num
        self.anchor_num = args.anchor_num
        self.max_ctx_length = int(args.max_ctx_length / 2)
        pass

    '''
    Extract the data from raw json file 
    '''

    def extract_data_from_json(self):
        '''
        Parse the json file and return the
        :return:
        '''
        cmnt_list, neg_cmnts_list = [], []
        src_token_list, src_action_list = [], []
        tgt_token_list, tgt_action_list = [], []
        pos_edits_list, neg_edits_list = [], []
        diff_url_list = []

        word_counter, lower_word_counter = Counter(), Counter()

        print("Sample file:", self.data_file)

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for idx, json_line in enumerate(tqdm(f)):

                # if idx % 100 == 0:
                #     print("== processed ", idx)

                article = json.loads(json_line.strip('\n'))
                # print(article['diff_url'])

                '''
                Json file format:
                =================
                revision_id: The revision ID
                parent_id: The parent revision ID
                timestamp: Timestamp
                diff_url: The wikipedia link to show the difference between previous and current version.
                page_title: The title of page.
                comment: Revision comment.
                src_token: List of tokens in before-editing version
                src_action: Action flags for each token in before-editing version. E.g., 0 represents no action; -1 represents removed token.
                tgt_token: List of tokens in after-editing version
                tgt_action: Action flags for each token in after-editing version. E.g., 0 represents no action; 1 represents added token.
                neg_cmnts: Negative samples of user comments in the same page.
                pos_edits: Edit sentences for comments.
                neg_edits: Negative edit sentences for comments.
                '''
                try:

                    # comment
                    comment = article['comment']
                    _, cmnt_tokens = tokenizeText(comment)
                    # cmnt_tokens = article['comment']
                    cmnt_list.append(cmnt_tokens)

                    # negative comments
                    # neg_cmnts = article['neg_cmnts']
                    neg_cmnts = []
                    for neg_cmnt in article['neg_cmnts']:
                        _, tokens = tokenizeText(comment)
                        neg_cmnts.append(tokens)
                    neg_cmnts_list.append(neg_cmnts)

                    # source tokens and actions
                    src_token = article['src_token']
                    src_token_list.append(src_token)
                    src_action_list.append(article['src_action'])

                    # target tokens and actions
                    tgt_token = article['tgt_token']
                    tgt_token_list.append(tgt_token)
                    tgt_action_list.append(article['tgt_action'])

                    # positive and negative edits
                    pos_edits = article['pos_edits']
                    pos_edits_list.append(pos_edits)

                    neg_edits = article['neg_edits']
                    neg_edits_list.append(neg_edits)

                    # diff url
                    diff_url = article['diff_url']
                    diff_url_list.append(diff_url)

                    # for counters
                    for word in cmnt_tokens + src_token + tgt_token + \
                                list(itertools.chain.from_iterable(neg_cmnts + pos_edits + neg_edits)):
                        word_counter[word] += 1
                        lower_word_counter[word.lower()] += 1

                except:
                    # ignore the index error
                    print("ERROR: Index Error", article['revision_id'])
                    continue

                # if idx >= 100:
                #     break

        return cmnt_list, neg_cmnts_list, src_token_list, src_action_list, tgt_token_list, tgt_action_list, \
               pos_edits_list, neg_edits_list, word_counter, lower_word_counter, diff_url_list

    '''
    Create dataset objects for wiki revision data.
    Arguments:
        args: arguments
        val_ratio: The ratio that will be used to get split validation dataset.
        shuffle: Whether to shuffle the data before split.
    '''

    def load_data(self, train_ratio, val_ratio):

        print("loading wiki data ...")

        # check the existence of data files
        if os.path.isfile(self.train_df_file) and os.path.isfile(self.test_df_file) and os.path.isfile(self.vocab_file):
            print("dataframe file exists:", self.train_df_file)
            train_df = pd.read_pickle(self.train_df_file)
            val_df = pd.read_pickle(self.val_df_file)
            test_df = pd.read_pickle(self.test_df_file)
            vocab_json = json.load(open(self.vocab_file))
        else:
            cmnts, neg_cmnts, src_tokens, src_actions, \
            tgt_tokens, tgt_actions, pos_edits, neg_edits, \
            word_counter, lower_word_counter, diff_url = self.extract_data_from_json()

            word2vec_dict = self.get_word2vec(word_counter)
            lower_word2vec_dict = self.get_word2vec(lower_word_counter)

            df = pd.DataFrame(
                {
                    'cmnt_words': cmnts, "neg_cmnts": neg_cmnts,
                    'src_tokens': src_tokens, "src_actions": src_actions,
                    'tgt_tokens': tgt_tokens, "tgt_actions": tgt_actions,
                    'pos_edits': pos_edits, 'neg_edits': neg_edits, 'diff_url': diff_url
                }
            )

            total_size = len(df)
            self.train_size = int(total_size * train_ratio)
            val_size = int(total_size * val_ratio)

            # test_ratio = 0.2
            # train_df, test_df = train_test_split(df,
            #                                     test_size=test_ratio, random_state=967898)
            train_df = df[:self.train_size]
            val_df = df[self.train_size:self.train_size + val_size]
            test_df = df[self.train_size + val_size:]

            print("saving data into pickle ...")
            train_df.to_pickle(self.data_dir + 'train_df.pkl')
            test_df.to_pickle(self.data_dir + 'test_df.pkl')
            val_df.to_pickle(self.data_dir + 'val_df.pkl')

            w2i = {w: i for i, w in enumerate(word_counter.keys(), 3)}
            NULL = "-NULL-"
            UNK = "-UNK-"
            ENT = "-ENT-"
            w2i[NULL] = 0
            w2i[UNK] = 1
            w2i[ENT] = 2

            # save word2vec dictionary
            vocab_json = {'word2idx': w2i, 'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}
            json.dump(vocab_json, open(self.vocab_file, 'w', encoding='utf-8'))

        return train_df, val_df, test_df, vocab_json

    '''
    batch padding
    '''

    def pad_batch(self, mini_batch, padding_size):
        mini_batch_size = len(mini_batch)
        # mean_sent_len = int(np.mean([len(x) for x in mini_batch]))
        main_matrix = np.zeros((mini_batch_size, padding_size), dtype=np.long)
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                try:
                    main_matrix[i, j] = mini_batch[i][j]
                except IndexError:
                    pass

        # transfer the tensor to LongTensor for some compatibility issues
        return Variable(torch.from_numpy(main_matrix).transpose(0, 1).type(torch.LongTensor))

    '''
    Generate minibatches from data frame
    '''

    def iterate_minibatches(self, df, batch_size, cur_epoch=-1, n_epoch=-1):

        cmnt_words = df.cmnt_words.tolist()
        neg_cmnts = df.neg_cmnts.tolist()
        src_tokens = df.src_tokens.tolist()
        src_actions = df.src_actions.tolist()
        tgt_tokens = df.tgt_tokens.tolist()
        tgt_actions = df.tgt_actions.tolist()
        pos_edits = df.pos_edits.tolist()
        neg_edits = df.neg_edits.tolist()
        diff_urls = df.diff_url.tolist()

        indices = np.arange(len(cmnt_words))
        np.random.shuffle(indices)

        cmnt_words = [cmnt_words[i] for i in indices]
        neg_cmnts = [neg_cmnts[i] for i in indices]
        src_tokens = [src_tokens[i] for i in indices]
        src_actions = [src_actions[i] for i in indices]
        tgt_tokens = [tgt_tokens[i] for i in indices]
        tgt_actions = [tgt_actions[i] for i in indices]
        pos_edits = [pos_edits[i] for i in indices]
        neg_edits = [neg_edits[i] for i in indices]
        diff_urls = [diff_urls[i] for i in indices]

        for start_idx in range(0, len(cmnt_words) - batch_size + 1, batch_size):

            # initialize batch variables
            batch_cmnt, batch_neg_cmnt, batch_src_tokens, batch_src_actions, \
            batch_tgt_tokens, batch_tgt_actions, batch_pos_edits, batch_neg_edits, batch_diffurls = [], [], [], [], [], [], [], [], []
            for i in range(start_idx, start_idx + batch_size):
                batch_cmnt.append(cmnt_words[i])
                batch_neg_cmnt.append(neg_cmnts[i])
                batch_src_tokens.append(src_tokens[i])
                batch_src_actions.append(src_actions[i])
                batch_tgt_tokens.append(tgt_tokens[i])
                batch_tgt_actions.append(tgt_actions[i])
                batch_pos_edits.append(pos_edits[i])
                batch_neg_edits.append(neg_edits[i])
                batch_diffurls.append(diff_urls[i])

            yield batch_cmnt, batch_neg_cmnt, batch_src_tokens, batch_src_actions, batch_tgt_tokens, batch_tgt_actions, batch_pos_edits, batch_neg_edits, batch_diffurls

    def get_datafile(self):
        return self.data_file

    def get_tffile(self):
        return self.tf_file

    def get_weight_file(self):
        return self.weight_file

    def get_train_size(self):
        return self.train_size

    def get_word2vec(self, word_counter):
        glove_path = os.path.join(self.glove_dir, "glove.6B.{}d.txt".format(self.glove_vec_size))
        print('----glove_path', glove_path)
        sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
        total = sizes['6B']
        word2vec_dict = {}
        with open(glove_path, 'r', encoding='utf-8') as fh:
            for line in tqdm(fh, total=total):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in word_counter:
                    word2vec_dict[word] = vector
                elif word.capitalize() in word_counter:
                    word2vec_dict[word.capitalize()] = vector
                elif word.lower() in word_counter:
                    word2vec_dict[word.lower()] = vector
                elif word.upper() in word_counter:
                    word2vec_dict[word.upper()] = vector

        print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter),
                                                                            glove_path))
        return word2vec_dict
