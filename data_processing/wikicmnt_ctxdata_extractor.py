#!/usr/bin/env python

import sys

sys.path.append('../')
from wiki_util import *


def extract_json(raw_file, ctx10_file, output_file, ctx_window=10, negative_edit_num=10):
    # load tokenized comment and neg_comments from ctx10_file
    print("Loading tokenized comment and neg_comments from ctx10_file")
    cmnt_dict = {}
    with open(ctx10_file, 'r', encoding='utf-8') as f:
        for idx, json_line in enumerate(tqdm(f)):
            article = json.loads(json_line.strip('\n'))
            rev_id = article["revision_id"]
            comment = article["comment"]
            neg_cmnts = article["neg_cmnts"]
            cmnt_dict[rev_id] = (comment, neg_cmnts)

    json_file = open(output_file, "w", buffering=1, encoding='utf-8')
    with open(raw_file, 'r', encoding='utf-8') as f:
        for idx, json_line in enumerate(tqdm(f)):

            article = json.loads(json_line.strip('\n'))

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
            rev_id = article["revision_id"]
            parent_id = article["parent_id"]
            timestamp = article["timestamp"]
            diff_url = article["diff_url"]
            page_title = article["page_title"]

            # comment = word_tokenize(article["comment"])
            # neg_comments = [word_tokenize(cmnt) for cmnt in article['neg_comments']]
            # lookup comment and neg_comments from dictionary
            comment, neg_cmnts = cmnt_dict[rev_id]

            src_text = article["src_text"]
            tgt_text = article["tgt_text"]

            src_sents = article["src_sents"]
            src_tokens = article["src_tokens"]

            tgt_sents = article["tgt_sents"]
            tgt_tokens = article["tgt_tokens"]

            src_token_diff = article["src_token_diff"]
            tgt_token_diff = article["tgt_token_diff"]

            # src_sents, src_tokens = tokenizeText(src_text)
            # tgt_sents, tgt_tokens = tokenizeText(tgt_text)

            # extract the offset of the changed tokens in both src and tgt
            # src_token_diff, tgt_token_diff = diffRevision(src_tokens, tgt_tokens)

            src_ctx_tokens, src_action = extContext(src_tokens, src_token_diff, ctx_window)
            tgt_ctx_tokens, tgt_action = extContext(tgt_tokens, tgt_token_diff, ctx_window)

            # src_sent_diff = findSentDiff(src_sents, src_tokens, src_token_diff)
            tgt_sent_diff = findSentDiff(tgt_sents, tgt_tokens, tgt_token_diff)

            # generate the positive edits
            pos_edits = [tgt_sents[i] for i in tgt_sent_diff]

            # generate negative edits
            neg_edits_idx = [i for i in range(len(tgt_sents)) if i not in tgt_sent_diff]
            if negative_edit_num > len(neg_edits_idx):
                sampled_neg_edits_idx = neg_edits_idx
            else:
                sampled_neg_edits_idx = random.sample(neg_edits_idx, negative_edit_num)
            neg_edits = [tgt_sents[i] for i in sampled_neg_edits_idx]

            if (len(src_token_diff) > 0 or len(tgt_token_diff) > 0):
                json_dict = {"revision_id": rev_id, "parent_id": parent_id, "timestamp": timestamp, \
                             "diff_url": diff_url, "page_title": page_title, \
                             "comment": comment, "src_token": src_ctx_tokens, "src_action": src_action, \
                             "tgt_token": tgt_ctx_tokens, "tgt_action": tgt_action, \
                             "neg_cmnts": neg_cmnts, "neg_edits": neg_edits, "pos_edits": pos_edits
                             }

                json_str = json.dumps(json_dict,
                                      indent=None, sort_keys=False,
                                      separators=(',', ': '), ensure_ascii=False)
                json_file.write(json_str + '\n')


def main():
    root_path = "../dataset/raw/"

    ctx_window = int(sys.argv[1])
    raw_file = root_path + "enwiki-sample_output_raw.json"
    ctx10_file = root_path + "wikicmnt_ctx10.json"
    output_file = root_path + "wikicmnt_ctx" + str(ctx_window) + ".json"

    extract_json(raw_file, ctx10_file, output_file, ctx_window=ctx_window)


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    time_elapsed = datetime.now() - start_time
