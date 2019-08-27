#!/usr/bin/env python
import json
import sys
sys.path.append('../')
from wiki_util import *
from process_data import word_tokenize

def extract_json(input_file, output_file, ctx_window=10, negative_edit_num=10):

    json_file = open(output_file, "w", buffering=1, encoding='utf-8')
    with open(input_file, 'r', encoding='utf-8') as f:
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

            # # tokenize comment
            # cmnt_tokens = word_tokenize(article['comment'])
            # cmnt_list.append(cmnt_tokens)
            #
            # # negative comments
            # neg_cmnts = [word_tokenize(cmnt) for cmnt in article['neg_comments']]
            # neg_cmnts_list.append(neg_cmnts)

            comment = word_tokenize(article["comment"])
            neg_comments = [word_tokenize(cmnt) for cmnt in article['neg_comments']]

            src_text = article["src_text"]
            tgt_text = article["tgt_text"]

            src_sents = article["src_sents"]
            src_tokens = article["src_tokens"]

            tgt_sents = article["tgt_sents"]
            tgt_tokens = article["tgt_tokens"]

            src_token_diff = article["src_token_diff"]
            tgt_token_diff = article["tgt_token_diff"]

            #src_sents, src_tokens = tokenizeText(src_text)
            #tgt_sents, tgt_tokens = tokenizeText(tgt_text)

            # extract the offset of the changed tokens in both src and tgt
            #src_token_diff, tgt_token_diff = diffRevision(src_tokens, tgt_tokens)

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
                             "neg_cmnts": neg_comments, "neg_edits": neg_edits, "pos_edits": pos_edits
                             }

                json_str = json.dumps(json_dict,
                                      indent=None, sort_keys=False,
                                      separators=(',', ': '), ensure_ascii=False)
                json_file.write(json_str + '\n')

def main():

    root_path = "../dataset/raw/"
    data_path = root_path + "split_files/"
    output_path = root_path + "output/"

    file_idx = int(sys.argv[1])

    dump_list = sorted(glob.glob(data_path + "*.json"))
    dump_file = dump_list[file_idx - 1]
    file_name = os.path.basename(dump_file)
    output_file = output_path + file_name[:-5] + '_output.json'

    # original way
    # input_file = root_path + "enwiki-sample_output_raw.json"
    # output_file = root_path + "wikicmnt.json"

    extract_json(dump_file, output_file, ctx_window=10)

if __name__ == '__main__':
    start_time = datetime.now()
    main()
    time_elapsed = datetime.now() - start_time