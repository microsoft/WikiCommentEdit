import difflib
import glob
import html
import random
import re

import spacy
from tqdm import tqdm

# initialize the spacy
nlp = spacy.load('en')

'''
Extract the contents by delimitors
'''
def extract_with_delims(content, start_delim, end_delim, start_idx):
    delims_start = content.find(start_delim, start_idx)
    if delims_start == -1:
        return '', start_idx

    delims_end = content.find(end_delim, start_idx)
    if delims_end == -1:
        return '', start_idx

    if delims_end <= delims_start:
        return '', start_idx

    delims_start += len(start_delim)

    return content[delims_start:delims_end], delims_end

'''
Extract the contents of revisions, e.g., revision_id, parent_id, user_name, comment, text
'''
def extract_data(revision_part):
    rev_id, next_idx = extract_with_delims(revision_part, "<id>", "</id>", 0)
    parent_id, next_idx = extract_with_delims(revision_part, "<parentid>", "</parentid>", next_idx)
    timestamp, next_idx = extract_with_delims(revision_part, "<timestamp>", "</timestamp>", next_idx)
    username, next_idx = extract_with_delims(revision_part, "<username>", "</username>", next_idx)
    userid, next_idx = extract_with_delims(revision_part, "<id>", "</id>", next_idx)
    # For annoymous user, the ip address will be used instead of the user name and id
    userip, next_idx = extract_with_delims(revision_part, "<ip>", "</ip>", next_idx)
    comment, next_idx = extract_with_delims(revision_part, "<comment>", "</comment>", next_idx)
    text, next_idx = extract_with_delims(revision_part, "<text xml:space=\"preserve\">", "</text>", next_idx)
    return (rev_id, parent_id, timestamp, username, userid, userip, comment, text)

'''
Extract the revision text buffer, which has the format "<revision> ... </revision>".
'''
def split_records(wiki_file, chunk_size=150 * 1024):
    
    text_buffer = ""    
    cur_index = 0

    while True:
        chunk = wiki_file.read(chunk_size)

        if chunk:
            text_buffer += chunk

        cur_index = 0
        REVISION_START = "<revision>"
        REVISION_END = "</revision>"
        PAGE_START = "<page>"
        PAGE_TITLE_START = "<title>"
        PAGE_TITLE_END = "</title>"
        while True:
            page_start_index = text_buffer.find(PAGE_START, cur_index)
            if page_start_index != -1:
                # update the current page title/ID
                page_title, _ = extract_with_delims(text_buffer, PAGE_TITLE_START, PAGE_TITLE_END, 0)
                if not page_title:
                    # no complete page title
                    break
                    #logging.debug("Error: page information is cut. FIX THIS ISSUE!!!")

            # find the revision start position
            revision_start_index = text_buffer.find(REVISION_START, cur_index)

            # No revision in the buffer, continue loading data
            if revision_start_index == -1:
                break

            # find the revision end position
            revision_end_index = text_buffer.find(REVISION_END, revision_start_index)

            # No complete page in buffer
            if revision_end_index == -1:
                break

            yield page_title, text_buffer[revision_start_index:revision_end_index + len(REVISION_END)]

            cur_index = revision_end_index + len(REVISION_END)

        # No more data
        if chunk == "":
            break

        if cur_index == -1:
            text_buffer = ""
        else:
            text_buffer = text_buffer[cur_index:]

def sampleNext(sample_ratio):
    return random.random() < sample_ratio

def cleanCmntText(comment):
    filter_words = []
    comment = comment.replace("(edited with [[User:ProveIt_GT|ProveIt]]", "")
    #comment = re.sub("(edited with \[\[User\:ProveIt_GT\|ProveIt\]\]", "", comment)
    return comment
    

def checkComment(comment, comment_tokens, min_comment_length):

    if len(comment_tokens) < min_comment_length:
        return False

    filter_words = ["[[Project:AWB|AWB]]", "[[Project:AutoWikiBrowser|AWB]]", "Undid revision"]
    if any(word in comment for word in filter_words):
        return False

    return True

'''
clean the wiki text
E.g. "[[link name]] a&quot; bds&quot; ''markup''" to "link name a bds markup"
'''
def cleanWikiText(wiki_text):

    # replace link: [[link_name]] and quotes
    wiki_text = re.sub("\[\[", "", wiki_text)
    wiki_text = re.sub("\]\]", "", wiki_text)
    wiki_text = re.sub("''", "", wiki_text)

    # replace '<', '>', '&'    
    # wiki_text = re.sub("&quot;", "", wiki_text)
    # wiki_text = re.sub("&lt;", "<", wiki_text)
    # wiki_text = re.sub("&gt;", ">", wiki_text)
    # wiki_text = re.sub("&amp;", "&", wiki_text)

    # use html unescape to decode the html special characters
    wiki_text = html.unescape(wiki_text)

    return wiki_text

def tokenizeText(text):
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    sent_tokens = []
    tokens = []
    for sent in sentences:
        sent_doc = nlp(sent)
        token_one_sent = [token.string.strip() for token in sent_doc]
        sent_tokens.append(token_one_sent)
        tokens += token_one_sent
    return sent_tokens, tokens

# return the difference indices starting at 0.
def diffRevision(parent_sent_list, sent_list):

    # make diff
    origin_start_idx, origin_start_idx = -1, -1
    target_start_idx, target_end_idx = -1, -1
    origin_diff_list, target_diff_list = [], []
    for line in difflib.context_diff(parent_sent_list, sent_list, 'origin', 'target'):
        
        #print(line)
        # parse the origin diff line range: e.g., --- 56,62 ----
        if line.startswith("*** ") and line.endswith(" ****\n"):
            target_start_idx, target_end_idx = -1, -1 # reset the target indices
            range_line = line[4:-6]
            if ',' not in range_line:
                origin_start_idx = int(range_line)
                origin_end_idx = origin_start_idx
            else:
                origin_start_idx, origin_end_idx = [int(i) for i in range_line.split(',')]
            origin_sent_idx = origin_start_idx
            continue

        # parse the diff line range: e.g., --- 56,62 ----
        if line.startswith("--- ") and line.endswith(" ----\n"):
            origin_start_idx, origin_end_idx = -1, -1 # reset the origin indices
            range_line = line[4:-6]
            if ',' not in range_line:
                target_start_idx = int(range_line)
                target_end_idx = target_start_idx
            else:
                target_start_idx, target_end_idx = [int(i) for i in range_line.split(',')]
            target_sent_idx = target_start_idx
            continue

        if origin_start_idx >= 0:
            if len(line.strip('\n')) == 0:
                continue
            elif line.startswith('-') or line.startswith('!'):
                origin_diff_list.append(origin_sent_idx - 1) # adding the index starting at 0
                origin_sent_idx += 1
            else:
                origin_sent_idx += 1

        if target_start_idx >= 0:
            if len(line.strip('\n')) == 0:
                continue
            elif line.startswith('+') or line.startswith('!'):
                target_diff_list.append(target_sent_idx - 1) # adding the index starting at 0
                target_sent_idx += 1
            else:
                target_sent_idx += 1

    #print("Extracted Diff:", diff_sent_list)
    return origin_diff_list, target_diff_list

def findSentDiff(sents, tokens, token_diff):
    diff_idx, sent_idx, token_offset = 0, 0, 0
    diff_sents = set()
    if len(token_diff) == 0 or len(sents) == 0:
        return list(diff_sents)

    token_offset = len(sents[0])
    while diff_idx < len(token_diff) and sent_idx < len(sents):
        if token_offset >= token_diff[diff_idx]:
            cur_token = tokens[diff_idx]
            # avoid the case that one more sentence added because only one markup included.
            if len(cur_token) > 1:
                diff_sents.add(sent_idx)
            diff_idx += 1
        else:
            sent_idx += 1
            token_offset += len(sents[sent_idx])
        
    return list(diff_sents)


def extContext(tokens, token_diff, ctx_window):
    '''
    Extend the context into the token difference

    :param token_diff: a list of tokens which represents the difference between previous version and current version.
           ctx_window: the size of context before and after the edits
    :return:

    For example: token_diff = [2, 3, 4, 11, 12, 13, 14, 16] and ctx_window = 2
    The function will return ctx_tokens = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
                             action =     [0, 0, 1, 1, 1, 0, 0, 0,  0,  1,  1,  1,  1,  0,  1,  0,  0]
    '''


    ctx_set = set(token_diff)

    for idx in token_diff:
        for i in range(idx - ctx_window, idx + ctx_window + 1):
            if i < 0 or i >= len(tokens):
                continue
            ctx_set.add(i)

    action = []
    ctx_token = []
    ctx_token_idx = sorted(list(ctx_set))
    diff_set = set(token_diff)

    for i in ctx_token_idx:
        ctx_token.append(tokens[i])
        if i in diff_set:
            action.append(1)
        else:
            action.append(0)

    return ctx_token, action



# def findDiffSents(sents, diff_list):
#     token_idx = 0
#     start_sent_idx, end_sent_idx = -1, -1
#     for i, sent in enumerate(sents):
#         token_idx += len(sent)
#         if start_sent_idx < 0 and diff_list[0] < token_idx:
#             start_sent_idx = i
#         if end_sent_idx < 0 and diff_list[-1] < token_idx:
#             end_sent_idx = i
#         if start_sent_idx >= 0 and end_sent_idx >= 0:
#             break
#     return start_sent_idx, end_sent_idx

def extDiffSents(sents, start, end, max_tokens=200, max_sents_add=2):
    token_size = 0
    for i in range(start, end + 1):
        token_size += len(sents[i])

    context_sents = sents[start:end + 1]
    sents_head_added, sents_tail_added = 0, 0
    while token_size <= max_tokens and len(context_sents) < len(sents)\
        and sents_head_added < max_sents_add and sents_tail_added < max_sents_add:
        if start > 0:
            start -= 1
            insert_sent = sents[start]
            context_sents.insert(0, insert_sent)
            token_size += len(insert_sent)
            sents_head_added += 1

        if end < len(sents) - 1:
            end += 1
            insert_sent = sents[end]
            context_sents.append(insert_sent)
            token_size += len(insert_sent)
            sents_tail_added += 1

    diff_offset = sum([len(sents[i]) for i in range(start)])
    return context_sents, diff_offset

def mapDiffContext(sents, start_sent, end_sent):
    
    start_idx = -1
    end_idx = -1
    start_sent_str = " ".join(start_sent)
    end_sent_str = " ".join(end_sent)
    for i, sent in enumerate(sents):
        sent_str = " ".join(sent)
        if start_idx < 0 and start_sent_str in sent_str:
            start_idx = i
        
        if end_sent_str in sent_str:
            end_idx = i
            break
    
    # if start_idx == -1:
    #     start_idx = 0

    # if end_idx == -1:
    #     context_sents = sents[start_idx:]
    # else:
    #     context_sents = sents[start_idx:end_idx + 1]
    if start_idx == -1 or end_idx == -1:
        return None, None

    diff_offset = sum([len(sents[i]) for i in range(start_idx)])
    return sents[start_idx:end_idx + 1], diff_offset


# def extDiffContextInt(target_context_sents, target_sents, target_start, target_end, token_size, max_tokens=200, max_sents_add=2):
#     sents_head_added, sents_tail_added = 0, 0
    
#     return target_context_sents

def calTokenSize(sents):
    return sum([len(sent) for sent in sents])

def isSameSent(origin_sent, target_sent):

    origin_size = len(origin_sent)
    target_size = len(target_sent)

    if origin_size != target_size:
        return False

    for i in range(origin_size):
        if origin_sent[i] != target_sent[i]:
            return False
    return True

def stripContext(origin_sents, target_sents, max_token_size=200):

    start_match = True
    end_match = True
    origin_token_size = calTokenSize(origin_sents)
    target_token_size = calTokenSize(target_sents)
    diff_offset = 0
    while origin_token_size > max_token_size or target_token_size > max_token_size:

        if len(origin_sents) == 0 or len(target_sents) == 0:
                break

        if start_match and isSameSent(origin_sents[0], target_sents[0]):
            # remove the sentence from both origin and target
            sent_size = len(origin_sents[0])
            origin_sents = origin_sents[1:]
            target_sents = target_sents[1:]
            diff_offset += sent_size
            origin_token_size -= sent_size
            target_token_size -= sent_size
        else:
            start_match = False

        if len(origin_sents) == 0 or len(target_sents) == 0:
            break

        if end_match and isSameSent(origin_sents[-1], target_sents[-1]):
            sent_size = len(origin_sents[-1])
            origin_sents = origin_sents[:-1]
            target_sents = target_sents[:-1]
            origin_token_size -= sent_size
            target_token_size -= sent_size
        else:
            end_match = False

        if not start_match and not end_match:
            break

    if origin_token_size > max_token_size or target_token_size > max_token_size:
        origin_sents, target_sents = None, None

    return origin_sents, target_sents, diff_offset


def extractDiffContext(origin_sents, target_sents, origin_diff, target_diff, max_tokens=200):

    origin_context, target_context, diff_offset = stripContext(origin_sents, target_sents)
    if origin_context != None and target_context != None:
        origin_diff = [i - diff_offset for i in origin_diff]
        target_diff = [i - diff_offset for i in target_diff]

        # fix the issue in the case that the appended dot belonging to current sentence is marked as the previous sentence. See example:
        # + .
        # + Warren
        # + ,
        # ...
        # + -
        # + syndicalists
        #   .
        if len(target_diff) > 0 and target_diff[0] == -1:
            target_diff = target_diff[1:]
    
    return origin_context, target_context, origin_diff, target_diff


def filterRevision(comment, diff_list, max_sent_length):
    filter_pattern = "^.*\s*\W*(revert|undo|undid)(.*)$"
    filter_regex = re.compile(filter_pattern, re.IGNORECASE)
    
    if len(diff_list) == 0 or len(diff_list) > max_sent_length:
        return True
    comment = comment.strip()
    if not comment.startswith('/*'):
        return True
    elif comment.startswith('/*') and comment.endswith('*/'):
        return True
    filter_match = filter_regex.match(comment)
    if filter_match:
        return True
    
    return False
    #return filter_match

comment_pattern = "^/\*(.+)\*/(.*)$"
comment_regex = re.compile(comment_pattern)
def extractSectionTitle(comment):
    sect_title, sect_cmnt = '', comment
    comment_match = comment_regex.match(comment)
    if comment_match:
        sect_title = comment_match.group(1).strip()
        sect_cmnt = html.unescape(comment_match.group(2).strip()).strip()
    return sect_title, sect_cmnt

def extractSectionText(text, sect_title):
    sect_content = ''
    text_match = re.search('(=+)\s*' + sect_title + '\s*(=+)', text)

    if text_match:
        sect_sign = text_match.group(1)
        sect_sign_end = text_match.group(2)

        if sect_sign != sect_sign_end:
            print("ALERT: Section Data Corrupted!! Skip!!")
            return sect_content
            
        sect_start = text_match.regs[2][1]
        remain_sect = text[sect_start:].strip().strip('\n')

        # TODO: Fix the bug of comparing the ===Section Title=== after ==Section==
        next_sect_match = re.search(sect_sign + '.*' + sect_sign, remain_sect)
        if next_sect_match:
            sect_end = next_sect_match.regs[0][0]
            sect_content = remain_sect[:sect_end].strip().strip('\n')
        else:
            sect_content = remain_sect

    return sect_content


'''
Merge the sample outputs of all the dump files
'''
def mergeOutputs(output_path):

    # merge sample results
    print("Merging the sampled outputs from each files ...")
    sample_list = glob.glob(output_path + '*.json')
    sample_file = open(output_path + 'wikicmnt.json', "w", encoding='utf-8')
    for fn in tqdm(sample_list):
        with open(fn, 'r', encoding='utf-8') as fi:
            sample_file.write(fi.read())