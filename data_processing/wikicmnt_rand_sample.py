#!/usr/bin/env python

import os
import sys
import argparse
import bz2    
import glob
from datetime import datetime
import random
import difflib
import nltk
import re
import threading
import logging
import operator
import html
import spacy
import en_core_web_sm
import json
from wiki_util import *
import random
from random import shuffle

def cal_total_line(input_file):
    total_line = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for json_line in tqdm(f):
            total_line += 1
    return total_line

def rand_sample(input_file, output_file, sample_num, total_num):
    sample_indices = random.sample(range(1, total_num), sample_num)
    sample_file = open(output_file, "w", encoding='utf-8')
    sample_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, json_line in enumerate(tqdm(f)):
            # if idx in sample_indices:
            #     sample_list.append(json_line)

            j = json.loads(json_line)
            tgt_sent_len = len(j['tgt_sents'])
            tgt_diff_sent_size = len(j['tgt_sent_diff'])
            if tgt_sent_len - tgt_diff_sent_size >= 10:
                sample_list.append(json_line)

            # sample_list.append(json_line)
            # if idx == sample_num - 1:
            #     break

    print("Start to shuffle sample list ...")
    shuffle(sample_list)

    print("Start to write output ...")
    for line in tqdm(sample_list):
        sample_file.write(line)

def main():

    data_folder = "../data/"
    
    #input_file = data_folder + "wikicmnt.json"
    input_file = data_folder + "wiki_comment_orig.json"
    output_file = data_folder + "wiki_comment.json"
    sample_num = 260000
    #sample_num = 500000
    #total_num = cal_total_line(input_file)
    total_num = 786886
    print("total line:", total_num)
    rand_sample(input_file, output_file, sample_num, total_num)

if __name__ == '__main__':
    start_time = datetime.now()
    main()
    time_elapsed = datetime.now() - start_time
    logging.debug('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))