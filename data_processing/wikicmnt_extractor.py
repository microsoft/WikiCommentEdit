#!/usr/bin/env python
import argparse
import sys

sys.path.append('../')
from wiki_util import *
from wikicmnt_extractor_st import randSampleRev

# arguments
parser = argparse.ArgumentParser(description='Wiki Extractor')
parser.add_argument('--data_path', type=str, default="../data/raw/", help='the data directory')
parser.add_argument('--output_path', type=str, default="../data/processed/", help='the sample output path')
parser.add_argument('--min_page_tokens', type=int, default=50,
                    help='the minimum size of tokens in page to extract [default: 100]')
parser.add_argument('--max_page_tokens', type=int, default=2000,
                    help='the maximum size of tokens in page to extract [default: 1000]')
parser.add_argument('--min_cmnt_length', type=int, default=8,
                    help='the minimum words contained in the comments [default: 8]')
parser.add_argument('--ctx_window', type=int, default=5, help='the window size of context [default: 5]')
parser.add_argument('--sample_ratio', type=float, default=0.01, help='the ratio of sampling [default: 0.001]')
parser.add_argument('--threads', type=int, default=3, help='the number of sampling threads [default: 5]')
parser.add_argument('--single_thread', type=int, default=0,
                    help='the dump file index when using single thread mode. If the index equals to zero, '
                         'use multi-thread mode to proprocess all the dump files [default: 0]')
parser.add_argument('--user_stat', type=bool, default=False, help='whether to do user statistics')
parser.add_argument('--merge_only', action='store_true', default=False, help='merge the results only')
parser.add_argument('--neg_cmnt_num', type=int, default=10,
                    help='how many negative comments sampled for ranking problem [default: 10]')
parser.add_argument('--count_revision_only', type=bool, default=False,
                    help='count the revision only without sampling anything [default: False]')
args = parser.parse_args()

# create sample output folder if it doesn't exist
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# logging configuration
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)s) %(message)s',
                    )

'''
WikiSampleTask class contains a list of dump files to be sampled.
The assign_task function will be called by workers to grab a task.
'''


class WikiSampleTask(object):
    def __init__(self, dump_list):
        self.lock = threading.Lock()
        self.dump_list = dump_list
        self.total_num = len(dump_list)

    def assign_task(self):
        logging.debug('Assign tasks ... Waiting for lock')
        self.lock.acquire()
        dump_name = None
        cur_progress = None
        try:
            # logging.debug('Acquired lock')
            if len(self.dump_list) > 0:
                dump_name = self.dump_list.pop(0)
                cur_progress = self.total_num - len(self.dump_list)
        finally:
            self.lock.release()
        return dump_name, cur_progress, self.total_num


'''
worker is main function for each thread.
'''


def worker(work_id, tasks):
    logging.debug('Starting.')
    output_file = args.data_path + 'sample/enwiki_sample_' + str(work_id) + '.json'
    # grab one task from task_list
    while 1:
        dump_file, cur_progress, total_num = tasks.assign_task()
        if not dump_file:
            break
        logging.debug('Assigned task (' + str(cur_progress) + '/' + str(total_num) + '): ' + str(dump_file))
        # start to sample the dump file
        output_file = args.output_path + 'enwiki-sample-' + os.path.basename(dump_file)[27:-4] + '.json'
        randSampleRev(work_id, dump_file, output_file, args.sample_ratio, args.min_cmnt_length, args.ctx_window,
                      args.neg_cmnt_num)
    logging.debug('Exiting.')


def initLogger(file_idx):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # or whatever

    handler = logging.FileHandler('extractor.txt', 'a', 'utf-8')  # or whatever
    # handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter = logging.Formatter('%(asctime)s' + '[' + str(file_idx) + '] - %(message)s')  # or whatever
    logger.addHandler(handler)
    return logger


def main():
    # single file mode
    if args.single_thread:

        logger = initLogger(args.single_thread)

        data_path = args.data_path
        output_path = args.output_path

        dump_list = sorted(glob.glob(data_path + "*.bz2"))
        print(dump_list)
        dump_file = dump_list[args.single_thread - 1]
        output_file = output_path + 'enwiki-sample-' + os.path.basename(dump_file)[27:-4] + '.json'

        # create sample output folder if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print("start to preprocess dump file:", str(dump_file))
        logger.info("[" + str(args.single_thread) + "] Start to sample dump file " + dump_file)
        randSampleRev(args.single_thread, dump_file, output_file, args.sample_ratio, args.min_cmnt_length,
                      args.ctx_window, args.neg_cmnt_num)
        return

    if not args.merge_only:
        dump_list = glob.glob(args.data_path + "*.bz2")
        # # testing
        # dump_list = dump_list[:5]
        dump_num = len(dump_list)
        logging.debug("Samping revisions from " + str(dump_num) + " dump files")

        task = WikiSampleTask(dump_list)
        threads = []
        for i in range(args.threads):
            t = threading.Thread(target=worker, args=(i, task))
            threads.append(t)
            t.start()

        logging.debug('Waiting for worker threads')
        main_thread = threading.currentThread()
        for t in threading.enumerate():
            if t is not main_thread:
                t.join()

        logging.debug('Merging the sample outputs from each dump file')

    # merge the result
    mergeOutputs(args.output_path)


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    time_elapsed = datetime.now() - start_time
    logging.debug('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
