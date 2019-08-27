Wikipedia Revision Data Extractor
======

This is a python implementation of the data extractor tool set for wikipedia revision data, as described in our EMNLP 2019 paper:

**Modeling the Relationship between User Comments and Edits in Document Revision**, Xuchao Zhang, Dheeraj Rajagopal, Michael Gamon, Sujay Kumar Jauhar and ChangTien Lu, 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), Hongkong, China, Nov 3-7, 2019.


We provide three tools to extract and preprocess the wikipedia revision history data from scratch:
- download entire enwiki revision dumps from wikipedia
- extract the revision data for comment modeling task from wiki dump files
- extract the summeration task dataset from wiki dump

Note: The collected wikipedia revision data can be used as the input for the proposed models in our EMNLP paper or used individually for other tasks.

## Requirements
- python 3.6
- [tqdm](https://github.com/noamraph/tqdm)
- [numpy v1.13+](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [spacy v2.0.11](https://spacy.io/)
- and some basic packages.


## Usage
### Download Wiki dump files
First, choose a dump such as ```https://dumps.wikimedia.org/enwiki/20190801``` (the latest version of wiki dump when our code is released). You can check all the information related to dump files from this page such as the list of files generated in this dump. Then download a machine-readable dump status file ```dumpstatus.json``` from the Wikipedia dump page. Next copy the status file into the default data path, e.g., ```./data/```.

###### Important Note:
* Check the dump files must contain the complete page edit history: ```All pages with complete page edit history (.bz2)```. The edit history is sometimes skipped by some specific versions.
* Always choose the recent dumps since Wikipedia cleans the old dumps and make the old one deprecated.


Finally, run our wiki dump download script to download dump files as follows:
```
python wiki_dump_download.py --data_path="./data/raw/" --compress_type="bz2" --threads=3
```
You need to specify the data path and compress type (by default choose bz2 ). Since the download process will be extremely slow, you can use multiple threads to download the dump files. However, Wikipedia only allows three http connections to download simultaneously for each IP address. The maximum threads I recommend is three unless you can assign different IP address for each threads.

At the beginning of download process, all the files are listed with unique Ids as follows:

```
All files to download ...
1 enwiki-20190801-pages-meta-history1.xml-p1043p2036.bz2
2 enwiki-20190801-pages-meta-history1.xml-p10675p11499.bz2
3 enwiki-20190801-pages-meta-history1.xml-p10p1042.bz2
4 enwiki-20190801-pages-meta-history1.xml-p11500p12310.bz2
...
...
648 enwiki-20190801-pages-meta-history9.xml-p2326206p2336422.bz2
```
Usually, the entire download process takes one to two days to be done. You can download each file individually by specifying the ```--start``` and ```--end``` parameters. You can also use ```--verify``` parameter to verify the completeness of your dump files.


### Revision Data Preprocessing

For preprocessing the revision data, we provide both single-thread and multi-thread versions.

To preprocess a single dump file, we specify the file index of the dump file by the parameter ```--single_thread``` as follows:
```
python3 wikicmnt_extractor.py --single_thread=5
```
Here the number 5 in the example means the 5th dump file in the data folder of dump files.

To preprocess multiple dump files,
```
python3 wikicmnt_extractor.py --threads=10
```

You need to specify some common parameters:
```
--data_path="../data/raw/"
--output_path="../data/processed/"
--sample_ratio=0.1
--ctx_window=5
--min_page_tokens=50
--max_page_tokens=2000
--min_cmnt_length=8
```
Last, if you use the single thread mode to generate the processed files one by one, you need to merge the outputs of all the dump files together by running the following command:
```
python3 wikicmnt_extractor.py --merge_only
```
The output of the command is a processed file ```wikicmnt.json``` which includes all the processed data.

## Author

If you have any troubles or questions, please contact [Xuchao Zhang](xuczhang@gmail.com).

August, 2019
