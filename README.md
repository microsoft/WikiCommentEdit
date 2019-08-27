Modeling the Relationship Between Comments and Edits in Document Revisions
======

This is a pytorch implementation of modeling the relationship between comments and edits for wikipedia revision data, as described in our EMNLP 2019 paper:

**Modeling the Relationship between User Comments and Edits in Document Revision**, Xuchao Zhang, Dheeraj Rajagopal, Michael Gamon, Sujay Kumar Jauhar and ChangTien Lu, 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), Hongkong, China, Nov 3-7, 2019.

Two distinct but related tasks are proposed in this work:
- **Comment Ranking**: ranking a list of comments based on their relevance to a specific edit
- **Edit Anchoring**: anchoring a comment to the most relevant sentences in a document

## Requirements
- python 3.6
- [tqdm](https://github.com/noamraph/tqdm)
- [pytorch 0.4.0](https://pytorch.org/)
- [numpy v1.13+](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [spacy v2.0.11](https://spacy.io/)
- and some basic packages.


## Usage
### Data Preparation
- Download the raw data from Wikipedia and generate the preprocessed data file ```wikicmnt.json``` by the steps in our [Data Extractor](./data_processing/README.md). And leave the generated data file ```wikicmnt.json``` in the dataset folder such as ```./data/processed/```.

  <!-- For demo purpose, we include a small dump file ```enwiki-20181001-pages-meta-history24.xml-p33948436p33952815.bz2``` (17MB) in the ```./dataset/``` folder. -->

- Download the glove embeddings from ```https://nlp.stanford.edu/projects/glove/```. And copy these files into the folder you specified in the parameter such as ```./dataset/glove/```.

### Training
In this section, we try to train the models for both comment ranking and edit anchoring tasks individually or jointly. Before training a model, you need to check whether the ```wikicmnt.json``` file existing in the ```--data_path```.

##### Comment Ranking Training
To train the model for comment ranking task, run the following command:
```
python3 main.py --cr_train --data_path="./data/processed/"
```

##### Edit Anchoring Training
To train the model for edit anchoring task, run the following command:
```
python3 main.py --ea_train --data_path="./data/processed/"
```
##### Jointly train on both Comment Ranking & Edit Anchoring tasks
Train a multi-task model with default parameters is simply combining the parameters of individual task together:
```
python3 main.py --cr_train --ea_train --data_path="./dataset/processed/"
```
The common parameters you can change:
```
--data_path="./dataset/"
--glove_path="./dataset/glove/"
--epoch=20
--batch_size=10
--word_embd_size=100
```

The best model is saved to default folder ```./checkpoint/```.

### Test
Test saved model. The following default metrics are presented: P@1, P@3, MRR and NDCG
```
python main.py --test -checkpoint="./checkpoint/saved_best_model.pt"
```

### Full Usage Options
A full options of our code are listed
```
usage: main.py [-h] [--data_path DATA_PATH]
               [--checkpoint_path CHECKPOINT_PATH] [--glove_path GLOVE_PATH]
               [--log_interval LOG_INTERVAL] [--test_interval TEST_INTERVAL]
               [--save_best SAVE_BEST] [--lr LR] [--ngpu NGPU]
               [--word_embd_size WORD_EMBD_SIZE]
               [--max_ctx_length MAX_CTX_LENGTH]
               [--max_diff_length MAX_DIFF_LENGTH]
               [--max_cmnt_length MAX_CMNT_LENGTH] [--ctx_mode CTX_MODE]
               [--rnn_model] [--epoch EPOCH] [--start_epoch START_EPOCH]
               [--batch_size BATCH_SIZE] [--cr_train] [--ea_train]
               [--no_action] [--no_attention] [--no_hadamard]
               [--src_train SRC_TRAIN] [--train_ratio TRAIN_RATIO]
               [--val_ratio VAL_RATIO] [--val_size VAL_SIZE]
               [--manualSeed MANUALSEED] [--test] [--case_study]
               [--resume PATH] [--seed SEED] [--gpu GPU]
               [--checkpoint CHECKPOINT] [--rank_num RANK_NUM]
               [--anchor_num ANCHOR_NUM] [--use_target_only] [--predict]
               [--pred_cmnt PRED_CMNT] [--pred_ctx PRED_CTX]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        the data directory
  --checkpoint_path CHECKPOINT_PATH
                        the checkpoint directory
  --glove_path GLOVE_PATH
                        the glove directory
  --log_interval LOG_INTERVAL
                        how many steps to wait before logging training status
                        [default: 100]
  --test_interval TEST_INTERVAL
                        how many steps to wait before testing [default: 1000]
  --save_best SAVE_BEST
                        whether to save when get best performance
  --lr LR               learning rate, default=0.5
  --ngpu NGPU           number of GPUs to use
  --word_embd_size WORD_EMBD_SIZE
                        word embedding size
  --max_ctx_length MAX_CTX_LENGTH
                        the maximum words in the context [default: 300]
  --max_diff_length MAX_DIFF_LENGTH
                        the maximum words in the revision difference [default:
                        200]
  --max_cmnt_length MAX_CMNT_LENGTH
                        the maximum words in the comment [default: 30]
  --ctx_mode CTX_MODE   whether to use change context in training [default:
                        True]
  --rnn_model           use rnn baseline model
  --epoch EPOCH         number of epoch, default=10
  --start_epoch START_EPOCH
                        resume epoch count, default=1
  --batch_size BATCH_SIZE
                        input batch size
  --cr_train            whether to training the comment rank task
  --ea_train            whether to training the revision anchoring task
  --no_action           whether to use action encoding to train the model
  --no_attention        whether to use mutual attention to train the model
  --no_hadamard         whether to use hadamard product to train the model
  --src_train SRC_TRAIN
                        whether to training the comment rank task without
                        before-editing version
  --train_ratio TRAIN_RATIO
                        ratio of training data in the entire data [default:
                        0.7]
  --val_ratio VAL_RATIO
                        ratio of validation data in the entire data [default:
                        0.1]
  --val_size VAL_SIZE   force the size of validation dataset, the parameter
                        will disgard the setting of val_ratio [default: -1]
  --manualSeed MANUALSEED
                        manual seed
  --test                use test model
  --case_study          use case study mode
  --resume PATH         path saved params
  --seed SEED           random seed
  --gpu GPU             gpu to use for iterate data, -1 mean cpu [default: -1]
  --checkpoint CHECKPOINT
                        filename of model checkpoint [default: None]
  --rank_num RANK_NUM   the number of ranking comments
  --anchor_num ANCHOR_NUM
                        the number of ranking comments
  --use_target_only     use target context only in model
  --predict             predict the sentence given
  --pred_cmnt PRED_CMNT
                        the comment of prediction
  --pred_ctx PRED_CTX   the context of prediction

```


## Author

If you have any troubles or questions, please contact [Xuchao Zhang]().

August, 2018


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
