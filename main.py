import argparse
import time

from eval import case_study, predict
from process_data import load_glove_weights
from train import *
from wikicmnt_dataset import Wiki_DataSet
from wikicmnt_model import CmntModel

#############################################################################################
# ArgumentParser
#############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/processed/', help='the data directory')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/', help='the checkpoint directory')
parser.add_argument('--glove_path', type=str, default='./data/glove/', help='the glove directory')

# model
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many steps to wait before logging training status [default: 100]')
parser.add_argument('--test_interval', type=int, default=1,
                    help='how many steps to wait before testing [default: 1000]')
parser.add_argument('--save_best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate, default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--word_embd_size', type=int, default=100, help='word embedding size')
parser.add_argument('--max_ctx_length', type=int, default=300, help='the maximum words in the context [default: 300]')
parser.add_argument('--max_diff_length', type=int, default=300,
                    help='the maximum words in the revision difference [default: 200]')
parser.add_argument('--max_cmnt_length', type=int, default=30, help='the maximum words in the comment [default: 30]')
parser.add_argument('--ctx_mode', type=bool, default=True,
                    help='whether to use change context in training [default: True]')

# training
parser.add_argument('--epoch', type=int, default=10, help='number of epoch, default=10')
parser.add_argument('--start_epoch', type=int, default=1, help='resume epoch count, default=1')
parser.add_argument('--batch_size', type=int, default=10, help='input batch size')

parser.add_argument('--cr_train', action='store_true', default=False, help='whether to training the comment rank task')
parser.add_argument('--ea_train', action='store_true', default=False,
                    help='whether to training the revision anchoring task')

# ablation testing
parser.add_argument('--no_action', action='store_true', default=False,
                    help='whether to use action encoding to train the model')
parser.add_argument('--no_attention', action='store_true', default=False,
                    help='whether to use mutual attention to train the model')
parser.add_argument('--no_hadamard', action='store_true', default=False,
                    help='whether to use hadamard product to train the model')

parser.add_argument('--src_train', type=bool, default=False,
                    help='whether to training the comment rank task without before-editing version')
parser.add_argument('--train_ratio', type=int, default=0.7,
                    help='ratio of training data in the entire data [default: 0.7]')
parser.add_argument('--val_ratio', type=int, default=0.1,
                    help='ratio of validation data in the entire data [default: 0.1]')
parser.add_argument('--val_size', type=int, default=10000,
                    help='force the size of validation dataset, the parameter will disgard the setting of val_ratio [default: -1]')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--test', action='store_true', default=False, help='use test model')
parser.add_argument('--case_study', action='store_true', default=False, help='use case study mode')
parser.add_argument('--resume', default='./checkpoints/model_best.tar', type=str, metavar='PATH',
                    help='path saved params')
parser.add_argument('--seed', type=int, default=1111, help='random seed')

# device
parser.add_argument('--gpu', type=int, default=-1, help='gpu to use for iterate data, -1 mean cpu [default: -1]')

parser.add_argument('--checkpoint', type=str, default=None, help='filename of model checkpoint [default: None]')

parser.add_argument('--rank_num', type=int, default=5, help='the number of ranking comments')
parser.add_argument('--anchor_num', type=int, default=5, help='the number of ranking comments')
parser.add_argument('--use_target_only', action='store_true', default=False, help='use target context only in model')

# single case prediction
parser.add_argument('--predict', action='store_true', default=False, help='predict the sentence given')
parser.add_argument('--pred_cmnt', type=str, default=None, help='the comment of prediction')
parser.add_argument('--pred_ctx', type=str, default=None, help='the context of prediction')

args = parser.parse_args()

if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

print(os.name)
sys.stdout.flush()

# load data
dataset = Wiki_DataSet(args)
train_df, val_df, test_df, vocab_json = dataset.load_data(train_ratio=args.train_ratio, val_ratio=args.val_ratio)
w2i = vocab_json['word2idx']

print('----')
print('n_train', train_df.shape[0])
print('n_val', val_df.shape[0])
print('n_test', test_df.shape[0])
print('vocab_size:', len(w2i))

# load glove
glove_embd_w = torch.from_numpy(load_glove_weights(args.glove_path, args.word_embd_size, len(w2i), w2i)).type(
    torch.FloatTensor)
# save_pickle(glove_embd_w, './pickle/glove_embd_w.pickle')

args.vocab_size_w = len(w2i)
args.pre_embd_w = glove_embd_w
args.filters = [[1, 5]]
args.out_chs = 100

# generate save directory
base_name = os.path.basename(os.path.normpath(args.data_path))
if args.cr_train and not args.ea_train:
    task_str = "cr"
elif args.ea_train and not args.cr_train:
    task_str = "ea"
elif args.cr_train and args.ea_train:
    task_str = "mt"
folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "_" + base_name + "_" + task_str
if args.no_action:
    folder_name += "_noaction"
if args.word_embd_size == 300:
    folder_name += "_d300"
args.save_dir = os.path.join(args.checkpoint_path, folder_name)
print("Save to ", args.save_dir)
sys.stdout.flush()

# initialize model
model = CmntModel(args)

if args.checkpoint is not None:
    print('\nLoading model from {}...'.format(args.checkpoint))
    model.load_state_dict(torch.load(args.checkpoint))

if torch.cuda.is_available() and os.name != 'nt':
    print('use cuda')
    model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0])

# optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.5)
# optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
# optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))

if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume))

print(model)
print('parameters-----')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.size())

if args.predict:
    print('Prediction mode')
    print('#Comment:', args.pred_cmnt)
    print('#Context:', args.pred_ctx)
    predict(args.pred_cmnt, args.pred_ctx, w2i, model, args.max_ctx_length)
elif args.test:
    print('Test mode')
    eval(dataset, test_df, w2i, model, args)
elif args.case_study:
    start_time = time.time()
    case_study(dataset, test_df, w2i, model, args)
else:
    print('Train mode')
    start_time = time.time()
    train(args, model, dataset, train_df, val_df, optimizer, w2i, \
          n_epoch=args.epoch, start_epoch=args.start_epoch, batch_size=args.batch_size)
    print("Training duration: %s seconds" % (time.time() - start_time))
print('finish')
