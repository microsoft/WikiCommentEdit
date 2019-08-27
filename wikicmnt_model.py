import torch
import torch.nn as nn
import torch.nn.functional as F


# from layers.highway import Highway

class WordEmbedding(nn.Module):
    '''
    In : (N, sentence_len)
    Out: (N, sentence_len, embd_size)
    '''

    def __init__(self, args, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size_w, args.word_embd_size)
        if args.pre_embd_w is not None:
            self.embedding.weight = nn.Parameter(args.pre_embd_w, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)


class CmntModel(nn.Module):
    def __init__(self, args):
        super(CmntModel, self).__init__()
        self.batch_size = args.batch_size
        self.embd_size = args.word_embd_size
        self.cmnt_length = args.max_cmnt_length
        self.d = self.embd_size  # word_embedding
        self.no_action = args.no_action
        self.no_attention = args.no_attention
        self.no_hadamard = args.no_hadamard
        # self.d = self.embd_size # only word_embedding

        # self.char_embd_net = CharEmbedding(args)
        self.word_embd_net = WordEmbedding(args)
        # self.highway_net = Highway(self.d)
        self.ctx_embd_layer = nn.GRU(self.d, self.d, bidirectional=True, dropout=0.2, batch_first=True)

        self.W = nn.Linear(6 * self.d + 1, 1, bias=False)

        # self.W2_noact = nn.Linear(2 * self.d, 1, bias=False)
        # self.W2 = nn.Linear(2 * self.d + 1, 1, bias=False)

        # weights for attention layer
        if self.no_hadamard and self.no_action:
            # (1, 1)
            self.W2_nhna = nn.Linear(1, 1, bias=False)
        elif self.no_hadamard and not self.no_action:
            # (2, 1)
            self.W2_nha = nn.Linear(2, 1, bias=False)
        elif not self.no_hadamard and self.no_action:
            # (2d, 1)
            self.W2_hna = nn.Linear(2 * self.d, 1, bias=False)
        elif not self.no_hadamard and not self.no_action:
            # (2d+1, 1)
            self.W2_ha = nn.Linear(2 * self.d + 1, 1, bias=False)

        self.modeling_layer = nn.GRU(8 * self.d, self.d, num_layers=2, bidirectional=True, dropout=0.2,
                                     batch_first=True)

        # Linear function for comment ranking
        self.rank_linear = nn.Linear(self.cmnt_length * 2, 1, bias=True)
        self.rank_ctx_linear = nn.Linear(self.cmnt_length * 4, 1, bias=True)

        # Linear function for edit anchoring
        self.anchor_linear = nn.Linear(self.cmnt_length * 2, 2, bias=True)

        self.use_target_only = args.use_target_only
        self.ctx_mode = 1
        # self.p2_lstm_layer = nn.GRU(2*self.d, self.d, bidirectional=True, dropout=0.2, batch_first=True)

    def build_contextual_embd(self, x_w):
        # 1. Word Embedding Layer
        embd = self.word_embd_net(x_w)  # (N, seq_len, embd_size)

        # 2. Highway Networks for 1.
        # embd = self.highway_net(word_embd) # (N, seq_len, d=embd_size)

        # 3. Contextual Embedding Layer
        ctx_embd_out, _h = self.ctx_embd_layer(embd)
        return ctx_embd_out

    def build_cmnt_sim(self, embd_context, embd_cmnt, embd_action, batch_size, T, J):

        shape = (batch_size, T, J, 2 * self.d)  # (N, T, J, 2d)
        embd_context_ex = embd_context.unsqueeze(2)  # (N, T, 1, 2d)
        embd_cmnt_ex = embd_cmnt.unsqueeze(1)  # (N, 1, J, 2d)

        # action embedding
        embd_action_ex = embd_action.float().unsqueeze(2).unsqueeze(2)
        embd_action_ex = embd_action_ex.expand((batch_size, T, J, 1))

        if self.no_hadamard:

            if self.no_action:
                raise Exception('no hadamard cannot be used with -no_action simultaneously')
            # use inner product to replace the hadamard product
            # generate (N, T, J, 1)
            embd_cmnt_ex = embd_cmnt_ex.permute(0, 2, 3, 1)  # (N, J, 2d, 1)
            # batch1 = torch.randn(10, 3, 4)
            # batch2 = torch.randn(10, 4, 5)
            # (N, T, 1, 2d) * (N, J, 2d, 1) => (N, T, J, 1)
            a_dotprod_mul_b = torch.einsum('ntid,njdi->ntji', [embd_context_ex, embd_cmnt_ex])

            # no hadamard & action
            cat_data = torch.cat((a_dotprod_mul_b, embd_action_ex), 3)  # (N, T, J, 2), [h◦u; a]
            S = self.W2_nha(cat_data).view(batch_size, T, J)  # (N, T, J)
        else:
            embd_context_ex = embd_context_ex.expand(shape)  # (N, T, J, 2d)

            embd_cmnt_ex = embd_cmnt_ex.expand(shape)  # (N, T, J, 2d)
            a_elmwise_mul_b = torch.mul(embd_context_ex, embd_cmnt_ex)  # (N, T, J, 2d)

            if self.no_action:
                # hadamard & no action
                S = self.W2_hna(a_elmwise_mul_b).view(batch_size, T, J)  # (N, T, J)
            else:
                # hadamard & action
                cat_data = torch.cat((a_elmwise_mul_b, embd_action_ex), 3)  # (N, T, J, 2d + 1), [h◦u; a]
                S = self.W2_ha(cat_data).view(batch_size, T, J)  # (N, T, J)

        if self.no_attention:
            # without using attention, simply use the mean of similarity matrix in edit dimension
            S_cmnt = torch.mean(S, 1)
        else:
            # attention implementation:
            # b: attention weights on the context
            b = F.softmax(torch.max(S, 2)[0], dim=-1)  # (N, T)
            S_cmnt = torch.bmm(b.unsqueeze(1), S)  # (N, 1, J) = bmm( (N, 1, T), (N, T, J) )
            S_cmnt = S_cmnt.squeeze(1)  # (N, J)

        # max implementation
        # S_cmnt = torch.max(S, 1)[0]

        # c: attention weights on the comment
        # c = torch.max(S, 1)[0] # (N, J)
        # S_cmnt = c * S_cmnt # (N, J) = (N, J) * (N, J)

        # c2q = torch.bmm(F.softmax(S, dim=-1), embd_cmnt) # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        # c2q = torch.bmm(F.softmax(S, dim=-1), embd_cmnt) # (N, J, 1) = bmm( (N, J, T), (N, T, 1) )

        return S_cmnt, S

    # cmnt_words, neg_cmnt_words, src_diff_words, tgt_diff_words
    def forward(self, cmnt, src_token, src_action, tgt_token, tgt_action, cr_mode=True, cl_mode=False):

        batch_size = cmnt.size(0)
        T = src_token.size(1)  # sentence length = 100 (word level)
        # C = src_token.size(1) # context sentence length = 200 (word level)
        J = cmnt.size(1)  # cmnt sentence length = 30  (word level)

        # ####################################################################################
        # 1. Word Embedding Layer
        # 2. Contextual Embedding Layer (GRU)
        ######################################################################################
        embd_src_diff = self.build_contextual_embd(src_token)  # (N, T, 2d)
        embd_tgt_diff = self.build_contextual_embd(tgt_token)  # (N, T, 2d)

        if cl_mode:
            return embd_src_diff + embd_tgt_diff  # (N, T, 2d)

        embd_cmnt = self.build_contextual_embd(cmnt)  # (N, J, 2d)

        # if self.ctx_mode:
        #     embd_src_ctx = self.build_contextual_embd(src_ctx) # (N, C, 2d)
        #     embd_tgt_ctx = self.build_contextual_embd(tgt_ctx) # (N, C, 2d)

        # ####################################################################################
        # 3. Similarity Layer
        ######################################################################################

        S_src_diff, _ = self.build_cmnt_sim(embd_src_diff, embd_cmnt, src_action, batch_size, T, J)  # (N, J)
        S_tgt_diff, _ = self.build_cmnt_sim(embd_tgt_diff, embd_cmnt, tgt_action, batch_size, T, J)  # (N, J)
        S_diff = torch.cat((S_src_diff, S_tgt_diff), 1)  # (N, 2J)

        # if self.ctx_mode:
        #     S_src_ctx, _ = self.build_cmnt_sim(embd_src_ctx, embd_cmnt, batch_size, C, J)
        #     S_tgt_ctx, _ = self.build_cmnt_sim(embd_tgt_ctx, embd_cmnt, batch_size, C, J)
        #     S_ctx = torch.cat((S_src_ctx, S_tgt_ctx), 1) # (N, 2J)
        #     score = self.rank_ctx_linear(torch.cat((S_diff, S_ctx), 1)) # (N, 2J) -> (N, 1)
        # else:
        if cr_mode:
            result = self.rank_linear(S_diff)  # (N, 2J) -> (N, 1)
        else:
            result = self.anchor_linear(S_diff)  # (N, 2J) -> (N, 2)

        # if self.use_target_only:
        #     S_diff = S_tgt_diff # (N, J)
        # else:
        #     #S_diff = S_src_diff + S_tgt_diff # (N, J)
        #     S_diff = torch.cat((S_src_diff, S_tgt_diff), 1)
        #     #S = (torch.cat((S_src_diff, S_tgt_diff), 1)

        return result, S_diff
