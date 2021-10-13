import torch
import torch.nn as nn
import numpy as np
from r2r_param_multiloss import args
import os
import r2r_utils
import model
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, Adamax
from optim import AdamW

# TODO move to r2r_utils
def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.speaker_featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.speaker_angle_feat_size] = self.drop3(x[..., :-args.speaker_angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.speaker_angle_feat_size] = self.drop3(feature[..., :-args.speaker_angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class SpeakerDecoderForUniter(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio, uniter_hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

        self.uniter_projection = nn.Linear(uniter_hidden_size, hidden_size)

    def forward(self, words, ctx, ctx_mask):

        # for uniter ctx
        batch_size = words.size()[0]
        h0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        ctx = self.uniter_projection(ctx)

        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit


class Speaker():
    env_actions = {
        'left': (0,-1, 0), # left
        'right': (0, 1, 0), # right
        'up': (0, 0, 1), # up
        'down': (0, 0,-1), # down
        'forward': (1, 0, 0), # forward
        '<end>': (0, 0, 0), # <end>
        '<start>': (0, 0, 0), # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, listener, tok):
        self.env = env
        # TODO for debug
        # self.feature_size = 0
        if args.features == 'none':
            self.feature_size = 0
        else:
            self.feature_size = self.env.feature_size
        self.tok = tok
        self.tok.finalize()
        self.listener = listener

        # Model
        print("VOCAB_SIZE", self.tok.vocab_size())
        self.encoder = SpeakerEncoder(self.feature_size+args.speaker_angle_feat_size, args.rnn_dim, args.speaker_dropout, bidirectional=args.bidir).cuda()
        self.decoder = SpeakerDecoder(self.tok.vocab_size(), args.wemb, self.tok.word_to_index['<PAD>'],
                                            args.rnn_dim, args.speaker_dropout).cuda()
        print(args.speaker_optim)
        if args.speaker_optim == 'rms':
            print("Optimizer: Using RMSProp")
            speaker_optimizer = torch.optim.RMSprop
        elif args.speaker_optim == 'adam':
            print("Optimizer: Using Adam")
            speaker_optimizer = torch.optim.Adam
        elif args.speaker_optim == 'sgd':
            print("Optimizer: sgd")
            speaker_optimizer = torch.optim.SGD
        else:
            assert False

        self.encoder_optimizer = speaker_optimizer(self.encoder.parameters(), lr=args.speaker_lr)
        self.decoder_optimizer = speaker_optimizer(self.decoder.parameters(), lr=args.speaker_lr)

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])

        self.aug_path_cache = r2r_utils.read_aug_path_cache(args.aug_path_cahche)


    def train_with_transformer(self, iters):
        for i in range(iters):
            self.env.reset()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss = self.teacher_forcing_with_transformer(train=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

    def train(self, iters):
        for i in range(iters):
            self.env.reset()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss = self.teacher_forcing(train=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

    def get_insts(self, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts = self.infer_batch()  # Get the insts of the result
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.tok.shrink(inst)  # Shrink the words
        return path2inst

    def valid(self, *aargs, **kwargs):
        """

        :param iters:
        :return: path2inst: path_id --> inst (the number from <bos> to <eos>)
                 loss: The XE loss
                 word_accu: per word accuracy
                 sent_accu: per sent accuracy
        """
        path2inst = self.get_insts(*aargs, **kwargs)

        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 1 if args.fast_train else 3     # Set the iter to 1 if the fast_train (o.w. the problem occurs)
        metrics = np.zeros(3)
        for i in range(N):
            self.env.reset()
            metrics += np.array(self.teacher_forcing(train=False))
        metrics /= N

        return (path2inst, *metrics)

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12   # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                    # print("UP")
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                    # print("DOWN")
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print("RIGHT")
                    # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def _teacher_action(self, obs, ended, tracker=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.feature_size + args.speaker_angle_feat_size), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                if self.feature_size == 0:
                    candidate_feat[i, :] = c['feature'][-args.speaker_angle_feat_size:] # angle feat
                else:
                    cand_angle_feat = r2r_utils.angle_feature(c['heading'], c['elevation'], args.speaker_angle_feat_size)
                    candidate_feat[i, :] = np.concatenate((c['feature'][:-args.angle_feat_size], cand_angle_feat), -1)
        return torch.from_numpy(candidate_feat).cuda()

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.speaker_angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            angle_feat = self.env.speaker_angle_feature[ob['viewIndex']]
            # cand_angle_feat = r2r_utils.angle_feature(ob['heading'], ob['elevation'], args.speaker_angle_feat_size)
            features[i, :, :] = np.concatenate((ob['feature'][:, :-args.angle_feat_size], angle_feat), -1)
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def from_shortest_path_cached(self):
        batch_size = self.env.batch_size
        lengths = np.zeros(batch_size, np.int64)
        aug_path_info = []

        # first_feat = np.zeros((batch_size, self.feature_size + args.speaker_angle_feat_size), np.float32)
        # for i, ob in enumerate(obs):
        #     first_feat[i, -args.speaker_angle_feat_size:] = r2r_utils.angle_feature(ob['heading'], ob['elevation'],
        #                                                                             args.speaker_angle_feat_size)
        # first_feat = torch.from_numpy(first_feat).cuda()

        for i, batch in enumerate(self.env.batch):
            instr_id = batch['instr_id']
            path_cache = self.aug_path_cache[instr_id]

            aug_path_info.append(path_cache)
            lengths[i] = len(path_cache)

        max_len = max(lengths)
        img_feats = np.zeros((batch_size, max_len, args.views, self.feature_size + args.speaker_angle_feat_size), dtype=np.float32)
        cand_feats = np.zeros((batch_size, max_len, self.feature_size + args.speaker_angle_feat_size), dtype=np.float32)

        for i, batch in enumerate(self.env.batch):
            scan = batch['scan']
            aug_cache = aug_path_info[i]
            goal_img_feat = np.concatenate(
                (self.env.env.features[scan + '_' + aug_cache[-1]['viewpoint']],
                 self.env.speaker_angle_feature[aug_cache[-1]['viewIndex']]), -1)

            for step in range(max_len):
                if step < lengths[i]:
                    ob = aug_cache[step]
                    viewpoint = ob['viewpoint']
                    long_id = scan + '_' + viewpoint
                    img_feats[i, step, ...] = np.concatenate(
                        (self.env.env.features[long_id], self.env.speaker_angle_feature[ob['viewIndex']]), -1)

                    if ob['teacher_action'] != -1:
                        teacher_cand = ob['candidate'][ob['teacher_action']]
                        # cand_long_id = scan + '_' + teacher_cand['viewpointId']
                        cand_visual_feat = self.env.env.features[long_id][teacher_cand['pointId']]
                        cand_angle_feat = r2r_utils.angle_feature(teacher_cand['heading'], teacher_cand['elevation'],
                                                                  args.speaker_angle_feat_size)
                        cand_feats[i, step, ...] = np.concatenate((cand_visual_feat, cand_angle_feat), -1)
                else:
                    img_feats[i, step, ...] = goal_img_feat

        return (torch.from_numpy(img_feats).contiguous().cuda(), torch.from_numpy(cand_feats).contiguous().cuda()), lengths


    def from_shortest_path(self, viewpoints=None, get_first_feat=False):
        """
        :param viewpoints: [[], [], ....(batch_size)]. Only for dropout viewpoint
        :param get_first_feat: whether output the first feat
        :return:
        """
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        first_feat = np.zeros((len(obs), self.feature_size+args.speaker_angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -args.speaker_angle_feat_size:] = r2r_utils.angle_feature(ob['heading'], ob['elevation'], args.speaker_angle_feat_size)
        first_feat = torch.from_numpy(first_feat).cuda()
        while not ended.all():
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self._feature_variable(obs)) # bs * 36 * 2176
            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action
            can_feats.append(self._candidate_variable(obs, teacher_action)) # bs * max * 2176
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052
        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            return (img_feats, can_feats), length

    def gt_words(self, obs):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        return torch.from_numpy(seq_tensor).cuda()

    def teacher_forcing(self, train=True, features=None, insts=None, for_listener=False):
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Get Image Input & Encode
        if features is not None:
            # It is used in calulating the speaker score in beam-search
            assert insts is not None
            (img_feats, can_feats), lengths = features
            ctx = self.encoder(can_feats, img_feats, lengths)
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs()
            batch_size = len(obs)
            (img_feats, can_feats), lengths = self.from_shortest_path()      # Image Feature (from the shortest path)
            ctx = self.encoder(can_feats, img_feats, lengths)
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        ctx_mask = length2mask(lengths)

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs)                                       # Language Feature

        # Decode
        logits, _, _ = self.decoder(insts, ctx, ctx_mask, h_t, c_t)

        # Because the softmax_loss only allow dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        logits = logits.permute(0, 2, 1).contiguous()
        loss = self.softmax_loss(
            input  = logits[:, :, :-1],         # -1 for aligning
            target = insts[:, 1:]               # "1:" to ignore the word <BOS>
        )

        if for_listener:
            return self.nonreduced_softmax_loss(
                input  = logits[:, :, :-1],         # -1 for aligning
                target = insts[:, 1:]               # "1:" to ignore the word <BOS>
            )

        if train:
            return loss
        else:
            # Evaluation
            _, predict = logits.max(dim=1)                                  # BATCH, LENGTH
            gt_mask = (insts != self.tok.word_to_index['<PAD>'])
            correct = (predict[:, :-1] == insts[:, 1:]) * gt_mask[:, 1:]    # Not pad and equal to gt
            correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
            word_accu = correct.sum().item() / gt_mask[:, 1:].sum().item()     # Exclude <BOS>
            sent_accu = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size  # Exclude <BOS>
            return loss.item(), word_accu, sent_accu

    def infer_batch(self, sampling=False, train=False, featdropmask=None):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                     log_probs(torch, requires_grad, [batch,max_len])
                                     hiddens(torch, requires_grad, [batch, max_len, dim})
                      And if train: the log_probs and hiddens are detached
                 if not sampling: returns insts(np, [batch, max_len])
        """
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Image Input for the Encoder
        # obs = self.env._get_obs()
        # batch_size = len(obs)
        batch_size = self.env.batch_size

        # Get feature
        # viewpoints_list = [list() for _ in range(batch_size)]
        # (img_feats, can_feats), lengths = self.from_shortest_path(viewpoints=viewpoints_list)
        (img_feats, can_feats), lengths = self.from_shortest_path_cached()      # Image Feature (from the shortest path)
        # assert torch.abs((img_feats-img_feats2)).sum()==0
        # assert torch.abs((can_feats2-can_feats)).sum() ==0
        # assert np.abs((lengths-lengths2)).sum()==0
        # This code block is only used for the featdrop.
        if featdropmask is not None:
            img_feats[..., :-args.speaker_angle_feat_size] *= featdropmask
            can_feats[..., :-args.speaker_angle_feat_size] *= featdropmask

        # Encoder
        ctx = self.encoder(can_feats, img_feats, lengths,
                           already_dropfeat=(featdropmask is not None))
        ctx_mask = length2mask(lengths)

        # Decoder
        words = []
        log_probs = []
        hidden_states = []
        entropies = []
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        ended = np.zeros(batch_size, np.bool)
        word = np.ones(batch_size, np.int64) * self.tok.word_to_index['<BOS>']    # First word is <BOS>
        word = torch.from_numpy(word).view(-1, 1).cuda()
        for i in range(args.maxDecode):
            # Decode Step
            logits, h_t, c_t = self.decoder(word, ctx, ctx_mask, h_t, c_t)      # Decode, logits: (b, 1, vocab_size)

            # Select the word
            logits = logits.squeeze()                                           # logits: (b, vocab_size)
            logits[:, self.tok.word_to_index['<UNK>']] = -float("inf")          # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    hidden_states.append(h_t.squeeze())
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    hidden_states.append(h_t.squeeze().detach())
                    entropies.append(m.entropy().detach())
            else:
                values, word = logits.max(1)

            # Append the word
            cpu_word = word.cpu().numpy()
            cpu_word[ended] = self.tok.word_to_index['<PAD>']
            words.append(cpu_word)

            # Prepare the shape for next step
            word = word.view(-1, 1)

            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.word_to_index['<EOS>'])
            if ended.all():
                break

        if train and sampling:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(hidden_states, 1), torch.stack(entropies, 1)
        else:
            return np.stack(words, 1)       # [(b), (b), (b), ...] --> [b, l]

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            # print(name)
            # print(list(model.state_dict().keys()))
            # for key in list(model.state_dict().keys()):
            #     print(key, model.state_dict()[key].size())
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1

