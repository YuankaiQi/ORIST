"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VQA model
"""
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
# from torch.nn import LayerNorm
from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel
from speaker import SpeakerDecoderForUniter

class SimpleClassifier(nn.Module):
    def __init__(self,config):
        super(SimpleClassifier, self).__init__()
        import json
        with open(config,'r') as f:
            data = json.load(f)
            self.hidden_size = data['hidden_size']
            self.initializer_range= data['initializer_range']
        self.critic = nn.Sequential(  # for rl
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            GELU(),
            LayerNorm(self.hidden_size * 2, eps=1e-12),
            nn.Linear(self.hidden_size * 2, 1)
        )
        self.apply(self.init_weights)

    def forward(self, x):
        return self.critic(x)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class UniterForR2R(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        # self.r2r_output = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size*2),
        #     GELU(),
        #     LayerNorm(config.hidden_size*2, eps=1e-12),
        #     nn.Linear(config.hidden_size*2, 1)
        # )
        self.r2r_output = nn.Linear(config.hidden_size, 1)
        # self.critic = nn.Sequential(  # for rl
        #     nn.Linear(config.hidden_size, config.hidden_size * 2),
        #     GELU(),
        #     LayerNorm(config.hidden_size * 2, eps=1e-12),
        #     nn.Linear(config.hidden_size * 2, 1)
        # )
        self.critic = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def forward(self, rl, input_ids=None,position_ids=None,img_feat=None,
                    img_pos_feat=None, orien_feat=None,attn_masks=None,gather_index=None,
                    candidate_leng=None, instr_state=None):

        if position_ids.size(0)>1:
            position_ids = position_ids[0].unsqueeze(0)

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index, orien_feat=orien_feat,
                                      output_all_encoded_layers=False,
                                      instr_state=instr_state,candi_leng=candidate_leng)
        instra_state = sequence_output[:,0,:] # check dims
        pooled_output = self.uniter.pooler(sequence_output)
        vln_logit = self.r2r_output(pooled_output).squeeze()

        # re-organize logit
        logit = torch.zeros(len(candidate_leng), max(candidate_leng)).cuda()
        tmp_start = 0
        for li, num in enumerate(candidate_leng):
            logit[li, 0:num] = vln_logit[tmp_start:sum(candidate_leng[0:li + 1])]
            tmp_start += num

        if rl == False:
            return logit, None, instra_state
        else:
            # re-organize state
            newstate, bstart = [], 0
            for candi_num in candidate_leng:
                tmp = pooled_output[bstart:candi_num + bstart - 1, :].unsqueeze(dim=0).unsqueeze(
                    dim=0)  # exclude the last candidate
                newstate.append(F.avg_pool2d(tmp, (candi_num - 1, 1), stride=(1, 1)).squeeze())
                bstart += candi_num
            del tmp
            newstate = torch.stack(newstate, dim=0)
            reward = self.critic(newstate)
            return logit, reward, instra_state

# for multi task
class ProgressIndicator(nn.Module):
    def __init__(self, hidden_size):
        super(ProgressIndicator, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

class AnglePredictor(nn.Module):
    def __init__(self, hidden_size):
        super(AnglePredictor, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 4)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.fc2(h)
        return h

class RegionPredictor(nn.Module):
    def __init__(self, hidden_size):
        super(RegionPredictor, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 31)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

class NextRegionPredictor(nn.Module):
    def __init__(self, hidden_size):
        super(NextRegionPredictor, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 31)

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.fc2(h)
        return h

# 11-03 Update: remove cur region
class UniterForR2RAux(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, tok, args):
        super().__init__(config)
        self.args = args
        self.uniter = UniterModel(config, img_dim)

        self.r2r_output = nn.Linear(config.hidden_size, 1)

        self.no_history_state = args.no_history_state
        if args.progress_loss:
            self.progress_pred = ProgressIndicator(config.hidden_size)
        else:
            self.progress_pred = None

        if args.angle_loss:
            self.angle_pred = AnglePredictor(config.hidden_size)
        else:
            self.angle_pred = None

        if args.candi_region_loss:
            self.region_pred = RegionPredictor(config.hidden_size)
        else:
            self.region_pred = None

        if args.next_region_loss:
            self.next_region_pred = NextRegionPredictor(config.hidden_size)
        else:
            self.next_region_pred = None

        if args.target_region_loss:
            self.target_region_pred = NextRegionPredictor(config.hidden_size)
        else:
            self.target_region_pred = None

        if config.use_lstm:
            self.use_lstm = True
            self.lstm = nn.LSTMCell(config.hidden_size, config.hidden_size)
        else:
            self.use_lstm = False

        if config.use_speaker:
            self.use_speaker = True
            self.speaker_decoder = SpeakerDecoderForUniter(tok.vocab_size(), args.wemb, tok.word_to_index['<PAD>'],
                                                      args.rnn_dim, args.speaker_dropout, config.hidden_size)
        else:
            self.use_speaker = False
            self.speaker_decoder = None

        self.critic = nn.Linear(config.hidden_size, 1)

        self.apply(self.init_weights)


    def forward(self, rl, input_ids=None,position_ids=None,img_feat=None,
                    img_pos_feat=None, orien_feat=None,attn_masks=None,gather_index=None,
                    candidate_leng=None, instr_state=None, c_0=None, step=0,
                speaker=False, insts=None, instr_state_ctx=None,
                decode_mask=None, speaker_step=False):

        progress_value, angle_value, region_value, \
        next_region_value, target_region_value = None, None, None, None, None
        c_1 = None

        if speaker:
            return self.speaker_decoder(insts, instr_state_ctx, decode_mask)

        if position_ids.size(0)>1:
            position_ids = position_ids[0].unsqueeze(0)

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index, orien_feat=orien_feat,
                                      output_all_encoded_layers=False,
                                      instr_state=instr_state,candi_leng=candidate_leng, step=step)

        pooled_output = self.uniter.pooler(sequence_output)

        vln_logit = self.r2r_output(pooled_output).squeeze()

        newstate, bstart = [], 0
        for candi_num in candidate_leng:
            tmp = pooled_output[bstart:candi_num + bstart - 1, :].unsqueeze(dim=0).unsqueeze(
                dim=0)  # exclude the last candidate
            newstate.append(F.avg_pool2d(tmp, (candi_num - 1, 1), stride=(1, 1)).squeeze())
            bstart += candi_num
        del tmp
        newstate = torch.stack(newstate, dim=0)

        if self.no_history_state == False:
            if self.use_lstm:
                if step==0:
                    h_0 = torch.zeros_like(c_0).cuda()
                    instra_state, c_1 = self.lstm(newstate, (h_0, c_0))
                else:
                    instra_state, c_1 = self.lstm(newstate,(instr_state, c_0))
            else:
                instra_state = sequence_output[:, 0, :]  # check dims
        else:
            instra_state = [None] * len(candidate_leng)

        # re-organize logit
        logit = torch.zeros(len(candidate_leng), max(candidate_leng)).cuda()
        if not speaker_step and not rl:
            if self.args.angle_loss:
                angle_logits = self.angle_pred(pooled_output)
                angle_value = torch.zeros(len(candidate_leng), max(candidate_leng), 4).cuda()

            if self.args.candi_region_loss:
                region_logits = self.region_pred(pooled_output)
                region_value = torch.zeros(len(candidate_leng), max(candidate_leng), 31).cuda()

            if self.args.next_region_loss:
                next_region_value = self.next_region_pred(newstate)

            if self.args.target_region_loss:
                target_region_value = self.target_region_pred(newstate)

        tmp_start = 0
        for li, num in enumerate(candidate_leng):
            logit[li, 0:num] = vln_logit[tmp_start:sum(candidate_leng[0:li + 1])]
            if not speaker_step and not rl:
                if self.args.angle_loss:
                    angle_value[li, 0:num, ...] = angle_logits[tmp_start:sum(candidate_leng[0:li + 1])]
                if self.args.candi_region_loss:
                    region_value[li, 0:num, ...] = region_logits[tmp_start:sum(candidate_leng[0:li + 1])]
            tmp_start += num


        if speaker_step:
            return logit, instra_state, c_1

        if rl == False:
            if self.args.progress_loss:
                progress_value = self.progress_pred(newstate).squeeze()
            return logit, None, instra_state, progress_value, angle_value, region_value, \
                   next_region_value, target_region_value, c_1
        else:
            reward = self.critic(newstate)
            return logit, reward, instra_state, None, None, None, \
                   None, None, c_1

# class UniterForR2RAux(UniterPreTrainedModel):
#     """ Finetune UNITER for VQA
#     """
#     def __init__(self, config, img_dim, tok, args):
#         super().__init__(config)
#         self.uniter = UniterModel(config, img_dim)
#
#         self.r2r_output = nn.Linear(config.hidden_size, 1)
#         self.progress_pred = ProgressIndicator(config.hidden_size)
#         self.angle_pred = AnglePredictor(config.hidden_size)
#         self.region_pred = RegionPredictor(config.hidden_size)
#
#         self.next_region_pred = NextRegionPredictor(config.hidden_size)
#         self.target_region_pred = NextRegionPredictor(config.hidden_size)
#
#         if config.use_lstm:
#             self.use_lstm = True
#             self.lstm = nn.LSTMCell(config.hidden_size, config.hidden_size)
#         else:
#             self.use_lstm = False
#
#         if config.use_speaker:
#             self.use_speaker = True
#             self.speaker_decoder = SpeakerDecoderForUniter(tok.vocab_size(), args.wemb, tok.word_to_index['<PAD>'],
#                                                       args.rnn_dim, args.speaker_dropout, config.hidden_size)
#         else:
#             self.use_speaker = False
#             self.speaker_decoder = None
#
#         self.critic = nn.Linear(config.hidden_size, 1)
#
#         self.apply(self.init_weights)
#
#
#     def forward(self, rl, input_ids=None,position_ids=None,img_feat=None,
#                     img_pos_feat=None, orien_feat=None,attn_masks=None,gather_index=None,
#                     candidate_leng=None, instr_state=None, c_0=None, step=0,
#                 speaker=False, insts=None, instr_state_ctx=None,
#                 decode_mask=None, speaker_step=False):
#
#         if speaker:
#             return self.speaker_decoder(insts, instr_state_ctx, decode_mask)
#
#         if position_ids.size(0)>1:
#             position_ids = position_ids[0].unsqueeze(0)
#
#         sequence_output = self.uniter(input_ids, position_ids,
#                                       img_feat, img_pos_feat,
#                                       attn_masks, gather_index, orien_feat=orien_feat,
#                                       output_all_encoded_layers=False,
#                                       instr_state=instr_state,candi_leng=candidate_leng, step=step)
#
#         pooled_output = self.uniter.pooler(sequence_output)
#
#         vln_logit = self.r2r_output(pooled_output).squeeze()
#
#         newstate, bstart = [], 0
#         for candi_num in candidate_leng:
#             tmp = pooled_output[bstart:candi_num + bstart - 1, :].unsqueeze(dim=0).unsqueeze(
#                 dim=0)  # exclude the last candidate
#             newstate.append(F.avg_pool2d(tmp, (candi_num - 1, 1), stride=(1, 1)).squeeze())
#             bstart += candi_num
#         del tmp
#         newstate = torch.stack(newstate, dim=0)
#         if self.use_lstm:
#             if step==0:
#                 h_0 = torch.zeros_like(c_0).cuda()
#                 instra_state, c_1 = self.lstm(newstate, (h_0, c_0))
#             else:
#                 instra_state, c_1 = self.lstm(newstate,(instr_state, c_0))
#         else:
#             instra_state = sequence_output[:, 0, :]  # check dims
#
#         # re-organize logit
#         logit = torch.zeros(len(candidate_leng), max(candidate_leng)).cuda()
#         if not speaker_step and not rl:
#             angle_logits = self.angle_pred(pooled_output)
#             region_logits = self.region_pred(pooled_output)
#             next_region_value = self.next_region_pred(newstate)
#             target_region_value = self.target_region_pred(newstate)
#             region_value = torch.zeros(len(candidate_leng), max(candidate_leng), 31).cuda()
#             angle_value = torch.zeros(len(candidate_leng), max(candidate_leng), 4).cuda()
#
#         tmp_start = 0
#         for li, num in enumerate(candidate_leng):
#             logit[li, 0:num] = vln_logit[tmp_start:sum(candidate_leng[0:li + 1])]
#             if not speaker_step and not rl:
#                 angle_value[li, 0:num, ...] = angle_logits[tmp_start:sum(candidate_leng[0:li + 1])]
#                 region_value[li, 0:num, ...] = region_logits[tmp_start:sum(candidate_leng[0:li + 1])]
#             tmp_start += num
#
#         if not self.use_lstm:
#             c_1 = None
#
#         if speaker_step:
#             return logit, instra_state, c_1
#
#         if rl == False:
#             progress_value = self.progress_pred(newstate).squeeze()
#             return logit, None, instra_state, progress_value, angle_value, region_value, \
#                    next_region_value, target_region_value, c_1
#         else:
#             reward = self.critic(newstate)
#             return logit, reward, instra_state, None, None, None, \
#                    None, None, c_1

# class UniterForR2RAux(UniterPreTrainedModel):
#     """ Finetune UNITER for VQA
#     """
#     def __init__(self, config, img_dim):
#         super().__init__(config)
#         self.uniter = UniterModel(config, img_dim)
#
#         self.r2r_output = nn.Linear(config.hidden_size, 1)
#         self.progress_pred = ProgressIndicator(config.hidden_size)
#         self.angle_pred = AnglePredictor(config.hidden_size)
#         self.region_pred = RegionPredictor(config.hidden_size)
#
#         self.next_region_pred = RegionPredictor(config.hidden_size)
#         self.target_region_pred = RegionPredictor(config.hidden_size)
#
#         # self.lstm = nn.LSTMCell(config.hidden_size, config.hidden_size)
#
#         self.critic = nn.Linear(config.hidden_size, 1)
#
#         # if args.speaker_loss:
#         #     from speaker import SpeakerDecoderForUniter
#         #     self.speaker_decoder = SpeakerDecoderForUniter(tok.vocab_size(), args.wemb,
#         #                                               tok.word_to_index['<PAD>'],
#         #                                               args.rnn_dim, args.speaker_dropout, config.hidden_size)
#         self.apply(self.init_weights)
#
#
#     def forward(self, rl, input_ids=None,position_ids=None, img_feat=None,
#                     img_pos_feat=None, orien_feat=None,attn_masks=None,gather_index=None,
#                     candidate_leng=None, instr_state=None):
#
#         if position_ids.size(0) > 1:
#             position_ids = position_ids[0].unsqueeze(0)
#
#         sequence_output = self.uniter(input_ids, position_ids,
#                                       img_feat, img_pos_feat,
#                                       attn_masks, gather_index, orien_feat=orien_feat,
#                                       output_all_encoded_layers=False,
#                                       instr_state=instr_state, candi_leng=candidate_leng)
#
#         # newstate, bstart = [], 0
#         # for candi_num in candidate_leng:
#         #     tmp = sequence_output[bstart:candi_num + bstart - 1, 0, :].unsqueeze(dim=0).unsqueeze(
#         #         dim=0)  # exclude the last candidate
#         #     newstate.append(F.avg_pool2d(tmp, (candi_num - 1, 1), stride=(1, 1)).squeeze())
#         #     bstart += candi_num
#         # del tmp
#         instra_state = sequence_output[:, 0, :]  # check dims
#         # newstate = torch.stack(newstate, dim=0) # bsx768
#         # if instr_state is None:
#         #     h_0 = torch.zeros_like(c_0).cuda()
#         #     instra_state, c_1 = self.lstm(newstate, (h_0, c_0))
#         # else:
#         #     instra_state, c_1 = self.lstm(newstate,(instr_state, c_0))
#
#         pooled_output = self.uniter.pooler(sequence_output)
#
#         vln_logit = self.r2r_output(pooled_output).squeeze()
#         angle_logits = self.angle_pred(pooled_output)
#         region_logits = self.region_pred(pooled_output)
#
#         newstate, bstart = [], 0
#         for candi_num in candidate_leng:
#             tmp = pooled_output[bstart:candi_num + bstart - 1, :].unsqueeze(dim=0).unsqueeze(
#                 dim=0)  # exclude the last candidate
#             newstate.append(F.avg_pool2d(tmp, (candi_num - 1, 1), stride=(1, 1)).squeeze())
#             bstart += candi_num
#         del tmp
#
#         newstate = torch.stack(newstate, dim=0)
#         # new loss for region
#         next_region_value = self.next_region_pred(newstate)
#         target_region_value = self.target_region_pred(newstate)
#
#         # re-organize logit
#         logit = torch.zeros(len(candidate_leng), max(candidate_leng)).cuda()
#         angle_value = torch.zeros(len(candidate_leng), max(candidate_leng), 4).cuda()
#         region_value = torch.zeros(len(candidate_leng), max(candidate_leng), 31).cuda()
#
#         tmp_start = 0
#         for li, num in enumerate(candidate_leng):
#             logit[li, 0:num] = vln_logit[tmp_start:sum(candidate_leng[0:li + 1])]
#             angle_value[li, 0:num, ...] = angle_logits[tmp_start:sum(candidate_leng[0:li + 1])]
#             region_value[li, 0:num, ...] = region_logits[tmp_start:sum(candidate_leng[0:li + 1])]
#             tmp_start += num
#
#         if rl == False:
#             progress_value = self.progress_pred(newstate).squeeze()
#             return logit, None, instra_state, progress_value, angle_value, region_value, \
#                    next_region_value, target_region_value
#         else:
#             reward = self.critic(newstate)
#             return logit, reward, instra_state, None, angle_value, region_value, \
#                    next_region_value, target_region_value
#
#
#         # if rl == False:
#         #     progress_value = self.progress_pred(newstate).squeeze()
#         #     return logit, None, instra_state, progress_value, angle_value, region_value, \
#         #            next_region_value, target_region_value, c_1
#         # else:
#         #     reward = self.critic(newstate)
#         #     return logit, reward, instra_state, None, angle_value, region_value, \
#         #            next_region_value, target_region_value, c_1

# class UniterForR2RAux(UniterPreTrainedModel):
#     """ Finetune UNITER for VQA
#     """
#     def __init__(self, config, img_dim, tok, args):
#         super().__init__(config)
#         self.uniter = UniterModel(config, img_dim)
#
#         self.r2r_output = nn.Linear(config.hidden_size, 1)
#         self.progress_pred = ProgressIndicator(config.hidden_size)
#         self.angle_pred = AnglePredictor(config.hidden_size)
#         self.region_pred = RegionPredictor(config.hidden_size)
#
#         self.next_region_pred = NextRegionPredictor(config.hidden_size)
#         self.target_region_pred = NextRegionPredictor(config.hidden_size)
#
#         if config.use_lstm:
#             self.use_lstm = True
#             self.lstm = nn.LSTMCell(config.hidden_size, config.hidden_size)
#         else:
#             self.use_lstm = False
#
#         if config.use_speaker:
#             self.use_speaker = True
#             self.speaker_decoder = SpeakerDecoderForUniter(tok.vocab_size(), args.wemb, tok.word_to_index['<PAD>'],
#                                                       args.rnn_dim, args.speaker_dropout, config.hidden_size)
#         else:
#             self.use_speaker = False
#
#         self.critic = nn.Linear(config.hidden_size, 1)
#
#         self.apply(self.init_weights)
#
#
#     def forward(self, rl, input_ids=None,position_ids=None,img_feat=None,
#                     img_pos_feat=None, orien_feat=None,attn_masks=None,gather_index=None,
#                     candidate_leng=None, instr_state=None, c_0=None, step=0,
#                 speaker=False, insts=None, instr_state_ctx=None,
#                 decode_mask=None, speaker_step=False):
#
#         if speaker:
#             return self.speaker_decoder(insts, instr_state_ctx, decode_mask)
#
#         if position_ids.size(0)>1:
#             position_ids = position_ids[0].unsqueeze(0)
#
#         sequence_output = self.uniter(input_ids, position_ids,
#                                       img_feat, img_pos_feat,
#                                       attn_masks, gather_index, orien_feat=orien_feat,
#                                       output_all_encoded_layers=False,
#                                       instr_state=instr_state,candi_leng=candidate_leng, step=step)
#
#         pooled_output = self.uniter.pooler(sequence_output)
#
#         vln_logit = self.r2r_output(pooled_output).squeeze()
#         angle_logits = self.angle_pred(pooled_output)
#         region_logits = self.region_pred(pooled_output)
#
#         newstate, bstart = [], 0
#         for candi_num in candidate_leng:
#             tmp = pooled_output[bstart:candi_num + bstart - 1, :].unsqueeze(dim=0).unsqueeze(
#                 dim=0)  # exclude the last candidate
#             newstate.append(F.avg_pool2d(tmp, (candi_num - 1, 1), stride=(1, 1)).squeeze())
#             bstart += candi_num
#         del tmp
#         newstate = torch.stack(newstate, dim=0)
#         if self.use_lstm:
#             if step==0:
#                 h_0 = torch.zeros_like(c_0).cuda()
#                 instra_state, c_1 = self.lstm(newstate, (h_0, c_0))
#             else:
#                 instra_state, c_1 = self.lstm(newstate,(instr_state, c_0))
#         else:
#             instra_state = sequence_output[:, 0, :]  # check dims
#
#         # re-organize logit
#         logit = torch.zeros(len(candidate_leng), max(candidate_leng)).cuda()
#         if not speaker_step:
#             # new loss for region
#             next_region_value = self.next_region_pred(newstate)
#             target_region_value = self.target_region_pred(newstate)
#             angle_value = torch.zeros(len(candidate_leng), max(candidate_leng), 4).cuda()
#             region_value = torch.zeros(len(candidate_leng), max(candidate_leng), 31).cuda()
#
#         tmp_start = 0
#         for li, num in enumerate(candidate_leng):
#             logit[li, 0:num] = vln_logit[tmp_start:sum(candidate_leng[0:li + 1])]
#             if not speaker_step:
#                 angle_value[li, 0:num, ...] = angle_logits[tmp_start:sum(candidate_leng[0:li + 1])]
#                 region_value[li, 0:num, ...] = region_logits[tmp_start:sum(candidate_leng[0:li + 1])]
#             tmp_start += num
#
#         if speaker_step:
#             return logit, instra_state, c_1
#
#         if rl == False:
#             progress_value = self.progress_pred(newstate).squeeze()
#             return logit, None, instra_state, progress_value, angle_value, region_value, \
#                    next_region_value, target_region_value, c_1
#         else:
#             reward = self.critic(newstate)
#             return logit, reward, instra_state, None, angle_value, region_value, \
#                    next_region_value, target_region_value, c_1

class UniterForR2RAuxSpeaker(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, speaker_decoder=None):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)

        self.r2r_output = nn.Linear(config.hidden_size, 1)
        self.progress_pred = ProgressIndicator(config.hidden_size)
        self.angle_pred = AnglePredictor(config.hidden_size)
        self.region_pred = RegionPredictor(config.hidden_size)

        self.next_region_pred = RegionPredictor(config.hidden_size)
        self.target_region_pred = RegionPredictor(config.hidden_size)

        self.critic = nn.Linear(config.hidden_size, 1)
        self.speaker_decoder = speaker_decoder

        self.apply(self.init_weights)

    def infer_instr(self, insts, instr_state_ctx, decode_mask):
        return self.speaker_decoder(insts, instr_state_ctx, decode_mask)

    def forward(self, rl, input_ids=None,position_ids=None,img_feat=None,
                    img_pos_feat=None, orien_feat=None,attn_masks=None,gather_index=None,
                    candidate_leng=None, instr_state=None, ):

        if position_ids.size(0)>1:
            position_ids = position_ids[0].unsqueeze(0)

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index, orien_feat=orien_feat,
                                      output_all_encoded_layers=False,
                                      instr_state=instr_state,candi_leng=candidate_leng)
        instra_state = sequence_output[:,0,:] # check dims
        pooled_output = self.uniter.pooler(sequence_output)

        vln_logit = self.r2r_output(pooled_output).squeeze()
        angle_logits = self.angle_pred(pooled_output)
        region_logits = self.region_pred(pooled_output)

        newstate, bstart = [], 0
        for candi_num in candidate_leng:
            tmp = pooled_output[bstart:candi_num + bstart - 1, :].unsqueeze(dim=0).unsqueeze(
                dim=0)  # exclude the last candidate
            newstate.append(F.avg_pool2d(tmp, (candi_num - 1, 1), stride=(1, 1)).squeeze())
            bstart += candi_num
        del tmp
        newstate = torch.stack(newstate, dim=0)
        # new loss for region
        next_region_value = self.next_region_pred(newstate)
        target_region_value = self.target_region_pred(newstate)

        # re-organize logit
        logit = torch.zeros(len(candidate_leng), max(candidate_leng)).cuda()
        angle_value = torch.zeros(len(candidate_leng), max(candidate_leng), 4).cuda()
        region_value = torch.zeros(len(candidate_leng), max(candidate_leng), 31).cuda()

        tmp_start = 0
        for li, num in enumerate(candidate_leng):
            logit[li, 0:num] = vln_logit[tmp_start:sum(candidate_leng[0:li + 1])]
            angle_value[li, 0:num, ...] = angle_logits[tmp_start:sum(candidate_leng[0:li + 1])]
            region_value[li, 0:num, ...] = region_logits[tmp_start:sum(candidate_leng[0:li + 1])]
            tmp_start += num

        if rl == False:
            progress_value = self.progress_pred(newstate).squeeze()
            return logit, None, instra_state, progress_value, angle_value, region_value, next_region_value, target_region_value
        else:
            reward = self.critic(newstate)
            return logit, reward, instra_state, None, angle_value, region_value, next_region_value, target_region_value


# class UniterForR2RAuxSpeaker(UniterPreTrainedModel):
#     """ Finetune UNITER for VQA
#     """
#     def __init__(self, config, img_dim, speaker_decoder=None):
#         super().__init__(config)
#         self.uniter = UniterModel(config, img_dim)
#
#         self.r2r_output = nn.Linear(config.hidden_size, 1)
#
#         self.progress_pred = ProgressIndicator(config.hidden_size)
#         self.angle_pred = AnglePredictor(config.hidden_size)
#         self.region_pred = RegionPredictor(config.hidden_size)
#
#         self.critic = nn.Linear(config.hidden_size, 1)
#
#         self.speaker_decoder = speaker_decoder
#
#         self.apply(self.init_weights)
#
#     def forward(self, rl, input_ids=None,position_ids=None,img_feat=None,
#                     img_pos_feat=None, orien_feat=None,attn_masks=None,gather_index=None,
#                     candidate_leng=None, instr_state=None):
#
#         if position_ids.size(0)>1:
#             position_ids = position_ids[0].unsqueeze(0)
#
#         sequence_output = self.uniter(input_ids, position_ids,
#                                       img_feat, img_pos_feat,
#                                       attn_masks, gather_index, orien_feat=orien_feat,
#                                       output_all_encoded_layers=False,
#                                       instr_state=instr_state,candi_leng=candidate_leng)
#         instra_state = sequence_output[:,0,:] # check dims
#         pooled_output = self.uniter.pooler(sequence_output)
#         vln_logit = self.r2r_output(pooled_output).squeeze()
#         angle_logits = self.angle_pred(pooled_output)
#         region_logits = self.region_pred(pooled_output)
#
#         # re-organize logit
#         logit = torch.zeros(len(candidate_leng), max(candidate_leng)).cuda()
#         angle_value = torch.zeros(len(candidate_leng), max(candidate_leng), 4).cuda()
#         region_value = torch.zeros(len(candidate_leng), max(candidate_leng), 31).cuda()
#
#         if rl == False:
#             progress_logits = self.progress_pred(pooled_output).squeeze()
#             progress_value = torch.zeros(len(candidate_leng), max(candidate_leng)).cuda()
#
#         tmp_start = 0
#         for li, num in enumerate(candidate_leng):
#             logit[li, 0:num] = vln_logit[tmp_start:sum(candidate_leng[0:li + 1])]
#             angle_value[li, 0:num, ...] = angle_logits[tmp_start:sum(candidate_leng[0:li + 1])]
#             region_value[li, 0:num, ...] = region_logits[tmp_start:sum(candidate_leng[0:li + 1])]
#             if rl == False:
#                 progress_value[li, 0:num] = progress_logits[tmp_start:sum(candidate_leng[0:li + 1])]
#
#             tmp_start += num
#
#         if rl == False:
#             # return logit, None, instra_state
#             # return logit, None, instra_state, progress_value, angle_value, region_value
#             return logit, None, instra_state, progress_value, angle_value, region_value
#         else:
#             # re-organize state
#             newstate, bstart = [], 0
#             for candi_num in candidate_leng:
#                 tmp = pooled_output[bstart:candi_num + bstart - 1, :].unsqueeze(dim=0).unsqueeze(
#                     dim=0)  # exclude the last candidate
#                 newstate.append(F.avg_pool2d(tmp, (candi_num - 1, 1), stride=(1, 1)).squeeze())
#                 bstart += candi_num
#             del tmp
#             newstate = torch.stack(newstate, dim=0)
#             reward = self.critic(newstate)
#             # return logit, reward, instra_state
#             return logit, reward, instra_state, None, angle_value, region_value


