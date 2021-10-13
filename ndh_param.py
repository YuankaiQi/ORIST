import argparse
import os
from os.path import abspath, dirname, exists, join
from utils.misc import parse_with_config

class BertParam:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--name', type=str, required=True)
        self.parser.add_argument('--project', type=str, required=True)
        self.parser.add_argument('--task', type=str, required=True)
        self.parser.add_argument('--resume', type=str, default=None)
        self.parser.add_argument('--log_every', type=int, default=-1)
        self.parser.add_argument("--eval_only", action="store_const", default=False, const=True)

        self.parser.add_argument('--local_rank', type=int, default=0)
        self.parser.add_argument('--print_step', action="store_const", default=False, const=True)
        self.parser.add_argument('--n_gpu', type=int, default=8, required=True)
        self.parser.add_argument('--num_train_epochs', type=int, default=400)
        self.parser.add_argument('--train', type=str, required=True, help='listner, speaker ...')
        self.parser.add_argument('--apex_level', type=str, default='O2')
        self.parser.add_argument('--fp16_tech', type=str, default='apex', help="apex, native")
        self.parser.add_argument('--parallel_tech', type=str, default=None, help="DataParallel, Distributed")
        self.parser.add_argument('--debug', action="store_const", default=False, const=True)
        self.parser.add_argument('--verbose', action="store_const", default=False, const=True)
        self.parser.add_argument('--fast_train', action="store_const", default=False, const=True)
        self.parser.add_argument('--candi_region_feat_cache_dir', type=str,
                                 default='/path/to/candidates_all_gt_bbox_feat_on_all_images_caffe/')
        self.parser.add_argument('--candi_whole_feat_cache_dir', type=str,
                                 default='/path/to/candidates_whole_img_feat_at_all_viewpoints_caffe/')
        self.parser.add_argument("--aug_path_cahche", type=str, default='aug_path_cache.json')
        # add for using pretrained model
        self.parser.add_argument('--use_angle_pi', dest='use_angle_pi', action="store_const", default=True, const=True)
        self.parser.add_argument('--use_angle_0', dest='use_angle_pi', action="store_const", const=False)

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=10, help='Max Action sequence')
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        self.parser.add_argument("--load", type=str, default=None)

        # for speaker
        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument('--wemb', type=int, default=256)
        self.parser.add_argument("--bidir", type=bool, default=True)
        self.parser.add_argument("--speaker_dropout", default=0.5, type=float,
                                 help="tune dropout regularization")
        self.parser.add_argument("--speaker_loss", action='store_const', default=False, const=True)
        self.parser.add_argument('--speaker_featdropout', type=float, default=0.4)
        # for region gt
        self.parser.add_argument('--region_cls_gt', type=str, default='matterport_utils/candidate_region_gt.json')
        self.parser.add_argument('--house_pano_info', type=str,
                                 default='matterport_utils/house_panos_gt.json')
        self.parser.add_argument("--add_whole_img_feat", action='store_const', default=False, const=True)
        self.parser.add_argument("--resume_optimizer", action='store_const', default=False, const=True)


        # More Paths from
        self.parser.add_argument("--aug", default=None)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--features", type=str, default='none')
        self.parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)

        # SSL configuration
        self.parser.add_argument("--selfTrain", dest='self_train', action='store_const', default=False, const=True)

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument("--paramSearch", dest='param_search', action='store_const', default=False, const=True)
        self.parser.add_argument("--submit", action='store_const', default=False, const=True)
        self.parser.add_argument("--beam", action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        # Training Configurations
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument("--candidate", dest="candidate_mask",
                                 action="store_const", default=False, const=True)
        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)
        self.parser.add_argument("--drop_region_feat", action='store_const', default=False, const=True)
        self.parser.add_argument("--region_drop_p", default=0.3, type=float)
        self.parser.add_argument("--progress_loss", action='store_const', default=False, const=True)
        self.parser.add_argument("--angle_loss", action='store_const', default=False, const=True)
        self.parser.add_argument("--candi_region_loss", action='store_const', default=False, const=True)
        self.parser.add_argument("--next_region_loss", action='store_const', default=False, const=True)
        self.parser.add_argument("--target_region_loss", action='store_const', default=False, const=True)

        self.parser.add_argument("--no_history_state", action='store_const', default=False, const=True)
        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')
        self.parser.add_argument('--use_lstm', dest='use_lstm', action="store_const", default=False, const=True)

        

        # Required parameters
        self.parser.add_argument("--model_config",
                            default=None, type=str,
                            help="json file for model architecture")
        self.parser.add_argument("--pretrained_model",
                            default=None, type=str,
                            help="pretrained model")

        # self.parser.add_argument(
        #     "--output_dir", default=None, type=str,
        #     help="The output directory where the model checkpoints will be "
        #          "written.")

        # Prepro parameters
        self.parser.add_argument('--max_txt_len', type=int, default=80,
                            help='max number of tokens in text (BERT BPE)')
        self.parser.add_argument('--conf_th', type=float, default=0.2,
                            help='threshold for dynamic bounding boxes '
                                 '(-1 for fixed)')
        self.parser.add_argument('--max_bb', type=int, default=30,
                            help='max number of bounding boxes')
        self.parser.add_argument('--min_bb', type=int, default=30,
                            help='min number of bounding boxes')
        self.parser.add_argument('--num_bb', type=int, default=30,
                            help='static number of bounding boxes')

        # training parameters
        self.parser.add_argument("--train_batch_size", default=2, type=int,
                            help="Total batch size for training. "
                                 "(batch by tokens)")
        self.parser.add_argument("--val_batch_size", default=1, type=int,
                            help="Total batch size for validation. "
                                 "(batch by tokens)")
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=-1,
                            help="Number of updates steps to accumualte before "
                                 "performing a backward/update pass.")
        self.parser.add_argument("--learning_rate", default=1e-5, type=float,
                            help="The initial learning rate for Adam.")
        self.parser.add_argument("--lr_mul", default=10.0, type=float,
                            help="multiplier for top layer lr")
        self.parser.add_argument("--optim", default='adam',
                            choices=['adam', 'adamax', 'adamw'],
                            help="optimizer")
        self.parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                            help="beta for adam optimizer")
        self.parser.add_argument("--dropout", default=0.1, type=float,
                            help="tune dropout regularization")
        self.parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="weight decay (L2) regularization")
        self.parser.add_argument("--grad_norm", default=2.0, type=float,
                            help="gradient clipping (-1 for no clipping)")
        self.parser.add_argument("--warmup_proportion", default=0.01, type=float,
                            help="Number of training steps to perform linear "
                                 "learning rate warmup for. (invsqrt decay)")

        # device parameters
        self.parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")
        self.parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit float precision instead "
                                 "of 32-bit")
        self.parser.add_argument('--n_workers', type=int, default=0,
                            help="number of data workers")
        self.parser.add_argument('--pin_mem', action='store_true', help="pin memory")
        # agent
        self.parser.add_argument(
            "--stop_feat", default="looking_to_target_vp", type=str,
            help="looking_to_target_vp, or zeros"
        )

        # can use config files
        self.parser.add_argument('--config', help='JSON config files')
        # for ndh
        self.parser.add_argument("--speaker_angleFeatSize", dest="speaker_angle_feat_size", type=int, default=128)
        self.parser.add_argument('--path_type', type=str, required=True, default='mixed', help='oracle, navigator, mixed')
        self.parser.add_argument("--mask_obj", action="store_const", default=False, const=True)
        # for multi loss
        self.parser.add_argument("--next_region_pred_w", default=0.2, type=float)
        self.parser.add_argument("--target_region_pred_w", default=0.2, type=float)
        self.args = parse_with_config(self.parser)

        # if exists(self.args.output_dir) and os.listdir(self.args.output_dir):
        #     print("Output directory ({}) already exists and is not "
        #                      "empty.".format(self.args.output_dir))

        # options safe guard
        if self.args.conf_th == -1:
            assert self.args.max_bb + self.args.max_txt_len + 2 <= 512
        else:
            assert self.args.num_bb + self.args.max_txt_len + 2 <= 512\


param = BertParam()
args = param.args
args.TRAIN_VOCAB = f'tasks/{args.task}/data/train_vocab.txt'
args.TRAINVAL_VOCAB = f'tasks/{args.task}/data/trainval_vocab.txt'

args.IMAGENET_FEATURES = f'img_features/ResNet-152-imagenet.h5'

