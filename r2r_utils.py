''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re

import string
import json
import numpy as np
import time
import math
import glob
import h5py
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import torch
# from bert_param import args
import collections
import torch.distributed as dist
from pytorch_pretrained_bert import BertTokenizer,WordpieceTokenizer
# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')

class VLNBertTokenizer(BertTokenizer):
    #never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
    def add_tokens(self,tokens, do_lower_case=False, return_new_word=False):
        maxIndex = len(self.vocab)-1
        added,added_word = 0, []
        for token in tokens:
            if do_lower_case:
                token = token.lower()
            if token in ['<PAD>', '<UNK>', '<SEP>','[SEP]','[sep]', '<EOS>', '<BOS>', '<pad>','<unk>','<eos>','<bos>','<sep>']:
                continue
            elif token not in self.vocab:
                added_word.append(token)
                self.vocab[token] = maxIndex+1
                maxIndex += 1
                added += 1
        # update
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        print('added %d word(s) into bert\'s vocab'%added)
        if return_new_word:
            return added_word


def create_links(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

def reduce_tensor(tensor,dst=0):
    rt = tensor.clone()
    dist.reduce(rt, dst, op=dist.ReduceOp.SUM)
    return rt

def check_param_across_gpus(model,n_gpu):
      for n, p in model.named_parameters():
          if p.requires_grad:
              # print('p.data type %s'%type(p.data))
              rt = [torch.zeros_like(p.data).cuda() for i in range(n_gpu)]
              dist.all_gather(rt,p.data.clone())
              for i in range(n_gpu-1):
                  if torch.sum(rt[0]-rt[i+1])!=0:
                      print('%s has different params on gpu 0 and gpu %d'%(n,i+1))

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    # print(scans)
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3], 
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits,task):
    """
    :param splits: A list of split.
        if the split is "something@5000", it will use a random 5000 data from the data
    :return:
    """
    import random
    data = []
    old_state = random.getstate()
    for split in splits:
        # It only needs some part of the dataset?
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json
        # if split in ['train', 'val_seen', 'val_unseen', 'test',
        #              'val_unseen_half1', 'val_unseen_half2', 'val_seen_half1', 'val_seen_half2']:       # Add two halves for sanity check
        if "/" not in split:
            with open(f'tasks/{task}/data/{task}_%s.json' % split,'r') as f:
                new_data = json.load(f)
        else:
            with open(split) as f:
                new_data = json.load(f)

        # Partition
        if number > 0:
            random.seed(0)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data

def update_bert_tokenizer(tokenizer, vocab_r2r):
# update vocab
    # check: special token such as pad, known, cls[101] sep[102] should not be used
    # ['[CLS]':101, '[SEP]':102, '[PAD]':0, '[MASK]':103, '[UNK]':100]
    # exclude task id:1-20
    unused_keys = [uk for uk in list(tokenizer.vocab.keys()) if uk.startswith('[unused')]
    # unused_keys = unused_keys[20:]  # exclude keys mapping to task id:1-20
    added = 0
    for new_key in vocab_r2r:
        if new_key in ['<PAD>','<UNK>','<EOS>']:
            continue
        if new_key in tokenizer.vocab:
            continue
        if len(unused_keys) == 0:
            break
        else:
            unused_key = unused_keys[-1]
            old_value = tokenizer.vocab.pop(unused_key)
            tokenizer.vocab[new_key] = old_value
            unused_keys = unused_keys[:-1]
            added += 1
    print('added %d new vocab'%added)
    return tokenizer


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character
  
    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        self.index_to_word = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
            new_w2i = defaultdict(lambda: self.word_to_index['<UNK>'])
            new_w2i.update(self.word_to_index)
            self.word_to_index = new_w2i
            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key
        old = self.vocab_size()
        self.add_word('<BOS>')
        assert self.vocab_size() == old+1
        print("OLD_VOCAB_SIZE", old)
        print("VOCAB_SIZE", self.vocab_size())
        if vocab is not None:
            print("VOACB", len(vocab))

    def finalize(self):
        """
        This is used for debug
        """
        self.word_to_index = dict(self.word_to_index)   # To avoid using mis-typing tokens

    def add_word(self, word):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()    # vocab_size() is the
        self.index_to_word[self.vocab_size()] = word

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def vocab_size(self):
        return len(self.index_to_word)

    def encode_sentence(self, sentence, max_length=None):
        if max_length is None:
            max_length = self.encoding_length
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = [self.word_to_index['<BOS>']]
        for word in self.split_sentence(sentence):
            encoding.append(self.word_to_index[word])   # Default Dict
        encoding.append(self.word_to_index['<EOS>'])

        if len(encoding) <= 2:
            return None
        #assert len(encoding) > 2

        if len(encoding) < max_length:
            encoding += [self.word_to_index['<PAD>']] * (max_length-len(encoding))  # Padding
        elif len(encoding) > max_length:
            encoding[max_length - 1] = self.word_to_index['<EOS>']                  # Cut the length with EOS

        return np.array(encoding[:max_length])

    def decode_sentence(self, encoding, length=None):
        sentence = []
        if length is not None:
            encoding = encoding[:length]
        for ix in encoding:
            if ix == self.word_to_index['<PAD>'] or ix == self.word_to_index['<EOS>']:
                break
            else:
                sentence.append(self.index_to_word[ix])
        return " ".join(sentence)

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
        end = np.argmax(np.array(inst) == self.word_to_index['<EOS>'])     # If no <EOS>, return empty string
        if len(inst) > 1 and inst[0] == self.word_to_index['<BOS>']:
            start = 1
        else:
            start = 0
        return inst[start: end]

# add task prefix
def build_vocab_from_file(files, min_count=5, do_lower_case=True):
    count = Counter()
    t = Tokenizer()
    for file in files:
        with open(file,'r') as f:
            data = json.load(f)

        for item in data:
            for _, instr in item.items():
                if do_lower_case: # In fact, do_lower_case does not work because 'split_sentence' always lower()
                    count.update(t.split_sentence(instr.lower()))
                else:
                    count.update(t.split_sentence(instr))
    vocab = []
    for word, num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    # print(vocab)
    return vocab

def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab, task=None):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits, task)

    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    # print(vocab)
    return vocab

# def write_bert_vocab(vocab, path):
#     print('Writing vocab of size %d to %s' % (len(vocab),path))
#     with open(path, 'w') as f:
#         for word, _ in vocab.items():
#             f.write("%s\n" % word)

def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, start_iter, iter, n_iters):
    now = time.time()
    s = now - since
    es = s / (iter-start_iter)*n_iters
    rs = es - s
    return 'run %s, still need %s' % (asMinutes(s), asMinutes(rs))

def read_img_features_h5(feature_store, args):
    import h5py
    args.views = 36
    # logging.info("Read from " + feature_store)
    features = {}
    f = h5py.File(feature_store, 'r')
    for k, v in f.items():
        features[k] = v[:]
    return features

def read_img_features(feature_store, args):
    import csv
    import base64
    from tqdm import tqdm

    print("Start loading the image feature")
    start = time.time()

    if "detectfeat" in args.features:
        views = int(args.features[10:])
    else:
        views = 36

    args.views = views

    tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
    features = {}
    with open(feature_store, "r") as tsv_in_file:     # Open the tsv file.
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
        for item in reader:
            long_id = item['scanId'] + "_" + item['viewpointId']
            features[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                                   dtype=np.float32).reshape((views, -1))   # Feature of long_id is (36, 2048)

    print("Finish Loading the image feature from %s in %0.4f seconds" % (feature_store, time.time() - start))
    return features

def read_candidates(candidates_store):
    import csv
    import base64
    from collections import defaultdict
    print("Start loading the candidate feature")

    start = time.time()

    TSV_FIELDNAMES = ['scanId', 'viewpointId', 'heading', 'elevation', 'next', 'pointId', 'idx', 'feature']
    candidates = defaultdict(lambda: list())
    items = 0
    with open(candidates_store, "r") as tsv_in_file:     # Open the tsv file.
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=TSV_FIELDNAMES)
        for item in reader:
            long_id = item['scanId'] + "_" + item['viewpointId']
            candidates[long_id].append(
                {'heading': float(item['heading']),
                 'elevation': float(item['elevation']),
                 'scanId': item['scanId'],
                 'viewpointId': item['next'],
                 'pointId': int(item['pointId']),
                 'idx': int(item['idx']) + 1,   # Because a bug in the precompute code, here +1 is important
                 'feature': np.frombuffer(
                     base64.decodestring(item['feature'].encode('ascii')),
                     dtype=np.float32)
                    }
            )
            items += 1

    for long_id in candidates:
        assert (len(candidates[long_id])) != 0

    assert sum(len(candidate) for candidate in candidates.values()) == items

    # candidate = candidates[long_id]
    # print(candidate)
    print("Finish Loading the candidates from %s in %0.4f seconds" % (candidates_store, time.time() - start))
    candidates = dict(candidates)
    return candidates

def add_exploration(paths):
    explore = json.load(open("tasks/R2R/data/exploration.json", 'r'))
    inst2explore = {path['instr_id']: path['trajectory'] for path in explore}
    for path in paths:
        path['trajectory'] = inst2explore[path['instr_id']] + path['trajectory']
    return paths

def angle_feature(heading, elevation, angle_feat_size):
    import math
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
                    dtype=np.float32)

def new_simulator():
    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60
    sys.path.append('build')
    import MatterSim
    sim = MatterSim.Simulator()
    sim.setNavGraphPath('/path/to/connectivity') 
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.init()

    return sim

def get_point_angle_feature(baseViewId=0, angle_feat_size=None):
    sim = new_simulator()

    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            sim.newEpisode('ZMojNkEp431', '2f4d90acd4024c269fb0efe49a8ac540', 0, math.radians(-30))
        elif ix % 12 == 0:
            sim.makeAction(0, 1.0, 1.0)
        else:
            sim.makeAction(0, 1.0, 0)

        state = sim.getState()
        assert state.viewIndex == ix

        heading = state.heading - base_heading

        feature[ix, :] = angle_feature(heading, state.elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(angle_feat_size):
    return [get_point_angle_feature(baseViewId,angle_feat_size) for baseViewId in range(36)]

# def read_img_features_h5(feature_store, args):
#
#     print('Read ' + feature_store)
#     start = time.time()
#     features = {}
#     f = h5py.File(feature_store, 'r')
#     for k, v in f.items():
#         features[k] = v[:]
#     f.close()
#     print("Finish Loading the h5 from %s in %0.4f seconds" % (feature_store, time.time() - start))
#     return features

def read_aug_path_cache(aug_path_cache):
    with open(aug_path_cache, 'r') as f:
        data = json.load(f)
    return data

def read_h5_cache(folder, level):
    files = glob.glob(folder+'*.h5')
    data = {}
    candi_vp_2_ids = {}
    print('loading candidate feature from %s'%folder)
    if level == 2:
        for file in tqdm(files):
            key, value, candi_vp_2_id = read_h5_group(file, level)
            data[key] = value
            candi_vp_2_ids[key] = candi_vp_2_id

        return data, candi_vp_2_ids
    elif level == 1: # read region features
        for file in tqdm(files):
            key, value = read_h5_group(file, level)
            # x,y,x,y,w*h -> x,y,x,y,w,h,w*h
            value[1] = np.concatenate((value[1][:,:4],value[1][:,2:4]-value[1][:,:2],value[1][:,4:]),axis=1)

            data[key] = value

        return data

def read_h5_vp_feat(feat_path):
    files = glob.glob(feat_path+'*.h5')
    results = {}
    print('loading vp feature from %s' % feat_path)
    for file in tqdm(files):
        f = h5py.File(file, 'r')
        scanVP = file.split('.')[0].split('/')[-1]
        bbox = f['bbox'][()]
        # x,y,x,y -> x,y,x,y,w,h,w*h
        img_w, img_h = f['image_w'][()], f['image_h'][()]
        bbox[:,0], bbox[:,2] = bbox[:,0]/img_w, bbox[:,2]/img_w
        bbox[:,1], bbox[:,3] = bbox[:,1]/img_h, bbox[:,3]/img_h
        newbbox = np.zeros((bbox.shape[0],7),dtype=np.float32)
        newbbox[:,:4] = bbox
        newbbox[:,4:6] = bbox[:,2:4]-bbox[:,:2]
        newbbox[:, 6] = newbbox[:,4]*newbbox[:,5]
        results[scanVP]=(f['features'][()], newbbox, f['num_boxes'][()])

    return results


def read_h5_group(fn, group_level):
    f = h5py.File(fn, 'r')
    if group_level == 1:
        result = []
        for tpk in f.keys(): # only 1 key
            result = [f[tpk]['region_feat'][()], f[tpk]['region_loc'][()],
                      f[tpk]['region_num'][()].item(),f[tpk]['region_label'][()]]
            break
        f.close()
        return tpk, result

    elif group_level == 2:
        result = []
        candi_vp_2_id = {}
        for tpk in f.keys():
            for i, seck in enumerate(f[tpk].keys()):
                result.append({ thdk:f[tpk][seck][thdk][()].item() for thdk in ['normalized_heading', 'elevation',
                        'pointId', 'idx']})
                result[-1]['scanId'] = f[tpk][seck]['scanId'][()]
                result[-1]['viewpointId'] = f[tpk][seck]['viewpointId'][()]
                candi_vp_2_id[f[tpk][seck]['viewpointId'][()]] = i
            break
        f.close()

        return tpk, result, candi_vp_2_id
    else:
        f.close()
        assert 1==0, 'do not support deeper groups'


def add_idx(inst):
    toks = Tokenizer.split_sentence(inst)
    return " ".join([str(idx)+tok for idx, tok in enumerate(toks)])

import signal
class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

from collections import OrderedDict

class Timer:
    def __init__(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def reset(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def tic(self, key):
        self.start[key] = time.time()

    def toc(self, key):
        delta = time.time() - self.start[key]
        if key not in self.cul:
            self.cul[key] = delta
        else:
            self.cul[key] += delta

    def step(self):
        self.iter += 1

    def show(self):
        total = sum(self.cul.values())
        for key in self.cul:
            print("%s, total time %0.2f, avg time %0.2f, part of %0.2f" %
                  (key, self.cul[key], self.cul[key]*1./self.iter, self.cul[key]*1./total))
        print(total / self.iter)


stop_word_list = [
    ",", ".", "and", "?", "!"
]


def stop_words_location(inst, mask=False):
    toks = Tokenizer.split_sentence(inst)
    sws = [i for i, tok in enumerate(toks) if tok in stop_word_list]        # The index of the stop words
    if len(sws) == 0 or sws[-1] != (len(toks)-1):     # Add the index of the last token
        sws.append(len(toks)-1)
    sws = [x for x, y in zip(sws[:-1], sws[1:]) if x+1 != y] + [sws[-1]]    # Filter the adjacent stop word
    sws_mask = np.ones(len(toks), np.int32)         # Create the mask
    sws_mask[sws] = 0
    return sws_mask if mask else sws

def get_segments(inst, mask=False):
    toks = Tokenizer.split_sentence(inst)
    sws = [i for i, tok in enumerate(toks) if tok in stop_word_list]        # The index of the stop words
    sws = [-1] + sws + [len(toks)]      # Add the <start> and <end> positions
    segments = [toks[sws[i]+1:sws[i+1]] for i in range(len(sws)-1)]       # Slice the segments from the tokens
    segments = list(filter(lambda x: len(x)>0, segments))     # remove the consecutive stop words
    return segments

def clever_pad_sequence(sequences, batch_first=True, padding_value=0):
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    max_len = max(seq.size()[0] for seq in sequences)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    if padding_value is not None:
        out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

import torch
def length2mask(length, size=None, device=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).to(device)
    return mask

def average_length(path2inst):
    length = []

    for name in path2inst:
        datum = path2inst[name]
        length.append(len(datum))
    return sum(length) / len(length)

def tile_batch(tensor, multiplier):
    _, *s = tensor.size()
    tensor = tensor.unsqueeze(1).expand(-1, multiplier, *(-1,) * len(s)).contiguous().view(-1, *s)
    return tensor

def viewpoint_drop_mask(viewpoint, seed=None, drop_func=None):
    local_seed = hash(viewpoint) ^ seed
    torch.random.manual_seed(local_seed)
    drop_mask = drop_func(torch.ones(2048).cuda())
    return drop_mask


class FloydGraph:
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)




