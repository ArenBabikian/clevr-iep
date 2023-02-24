# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import os
from sys import exit

import torch
from torch.autograd import Variable
import numpy as np
import h5py
from tqdm import tqdm
sys.path.append('.')
import iep.utils as utils
from iep.data import _dataset_to_tensor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--models_dir', type=str, default=None)
parser.add_argument('--use_gpu', default=1, type=int)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None) #  TEMP?

# For running on a single example
parser.add_argument('--question', default=None)#  TEMP?
parser.add_argument('--root_dir', type=str, default='../data/CLEVR_v1.0/images')#  TEMP?
parser.add_argument('--image_h5', type=str)#  TEMP?
parser.add_argument('--cnn_model', default='resnet101')#  TEMP?
parser.add_argument('--cnn_model_stage', default=3, type=int)#  TEMP?
parser.add_argument('--image_width', default=224, type=int)#  TEMP?
parser.add_argument('--image_height', default=224, type=int)#  TEMP?

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)

parser.add_argument('--questions_file', type=str, default=None)
parser.add_argument('--questions_interval', type=int, default=1)
parser.add_argument('--num_images', type=int, default=3000)
parser.add_argument('--image_features', type=str, default=None)
parser.add_argument('--only_extract_stem_feats', default=False, action='store_true')

# If this is passed, then save all predictions to this file
parser.add_argument('--output_dir', default=None)

models = {
    '9k': ('program_generator_9k.pt', 'execution_engine_9k.pt'),
    '18k': ('program_generator_18k.pt', 'execution_engine_18k.pt'),
     '700k_strong': ('program_generator_700k.pt', 'execution_engine_700k_strong.pt'),
    # 'lstm': 'lstm.pt',
    # 'cnn_lstm': 'cnn_lstm.pt'#,
    # 'cnn_lstm_sa': 'cnn_lstm_sa.pt',
    # 'cnn_lstm_sa_mlp': 'cnn_lstm_sa_mlp.pt'
}

def main(args):
  print()
  models_path = Path(args.models_dir)
  Path(args.output_dir).mkdir(exist_ok=True, parents=True)
  # results = {}
  
  for name, model_name in tqdm(models.items()):
    model = None
    vocab_path = None
    if type(model_name) is tuple:
      print('Loading program generator from ', model_name[0])
      program_generator, _ = utils.load_program_generator(models_path / model_name[0])
      print('Loading execution engine from ', model_name[1])
      execution_engine, _ = utils.load_execution_engine(models_path / model_name[1], verbose=False)
      if args.vocab_json is not None:
        new_vocab = utils.load_vocab(args.vocab_json)
        program_generator.expand_encoder_vocab(new_vocab['question_token_to_idx'])
      model = (program_generator, execution_engine)
      vocab_path = models_path / model_name[0]
    else:
      print('Loading baseline model from ', model_name)
      model, _ = utils.load_baseline(models_path / model_name)
      vocab_path = models_path / model_name
      if args.vocab_json is not None:
        new_vocab = utils.load_vocab(args.vocab_json)
        model.rnn.expand_vocab(new_vocab['question_token_to_idx'])

    #PRINT EMBEDDED VOCAB
    # import json
    # with open(f'/mnt/c/git/clevr-iep/{name}-vocab.json', 'w') as f:
    #   json.dump(load_vocab(vocab_path), f)
    # continue

    # TODO
    # run for a single model, given as cmd line arg, instead of a dictionary in this file

    # print(load_vocab(vocab_path)['answer_token_to_idx'])
    # exit()
    q_ids, scores, results = run_raw_images(args, model, vocab_path)

    # Save the derived data
    _split = os.path.splitext(args.questions_file)[-2].split('_')
    # save_path = f'{args.output_dir}/{name}_{_split[-2]}_{_split[-1]}__{args.questions_interval}.h5'
    # save_path = f'{args.output_dir}/{name}_{_split[-2]}_{args.num_images}__{args.questions_interval}.h5'
    save_path = f'{args.output_dir}/{name}_{q_ids[0]}_{args.num_images}__{args.questions_interval}.h5'
    with h5py.File(save_path, 'w') as f:
      f.create_dataset('question_ids', data=np.asarray(q_ids))
      f.create_dataset('scores', data=np.asarray(scores))
      f.create_dataset('results', data=np.asarray(results))


def load_vocab(path):
  return utils.load_cpu(path)['vocab']


def run_raw_images(args, model, vocab_path):
  dtype = torch.FloatTensor
  if args.use_gpu == 1:
    dtype = torch.cuda.FloatTensor

  print('Loading image features from ', args.image_features)
  f_img = h5py.File(args.image_features, 'r')
  # image_features = torch.FloatTensor(f['features'])
  arr = np.asarray(f_img['features'], dtype=np.float32)
  image_features = torch.tensor(arr).type(dtype)

  if args.only_extract_stem_feats:
    # (*) Run the model
    scores = None
    if type(model) is tuple:
      _, execution_engine = model
      execution_engine.type(dtype)
    else:
      exit(1)
      # UNHANDLED FOR NOW

    # (*) Set save path
    path = Path(args.image_features)
    grandparent = path.parent.parent.absolute()
    feats_stem_path = grandparent / 'stem-feats' / path.name
    if not os.path.isdir(os.path.dirname(feats_stem_path)):
        os.makedirs(os.path.dirname(feats_stem_path))
    
    # TODO no GPU support for now
    print('Saving image stem features at ', feats_stem_path)
    with h5py.File(feats_stem_path, 'w') as f:
      feat_dset = None
      for f_i, feats in tqdm(enumerate(image_features)):
        # (*) Get image features
        feats_var = feats.unsqueeze(0)
        feats_stem = execution_engine.stem(feats_var)
        if feat_dset is None:
          N = len(image_features)
          _, C, H, W = feats_stem.shape
          feat_dset = f.create_dataset('features', (N, C, H, W),
                                       dtype=np.float32)

        feat_dset[f_i] = feats_stem.cpu().detach().numpy()
    exit()

  print('Loading question from ', args.questions_file)
  f_q = h5py.File(args.questions_file, 'r')
  questions = _dataset_to_tensor(f_q['questions'])
  num_questions_in_dataset = len(questions)
  
  image_ids = _dataset_to_tensor(f_q['image_idxs'])
  num_questions_from_num_images = 0
  for id in image_ids:
    if id < args.num_images:
      num_questions_from_num_images += 1
  num_questions = min(num_questions_from_num_images, num_questions_in_dataset)
  question_ids = _dataset_to_tensor(f_q['orig_idxs'])

  # LOOP OVER IMAGES
  q_ids = []
  all_scores = []
  predicted = []
  counter = 0
  for i in tqdm(range(0, num_questions, args.questions_interval)):
    # Add counter if tqdm is not working
    counter+=1
    # if counter % 300 == 0 :
    #   print(f'{int(counter/300)}-', end='')
    # (*) Get image features
    img_id = image_ids[i]
    feats_var = image_features[img_id]
    feats_var = feats_var.unsqueeze(0)

    # (*) Get tokenized question
    question_encoded = questions[i]

    # Remove traoiling zeros (padding)
    # ind = len(question_encoded)
    # while question_encoded[ind-1] == 0:
    #   ind = ind - 1
    # question_encoded = question_encoded[:ind]

    question_encoded = torch.LongTensor(question_encoded).view(1, -1)
    # TODO ABOVE!!!!!!
    question_encoded = question_encoded.type(dtype).long()
    question_var = Variable(question_encoded)
    question_var.requires_grad = False

    # (*) Run the model
    scores = None
    predicted_program = None
    if type(model) is tuple:
      program_generator, execution_engine = model
      program_generator.type(dtype)
      execution_engine.type(dtype)
      predicted_programs = []
      predicted_programs = program_generator.reinforce_sample(
                                question_var,
                                temperature=args.temperature,
                                argmax=(args.sample_argmax == 1))
      # print(feats_var.size())
      # print(feats_var.dtype)
      # print(predicted_programs)
      # print(predicted_programs.dtype)
      scores = execution_engine(feats_var, predicted_programs)
    else:
      model.type(dtype)
      scores = model(question_var, feats_var)

    # (*) Find results
    # Currently only works for a single question at a time (batch_size=1)
    _, predicted_answer_idx = scores.data.cpu().max(dim=1)
    predicted_answer = [load_vocab(vocab_path)['answer_idx_to_token'][i.item()] for i in predicted_answer_idx]

    # (*) Save results
    all_scores.append(scores.detach().cpu().numpy())
    # predicted.extend(list(predicted_answer))
    predicted.extend(list(predicted_answer_idx))
    q_ids.append(question_ids[i])

    f_img.close()
    f_q.close()
  return q_ids, all_scores, predicted

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
