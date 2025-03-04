# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os, json
import h5py
import numpy as np
#from scipy.misc import imread, imresize
from imageio import imread
from PIL import Image
from pathlib import Path


import torch
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--min_index', default=0, type=int)
parser.add_argument('--max_index', default=None, type=int)
parser.add_argument('--output_h5_dir', required=True)
parser.add_argument('--output_h5_prefix', required=True)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)

parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--get_every_layer', default=0, type=int)

def build_model(args, dtype):
  if not hasattr(torchvision.models, args.model):
    raise ValueError('Invalid model "%s"' % args.model)
  if not 'resnet' in args.model:
    raise ValueError('Feature extraction only supports ResNets')
  cnn = getattr(torchvision.models, args.model)(pretrained=True)
  layers = [
    cnn.conv1,
    cnn.bn1,
    cnn.relu,
    cnn.maxpool,
  ]
  for i in range(args.model_stage):
    name = 'layer%d' % (i + 1)
    layers.append(getattr(cnn, name))
  model = torch.nn.Sequential(*layers)
  model.type(dtype)
  model.eval()
  return model


def run_batch(cur_batch, model):
  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
  # print(len(cur_batch))

  image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
  image_batch = (image_batch / 255.0 - mean) / std
  image_batch = torch.FloatTensor(image_batch)
  if args.use_gpu:
    image_batch = image_batch.cuda()

  with torch.no_grad():
    image_batch = torch.autograd.Variable(image_batch)

    # Calling the NN  
    if args.get_every_layer:
      input = image_batch
      # print(input.size())
      for layer in model:
        output = layer(input)
        # print(output.size())
        input = output
      feats = output
      # exit()
    else:
      feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

  return feats


def main(args):
  input_paths = []
  idx_set = set()
  for fn in os.listdir(args.input_image_dir):
    if not fn.endswith('.png'): continue
    middle_split_str = os.path.splitext(fn)[0].split('_')[-2]
    idx = int(middle_split_str) if middle_split_str.isdigit() else int(os.path.splitext(fn)[0].split('_')[-1])
    input_paths.append((os.path.join(args.input_image_dir, fn), idx))
    idx_set.add(idx)
  input_paths.sort(key=lambda x: x[1])
  assert len(idx_set) == len(input_paths)
  assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
  max_index = len(input_paths)
  if args.max_index is not None:
    max_index = min(len(input_paths), args.max_index)
  input_paths = input_paths[args.min_index:max_index]
  print(f'{len(input_paths)} images to be processed')
  dtype = torch.FloatTensor
  if args.use_gpu == 1:
    dtype = torch.cuda.FloatTensor

  model = build_model(args, dtype)

  img_size = (args.image_height, args.image_width)

  dataset_path = f'{args.output_h5_dir}/{args.output_h5_prefix}_{args.min_index}_{max_index}.h5'
  dataset_ids_path = f'{args.output_h5_dir}/{args.output_h5_prefix}_ids_{args.min_index}_{max_index}.h5'

  Path(dataset_path).parent.mkdir(exist_ok=True, parents=True)
  with h5py.File(dataset_path, 'w') as f:
    feat_dset = None
    i0 = 0
    cur_batch = []
    all_ids = []
    for i, (path, idx) in enumerate(input_paths):
      img = imread(path, pilmode='RGB')
      img = np.array(Image.fromarray(img).resize((img_size), Image.BICUBIC))
      img = img.transpose(2, 0, 1)[None]
      cur_batch.append(img)
      all_ids.append(idx)
      if len(cur_batch) == args.batch_size:
        feats = run_batch(cur_batch, model)
        # TODO? save to tensor file
        if feat_dset is None:
          N = len(input_paths)
          _, C, H, W = feats.shape
          feat_dset = f.create_dataset('features', (N, C, H, W),
                                       dtype=np.float32)
        i1 = i0 + len(cur_batch)
        feat_dset[i0:i1] = feats
        i0 = i1
        print('Processed %d / %d images' % (i1, len(input_paths)))
        cur_batch = []
    if len(cur_batch) > 0:
      # do the last batch of remaining images, which are less in number than batch_size
      feats = run_batch(cur_batch, model)
      i1 = i0 + len(cur_batch)
      feat_dset[i0:i1] = feats
      print('Processed %d / %d images' % (i1, len(input_paths)))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
