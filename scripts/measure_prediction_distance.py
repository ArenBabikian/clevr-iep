# read the h5 file and gather the prediction data


# scores (float arrays for each image, of length corresponding to all seen responses)
# compare using MSE

# results (int value [0:N-1] for each image, where N is the number of different possible results):
# make one-hot encoding
# compare using BCE loss
# can even compare to GT

# OR we can make a measurement as to which percentage of images yield the same result.
# note thatthis is ONE-hot encoding, and not MULTI-hot, so we can just compare the int value of the result.


# COMPARISONS:
# gat-real
# gat-random1
# gat-randommore1
# resnet101 (ground truth we tried tolearn from)

# vs.

# 18k
# 9k
# 700k_strong
# cnn_lstm
# cnn_lstm_sa
# cnn_lstm_sa_mlp
# Ground truth (get this directly from the CLEVR_v1.0/questions/CLEVR_val_questions.json file, by querying question id)

import argparse
import os
import h5py
import torch
import iep.utils as utils
import json

parser = argparse.ArgumentParser()
parser.add_argument('--encoder_dir', type=str, default='data/answers')
# parser.add_argument('--questions_max', type=int, default=30000)
parser.add_argument('--questions_interval', type=int, default=1)

parser.add_argument('--images_min', default=0, type=int)
parser.add_argument('--images_max', default=300, type=int)
parser.add_argument('--vocab_json', default=None)
parser.add_argument('--questions_json', default=None)
parser.add_argument('--answers_name', default=None)

# CLEVR
encoder_targets = ['resnet101']
encoder_sources = ['learned', 'random1', 'randomwithreplacement1', 'learned_from_ans_9k_mse', 'learned_from_ans_9k_e']
# encoder_sources = ['learned', 'random1', 'randomwithreplacement1']
# decoders = ['18k', '9k', '700k_strong', 'lstm', 'cnn_lstm', 'cnn_lstm_sa', 'cnn_lstm_sa_mlp']
decoders = ['9k', '18k', '700k_strong']

#DISTINCT
encoder_targets = ['resnet101']
encoder_sources = ['GatAllClevrRand1', 'GatAllClevrRandWRep1', 'GatAllClevrReal', 'GatDisClevrRand1', 'GatDisClevrRandWRep1', 'GatDisClevrReal']
# encoder_sources = ['GatDisClevrReal']
decoders = ['9k', '18k', '700k_strong']

# variables
encoder_str_len = len(max(encoder_sources, key=len))
answers_map = {"True":"yes", "False":"no", "1":"1", "0":"0"}

def main(args):

    def measureAccuracy(f_src, f_tgt):
        correct_preditions = 0
        correct_tgt_wrt_gt = 0
        correct_src_wrt_gt = 0
        total_predictions = len(f_tgt['results'])

        # (*) invert map
        vocab_raw = utils.load_vocab(args.vocab_json)["answer_token_to_idx"]
        vocab = {}
        for k, v in vocab_raw.items():
            vocab[v] = k
        # print(vocab)

        with open(args.questions_json, 'r') as f:
            all_questions = json.load(f)["questions"]

        # (*) Debugging
        # for id in range(16):
        #     # print(f_tgt['results'][id])
        #     # print(f_tgt['question_ids'][id])
        #     res_gt = all_questions[id]["answer"]
        #     print(res_gt)

        for id in range(len(f_tgt['question_ids'])):
            res_tgt_id = f_tgt['results'][id]
            res_tgt = vocab[res_tgt_id]
            res_src_id = f_src['results'][id]
            res_src = vocab[res_src_id]

            res_gt = all_questions[id]["answer"]
            res_gt_str = str(res_gt)

            # (*) Debugging
            # print(f'{id} : t={res_tgt}({res_tgt_id}), s={res_src}({res_src_id}), g={res_gt_str}({answers_map[res_gt_str]}) |',end="")
            # print(type(res_gt))
            # print(type(res_tgt))
            # print("------")
            # exit()

            if res_tgt == res_src:
                correct_preditions += 1
            if res_tgt == answers_map[res_gt_str]:
                correct_tgt_wrt_gt += 1
            if res_src == answers_map[res_gt_str]:
                correct_src_wrt_gt += 1

        tgtVsSrc = correct_preditions/total_predictions
        tgtVsGt = correct_tgt_wrt_gt/total_predictions
        srcVsGt = correct_src_wrt_gt/total_predictions
        return {"tgt/src":round(tgtVsSrc, 3),
        "tgt/gt":round(tgtVsGt, 3),
        "src/gt":round(srcVsGt, 3)}

    def measureCrossEntropy(t_src, f_tgt):
        s_src = torch.nn.Softmax(dim=1)(t_src)
        t_tgt = torch.LongTensor(list(f_tgt['results']))
        return round(torch.nn.CrossEntropyLoss()(s_src, t_tgt), 2)

    distance_functions = {
        # 'mse': ((1, 1), lambda x, y : round(torch.nn.MSELoss()(x, y).item(), 2)),
        # 'crossEntropy': ((1, 0), measureCrossEntropy),
        'accuracy': ((0, 0), measureAccuracy)
    }

    for d in decoders:
        print(f'IMAGE RANGE = [{args.images_min}..{args.images_max}]')
        print(f'DECODER = {d}')
        decoder_file_name = f'{d}_{args.images_min}_{args.images_max}__{args.questions_interval}.h5'
        results = {}
        q = args.answers_name
            
        print(f'  QUESTION = {q}')
        for tgt in encoder_targets:

            tgt_path = os.path.join(args.encoder_dir, f'{q}/{tgt}', decoder_file_name)
            f_tgt = h5py.File(tgt_path, 'r')

            for src in encoder_sources:
                src_path = os.path.join(args.encoder_dir, f'{q}/{src}', decoder_file_name)
                f_src = h5py.File(src_path, 'r')
                # Iterate through the images
                # print(f_tgt['question_ids'])
                # print(f_src['question_ids'])
                for i in range(len(f_src['question_ids'])):
                    assert f_tgt['question_ids'][i] == f_src['question_ids'][i]

                src_tensor = torch.squeeze(torch.Tensor(f_src['scores']))
                src_tensor.requires_grad = True
                tgt_tensor = torch.squeeze(torch.Tensor(f_tgt['scores']))

                for name, func in distance_functions.items():
                    x1 = src_tensor if func[0][0] else f_src
                    x2 = tgt_tensor if func[0][1] else f_tgt
                    results[name] = func[1](x1, x2)                

                f_src.close()

                # PRINT
                print(f'    {tgt} vs.{src.ljust(encoder_str_len)}: (', end='')
                for name, value in results.items():
                    print(f'{name}={value}, ', end='')
                print(')')
            f_tgt.close()
            print('---')
        # print(results)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
