
import argparse
import os
from sys import exit
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import math
from statistics import mean 
import analysis_util
from numpy import dot
from numpy.linalg import norm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--input_scene_file', default='../output/CLEVR_scenes.json',
    help="JSON file containing ground-truth scene information for all images " +
         "from render_images.py")
parser.add_argument('--questions_dir', default=None) # gt questions
parser.add_argument('--data_dir', default=None)
parser.add_argument('--parallels_dir', default=None)
parser.add_argument('--images_min', default=0, type=int)
# parser.add_argument('--images_max', default=300, type=int)
parser.add_argument('--figs_dir', default=None)

parser.add_argument('--save_manhattan', default=False, action='store_true')
parser.add_argument('--save_cosine_sim', default=False, action='store_true')
parser.add_argument('--save_f1', default=False, action='store_true')
parser.add_argument('--save_num_2_acc', default=False, action='store_true')
parser.add_argument('--save_accuracy_2_shape', default=False, action='store_true')

num_neurons = 1024*14*14
allowed_parallels = [0, 2147483647]

encoders = ['resnet101']
decoders = ['9k', '18k', '700k_strong']
suffixes = ['obj-cnt', 'obj-ex', 'rel-cnt', 'rel-ex']

# TEMP
# decoders = ['9k', '18k']
suffixes = ['obj-cnt']
# suffixes = ['rel-ex', 'obj-ex']
# suffix_2_decoder_spec = {'rel':('0', '100'), 'obj':('0', '1000')}
suf_2_bds = {'rel':('0', '100'), 'obj':('0', '250')}

def main(args):

    for en in encoders:
        six_obj_accs = {}
        all_f1_data = []
        for de in decoders:
            for su in suffixes:
                su_bds = suf_2_bds[su[:3]]

                #################
                # INITIAL SETUP
                #################
                images_max = int(suf_2_bds[su[:3]][1])
                # Display info
                print(f'>>> enc={en}, dec={de}, qs={su}, ims=[{args.images_min}..{images_max}]')

                # tot_num_imgs = images_max-args.images_min
                # NOTE NO NEED SINCE ACURACIES ARECOMING FROM SAVE FILE
                # vocab = utils.load_vocab(args.vocab_json)["answer_idx_to_token"]
                #
                # # get answers file
                # decoder_file_name = f'{de}_{suffix_2_decoder_spec[su[:3]][0]}_{suffix_2_decoder_spec[su[:3]][1]}__1.h5'
                # ans_path = os.path.join(args.encoder_dir, f'answers-{su}/{en}', decoder_file_name)
                # f_ans = h5py.File(ans_path, 'r')

                # get question ground-truth file
                # with open(f'{args.questions_dir}/questions-{su}.json', 'r') as f:
                questions_file_path = f'{args.questions_dir}/questions/{su}.json'
                with open(questions_file_path, 'r') as f:
                    gt_questions = json.load(f)["questions"]

                # check if "question_index" is the same as the id in the list
                for q_id in range(len(gt_questions)):
                    assert q_id == gt_questions[q_id]["question_index"]
                
                # image ground truth    
                with open(args.input_scene_file, 'r') as f:
                    scene_data = json.load(f)
                    all_scenes = scene_data['scenes']

                #################
                # MEASUREMENT SETUP
                #################

                # GATHER IMAGE-TO-ACCURACY INFO
                # acc_save_dir = f'{args.data_dir}/accuracy/{de}'
                # acc_save_path = f'{acc_save_dir}/{su}-{suffix_2_decoder_spec[su[:3]][0]}-{suffix_2_decoder_spec[su[:3]][1]}.json'
                acc_save_dir = f'{args.data_dir}/accuracy/{en}/{su}'
                acc_save_path = f'{acc_save_dir}/{de}_{su_bds[0]}_{su_bds[1]}__1.json'
                if os.path.exists(acc_save_path):
                    # USE SAVED FILE FOR ACCURACY INFO
                    with open(acc_save_path, 'r') as f:
                        all_accuracy_data = json.load(f)
                    image_2_accuracy = all_accuracy_data[:images_max]
                    # MAJOR SHORTCUT/ASSUMPTION BELOW
                    for i in range(args.images_min):
                        image_2_accuracy[i] = None
                    
                else:
                    # the files shold be generated from the analyse_image_subsets.py file
                    exit(1)

                # GATHER image-to-covered-shapes map
                image_2_shapes = {}
                shape_2_image = {}
                shape_2_id = {}
                id_2_shape = {}
                for num_parallels in allowed_parallels:
                    with open(f'{args.parallels_dir}/data{num_parallels}pars.json', 'r') as f:
                        all_neigh_occurrences = json.load(f)

                    image_2_shapes[str(num_parallels)] = []
                    shape_2_image[str(num_parallels)] = {}
                    shape_2_id[str(num_parallels)] = {}
                    id_2_shape[str(num_parallels)] = {}
                    shape_id_cnt = 0
                    for img_id, acc_info in enumerate(image_2_accuracy):
                        if acc_info == None:
                            image_2_shapes[str(num_parallels)].append(None)
                        else:
                            image_2_shapes[str(num_parallels)].append(all_neigh_occurrences[str(img_id)])

                            shape_str = ''.join([str(z) for z in all_neigh_occurrences[str(img_id)]])
                            if shape_str in shape_2_image[str(num_parallels)]:
                                shape_2_image[str(num_parallels)][shape_str].append(img_id)
                            else:
                                shape_2_image[str(num_parallels)][shape_str] = [img_id]

                            if shape_str not in shape_2_id[str(num_parallels)]:
                                shape_2_id[str(num_parallels)][shape_str] = shape_id_cnt
                                id_2_shape[str(num_parallels)][shape_id_cnt] = all_neigh_occurrences[str(img_id)]
                                shape_id_cnt += 1

                #################
                # DATA ANALYSIS
                #################

                # THE TWO RELVANT MAPS NOW ARE:
                # image_2_accuracy, image_2_shapes

                # Num objects in image vs accuracy
                num_objects_list = [0 for _ in range(11)]
                specific_shape_2_accuracy = [[] for _ in image_2_shapes["0"][0]]

                manhattan_dist_2_acc_diff = []
                cosine_sim_2_acc_diff = []
                num_objects_2_accuracies = []

                accuracies = [] # (shape_id, accuracies)
                num_objects_2_f1_info = []

                for im_id, acc_info in tqdm(enumerate(image_2_accuracy)):
                    if acc_info != None:
                        # ACC
                        accuracy = (acc_info[f'cor'] / acc_info[f'tot'])

                        # GT
                        gt_scene = all_scenes[im_id]
                        num_objects = len(gt_scene["objects"])
                        num_objects_list[num_objects] += 1

                        # SHAPE
                        shape_info = image_2_shapes["0"][im_id] #TODO, 0 is the num_parallels

                        num_shapes = sum(shape_info)
                        dif_shapes = len(list(filter(lambda x : (x != 0), shape_info)))

                        # accuracies and shape_ids
                        shape_str = ''.join([str(z) for z in shape_info])
                        shape_id = shape_2_id['0'][shape_str]
                        accuracies.append((shape_id, accuracy))

                        # specific_shape_2_accuracy
                        for sh_id, sh_ex in enumerate(shape_info):
                            if sh_ex != 0:
                                specific_shape_2_accuracy[sh_id].append(accuracy)

                        # ADD INFOR TO LISTS                        
                        analysis_util.addF1Info(num_objects, acc_info, num_objects_2_f1_info)
                        analysis_util.add2list(num_objects, accuracy, num_objects_2_accuracies)
                        # analysis_util.add2list(num_shapes, accuracy, num_shapes_2_accuracies)
                        # analysis_util.add2list(dif_shapes, accuracy, dif_shapes_2_accuracies)

                        # SHAPE PAIR ANALYSIS
                        if not args.save_num_2_acc and not args.save_manhattan:
                            continue
                        num_cos_sim_buckets = 20
                        for im_id_2 in range(im_id+1, len(image_2_accuracy)):
                            acc_info_2 = image_2_accuracy[im_id_2]
                            if acc_info_2 == None:
                                continue
                            # print(f'{im_id_2}-{acc_info_2}')

                            # get manhattan distance
                            shape_info_2 = image_2_shapes["0"][im_id_2] #TODO, 0 is the num_parallels
                            man_dist = 0
                            for shape_id, num_shape_1 in enumerate(shape_info):
                                num_shape_2 = shape_info_2[shape_id]
                                man_dist += abs(num_shape_1-num_shape_2)

                            # get cosine similarity
                            a = np.array(shape_info)
                            b = np.array(shape_info_2)
                            cos_sim = dot(a, b)/(norm(a)*norm(b))
                            cos_sim_bucket = math.floor(cos_sim * num_cos_sim_buckets)

                            # # get student-t distribution
                            # TODO

                            # get accuracy diff
                            accuracy_2 = (acc_info_2[f'cor'] / acc_info_2[f'tot'])
                            acc_diff = abs(accuracy-accuracy_2)

                            # add to correspoding lists
                            analysis_util.add2list(man_dist, acc_diff, manhattan_dist_2_acc_diff)
                            analysis_util.add2list(cos_sim_bucket, acc_diff, cosine_sim_2_acc_diff)
                
                # DEBUGGING
                # print(num_objects_list)
                # printMap(num_objects_2_accuracies)
                # printMap(num_shapes_2_accuracies)
                # printMap(dif_shapes_2_accuracies)
                # printMap(specific_shape_2_accuracy)
                # printMap(manhattan_dist_2_acc_diff, None, 0)
                # printMap(num_objects_2_accuracies, 50)
                # six_obj_accs[f'{de}-{su}'] = num_objects_2_accuracies[6]
                # continue

                # PAIRWISE ANALYSIS
                # TODO the saveFig calls ight be outdated. Might need to replace with saveBox
                if args.save_manhattan:
                    analysis_util.saveFig(args, manhattan_dist_2_acc_diff, f"{en}/shape_analysis/manhattan-2-diff", f'{su}-{images_max}/{de}', xl='manhattan distance', yl='accuracy difference')
                if args.save_cosine_sim:
                    analysis_util.saveFig(args, cosine_sim_2_acc_diff, f"{en}/shape_analysis/cos-sim-2-diff", f'{su}-{images_max}/{de}', xl='cosine similarity bucket', yl='accuracy difference')

                # INDIVIDUAL ANALYSIS
                if args.save_accuracy_2_shape:
                    sorted_pairs = sorted(accuracies, key=lambda pair:pair[1])

                    sorted_acc=[[],[]]
                    sorted_acc[0] = [s for (s, _) in sorted_pairs]
                    sorted_acc[1] = [a for (_, a) in sorted_pairs]

                    shape_id_2_accuracies = []
                    for (id, acc) in accuracies:
                        analysis_util.add2list(id, acc, shape_id_2_accuracies)

                    sorted_s_i_2_a = sorted(zip(list(range(len(shape_id_2_accuracies))), shape_id_2_accuracies), key=lambda pair: mean(pair[1]))
                    sorted_a_2_s_i = [(a, s) for (s, a) in sorted_s_i_2_a]
                    # analysis_util.printPairsList(sorted_s_i_2_a)

                    num_diff_shapes = len(id_2_shape['0'][0])
                    # individual shape analysis
                    acc_2_unique_shape_freq = {}
                    for a in sorted_acc[1]:
                        acc_2_unique_shape_freq[a]=None
                    num_acc_2_unique_shape_freq = len(acc_2_unique_shape_freq)

                    for k in acc_2_unique_shape_freq.keys():
                        acc_2_unique_shape_freq[k] = [0 for _ in range(num_diff_shapes)]
                    # analysis_util.printMapSimple(acc_2_unique_shape_freq)

                    for (s_i_x, a_x) in sorted_pairs:
                        shapes = id_2_shape['0'][s_i_x]
                        # print(a_x)
                        for i_unique_sh, num_unique_sh in enumerate(shapes):
                            acc_2_unique_shape_freq[a_x][i_unique_sh] += num_unique_sh
                    # analysis_util.printMapSimple(acc_2_unique_shape_freq)

                    ##### Save as CSV
                    # TODO refactor to util
                    save_path = f"{args.figs_dir}/{en}/shape_analysis/accuracy_vs_shape/{su}-{images_max}/{de}.csv"
                    if not os.path.isdir(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    to_print = []
                    header = ['accuracy'] + [i for i in range(num_diff_shapes)]
                    to_print.append(header)
                    for k, v in acc_2_unique_shape_freq.items():
                        to_print.append([k]+v)

                    np.savetxt(save_path, to_print, delimiter=',', fmt='%s')
                    print(f'    Saving csv at {save_path}')
                    #####

                if args.save_num_2_acc:
                    analysis_util.saveFig(args, num_objects_2_accuracies, f"{en}/shape_analysis/objects-2-accuracies", f'{su}-{images_max}/{de}', xl='number of objects in the image', yl='accuracy difference')

                f1_results = analysis_util.gatherF1Data(num_objects_2_f1_info)
                # analysis_util.saveF1Fig(args, f1_results, en, f'{su}-{images_max}/{de}')
                f1_xs, f1_scores, _ = f1_results

                f1_data = {'label':f'{de}-{su}', 'x':f1_xs, 'y':f1_scores}
                all_f1_data.append(f1_data)
        
        # # Save 6-object-accs to file
        # x = f'{args.figs_dir}/six_obj_accs.json'
        # with open(x, 'w') as f:
        #     json.dump(six_obj_accs, f)
        # print(f'    Saving 6-obj accuracies at {x}')

        if args.save_f1:
            # create global f1 figure
            # TODO refactor to util
            path_dir = f'{args.figs_dir}/{en}/shape_analysis/f1-score'
            path_file = f'{path_dir}/all-f1-{images_max}.png'
            if not os.path.isdir(os.path.dirname(path_file)):
                os.makedirs(os.path.dirname(path_file))
            path_file_zoom = f'{path_dir}/all-f1-{images_max}-zoom.png'
            for d in all_f1_data:
                plt.plot(d['x'], d['y'], label=d['label'], marker="o", markersize=7)
            plt.xlabel('num objects')
            plt.ylabel('f1 score')
            plt.legend()
            plt.savefig(path_file)
            print(f'    Saving plot at {path_file}')
            plt.ylim(0.9, 1)
            plt.savefig(path_file_zoom)
            plt.clf()
            print(f'    Saving plot at {path_file_zoom}')

    exit()

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
