# read the h5 file and gather the prediction data


# scores (float arrays for each image, of length corresponding to all seen responses)
# compare using MSE

# results (int value [0:N-1] for each image, where N is the number of different possible results):
# make one-hot encoding
# compare using BCE loss
# can even compare to GT

# OR we can make a measurement as to which percentage of images yield the same result.
# note thatthis is ONE-hot encoding, and not MULTI-hot, so we can just compare the int value of the result.


# We take the Resnet101 feature representations of the images.
# 1. we get ACCURACY measurements for ALL images (i.e. ALL questions)
# 2. NEURON COVERAGE, with some threshold criteria (at resnet101 feature level)
# 2.1 compare best 100 to worst 100
# 3. SHAPE COVERAGE, 

import argparse
import os
import h5py
import torch
import iep.utils as utils
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import copy
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--encoder_dir', type=str, default='data/answers')

parser.add_argument('--images_min', default=0, type=int)
parser.add_argument('--images_max', default=300, type=int)
parser.add_argument('--vocab_json', default=None)
parser.add_argument('--questions_json', default=None) # gt questions
parser.add_argument('--data_json', default=None)
parser.add_argument('--figs_path', default=None)
parser.add_argument('--save_filename', default=None)

# NEURON COVERAGE
parser.add_argument('--features_dir', type=str, default=None)
parser.add_argument('--rankings_path', default=None)

parser.add_argument('--encoder', default=None)
parser.add_argument('--decoder_name', default=None)
parser.add_argument('--answers_name', default=None)
parser.add_argument('--shape_analysis', default='00')

num_neurons = 1024*14*14
allowed_parallels = [0, 2147483647]


########################################


def findMedianSample(list_of_series, img_2_accuracy, name, color):
    list_of_running_acc = []
    for serie in list_of_series:
        image_seq = serie['img_seq']
        list_of_running_acc.append(measureRunningAccuracy(image_seq, img_2_accuracy))
    
    # auc = area under curve
    list_of_auc = []
    final_acc = list_of_running_acc[-1][-1]
    for serie in list_of_running_acc:
        list_of_auc.append(sum([abs(x-final_acc) for x in serie]))

    median_index = list_of_auc.index(np.percentile(list_of_auc, 50, interpolation='nearest'))
    median_seq = list_of_series[median_index]['img_seq']
    series_median = {'label':f'median-{name}', 'img_seq':median_seq, 'alpha':1, 'color': color}

    return series_median

def saveAccuracyPlot(plot_info, ylim):
    list_of_plot_info = plot_info['series_info']
    fig_name = plot_info['fig_name']

    # path_dir = f'{args.figs_path}/{args.save_subdir}/{fig_name}-accuracy.png'
    path_dir = f'{args.figs_path}/{args.encoder}/{fig_name}'
    path_file = f'{path_dir}/{args.save_filename}.png'
    if not os.path.isdir(os.path.dirname(path_file)):
        os.makedirs(os.path.dirname(path_file))

    for series_info in list_of_plot_info:
        plt.plot(series_info['series'], 
            label=series_info['label'] if 'label' in series_info else None,
            alpha=series_info['alpha'] if 'alpha' in series_info else 1,
            color=series_info['color'] if 'color' in series_info else None,
            linestyle=series_info['linestyle'] if 'linestyle' in series_info else 'solid'
        )
    plt.xlabel('Number of images')
    plt.ylabel('Accuracy')
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    legend_pos = "upper right" if fig_name.startswith("conv") else "lower right"
    plt.legend(loc=legend_pos)
    plt.savefig(path_file)
    plt.clf()
    print(f'    Saving plot at {path_file}')


def createAccuracyPlot(plot_info, img_2_accuracy):
    # This adds everything to one plot. Cannot distinguish the different subcategories of series
    # {fig_name: ...
    # series_info: {[{name: ..., img_seq:[], color: ...}, {}]}
    # }

    # GATHER running accuracy
    base_name = plot_info['fig_name']
    plot_info['fig_name'] = f"accu-{base_name}"
    series_details = plot_info['series_info']
    for serie in series_details:
        image_seq = serie['img_seq']
        serie['series'] = measureRunningAccuracy(image_seq, img_2_accuracy)

    # GATHER convergence
    plot_info_conv = copy.deepcopy(plot_info)

    plot_info_conv['fig_name'] = f'conv-{base_name}'
    series_details_conv = plot_info_conv['series_info']

    for serie in series_details_conv:
        final_acc = serie['series'][-1]
        serie['series'] = [abs(x-final_acc) for x in serie['series']]

    # Add reference line to runing accuracy
    final_acc = series_details[0]['series'][-1]
    ref_series = [final_acc for _ in range(len(series_details[0]['img_seq']))]
    ref_plot_info = {'label':'Reference', 'series':ref_series, 'alpha':1, 'color':'red'}
    series_details.append(ref_plot_info)

    # set ylim
    ylim, ylim_conv = None, None
    if final_acc > 0.97:
        ylim = (0.94, 1.005)
        ylim_conv = (-0.005, 0.06)
    # Save plots
    saveAccuracyPlot(plot_info, ylim)
    saveAccuracyPlot(plot_info_conv, ylim_conv)

def getImageSequence(threshold, num_feats, all_feat_flat, ranking_type):
    active_neurons = torch.zeros(num_feats, dtype=torch.uint8) # 1 if active, 0 if not active
    remaining_image_ids = set(range(args.images_min, args.images_max)) # TODO will cause issues if images_min != 0
    num_images = args.images_max-args.images_min # might need a slight change related to upper bound
    top_images = torch.IntTensor(num_images)

    # temp_num_active_neurons = [] TODO

    # for _ in tqdm(range(num_images), f'{ranking_type}{threshold}'):
    for i in range(num_images):
        # best_active_new_neurons = torch.ones(num_feats, dtype=torch.uint8) * 2 # 1 if active, 0 if not active
        # best_image_id = -1

        # INITIALIZE WITH FIRST ELIGIBLE IMAGE
        starting_id = list(remaining_image_ids)[0]
        best_active_new_neurons = ((all_feat_flat[starting_id] > threshold) - active_neurons) == 1

        list_eligible_ids = [starting_id]
        for cur_image_id in remaining_image_ids:
            
            # which neurons are active for cur_image_id?
            # TENSOR 1 if active, 0 if not active
            cur_active_neurons = (all_feat_flat[cur_image_id] > threshold)

            # see how many NEW neurons it covers
            cur_active_new_neurons = (cur_active_neurons - active_neurons) == 1

            # To optimize number ofnewly activated neurons
            num_cur_new = torch.sum(cur_active_new_neurons)
            num_best_new = torch.sum(best_active_new_neurons) 

            if num_cur_new == num_best_new:
                list_eligible_ids.append(cur_image_id)
                continue
            
            if ((ranking_type == "best" and num_cur_new > num_best_new) or 
                (ranking_type == "worst" and num_cur_new < num_best_new)):
                best_active_new_neurons = cur_active_new_neurons
                list_eligible_ids = [cur_image_id]

            if ranking_type != "best" and ranking_type != "worst":
                exit(1)

        # best image has been selected
        assert torch.sum((active_neurons + best_active_new_neurons) > 1) == 0
        active_neurons = (active_neurons + best_active_new_neurons) > 0

        selected_id = list_eligible_ids[-1] # TODO
        selected_id = random.choice(list_eligible_ids)
        remaining_image_ids.remove(selected_id)
        # top_images.append((args.images_min+selected_id))
        top_images[i] = (args.images_min+selected_id)

        # temp_num_active_neurons.append(torch.sum(active_neurons)) TODO
    # return (top_images, temp_num_active_neurons) TODO
    return (top_images, None)


def getShapeCoverage(ranking_type, image_2_shapes_map, num_orderings_needed):

    tot_num_shapes = len(image_2_shapes_map[-1])

    clean_neighs_list = image_2_shapes_map[args.images_min:] # TODO got to watch out for this
    neighs_tensor = torch.Tensor(clean_neighs_list)

    num_images = args.images_max-args.images_min
    # orderings = []
    orderings = torch.IntTensor(num_orderings_needed, num_images)
    for i in tqdm(range(num_orderings_needed)):
        ordering, temp_num_active_neurons = getImageSequence(0, tot_num_shapes, neighs_tensor, ranking_type)
        # orderings.append(ordering)
        orderings[i] = ordering

    # addPdfToPlot(temp_num_active_neurons, num_parallels, ranking_type) # IMPORTANT

    return orderings, None
    # return top_images, temp_num_active_neurons


def measureRunningAccuracy(image_order, image_2_accuracy):
    correct_so_far = 0
    total_so_far = 0
    running_accuracy=[]
    for id in image_order:
        current_correct = image_2_accuracy[id]["cor"]
        correct_so_far += current_correct
        current_total = image_2_accuracy[id]["tot"]
        total_so_far += current_total
        # running_accuracy.append(correct_so_far/total_so_far)
        running_accuracy.append(round((correct_so_far/total_so_far), 3))
    return running_accuracy


def main(args):

    #################
    # INITIAL SETUP
    #################

    e = args.encoder
    d = args.decoder_name
    q = args.answers_name
    tot_num_imgs = args.images_max-args.images_min
    vocab = utils.load_vocab(args.vocab_json)["answer_idx_to_token"]

    # Display info
    dec_short = d.split('_0')[0]
    print(f'>>> enc={e}, dec={dec_short}, qs={q}, ims=[{args.images_min}..{args.images_max}]')

    # get answers file
    decoder_file_name = f'{args.decoder_name}.h5'
    ans_path = os.path.join(args.encoder_dir, f'{q}/{e}', decoder_file_name)
    f_ans = h5py.File(ans_path, 'r')

    # get ground-truth file
    with open(args.questions_json, 'r') as f:
        gt_questions = json.load(f)["questions"]

    # check if "question_index" is the same as the id in the list
    for q_id in range(len(gt_questions)):
        assert q_id == gt_questions[q_id]["question_index"]

    # res_gt = gt_questions[id]["answer"]
    # res_gt_str = str(res_gt)
    # print(len(f_ans['question_ids']))
    # exit()

    #################
    # MEASUREMENT SETUP
    #################

    # GATHER IMAGE-TO-ACCURACY INFO
    dec_split = d.split('_')
    acc_save_dir = f'{args.data_json}/accuracy/{dec_short}'
    acc_save_path = f'{acc_save_dir}/{q[8:]}-{dec_split[-4]}-{dec_split[-3]}.json'

    if os.path.exists(acc_save_path):
        # USE SAVED FILE FOR ACCURACY INFO
        with open(acc_save_path, 'r') as f:
            all_accuracy_data = json.load(f)
        image_2_accuracy = all_accuracy_data[:args.images_max]
        for i in range(args.images_min):
            image_2_accuracy[i] = None
        
    else:
        # GATHER IMAGE ACCURACY INFO FROM QUESTION RESULTS FILES

        # GATHER image-to-related-questions-ids map
        image_ids = list(range(args.images_max))
        image_2_related_q_idx = [[] for _ in image_ids]
        for qu in gt_questions:
            if qu["image_index"] in range(args.images_min, args.images_max):
                image_2_related_q_idx[qu['image_index']].append(qu["question_index"]) 
                # TODO could possibly simplify this by just looking at the list id

        # GATHER image-to-accuracy map
        image_2_accuracy = [{"cor":None, "tot":None} for _ in image_2_related_q_idx]
        for img_id, related_qs in enumerate(image_2_related_q_idx):
            correct_preditions = 0
            total_predictions = 0
            for q_id_2 in related_qs:
                ms_ans = f_ans["results"][q_id_2]
                ms_ans_str = vocab[ms_ans]
                gt_ans_str = gt_questions[q_id_2]["answer"]

                total_predictions += 1
                if ms_ans_str == gt_ans_str:
                    correct_preditions += 1

            if len(related_qs) == 0:
                image_2_accuracy[img_id] = None
            else:
                image_2_accuracy[img_id]["cor"] = correct_preditions
                image_2_accuracy[img_id]["tot"] = total_predictions

        # # TEMPORARY - Print accuracy info to file, for reuse
        acc_save_dir = f'{args.data_json}/accuracy/{dec_short}'
        if not os.path.isdir(acc_save_dir):
            os.makedirs(acc_save_dir)
        acc_save_path = f'{acc_save_dir}/{q[8:]}-{args.images_min}-{args.images_max}.json'
        with open(acc_save_path, 'w') as f:
            json.dump(image_2_accuracy, f)
        print(f'    Saving accuracies at {acc_save_path}')
        exit()
        # TEMPORARY - END

    # MEASURE AND PRINT GLOBAL ACCURACY
    if args.shape_analysis[0] == '1':
        #OBSOLETE TODO remove
        # acc = measureGlobalAccuracy(f_ans, gt_questions, vocab, args.images_min, args.images_max)
        # general_accuracy = acc['cor']/acc['tot']
        # print(f"  Global Accuracy = {general_accuracy} ({acc['cor']}/{acc['tot']})")

        cor, tot = 0, 0
        for acc_info in image_2_accuracy:
            if acc_info != None:
                cor += acc_info['cor']
                tot += acc_info['tot']
        general_accuracy = cor / tot
        print(f"  Global Accuracy = {general_accuracy} ({cor}/{tot})")

    if args.shape_analysis[1] == '0':
        exit()

    # GATHER image-to-covered-shapes map
    # TODO move this down, and only look at this data if the ordering is not taken from a file
    image_2_shapes = {}
    for num_parallels in allowed_parallels:
        with open(f'{args.data_json}/data{num_parallels}pars.json', 'r') as f:
            all_neigh_occurrences = json.load(f)

        image_2_shapes[str(num_parallels)] = []
        for img_id, acc_info in enumerate(image_2_accuracy):
            if acc_info == None:
                image_2_shapes[str(num_parallels)].append(None)
            else:
                image_2_shapes[str(num_parallels)].append(all_neigh_occurrences[str(img_id)])

    # TODO GATHER image-to-covered-neurons map

    #################
    # RELEVANT IMAGE ORDERINGS
    #################
    clean_accuracies = image_2_accuracy[args.images_min:]

    ###################
    # OPTIMAL ACCURACIES
    series_extremum = []
    plot_max_min = {'fig_name':'shape-max-min', 'series_info':[]}
    indices = list(range(len(clean_accuracies)))
    sort_key = lambda y:y[1]["cor"]/y[1]["tot"]

    # BEST
    decreasing_acc_idx = [x for x, _ in sorted(zip(indices, clean_accuracies), key=sort_key, reverse=True)]
    dec_acc_info = {'label':'best_accuracy', 'img_seq':decreasing_acc_idx, 'color':'red', 'linestyle':'dashed'}
    series_extremum.append(dec_acc_info)

    # WORST
    increasing_acc_idx = [x for x, _ in sorted(zip(indices, clean_accuracies), key=sort_key)]
    inc_acc_info = {'label':'worst_accuracy', 'img_seq':increasing_acc_idx, 'color':'red', 'linestyle':'dashdot'}
    series_extremum.append(inc_acc_info)

    # PLOT
    # plot_max_min['series_info'] = series_extremum
    # createAccuracyPlot(plot_max_min, image_2_accuracy)

    ###################
    # RANDOM ACCURACIES
    num_random_plots = 300 # TODO 1000
    series_random = []
    plot_random = {'fig_name':f'random-only-{num_random_plots}', 'series_info':[]}
    for _ in range(num_random_plots):
        image_ids = list(range(args.images_min, args.images_max))
        random.shuffle(image_ids)
        random_plot_info = {'label':None, 'img_seq':image_ids, 'alpha':0.05, 'color':'black'}
        series_random.append(random_plot_info)
    median_random = [findMedianSample(series_random, image_2_accuracy, 'random', 'gold')]
    
    if True:
        plot_random['series_info'] = series_random + series_extremum + median_random
        createAccuracyPlot(plot_random, image_2_accuracy)

    ###################
    # SHAPE COVERAGE
    print('  Measuring Shape Coverage')
    # clean_shapes = image_2_shapes[args.images_min:]
    series_shape = []
    plot_shape_all = {'fig_name':'shape-all', 'series_info':[]}
    config_2_color = {
        (0, 'best'):('lime', 'darkgreen'),
        (2147483647, 'best'):('deepskyblue', 'darkblue'),
        (0, 'worst'):('orange', 'sienna'),
        (2147483647, 'worst'):('magenta', 'darkmagenta'),
    }
    medians_shape = []
    for pars in allowed_parallels:
        for t in ["best", "worst"]:
            main_color = config_2_color[(pars, t)][0]
            median_color = config_2_color[(pars, t)][1]
            plot_shape_current = {'fig_name':f'shape-{t}-{pars}', 'series_info':[]}

            ordering_dir = f'{args.data_json}/ordering/{t}-{pars}'
            ordering_path = f'{ordering_dir}/{args.images_min}-{args.images_max}.pt'

            if os.path.exists(ordering_path):
                # USE SAVED FILE FOR shape-based ORDERING INFO
                all_image_orderings = torch.load(ordering_path).tolist()
                image_orderings = random.sample(all_image_orderings, num_random_plots)
            else:
                image_orderings_pt, _ = getShapeCoverage(t, image_2_shapes[str(pars)], 500)
                # 2000 replaced by num_random_plots
                # 2000 taking too much time for 1000 images, so we do with 500

                # TEMPORARY - SAVE 2000 shape-based orderings in json
                if not os.path.isdir(ordering_dir):
                    os.makedirs(ordering_dir)
                s_time = time.time()
                torch.save(image_orderings_pt, ordering_path)
                print(f'    Saved shape-based orderings at {ordering_path} in {time.time()-s_time}')
                continue
                # TEMPORARY - END
                image_orderings = image_orderings_pt.tolist()

            current_series_shape = []
            for ordering in image_orderings:
                shape_plot_info = {'label':None, 'img_seq':ordering, 'alpha':0.05, 'color': main_color}
                current_series_shape.append(shape_plot_info)

            median_current = findMedianSample(current_series_shape, image_2_accuracy, f'{t}-{pars}', median_color)

            # generate current series + random
            plot_shape_current['series_info'] = series_random + series_extremum + current_series_shape + median_random + [median_current]
            
            createAccuracyPlot(plot_shape_current, image_2_accuracy)

            # Management for shape-all
            medians_shape.append(median_current)
            series_shape.extend(current_series_shape)

    plot_shape_all['series_info'] = series_random + series_extremum + series_shape + median_random + medians_shape
    # savePdfPlot("shape") # IMPORTANTY
    createAccuracyPlot(plot_shape_all, image_2_accuracy)

    exit()

    # TODO TEMPORARILY COMMENTED OUT
    # # NEURON COVERAGE
    # # for thresh in [0, 0.5, 1, 2]:
    # print('  Measuring Neuron Coverage')
    # for thresh in [0, 0.5, 1]:
    #     for t in ["best", "worst"]:
    #         # TODO TEMPORARY the *0.05
    #         top_neuron_coverage_images, _ = getNeuronCoverage(thresh, t, int(0.01*tot_num_imgs))
    #         sample_map_2_imgs[f'{t}_neu_cvg_thresh={thresh}'] = top_neuron_coverage_images

    #         # print(f'Thresh = {thresh}, Type={t}')
    #         # print(f'{t}{thresh}, ')
    # savePdfPlot("neuron")

    # TODO For now, no analysis of scores for now. 
    # See measur_prediction_distance.py file for examples


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
