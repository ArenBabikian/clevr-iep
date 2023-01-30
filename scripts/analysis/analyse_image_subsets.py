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

from sys import exit
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
parser.add_argument('--questions_dir', default=None) # gt questions
parser.add_argument('--data_dir', default=None)
parser.add_argument('--figs_dir', default=None)
parser.add_argument('--save_filename', default=None)
parser.add_argument('--num_random_plots', default=500, type=int)

# NEURON COVERAGE
parser.add_argument('--features_dir', type=str, default=None)
parser.add_argument('--rankings_path', default=None)

parser.add_argument('--encoder', default=None)
parser.add_argument('--decoder_name', default=None)
parser.add_argument('--answers_name', default=None)
parser.add_argument('--suffix', default=None)


parser.add_argument('--parallels_dir', default=None)
parser.add_argument('--print_global', default=False, action='store_true')
parser.add_argument('--shape_analysis', default=False, action='store_true')
parser.add_argument('--plot_random', default=False, action='store_true')
parser.add_argument('--plot_shape_cvg', default=False, action='store_true')

num_neurons = 1024*14*14
allowed_parallels = [0, 2147483647]
suf_2_bds = {'rel':('0', '100'), 'obj':('0', '250')}


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

    # path_dir = f'{args.figs_dir}/{args.save_subdir}/{fig_name}-accuracy.png'
    path_dir = f'{args.figs_dir}/{args.encoder}/3-orderings/{fig_name}'
    path_file = f'{path_dir}/{args.suffix}-{args.images_max}/{args.decoder_name}.png'
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
    plot_info['fig_name'] = f"accuracy/{base_name}"
    series_details = plot_info['series_info']
    for serie in series_details:
        image_seq = serie['img_seq']
        serie['series'] = measureRunningAccuracy(image_seq, img_2_accuracy)

    # GATHER convergence
    plot_info_conv = copy.deepcopy(plot_info)

    plot_info_conv['fig_name'] = f'convergence/{base_name}'
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

def getImageSequence(threshold, num_feats, all_feat_flat, ranking_type, initial_image_id_set, ids_to_rm):
    active_neurons = torch.zeros(num_feats, dtype=torch.uint8) # 1 if active, 0 if not active
    remaining_image_ids = set(range(args.images_min, args.images_max)) # TODO will cause issues if images_min != 0
    for id in ids_to_rm:
        remaining_image_ids.remove(id)
    eligible_image_ids = remaining_image_ids.copy()
    num_images = args.images_max-args.images_min # might need a slight change related to upper bound
    top_images = torch.IntTensor(num_images)

    # temp_num_active_neurons = [] TODO

    starting_index = 0
    if initial_image_id_set != None:
        initial_image_id = random.choice(initial_image_id_set)
        remaining_image_ids.remove(initial_image_id)
        top_images[0] = initial_image_id
        starting_index = 1
        #also set the covered neurons
        active_neurons = (all_feat_flat[initial_image_id] > threshold)

    # for _ in tqdm(range(num_images), f'{ranking_type}{threshold}'):
    # for i in range(starting_index, num_images):
    for i in range(starting_index, num_images-len(ids_to_rm)):

        # best_active_new_neurons = torch.ones(num_feats, dtype=torch.uint8) * 2 # 1 if active, 0 if not active
        # best_image_id = -1

        # INITIALIZE WITH FIRST ELIGIBLE IMAGE
        starting_id = list(remaining_image_ids)[0]
        best_active_new_neurons = ((all_feat_flat[starting_id] > threshold) - active_neurons) == 1
        num_best_new = torch.sum(best_active_new_neurons) 

        list_eligible_ids = [starting_id]
        for cur_image_id in remaining_image_ids:
            
            # which neurons are active for cur_image_id?
            # TENSOR 1 if active, 0 if not active
            cur_active_neurons = (all_feat_flat[cur_image_id] > threshold)

            # which NEW neurons it covers
            cur_active_new_neurons = (cur_active_neurons - active_neurons) == 1

            # To optimize number of newly activated neurons
            num_cur_new = torch.sum(cur_active_new_neurons)

            if num_cur_new == num_best_new:
                list_eligible_ids.append(cur_image_id)
                continue
            
            if ((ranking_type == "best" and num_cur_new > num_best_new) or 
                (ranking_type == "worst" and num_cur_new < num_best_new)):
                num_best_new = num_cur_new
                list_eligible_ids = [cur_image_id]

            if ranking_type != "best" and ranking_type != "worst":
                exit(1)

        # select best image among possibilities
        selected_id = random.choice(list_eligible_ids)
        remaining_image_ids.remove(selected_id)
        top_images[i] = (args.images_min+selected_id)

        # which new neurons are covered by the selected id
        sel_active_neurons = (all_feat_flat[selected_id] > threshold)
        active_new_neurons_for_selection = (sel_active_neurons - active_neurons) == 1

        # Edit the global active neurons
        assert torch.sum((active_neurons + active_new_neurons_for_selection) > 1) == 0
        active_neurons = (active_neurons + active_new_neurons_for_selection) > 0

        # temp_num_active_neurons.append(torch.sum(active_neurons)) TODO
    # return (top_images, temp_num_active_neurons) TODO
    return (top_images, None)

def getShapeCoverage(ranking_type, image_2_shapes_map, num_orderings_needed, ids_to_rm, initial_image_id_set = None):

    tot_num_shapes = len(image_2_shapes_map[-1])

    clean_neighs_list = image_2_shapes_map[args.images_min:] # TODO got to watch out for this
    for id in ids_to_rm:
        clean_neighs_list[id] = [-1 for _ in range(tot_num_shapes)]
    neighs_tensor = torch.Tensor(clean_neighs_list)

    num_images = args.images_max-args.images_min
    # orderings = []
    orderings = torch.IntTensor(num_orderings_needed, num_images)
    for i in tqdm(range(num_orderings_needed)):
        ordering, temp_num_active_neurons = getImageSequence(0, tot_num_shapes, neighs_tensor, ranking_type, initial_image_id_set, ids_to_rm)
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
    su = args.suffix
    su_bds = suf_2_bds[su[:3]]
    tot_num_imgs = args.images_max-args.images_min
    vocab = utils.load_vocab(args.vocab_json)["answer_idx_to_token"]

    # Display info
    # dec_short = d.split('_0')[0]
    print(f'>>> enc={e}, dec={d}, qs={q}, ims=[{args.images_min}..{args.images_max}]')

    # # NOTE NO NEED SINCE ACURACIES ARECOMING FROM SAVE FILE
    # # get answers file
    # decoder_file_name = f'{args.decoder_name}.h5'
    # ans_path = os.path.join(args.encoder_dir, f'{q}/{e}', decoder_file_name)
    # f_ans = h5py.File(ans_path, 'r')

    # get ground-truth file
    # with open(args.questions_json, 'r') as f:
    questions_file_path = f'{args.questions_dir}/questions/{su}.json'
    with open(questions_file_path, 'r') as f:
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
    acc_save_dir = f'{args.data_dir}/accuracy/{e}/{su}'
    acc_save_path = f'{acc_save_dir}/{d}_{su_bds[0]}_{su_bds[1]}__1.json'

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
        image_2_accuracy = [{"cor":None, "tot":None, "y-cor":None, "y-tot":None, "n-cor":None, "n-tot":None} for _ in image_2_related_q_idx]
        for img_id, related_qs in enumerate(image_2_related_q_idx):
            correct_preditions = 0
            total_predictions = 0
            yes_correct_preds, yes_total_preds, no_correct_preds, no_total_preds = 0, 0, 0, 0
            for q_id_2 in related_qs:
                ms_ans = f_ans["results"][q_id_2]
                ms_ans_str = vocab[ms_ans]
                gt_ans_str = gt_questions[q_id_2]["answer"]

                total_predictions += 1
                if ms_ans_str == gt_ans_str:
                    correct_preditions += 1

                # yes or 1 anwers
                if gt_ans_str == "yes" or gt_ans_str == "1":
                    yes_total_preds += 1
                    if ms_ans_str == gt_ans_str:
                        yes_correct_preds += 1
                else:
                    no_total_preds += 1
                    if ms_ans_str == gt_ans_str:
                        no_correct_preds += 1

            if len(related_qs) == 0:
                image_2_accuracy[img_id] = None
            else:
                image_2_accuracy[img_id]["cor"] = correct_preditions
                image_2_accuracy[img_id]["tot"] = total_predictions
                image_2_accuracy[img_id]["y-cor"] = yes_correct_preds
                image_2_accuracy[img_id]["y-tot"] = yes_total_preds
                image_2_accuracy[img_id]["n-cor"] = no_correct_preds
                image_2_accuracy[img_id]["n-tot"] = no_total_preds

        # # TEMPORARY - Print accuracy info to file, for reuse
        acc_save_dir = f'{args.data_dir}/accuracy/{dec_short}'
        if not os.path.isdir(acc_save_dir):
            os.makedirs(acc_save_dir)
        acc_save_path = f'{acc_save_dir}/{q[8:]}-{args.images_min}-{args.images_max}.json'
        with open(acc_save_path, 'w') as f:
            json.dump(image_2_accuracy, f)
        print(f'    Saving accuracies at {acc_save_path}')
        exit()
        # TEMPORARY - END

    # MEASURE AND PRINT GLOBAL ACCURACY
    if args.print_global:
        def printAccuracy(prefix):
            cor, tot = 0, 0
            for acc_info in image_2_accuracy:
                if acc_info != None:
                    cor += acc_info[f'{prefix}cor']
                    tot += acc_info[f'{prefix}tot']
            general_accuracy = round(cor / tot, 3)
            name = {"":"Global", "y-":"Yes/1", "n-":"No/0"}
            print(f"  {name[prefix]} Accuracy = {general_accuracy} ({cor}/{tot})")

        printAccuracy("")
        printAccuracy("y-")
        printAccuracy("n-")

    if not args.shape_analysis:
        exit()

    # GATHER image-to-covered-shapes map
    # TODO move this down, and only look at this data if the ordering is not taken from a file
    image_2_shapes = {}
    for num_parallels in allowed_parallels:
        with open(f'{args.parallels_dir}/data{num_parallels}pars.json', 'r') as f:
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

    ###################
    # OPTIMAL ACCURACIES
    series_extremum = []
    clean_accuracies = image_2_accuracy[args.images_min:args.images_max]
    indices = list(range(len(clean_accuracies))) # problem if images_min != 0
    items_to_pop = []
    for ind in range(len(clean_accuracies)):
        if clean_accuracies[ind] == None:
            items_to_pop.append(ind)

    for ind in items_to_pop:
        clean_accuracies.pop(ind)
        indices.pop(ind)

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
    # plot_max_min = {'fig_name':'shape-max-min', 'series_info':[]}
    # plot_max_min['series_info'] = series_extremum
    # createAccuracyPlot(plot_max_min, image_2_accuracy)

    ###################
    # RANDOM ACCURACIES
    series_random = []
    plot_random = {'fig_name':f'random-only-{args.num_random_plots}', 'series_info':[]}
    for _ in range(args.num_random_plots):
        image_ids = list(range(args.images_min, args.images_max))
        for ind in items_to_pop:
            image_ids.pop(ind)
        random.shuffle(image_ids)
        random_plot_info = {'label':None, 'img_seq':image_ids, 'alpha':0.05, 'color':'black'}
        series_random.append(random_plot_info)
    median_random = [findMedianSample(series_random, image_2_accuracy, 'random', 'gold')]
    
    if args.plot_random:
        plot_random['series_info'] = series_random + series_extremum + median_random
        createAccuracyPlot(plot_random, image_2_accuracy)

    ###################
    # SHAPE COVERAGE
    if args.plot_shape_cvg:
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

                ordering_dir = f'{args.data_dir}/ordering/{t}-{pars}'
                ordering_path = f'{ordering_dir}/ids{args.images_min}_{args.images_max}.pt'

                if os.path.exists(ordering_path):
                    # USE SAVED FILE FOR shape-based ORDERING INFO
                    all_image_orderings = torch.load(ordering_path).tolist()
                    image_orderings = random.sample(all_image_orderings, args.num_random_plots)
                else:
                    image_orderings_pt, _ = getShapeCoverage(t, image_2_shapes[str(pars)], 2000, items_to_pop)
                    # 2000 replaced by args.num_random_plots
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
    
    ################
    # UNUSED FOR NOW
    ################
    exit() 
    ###################
    # SHAPE COVERAGE W/ SPECIFIC INPUT
    print('  Measuring Shape Coverage With Specific Starting Image')
    # clean_shapes = image_2_shapes[args.images_min:]
    config_2_color = {
        (0, 'worst'):('lime', 'darkgreen'),
        (2147483647, 'worst'):('lime', 'darkgreen'),
        (0, 'global'):('orange', 'sienna'),
        (2147483647, 'global'):('orange', 'sienna'),
        (0, 'best'):('magenta', 'darkmagenta'),
        (2147483647, 'best'):('magenta', 'darkmagenta'),
    }
    # get global accuracy
    all_cor, all_tot= 0, 0
    for acc in clean_accuracies:
        all_cor += acc['cor']
        all_tot += acc['tot']
    global_acc = (all_cor / all_tot)
    for pars in allowed_parallels: # TODO allowed_oaralels

        series_starting = []
        plot_all_starting_points = {'fig_name':f'starting-all-{pars}', 'series_info':[]}
        medians_starting = []

        for s in ["worst", "global", "best"]:
            main_color = config_2_color[(pars, s)][0]
            median_color = config_2_color[(pars, s)][1]
            plot_starting_current = {'fig_name':f'starting-{s}-{pars}', 'series_info':[]}

            ordering_dir = f'{args.data_dir}/ordering-start-worst/{s}/{dec_short}-{pars}/{q[8:]}'
            ordering_path = f'{ordering_dir}/{args.images_min}-{args.images_max}.pt'

            if os.path.exists(ordering_path):
                # USE SAVED FILE FOR shape-based ORDERING INFO
                all_image_orderings = torch.load(ordering_path).tolist()
                image_orderings = random.sample(all_image_orderings, args.num_random_plots)
            else:
                #################
                # FIND POSSIBLE STARTING IMAGES
                if s == "worst":
                    image_id_seq_to_query = increasing_acc_idx
                elif s == "global":
                    # This is only relevant for rel questions.
                    # For obj questions, it essentially is giving the same as "best"
                    sort_diff_key = lambda y:abs((y[1]["cor"]/y[1]["tot"]) - global_acc)
                    image_id_seq_to_query = [x for x, _ in sorted(zip(indices, clean_accuracies), key=sort_diff_key)]
                elif s == "best":
                    image_id_seq_to_query = decreasing_acc_idx
                
                # Get possible starting image ids
                image_accuracies_seq = [clean_accuracies[i]["cor"] / clean_accuracies[i]["tot"] for i in image_id_seq_to_query]
                possible_starting_ids = []
                for id, acc in enumerate(image_accuracies_seq):
                    if acc == image_accuracies_seq[0]:
                        #acc = ideal acc
                        possible_starting_ids.append(image_id_seq_to_query[id])
                    else:
                        break

                # case where <5% of the images have the ideal accuracy
                if len(possible_starting_ids) < int(0.05 * tot_num_imgs):
                    
                    for im_id in image_id_seq_to_query[len(possible_starting_ids):int(0.05*tot_num_imgs)]:
                        # add something here that maybe only adds an imaghe if accuracy is within 5% ofthe ideal
                        possible_starting_ids.append(im_id)

                # print(f'{s}: {[round(clean_accuracies[i]["cor"] / clean_accuracies[i]["tot"], 3) for i in possible_starting_ids]}')
                print(f'{s}: {len(possible_starting_ids)} ({image_accuracies_seq[0]}, {image_accuracies_seq[len(possible_starting_ids)-1]})')

                #################
                # GET SHAPE SEQ

                image_orderings_pt, _ = getShapeCoverage("worst", image_2_shapes[str(pars)], 500, possible_starting_ids) #TODO
                # 2000 replaced by args.num_random_plots
                # 2000 taking too much time for 1000 images, so we do with 500

                # TEMPORARY - SAVE 2000 shape-based orderings in json
                if not os.path.isdir(ordering_dir):
                    os.makedirs(ordering_dir)
                s_time = time.time()
                torch.save(image_orderings_pt, ordering_path)
                print(f'    Saved shape-based orderings with {s} starting image at {ordering_path} in {time.time()-s_time}')
                # continue
                # TEMPORARY - END
                image_orderings = image_orderings_pt.tolist()

            current_series_starting = []
            for ordering in image_orderings:
                starting_plot_info = {'label':None, 'img_seq':ordering, 'alpha':0.05, 'color': main_color}
                current_series_starting.append(starting_plot_info)

            median_current = findMedianSample(current_series_starting, image_2_accuracy, f'{s}-{pars}', median_color)

            # generate current series + random
            plot_starting_current['series_info'] = series_random + series_extremum + current_series_starting + median_random + [median_current]
            
            createAccuracyPlot(plot_starting_current, image_2_accuracy)

            # Management for shape-all
            medians_starting.append(median_current)
            series_starting.extend(current_series_starting)

        plot_all_starting_points['series_info'] = series_random + series_extremum + series_starting + median_random + medians_starting
        # savePdfPlot("shape") # IMPORTANTY
        createAccuracyPlot(plot_all_starting_points, image_2_accuracy)

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
