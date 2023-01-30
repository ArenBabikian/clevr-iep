import argparse
import os
import h5py
import iep.utils as utils
from tqdm import tqdm
import json
import analysis_util
import pingouin
from scipy import stats
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--questions_dir', default=None) # gt questions
parser.add_argument('--data_dir', type=str, default='data/answers')
parser.add_argument('--vocab_json', default=None)
parser.add_argument('--images_min', default=0, type=int)
parser.add_argument('--images_max', default=300, type=int)
parser.add_argument('--figs_dir', default=None)

parser.add_argument('--save_graph_2_accuracy', default=False, action='store_true')
parser.add_argument('--save_graph_2_accuracy_distrib', default=False, action='store_true')
parser.add_argument('--save_accuracy_2_shape', default=False, action='store_true')
parser.add_argument('--save_stat_sig', default=False, action='store_true')

parser.add_argument('--print_accuracy', default=False, action='store_true')

allowed_parallels = [0, 2147483647]

encoders = ['resnet101']
decoders = ['9k', '18k', '700k_strong']
suffixes = ['obj-cnt', 'obj-ex', 'rel-cnt', 'rel-ex']
graph_ids = ['6_000', '6_003', '6_312', '6_rand_nsga']
# graph_ids = ['6_rand_nsga']

# TEMP
# decoders = ['9k']
# # decoders = ['9k', '18k']
# decoders = ['700k_strong']
# suffixes = [ 'rel-ex']
suffixes = ['obj-cnt', 'obj-ex']
# suffixes = ['obj-ex']

suf_2_bds = {'rel':('0', '100'), 'obj':('0', '250')}

def main(args):

    # if args.save_accuracy_2_shapes:
    #     if not '6_rand_nsga' in suffixes:
    #         print("ERROR: Must look at '6_rand_nsga' if want to output save_accuracy_2_shape")
    #         exit()
    #     else:
    #         # GATHER graph_id-to-covered-shapes map
    #         pars_2_id_2_shape = {}
    #         for num_parallels in allowed_parallels:
    #             shapes_file = f'{args.data_dir}/data{num_parallels}pars.json'
    #             with open(shapes_file, 'r') as f:
    #                 all_neigh_occurrences = json.load(f)
    #             pars_2_id_2_shape[str(num_parallels)] = all_neigh_occurrences

    # if args.save_accuracy_2_shape and not args.shape_graph_2_shape:
    #     suffixes = ['6_rand_nsga']

    for en in encoders:
        all_f1_data = []
        for su in suffixes:
            for de in decoders:
                su_bds = suf_2_bds[su[:3]]

                graph_id_2_accuracies = {}
                for graph_id in graph_ids:

                    #################
                    # INITIAL SETUP
                    #################
                    images_max = int(suf_2_bds[su[:3]][1])
                    vocab = utils.load_vocab(args.vocab_json)["answer_idx_to_token"]

                    # Display info
                    # dec_short = d.split('_0')[0]
                    print(f'>>> enc={en}, dec={de}, qs={su}, graph={graph_id}, ims=[{args.images_min}..{images_max}]')

                    # get answers file
                    ans_path = f'{args.data_dir}/{graph_id}/answers/{en}/{su}/{de}_{su_bds[0]}_{su_bds[1]}__1.h5'
                    f_ans = h5py.File(ans_path, 'r')

                    # get question ground-truth file
                    questions_file_path = f'{args.questions_dir}/{graph_id}/questions/{su}.json'
                    with open(questions_file_path, 'r') as f:
                        gt_questions = json.load(f)["questions"]

                    # check if "question_index" is the same as the id in the list
                    for q_id in range(len(gt_questions)):
                        assert q_id == gt_questions[q_id]["question_index"]
                    
                    # image ground truth
                    scenes_json_path = f'{args.questions_dir}/{graph_id}/scenes.json'
                    with open(scenes_json_path, 'r') as f:
                        scene_data = json.load(f)
                        all_scenes = scene_data['scenes']

                    # for scene in all_scenes:
                    #     img_id = scene["image_index"]
                    #     num_obj = len(scene["objects"])

                    #################
                    # MEASUREMENT SETUP
                    #################

                    # GATHER IMAGE-TO-ACCURACY INFO
                    acc_save_dir = f'{args.data_dir}/{graph_id}/accuracy/{en}/{su}'
                    acc_save_path = f'{acc_save_dir}/{de}_{su_bds[0]}_{su_bds[1]}__1.json'

                    # TODO refactor this
                    if os.path.exists(acc_save_path):
                        # USE SAVED FILE FOR ACCURACY INFO
                        with open(acc_save_path, 'r') as f:
                            all_accuracy_data = json.load(f)
                        image_2_accuracy = all_accuracy_data[:images_max]
                        for i in range(args.images_min):
                            image_2_accuracy[i] = None
                        
                    else:
                        # GATHER IMAGE ACCURACY INFO FROM QUESTION RESULTS FILES

                        # GATHER image-to-related-questions-ids map
                        image_ids = list(range(images_max))
                        image_2_related_q_idx = [[] for _ in image_ids]
                        for qu in gt_questions:
                            middle_split_str = os.path.splitext(qu['image'])[0].split('_')[-2]
                            im_index = int(middle_split_str) if middle_split_str.isdigit() else qu["image_index"]
                            if im_index in range(args.images_min, images_max):
                                image_2_related_q_idx[im_index].append(qu["question_index"]) 
                                # TODO could possibly simplify this by just looking at the list id

                        # GATHER image-to-accuracy map
                        image_2_accuracy = [{"cor":None, "tot":None, "y-cor":None, "y-tot":None, "n-cor":None, "n-tot":None} for _ in image_2_related_q_idx]
                        for img_id, related_qs in enumerate(image_2_related_q_idx):
                            correct_preditions = 0
                            total_predictions = 0
                            yes_correct_preds, yes_total_preds, no_correct_preds, no_total_preds = 0, 0, 0, 0

                            #TODO
                            print(len(related_qs))
                            for q_id_2 in related_qs:
                                ms_ans = f_ans["results"][q_id_2]
                                ms_ans_str = vocab[ms_ans]
                                gt_ans_str = gt_questions[q_id_2]["answer"]
                                # print(f'{ms_ans_str}-{gt_ans_str} | ',end='')

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
                                # print()
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
                        continue
                        if not os.path.isdir(acc_save_dir):
                            os.makedirs(acc_save_dir)
                        with open(acc_save_path, 'w') as f:
                            json.dump(image_2_accuracy, f)
                        print(f'    Saving accuracies at {acc_save_path}')
                        continue
                        # TEMPORARY - END
                    graph_id_2_accuracies[graph_id] = image_2_accuracy

                    # MEASURE AND PRINT GLOBAL ACCURACY
                    def printAccuracy(prefix):
                        cor, tot = 0, 0
                        for acc_info in image_2_accuracy:
                            if acc_info != None:
                                cor += acc_info[f'{prefix}cor']
                                tot += acc_info[f'{prefix}tot']
                        general_accuracy = round(cor / tot, 3)

                        name = {"":"Global", "y-":"Yes/1", "n-":"No/0"}
                        print(f"  {name[prefix]} Accuracy = {general_accuracy} ({cor}/{tot})")

                    if args.print_accuracy:
                        printAccuracy("")
                        printAccuracy("y-")
                        printAccuracy("n-")

                #################
                # DATA ANALYSIS
                #################

                # THE TWO RELVANT MAPS NOW ARE:
                # graph_id_2_accuracies
                # pars_2_id_2_shape (if save_accuracy_2_shape)

                # #############
                # graph_id vs accuracies

                all_accuracies = []
                for ind, graph_id in enumerate(graph_ids):

                    acc_for_graph_id = graph_id_2_accuracies[graph_id]
                    for im_id, acc_info in tqdm(enumerate(acc_for_graph_id)):
                        if acc_info != None:
                            # ACC
                            accuracy = (acc_info[f'cor'] / acc_info[f'tot'])
                            analysis_util.add2list(ind, accuracy, all_accuracies)

                # ADD GT to list
                if args.save_graph_2_accuracy:
                    # with open('/mnt/c/git/clevr-iep/data/distinct2/figsPostRel-general2/six_obj_accs.json', 'r') as f:
                    #     gt_accs = json.load(f)[f'{de}-{su}']
                    # all_accuracies.append(gt_accs)
                    analysis_util.saveBox(args, all_accuracies, f'{en}/5-graph-2-accuracy-new', f'{su}-{images_max}/{de}', xl='', yl='accuracy', xt=graph_ids)

                if args.save_graph_2_accuracy_distrib:
                    analysis_util.saveDistrib(args, all_accuracies, f'{en}/5-graph-2-acc-distrib', f'{su}-{images_max}/{de}', xl='accuracy', yl='number of occurrences', xt=graph_ids)

                if args.save_stat_sig:
                    # we use the t-test since accuracy is between 0 and 1 (bounded),
                    # which means that mean and variance are also bounded.
                    # Arcuri p5, l2
                    path_dir = f'{args.figs_dir}/{en}/5-stat-sig'
                    path_file = f'{path_dir}/{su}-{images_max}/{de}.txt'
                    if not os.path.isdir(os.path.dirname(path_file)):
                        os.makedirs(os.path.dirname(path_file))

                    to_print = [">>>Statistical Significance<<<"]
                    for i_s, acc_s in enumerate(all_accuracies):
                        for j_t, acc_t in enumerate(all_accuracies[i_s+1:]):
                            t_stat, t_p_val = stats.ttest_ind(acc_s, acc_t)
                            to_print.append(('~~~~' if t_p_val>0.05 else '    ') + f'{graph_ids[i_s][:5]} v {graph_ids[i_s+j_t+1][:5]}: (t)(p={round(t_p_val, 3)}, odds={round(t_stat, 3)})')
                            
                            dffac = pingouin.mwu(acc_s, acc_t, tail='two-sided')
                            mwu_p=dffac["p-val"]["MWU"]
                            to_print.append(('~~~~' if mwu_p>0.05 else '    ') + f'{graph_ids[i_s][:5]} v {graph_ids[i_s+j_t+1][:5]}: (mwu)(p={round(mwu_p, 3)}, eff={round(dffac["CLES"]["MWU"], 3)})')
                            to_print.append('----------------------')
                    to_print.append(">>>End Statistical Significance<<<")

                    with open(path_file, 'w') as f:
                        f.write('\n'.join(to_print))
                    print(f'    Saving statistical significance measurements at {path_file}')
