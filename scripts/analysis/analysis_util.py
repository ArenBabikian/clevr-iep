import os
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics

# ####### F1

def addF1Info(ind, acc_info, l):
    if ind >= len(l):
        num_indices_to_add = ind + 1 - len(l)
        for _ in range(num_indices_to_add):
            l.append({"tp":0, "fn":0, "fp":0, "tn":0})
    
    l[ind]["tp"] += acc_info['y-cor']
    l[ind]["fn"] += acc_info['y-tot'] - acc_info['y-cor']
    l[ind]["tn"] += acc_info['n-cor']
    l[ind]["fp"] += acc_info['n-tot'] - acc_info['n-cor']

def measureF1Score(j):
    precision = j['tp']/(j['tp']+j['fp'])
    recall = j['tp']/(j['tp']+j['fn'])
    return 2 * (precision * recall) / (precision + recall)

def gatherF1Data(num_objects_2_f1_info):
    f1_scores = []
    f1_xs = []
    totals = {"tp":0, "fn":0, "fp":0, "tn":0}
    for i, j in enumerate(num_objects_2_f1_info):
        # t = j["tp"]+j["tn"]
        a = j["tp"]+j["tn"]+j["fn"]+j["fp"]
        if a == 0:
            continue

        for k in totals.keys():
            totals[k] += j[k]

        f1 = measureF1Score(j)
        # print(f'{i} objects: f1 = {f1}')
        f1_xs.append(i)
        f1_scores.append(f1)

    f1_tot_val = measureF1Score(totals)
    f1_totals = [f1_tot_val for _ in f1_xs]

    return (f1_xs, f1_scores, f1_totals)

def saveF1Fig(args, f1_results, encoder, filename):
    f1_xs, f1_scores, f1_totals = f1_results

    # Create fig
    path_dir = f'{args.figs_path}/{encoder}/f1-score'
    path_file = f'{path_dir}/{filename}.png'
    if not os.path.isdir(os.path.dirname(path_file)):
        os.makedirs(os.path.dirname(path_file))
    
    plt.plot(f1_xs, f1_scores, label = "score-per-num-obj")
    plt.plot(f1_xs, f1_totals, label="global f1 score")
    plt.xlabel('num objects')
    plt.ylabel('f1 score')
    plt.savefig(path_file)
    plt.clf()
    print(f'    Saving plot at {path_file}')

# ####### Add and print

def add2map(x,y, x_2_y):
    if x in x_2_y.keys():
        x_2_y[x].append(y)
    else:
        x_2_y[x] = [y]

def add2list(ind, item, l):
    if ind >= len(l):
        num_indices_to_add = ind - len(l)
        for _ in range(num_indices_to_add):
            l.append([])
        l.append([item])
    else:
        l[ind].append(item)

def printPairsList(l):
    print('>>>>>>')
    for x in l:
        print(x)
    print('<<<<<<<')

def printMapSimple(m):
    print('>>>>>>')
    for k, v in m.items():
        print(f'{k}: {v}')
    print('<<<<<<<')

def printMap(l, random_select = None, best_score = 1):
    # if type(x_2_y) == list:
    for x, data_list in enumerate(l):
        if data_list == []:
            continue
        new_data_list = data_list
        if random_select != None and random_select < len(data_list):
            new_data_list = random.sample(data_list, random_select)
        num_best = new_data_list.count(best_score)
        print(f"{x}: {statistics.mean(new_data_list)} ({len(new_data_list)}) ({num_best} {best_score}s ({round(num_best/len(new_data_list)*100, 1)}%))")
    # else:
    #     for x in sorted(list(x_2_y.keys())):
    #         print(f"{x}: {mean(x_2_y[x])} ({len(x_2_y[x])})")
    print("-----")

# ####### Save figures

def saveDistrib(args, l, fig_dir, filename, xl='x', yl='y', xt=None, num_bins=20):
    bins_min, bins_max = 1, 0
    for s in l:
        for i in s:
            bins_min = i if i < bins_min else bins_min
            bins_max = i if i > bins_max else bins_max

    bins = np.linspace(bins_min, bins_max, num_bins)
    for i, series in enumerate(l):
        # plt.hist(series, bins, alpha=0.33)
        y, binEdges = np.histogram(series, bins)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        plt.plot(bincenters, y, label=xt[i])
    plt.legend()
    saveFig(args,  fig_dir, filename, xl, yl)

def saveBox(args, l, fig_dir, filename, xl='x', yl='y', xt=None):
    x_ticks = list(range(len(l)))
    plt.boxplot(l, positions=x_ticks)
    plt.xticks(x_ticks, labels=xt)
    saveFig(args,  fig_dir, filename, xl, yl)

def saveFig(args, fig_dir, filename, xl, yl):
    path_dir = f'{args.figs_dir}/{fig_dir}'
    path_file = f'{path_dir}/{filename}.png'
    if not os.path.isdir(os.path.dirname(path_file)):
        os.makedirs(os.path.dirname(path_file))

    # for series_info in list_of_plot_info:
    #     plt.plot(series_info['series'], 
    #         label=series_info['label'] if 'label' in series_info else None,
    #         alpha=series_info['alpha'] if 'alpha' in series_info else 1,
    #         color=series_info['color'] if 'color' in series_info else None,
    #         linestyle=series_info['linestyle'] if 'linestyle' in series_info else 'solid'
    #     )
    plt.xlabel(xl)
    # if xt != None:
    #     plt.xticks(range(0, len(xt)), xt)
    plt.ylabel(yl)
    # plt.yscale('log')
    # plt.ylim(0, 0.02)
    # if ylim != None:
    #     plt.ylim(ylim[0], ylim[1])
    # legend_pos = "upper right" if fig_name.startswith("conv") else "lower right"
    # plt.legend(loc=legend_pos)
    # plt.legend()
    plt.savefig(path_file)
    plt.clf()
    print(f'    Saving plot at {path_file}')
