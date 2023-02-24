import argparse
import sys
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
sys.path.append('.')

# Data manipulation
import pandas as pd # for data manipulation

# Visualization
import plotly.express as px # for data visualization

# Skleran
from sklearn.manifold import MDS, TSNE # for MDS dimensionality reduction
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--feats_name', type=str, default=None)
parser.add_argument('--encoder', type=str, default=None)
parser.add_argument('--min_index', default=0, type=int)
parser.add_argument('--max_index', default=None, type=int)
parser.add_argument('--figs_dir', default=None)
parser.add_argument('--num_samples', default=100, type=int)
parser.add_argument('--num_deco_feats', default=64, type=int)

graph_ids = ['6_000', '6_003', '6_312', '6_rand_nsga']
colors = ['red', 'blue', 'green', 'black']

# graph_ids = ['6_rand_nsga', '6_312']

def main(args):

    all_colors = []
    int_rep_names = ['original', 'svd', 'pca', 'x-m', 'x-t', 's-m', 's-t', 'p-m', 'p-t']
    num_int_rep = 3

    # initialize all graph_reps
    mds2=MDS(n_components=2, 
        metric=True, 
        n_init=4, 
        max_iter=300, 
        verbose=0, 
        eps=0.001, 
        n_jobs=None, 
        random_state=42, 
        dissimilarity='euclidean')
    tsne2=TSNE(n_components=2, 
        init='random',
        n_iter=10000,
        verbose=0)

    name_2_approach = {'mds2':mds2, 'tsne2':tsne2}
    # name_2_approach = {'mds2':mds2, 'tsne2a':tsne2, 'tsne2b':tsne2, 'tsne2c':tsne2}
    all_graph_reps = {}
    for key in name_2_approach.keys():
        all_graph_reps[key] = [None for _ in range(num_int_rep)]
    
    # RARE EVENT SIMULATION
    rand_int_reps = None
    gid_2_all_inter_reps = {}
    rare_event_detection_approaches = [IsolationForest(), OneClassSVM()]

    for j, graph_id in enumerate(graph_ids):
        features_path = f'{args.data_dir}/{graph_id}/{args.feats_name}/k-{args.encoder}_{args.min_index}_{args.max_index}.h5'

        print('Loading image features from ', features_path)
        f_img = h5py.File(features_path, 'r')
        # image_features = torch.FloatTensor(f['features'])
        arr = np.asarray(f_img['features'], dtype=np.float32)[:args.num_samples]
        # print(arr.shape) # 250 images * 1024*14*14
        image_features = torch.FloatTensor(arr).flatten(start_dim=1)
        # print(image_features.size()) # 250 images * 1024*14*14 -> 250 images * 200704

        # Dimensionality reduction
        svd = TruncatedSVD(n_components=args.num_deco_feats, n_iter=5, algorithm='randomized') # 'arpack' # SPARSE
        pca = PCA(n_components=args.num_deco_feats, svd_solver='randomized') # ‘full’, ‘arpack’, ‘randomized’ # DENSE
        svd_fitted = svd.fit_transform(image_features)
        pca_fitted = pca.fit_transform(image_features)

        # visualisation points
        intermediate_rep = [image_features, svd_fitted, pca_fitted]
        all_int_reps_for_gid = intermediate_rep.copy()
        for i, irep in enumerate(intermediate_rep):
            for approach, features in all_graph_reps.items():
                f = name_2_approach[approach].fit_transform(irep)
                all_int_reps_for_gid.append(f)
                cur_fs = all_graph_reps[approach][i]
                all_graph_reps[approach][i] = f if cur_fs is None else np.vstack([cur_fs, f])

        # Add colors
        for _ in range(args.num_samples):
            all_colors.append(colors[j])

        #########
        # RARE EVENT DETECTION
        #########

        # TODO we need to do thge opposite:
        # first, train according to the images associated to a single graph, 
        # then check if the images associated to other (random) graphs are considered as inliers or outliers
        # We are expecting most of the random graphs (features) to be outliers
        if graph_id == '6_rand_nsga':
            for cur_gid, cur_int_reps in gid_2_all_inter_reps.items():
                print(f'  Preicting if {graph_id} fits in {cur_gid}')
                for red_app in rare_event_detection_approaches:
                    print(f'    Approach: {red_app}')
                    for int_rep_id, int_rep in enumerate(cur_int_reps):
                        # Fit
                        s = int_rep.shape
                        print(f'      Rep #{int_rep_id}, {int_rep_names[int_rep_id]} {s}: ', end='')
                        cl = red_app.fit(int_rep)
                        # Predict
                        out = cl.predict(all_int_reps_for_gid[int_rep_id])
                        num_in = sum(1 for prediction in out if prediction == 1)
                        num_out = sum(1 for prediction in out if prediction == -1)
                        print(f'{num_in} inliers, {num_out} outliers')
                print()
        else:
            # print('<<<<RARE EVENT DETECTION>>>>')
            gid_2_all_inter_reps[graph_id] = all_int_reps_for_gid

    # DISPLAY PLOTS
    for app_name in all_graph_reps.keys():
        print(f'<<<<{app_name}>>>>')
        num_sps = len(all_graph_reps[app_name])
        for sp_id, all_fs in enumerate(all_graph_reps[app_name]):
            plt.subplot(1, num_sps, sp_id+1)
            plt.title(int_rep_names[sp_id])
            plt.scatter(x=all_fs[:,0], y=all_fs[:,1], c=all_colors, alpha=0.5)
        plt.show()

    exit()

    # TODO Save plots
    path_dir = f'{args.figs_dir}/{args.encoder}/6-features-mds'
    path_file = f'{path_dir}/mds.png'
    if not os.path.isdir(os.path.dirname(path_file)):
        os.makedirs(os.path.dirname(path_file))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
