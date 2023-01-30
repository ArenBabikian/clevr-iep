
import json
import numpy as np

def main():
    allowed_parallels = [0, 2147483647]
    allowed_parallels = [0]
    for num_parallels in allowed_parallels:
        with open(f'/mnt/c/git/clevr-iep/data/data{num_parallels}pars.json', 'r') as f:
            all_neigh_occurrences = json.load(f)

        num_obj_2_images = [[] for _ in range(11)]
        # c = 0
        for i, n_occ in all_neigh_occurrences.items():
            # c += 1
            # if c == 100:
            #     break

            num_obj = sum(n_occ)
            # if num_obj in num_obj_2_num_images:
            num_obj_2_images[num_obj].append(i)
            # else:
            #     num_obj_2_num_images[num_obj] = 1

        # analyse per num_obj

        n_obj_2_n_diff_neighs_2_im_pair = [[[] for _ in range(11)] for _ in range(11)]

        for n_ob, ims in enumerate(num_obj_2_images):
            for im_s_i, im_s in enumerate(ims):
                neighs_s = np.array(all_neigh_occurrences[im_s])
                for im_t in ims[im_s_i+1:]:
                    neighs_t = np.array(all_neigh_occurrences[im_t])

                    num_diff_neighs_f = sum(abs(neighs_s-neighs_t)) / 2

                    assert int(num_diff_neighs_f) ==  num_diff_neighs_f
                    num_diff_neighs = int(num_diff_neighs_f)

                    n_obj_2_n_diff_neighs_2_im_pair[n_ob][num_diff_neighs].append((im_s, im_t))

        # PRINT
        print('NUM IMAGES w/ n differences in multihot shape encoding')
        for n_ob, ims in enumerate(num_obj_2_images):
            if n_ob < 3:
                continue
            num_diff_neighs = [len(n_obj_2_n_diff_neighs_2_im_pair[n_ob][i]) for i in range(11)]
            print(f'{n_ob} ({len(ims)}) : {np.array(num_diff_neighs)}')

        # Find 3 models we are looking for
        print("FOUND A TRIPLES")
        not_found = True
        for i1 in num_obj_2_images[6]:
            for i2 in num_obj_2_images[6]:
                for i3 in num_obj_2_images[6]:
                    c11 = (i1, i2) in n_obj_2_n_diff_neighs_2_im_pair[6][2]
                    c12 = (i2, i1) in n_obj_2_n_diff_neighs_2_im_pair[6][2]
                    
                    c21 = (i1, i3) in n_obj_2_n_diff_neighs_2_im_pair[6][4]
                    c22 = (i3, i1) in n_obj_2_n_diff_neighs_2_im_pair[6][4]
                    
                    c31 = (i3, i2) in n_obj_2_n_diff_neighs_2_im_pair[6][6]
                    c32 = (i2, i3) in n_obj_2_n_diff_neighs_2_im_pair[6][6]

                    if (c11 or c12) and (c21 or c22) and (c31 or c32):
                        not_found = False
                        break
                if not not_found:
                    break
            if not not_found:
                break
        
        for i in [i1, i2, i3]:
            print(f'{i} : {all_neigh_occurrences[i]}')

if __name__ == '__main__':
    main()