import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torch
import csv
from numpy.lib.stride_tricks import as_strided
import random
# import matplotlib.image as img
import itertools
import nibabel as nib
from glob import glob
from sklearn.model_selection import KFold, train_test_split
from typing import Tuple, Optional, Union, Sequence


def masked_image(kspace_data, acceleration_rate=4):
    _, nx, ny, _ = kspace_data.size()
    # masking
    mask = create_mask_for_mask_type('random', [0.08], [acceleration_rate])
    # data transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    img.imsave('test.png', mask)


def read_metadata(directory, num_red=11, kfold_num=5):
    k = kfold_num
    test_full_list = [[] for _ in range(k)]
    train_full_list = [[] for _ in range(k)]
    random_selection = [[] for _ in range(k)]
    ub_id_list = [[] for _ in range(k)]
    ub_random_selection = [[] for _ in range(k)]
    b_train = [[] for _ in range(k)]
    b_val = [[] for _ in range(k)]
    ub_train = [[] for _ in range(k)]
    ub_val = [[] for _ in range(k)]
    test_list = [[] for _ in range(k)]
    F_Y_list = []
    F_M_list = []
    F_O_list = []
    M_Y_list = []
    M_M_list = []
    M_O_list = []

    with open(directory) as metalist:
        datalist = csv.reader(metalist, delimiter=',')
        next(datalist)  # Skip the header row

        young_age = 40
        old_age = 65
        id_list = [row[:] for row in datalist]  # Extract the first column values
        for row in id_list:
            if row[1] == 'F':
                if int(row[2]) <= young_age:
                    F_Y_list.append(row)
                elif int(row[2]) > old_age:
                    F_O_list.append(row)
                else:
                    F_M_list.append(row)
            elif row[1] == 'M':
                if int(row[2]) <= young_age:
                    M_Y_list.append(row)
                elif int(row[2]) > old_age:
                    M_O_list.append(row)
                else:
                    M_M_list.append(row)

    all_lists = [F_Y_list, F_O_list, F_M_list, M_Y_list, M_O_list, M_M_list]

    ###### Kfold Operation ########
    test_num = 5 * k
    test_selection = [random.sample(lst, test_num) for lst in all_lists]
    kf = KFold(n_splits=k)

    for sublist in test_selection:
        single_testset = []
        for train_index, test_index in kf.split(sublist):
            X_train, X_test = [sublist[i] for i in train_index], [sublist[i] for i in test_index]
            single_testset.append(X_test)
        for num in range(k):
            test_full_list[num].append(single_testset[num])

    for num in range(k):
        train_full_list[num] = [[item for item in lst if item not in sublist] for lst, sublist in zip(all_lists, test_full_list[num])]

        # balanced dataset
        shortest_list = min(train_full_list[num], key=len)
        shortest_length = len(shortest_list)
        random_selection[num] = [random.sample(lst, shortest_length) for lst in train_full_list[num]]
        for sub_list in random_selection[num]:
            set_size = len(sub_list) // num_red
            for i in range(set_size):
                b_train[num].extend(sub_list[11 * i: 11 * i + 10])
                b_val[num].extend(sub_list[11 * i + 10: 11 * i + 11])

        # balanced dataset
        ub_id_list[num] = [item for item in id_list if item not in list(itertools.chain(*test_full_list[num]))]
        ub_random_selection[num] = random.sample(ub_id_list[num], 6 * shortest_length)
        ub_train[num], ub_val[num] = train_test_split(ub_random_selection[num], test_size=12, random_state=42)
        test_list[num] = list(itertools.chain(*test_full_list[num]))
    ###### check duplicates
        if has_duplicates([lst[0] for lst in test_list[num]]) == False:
            print('no duplicated subject in test list', str(num))
        else:
            print('warning: duplicated subject in test list', str(num))
    # unbalanced trainingset
        if has_duplicates([lst[0] for lst in ub_train[num]]) == False:
            print('no duplicated subject in ub_train list', str(num))
        else:
            print('warning: duplicated subject in ub_train list', str(num))
    # balanced trainingset
        if has_duplicates([lst[0] for lst in b_train[num]]) == False:
            print('no duplicated subject in b_train list', str(num))
        else:
            print('warning: duplicated subject in b_train list', str(num))


    print('train:', len(b_train[0]))
    print('val:', len(b_val[0]))
    print('test:', len(test_list[0]))
    print('ub_train:', len(ub_train[0]))
    print('ub_val:', len(ub_val[0]))
    print('ub_test:', len(test_list[0]))

    return b_train, b_val, ub_train, ub_val, test_list

def has_duplicates(lst):
    seen = set()
    for item in lst:
        if item in seen:
            return True
        seen.add(item)
    return False


def write_to_csv(data, filename):
    # Extract the directory path from the filename
    directory = os.path.dirname(filename)

    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in data:
            writer.writerow(item)
    print(f'{filename} written successfully.')


def read_raw_data(data_path, data_list):
    # read data and select slices based on patient id from data_list
    # output: normalised dataset(tensor):(slices_num, width,height)
    #         kspace dataset(tensor):(slices_num,width,height,2 stands for real and imaginary part)
    raw_dataset = []
    for patient_id in data_list[0:2]:
        image_path = glob(os.path.join(data_path, patient_id, 'PROCESSED', 'MPRAGE', 'T88_111', '*t88_gfc.img'))
        image_data = nib.load(image_path[0]).get_fdata()
        raw_image = np.transpose(image_data.squeeze(-1), (1, 0, 2))
        raw_slices = select_slice(raw_image, selected_percentage=0.7)
        raw_dataset.extend(raw_slices)
    tensor_raw_dataset = torch.tensor(raw_dataset)
    norm_raw_dataset, _, _ = normalize_instance(tensor_raw_dataset)
    complex_kspace_dataset, raw_kspace_dataset = im2kp(norm_raw_dataset)

    return norm_raw_dataset, raw_kspace_dataset







if __name__ == "__main__":
    meta_path = './Dataset/oasis_cross-sectional.csv'
    # data_path = '/remote/rds/groups/idcom_imaging/data/Brain/OASIS/oasis_cross_sectional_data/'
    # data_path = 'C:/Users\s2166007\Downloads\oasis_cross-sectional_disc1.tar\disc1'
    kfold_num = 5
    train_l, val_l, ub_train_l, ub_val_l, test_l = read_metadata(meta_path, num_red=11, kfold_num=kfold_num)
    # Write train_list, val_list, and test_list to separate CSV files
    for num in range(kfold_num):
        write_to_csv(train_l[num], f'M:/projects/ReconVerse-main-2/Dataset/oasis_balanced_{num}_train.csv')
        write_to_csv(val_l[num], f'M:\projects\ReconVerse-main-2/Dataset/oasis_balanced_{num}_val.csv')
        write_to_csv(test_l[num], f'M:\projects\ReconVerse-main-2/Dataset/oasis_balanced_{num}_test.csv')
        write_to_csv(ub_train_l[num], f'M:\projects\ReconVerse-main-2/Dataset/oasis_unbalanced_{num}_train.csv')
        write_to_csv(ub_val_l[num], f'M:\projects\ReconVerse-main-2/Dataset/oasis_unbalanced_{num}_val.csv')
        write_to_csv(test_l[num], f'M:\projects\ReconVerse-main-2/Dataset/oasis_unbalanced_{num}_test.csv')

    # train_target_image, train_kspace = read_raw_data(data_path, train_l)  #
    # masked_image(train_kspace)

    print('done')

    # train = raw_dataset(root,directory)
    # src_loader = torch.utils.data.DataLoader(train, batch_size=30, shuffle=True, num_workers=0)
    # for i, data in enumerate(src_loader):
    #     inputs, labels = data
    #     print(inputs.shape, labels)
