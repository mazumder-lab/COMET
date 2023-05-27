import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import random



def generate_multi_mnist_dataset(
    seed=0, 
    fashion_mnist=False
):
    random.seed(seed)
    np.random.seed(seed)

    if not fashion_mnist:
        data = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    else:
        with open("../../raw/multi_fashion_mnist/fashion_mnist_dataset.pkl", "rb") as f:
            data = pickle.load(f)
    
    _X_trainval, _y_trainval = data[0][0], data[0][1]
    _X_test, _y_test = data[1][0], data[1][1]

    (
        _X_train,
        _X_val,
        _y_train,
        _y_val            
    ) = train_test_split(
        _X_trainval, _y_trainval,
        train_size=5/6,
        shuffle=True, 
        stratify=_y_trainval, 
        random_state=seed       
    )

    n_train = _X_train.shape[0]
    n_val = _X_val.shape[0]
    n_test = _X_test.shape[0]

    max_hw = _X_train[0].shape[0] + 2 * 4

    X_val, y_val = [], [] 
    for i, image_top_left in tqdm(enumerate(_X_val)):
        generate_agg_images(i, image_top_left, max_hw, n_val, _X_val, _y_val, X_val, y_val)
    X_val = np.array(X_val) # OK
    y_val = np.array(y_val) # OK

    X_test, y_test = [], [] 
    for i, image_top_left in tqdm(enumerate(_X_test)):
        generate_agg_images(i, image_top_left, max_hw, n_test, _X_test, _y_test, X_test, y_test)
    X_test = np.array(X_test) # OK
    y_test = np.array(y_test) # OK

    X_train, y_train = [], [] 
    for i, image_top_left in tqdm(enumerate(_X_train)):
        generate_agg_images(i, image_top_left, max_hw, n_train, _X_train, _y_train, X_train, y_train)
    X_train = np.array(X_train) # OK
    y_train = np.array(y_train) # OK

    if not fashion_mnist:
        bpath = "../../raw/multi_mnist/"
    else:
        bpath = "../../raw/multi_fashion_mnist/"
    print(f"\nSaving shared features to {bpath}...")
    with open(bpath + "shared_features/shared_features_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(bpath + "shared_features/shared_features_val.pkl", "wb") as f:
        pickle.dump(X_val, f)  
    with open(bpath + "shared_features/shared_features_test.pkl", "wb") as f:
        pickle.dump(X_test, f)  

    print(f"\nSaving targets from first task to {bpath}...")
    if not fashion_mnist:
        task_name = "multi_mnist"
    else:
        task_name = "multi_fashion_mnist"
    # with open(bpath + f"{task_name}_targets/{task_name}_targets_train.csv", "wb") as f:
    #     pickle.dump(y_train, f)
    # with open(bpath + f"{task_name}_targets/{task_name}_targets_val.csv", "wb") as f:
    #     pickle.dump(y_val, f)  
    # with open(bpath + f"{task_name}_targets/{task_name}_targets_test.csv", "wb") as f:
    #     pickle.dump(y_test, f)  
    targets_train = pd.DataFrame(y_train)
    targets_train.to_csv(bpath + f"{task_name}_targets/{task_name}_targets_train.csv", index=False, header=False)
    targets_val = pd.DataFrame(y_val)
    targets_val.to_csv(bpath + f"{task_name}_targets/{task_name}_targets_val.csv", index=False, header=False)
    targets_test = pd.DataFrame(y_test)
    targets_test.to_csv(bpath + f"{task_name}_targets/{task_name}_targets_test.csv", index=False, header=False)


    
def generate_agg_images(i, image_top_left, max_hw, n, _X, _y, X, y):
        zero_cols = np.zeros((
            image_top_left.shape[0],
            max_hw - image_top_left.shape[0]
        ))
        zero_rows = np.zeros((
            max_hw - image_top_left.shape[0], 
            max_hw
        ))

        image_top_left_shifted = np.concatenate(
            [
                np.concatenate([image_top_left, zero_cols], axis=1), 
                zero_rows
            ], 
            axis=0
        )
        
        other_images_to_pick = [j for j in range(n) if j != i]
        randomly_selected_ints = np.random.randint(low=0, high=len(other_images_to_pick)-1, size=2)
        index1, index2 = (
            other_images_to_pick[randomly_selected_ints[0]],
            other_images_to_pick[randomly_selected_ints[1]]
        )
        image_bottom_right1 = _X[index1]
        image_bottom_right1_shifted = np.concatenate(
            [
                zero_rows,
                np.concatenate([zero_cols, image_bottom_right1], axis=1)
            ], 
            axis=0
        )
        image_bottom_right2 = _X[index2]
        image_bottom_right2_shifted = np.concatenate(
            [
                zero_rows,
                np.concatenate([zero_cols, image_bottom_right2], axis=1)
            ], 
            axis=0
        )

        agg_image1 = image_top_left_shifted + image_bottom_right1_shifted
        agg_image2 = image_top_left_shifted + image_bottom_right2_shifted

        X.append(agg_image1)
        y.append([_y[i], _y[index1]])

        X.append(agg_image2)
        y.append([_y[i], _y[index2]])    



if __name__ == "__main__":
    # data = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    # # print(data[0][1][0])
    # # plt.imshow(data[0][0][0])
    # # plt.show()

    # print(data[1][0].shape)

    # k = 36

    # test_data_pt1 = data[0][0][0]   # 5

    # zero_cols = np.zeros((
    #     test_data_pt1.shape[0],
    #     k - test_data_pt1.shape[0]
    # ))

    # test_data_pt1 = np.concatenate([test_data_pt1, zero_cols], axis=1)
    # print(test_data_pt1.shape)

    # zero_rows = np.zeros((
    #     k - test_data_pt1.shape[0], 
    #     k
    # ))

    # test_data_pt1 = np.concatenate([test_data_pt1, zero_rows], axis=0)
    # print(test_data_pt1.shape)

    # # plt.imshow(test_data_pt1)
    # # plt.show()


    # test_data_pt2 = data[0][0][4]   # 0
    # zero_cols = np.zeros((
    #     test_data_pt2.shape[0],
    #     k - test_data_pt2.shape[0]
    # ))
    # test_data_pt2 = np.concatenate([zero_cols, test_data_pt2], axis=1)
    # zero_rows = np.zeros((
    #     k - test_data_pt2.shape[0], 
    #     k
    # ))
    # test_data_pt2 = np.concatenate([zero_rows, test_data_pt2], axis=0)


    # agg = test_data_pt1 + test_data_pt2
    # plt.imshow(agg)
    # plt.show()


    # FAIRE cd data/utils/data_generators AVANT d'exécuter
    # PUIS /opt/homebrew/Caskroom/miniforge/base/envs/deep-learning/bin/python -m generate_mnist_dataset
    generate_multi_mnist_dataset(fashion_mnist=False)









    

