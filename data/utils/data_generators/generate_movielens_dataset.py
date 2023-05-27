import pandas as pd
from sklearn.model_selection import train_test_split


def generate_movielens_dataset(
    unprocessed_path,
    split_proportions=[0.9,0.125],   # train_prop for first train/test split, then val proportion for train/val split
    seed=0
):
    movielens_df = pd.read_csv(unprocessed_path, index_col=0)
    movielens_df.columns = ['user_id', 'movie_id', 'rating', 'has_watched']
    # print(movielens_df)
    # return
    # for the keras embedding layer to have the exact voc size
    movielens_df["user_id"] -= 1
    movielens_df["movie_id"] -= 1

    x_df = movielens_df[["user_id", "movie_id"]]
    y1_df, y2_df = movielens_df["has_watched"], movielens_df["rating"]

    (
        x_train_set, 
        x_test_set,
        y1_train_set, 
        y1_test_set,  
        y2_train_set, 
        y2_test_set,           
    ) = train_test_split(
        x_df, y1_df, y2_df,
        train_size=split_proportions[0],
        shuffle=True, 
        stratify=y1_df, 
        random_state=seed
    )

    (
        x_train_set,
        x_val_set,
        y1_train_set,
        y1_val_set,  
        y2_train_set,
        y2_val_set,                
    ) = train_test_split(
        x_train_set, y1_train_set, y2_train_set,
        test_size=split_proportions[1],
        shuffle=True, 
        stratify=y1_train_set, 
        random_state=seed       
    )

    bpath = "../../raw/movielens/"
    print(f"\nSaving shared features to {bpath}...")
    shared_features_train = pd.DataFrame(x_train_set)
    shared_features_train.to_csv(bpath + "shared_features/shared_features_train.csv", index=False, header=False)
    shared_features_val = pd.DataFrame(x_val_set)
    shared_features_val.to_csv(bpath + "shared_features/shared_features_val.csv", index=False, header=False)
    shared_features_test = pd.DataFrame(x_test_set)
    shared_features_test.to_csv(bpath + "shared_features/shared_features_test.csv", index=False, header=False)

    print(f"\nSaving targets from first task to {bpath}...")
    regression1_targets_train = pd.DataFrame(y1_train_set)
    regression1_targets_train.to_csv(bpath + "watch_classification_targets/watch_classification_targets_train.csv", index=False, header=False)
    regression1_targets_val = pd.DataFrame(y1_val_set)
    regression1_targets_val.to_csv(bpath + "watch_classification_targets/watch_classification_targets_val.csv", index=False, header=False)
    regression1_targets_test = pd.DataFrame(y1_test_set)
    regression1_targets_test.to_csv(bpath + "watch_classification_targets/watch_classification_targets_test.csv", index=False, header=False)

    print(f"\nSaving targets from second task to {bpath}...")
    regression2_targets_train = pd.DataFrame(y2_train_set)
    regression2_targets_train.to_csv(bpath + "rating_regression_targets/rating_regression_targets_train.csv", index=False, header=False)
    regression2_targets_val = pd.DataFrame(y2_val_set)
    regression2_targets_val.to_csv(bpath + "rating_regression_targets/rating_regression_targets_val.csv", index=False, header=False)
    regression2_targets_test = pd.DataFrame(y2_test_set)
    regression2_targets_test.to_csv(bpath + "rating_regression_targets/rating_regression_targets_test.csv", index=False, header=False)




if __name__ == "__main__":
    # run: /opt/homebrew/Caskroom/miniforge/base/envs/deep-learning/bin/python3.9 -m generate_movielens_dataset
    # move to this directory, then do python -m
    generate_movielens_dataset("../../raw/movielens_unprocessed/movielens1M_multitask.csv")