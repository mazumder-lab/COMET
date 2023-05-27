import pandas as pd
from sklearn.model_selection import train_test_split


def subsample_movielens_dataset(
    existing_dataset_bpath="../../raw/movielens/",
    subsample_dataset_bpath="../../raw/movielens_s/",
    split_proportion=1/15,   # train_prop for first train/test split, then val proportion for train/val split
    seed=0
):
    # TODO: finish this function; subsample only the ratings from users with a lot of ratings; same for movies
    # faire un groupby sur les users, faire un count des ratings >= 1 pr ces users
    # idem pr les films

    # TRAIN
    movielens_shared_features_train = pd.read_csv(
        existing_dataset_bpath + "shared_features/shared_features_train.csv",
        header=None
    )
    movielens_rating_regression_targets_train = pd.read_csv(
        existing_dataset_bpath + "rating_regression_targets/rating_regression_targets_train.csv",
        header=None
    )
    movielens_rating_regression_targets_train.rename(
        columns={0: "rating_regression_targets"}, 
        inplace=True
    )
    movielens_watch_classification_targets_train = pd.read_csv(
        existing_dataset_bpath + "watch_classification_targets/watch_classification_targets_train.csv",
        header=None
    )
    movielens_watch_classification_targets_train.rename(
        columns={0: "watch_classification_targets"}, 
        inplace=True
    )

    movielens_train = pd.concat(
        [
            movielens_shared_features_train, 
            movielens_rating_regression_targets_train,
            movielens_watch_classification_targets_train
        ], 
        axis=1
    )

    # we isolate the user ids of the users with sufficient nonnull ratings
    user_value_counts = movielens_train[
        movielens_train["rating_regression_targets"] > 0
    ][0].value_counts()
    # print(sum(movielens_train["ratings_regression_targets"] > 0)) # 787665 avec ratings explicites
    user_value_counts = user_value_counts[user_value_counts > user_value_counts.quantile(0.75)]
    user_indices_to_keep = list(user_value_counts.index)

    # we isolate the rows of these users in the dataset
    movielens_train_relevant_users = movielens_train[
        movielens_train[0].isin(user_indices_to_keep)
    ]

    # we isolate their nonnull ratings
    movielens_train_explicit_ratings = movielens_train_relevant_users[
        movielens_train_relevant_users["rating_regression_targets"] > 0
    ]
    # print(len(movielens_train_explicit_ratings))     # we are left with 500K nonnull ratings; let's sample a small proportion of them
    sample_proportion = len(movielens_train) / len(movielens_train_explicit_ratings) * split_proportion

    # final sampled explicit ratings from the train set:
    movielens_train_sub_explicit = movielens_train_explicit_ratings.sample(
        frac=sample_proportion
    )
    user_indices_train_sub = list(movielens_train_sub_explicit[0].unique())

    movielens_train_final_users = movielens_train[
        movielens_train[0].isin(user_indices_train_sub)
    ]
    movielens_train_implicit_ratings = movielens_train_final_users[
        movielens_train_final_users["watch_classification_targets"] < 1
    ]
    movielens_train_sub_implicit = movielens_train_implicit_ratings.sample(
        n=len(movielens_train_sub_explicit)
    )
    movielens_train_sub = pd.concat(
        [movielens_train_sub_explicit, movielens_train_sub_implicit],
        axis=0
    ).sample(frac=1)
    print(len(movielens_train_sub))


    # VALIDATION
    movielens_shared_features_val = pd.read_csv(
        existing_dataset_bpath + "shared_features/shared_features_val.csv",
        header=None
    )
    movielens_rating_regression_targets_val = pd.read_csv(
        existing_dataset_bpath + "rating_regression_targets/rating_regression_targets_val.csv",
        header=None
    )
    movielens_rating_regression_targets_val.rename(
        columns={0: "rating_regression_targets"}, 
        inplace=True
    )
    movielens_watch_classification_targets_val = pd.read_csv(
        existing_dataset_bpath + "watch_classification_targets/watch_classification_targets_val.csv",
        header=None
    )
    movielens_watch_classification_targets_val.rename(
        columns={0: "watch_classification_targets"}, 
        inplace=True
    )

    movielens_val = pd.concat(
        [
            movielens_shared_features_val, 
            movielens_rating_regression_targets_val,
            movielens_watch_classification_targets_val
        ], 
        axis=1
    ) 
    # print(len(movielens_val))

    movielens_val_relevant_rows = movielens_val[
        movielens_val[0].isin(user_indices_train_sub)
    ]
    # print(len(movielens_val_relevant_rows))
    sample_proportion = len(movielens_val) / len(movielens_val_relevant_rows) * split_proportion

    movielens_val_sub = movielens_val_relevant_rows.sample(
        frac=sample_proportion
    )
    print(len(movielens_val_sub))
 

    # TEST
    movielens_shared_features_test = pd.read_csv(
        existing_dataset_bpath + "shared_features/shared_features_test.csv",
        header=None
    )
    movielens_rating_regression_targets_test = pd.read_csv(
        existing_dataset_bpath + "rating_regression_targets/rating_regression_targets_test.csv",
        header=None
    )
    movielens_rating_regression_targets_test.rename(
        columns={0: "rating_regression_targets"}, 
        inplace=True
    )
    movielens_watch_classification_targets_test = pd.read_csv(
        existing_dataset_bpath + "watch_classification_targets/watch_classification_targets_test.csv",
        header=None
    )
    movielens_watch_classification_targets_test.rename(
        columns={0: "watch_classification_targets"}, 
        inplace=True
    )

    movielens_test = pd.concat(
        [
            movielens_shared_features_test, 
            movielens_rating_regression_targets_test,
            movielens_watch_classification_targets_test
        ], 
        axis=1
    )  

    movielens_test_relevant_rows = movielens_test[
        movielens_test[0].isin(user_indices_train_sub)
    ]
    # print(len(movielens_test_relevant_rows))
    sample_proportion = len(movielens_test) / len(movielens_test_relevant_rows) * split_proportion

    movielens_test_sub = movielens_test_relevant_rows.sample(
        frac=sample_proportion
    )
    print(len(movielens_test_sub))


    print(f"\nSaving shared features to {subsample_dataset_bpath}...")
    shared_features_train = pd.DataFrame(movielens_train_sub[[0,1]])
    shared_features_train.to_csv(subsample_dataset_bpath + "shared_features/shared_features_train.csv", index=False, header=False)
    shared_features_val = pd.DataFrame(movielens_val_sub[[0,1]])
    shared_features_val.to_csv(subsample_dataset_bpath + "shared_features/shared_features_val.csv", index=False, header=False)
    shared_features_test = pd.DataFrame(movielens_test_sub[[0,1]])
    shared_features_test.to_csv(subsample_dataset_bpath + "shared_features/shared_features_test.csv", index=False, header=False)

    print(f"\nSaving targets from first task to {subsample_dataset_bpath}...")
    regression1_targets_train = pd.DataFrame(movielens_train_sub["watch_classification_targets"])
    regression1_targets_train.to_csv(subsample_dataset_bpath + "watch_classification_s_targets/watch_classification_s_targets_train.csv", index=False, header=False)
    regression1_targets_val = pd.DataFrame(movielens_val_sub["watch_classification_targets"])
    regression1_targets_val.to_csv(subsample_dataset_bpath + "watch_classification_s_targets/watch_classification_s_targets_val.csv", index=False, header=False)
    regression1_targets_test = pd.DataFrame(movielens_test_sub["watch_classification_targets"])
    regression1_targets_test.to_csv(subsample_dataset_bpath + "watch_classification_s_targets/watch_classification_s_targets_test.csv", index=False, header=False)

    print(f"\nSaving targets from second task to {subsample_dataset_bpath}...")
    regression2_targets_train = pd.DataFrame(movielens_train_sub["rating_regression_targets"])
    regression2_targets_train.to_csv(subsample_dataset_bpath + "rating_regression_s_targets/rating_regression_s_targets_train.csv", index=False, header=False)
    regression2_targets_val = pd.DataFrame(movielens_val_sub["rating_regression_targets"])
    regression2_targets_val.to_csv(subsample_dataset_bpath + "rating_regression_s_targets/rating_regression_s_targets_val.csv", index=False, header=False)
    regression2_targets_test = pd.DataFrame(movielens_test_sub["rating_regression_targets"])
    regression2_targets_test.to_csv(subsample_dataset_bpath + "rating_regression_s_targets/rating_regression_s_targets_test.csv", index=False, header=False)
    


    # get the users sampled
    # get some 0 rating rows from these users in the original dataset
    # concat the rows

    return
    # print(user_value_counts.sum()) # renvoie bien len(movielens_total["ratings_regression_targets"] > 0)
    indices_to_keep = []
    max_val = int(len(movielens_total) * split_proportion * 4)
    current_sum = 0
    print(max_val)
    for user_index, explicit_ratings_count in user_value_counts.iteritems():
        current_sum += explicit_ratings_count
        indices_to_keep.append(user_index)
        if current_sum > max_val:
            break
    print(indices_to_keep)


    # print(user_value_counts[:50])
    # for i in range(len(user_value_counts))

    # print(movielens_total.info())
    # print((movielens_total["ratings_regression_targets"] > 0).sum())

    # print(movielens_df.info())
    return

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
    # move to this directory, then do python -m
    subsample_movielens_dataset()#"../../raw/movielens_unprocessed/movielens1M_multitask.csv")