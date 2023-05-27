import pandas as pd
from sklearn.model_selection import train_test_split


def subsample_MoE_synthetic_dataset(
    existing_dataset_bpath="../../raw/synthetic_regressions/",
    subsample_dataset_bpath="../../raw/synthetic_regressions_s/",
    split_proportion=10/34,   # we take 10/34 of the existing dataset (TODO: ou prendre 1/5?)
    seed=0
):

    x_mat_train = pd.read_csv(
        existing_dataset_bpath + "shared_features/shared_features_train.csv",
        header=None
    )
    y1_train = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression1_targets/synthetic_regression1_targets_train.csv",
        header=None        
    )
    y2_train = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression2_targets/synthetic_regression2_targets_train.csv",
        header=None
    )
    y3_train = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression3_targets/synthetic_regression3_targets_train.csv",
        header=None
    )
    y4_train = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression4_targets/synthetic_regression4_targets_train.csv",
        header=None
    )


    x_mat_val = pd.read_csv(
        existing_dataset_bpath + "shared_features/shared_features_val.csv",
        header=None
    )
    y1_val = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression1_targets/synthetic_regression1_targets_val.csv",
        header=None
    )
    y2_val = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression2_targets/synthetic_regression2_targets_val.csv",
        header=None
    )
    y3_val = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression3_targets/synthetic_regression3_targets_val.csv",
        header=None
    )
    y4_val = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression4_targets/synthetic_regression4_targets_val.csv",
        header=None
    )


    x_mat_test = pd.read_csv(
        existing_dataset_bpath + "shared_features/shared_features_test.csv",
        header=None
    )
    y1_test = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression1_targets/synthetic_regression1_targets_test.csv",
        header=None
    )
    y2_test = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression2_targets/synthetic_regression2_targets_test.csv",
        header=None
    )
    y3_test = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression3_targets/synthetic_regression3_targets_test.csv",
        header=None
    )
    y4_test = pd.read_csv(
        existing_dataset_bpath + "synthetic_regression4_targets/synthetic_regression4_targets_test.csv",
        header=None
    )


    (
        x_train_subset, 
        _,
        y1_train_subset, 
        _,  
        y2_train_subset, 
        _,    
        y3_train_subset, 
        _,  
        y4_train_subset, 
        _,         
    ) = train_test_split(
        x_mat_train, y1_train, y2_train, y3_train, y4_train,
        train_size=split_proportion,
        shuffle=True, 
        stratify=None,   # no stratification because we are not doing classification here
        random_state=seed
    )

    (
        x_val_subset, 
        _,
        y1_val_subset, 
        _,  
        y2_val_subset, 
        _,    
        y3_val_subset, 
        _,  
        y4_val_subset, 
        _,         
    ) = train_test_split(
        x_mat_val, y1_val, y2_val, y3_val, y4_val,
        train_size=split_proportion,
        shuffle=True, 
        stratify=None,   # no stratification because we are not doing classification here
        random_state=seed
    )

    (
        x_test_subset, 
        _,
        y1_test_subset, 
        _,  
        y2_test_subset, 
        _,    
        y3_test_subset, 
        _,  
        y4_test_subset, 
        _,         
    ) = train_test_split(
        x_mat_test, y1_test, y2_test, y3_test, y4_test,
        train_size=split_proportion,
        shuffle=True, 
        stratify=None,   # no stratification because we are not doing classification here
        random_state=seed
    )

    x_train_subset.to_csv(
        subsample_dataset_bpath + "shared_features/shared_features_train.csv", index=False, header=False
    )
    y1_train_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression1_s_targets/synthetic_regression1_s_targets_train.csv", index=False, header=False
    )
    y2_train_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression2_s_targets/synthetic_regression2_s_targets_train.csv", index=False, header=False
    )
    y3_train_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression3_s_targets/synthetic_regression3_s_targets_train.csv", index=False, header=False
    )
    y4_train_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression4_s_targets/synthetic_regression4_s_targets_train.csv", index=False, header=False
    )

    x_val_subset.to_csv(
        subsample_dataset_bpath + "shared_features/shared_features_val.csv", index=False, header=False
    )
    y1_val_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression1_s_targets/synthetic_regression1_s_targets_val.csv", index=False, header=False
    )
    y2_val_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression2_s_targets/synthetic_regression2_s_targets_val.csv", index=False, header=False
    )
    y3_val_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression3_s_targets/synthetic_regression3_s_targets_val.csv", index=False, header=False
    )
    y4_val_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression4_s_targets/synthetic_regression4_s_targets_val.csv", index=False, header=False
    )

    x_test_subset.to_csv(
        subsample_dataset_bpath + "shared_features/shared_features_test.csv", index=False, header=False
    )
    y1_test_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression1_s_targets/synthetic_regression1_s_targets_test.csv", index=False, header=False
    )
    y2_test_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression2_s_targets/synthetic_regression2_s_targets_test.csv", index=False, header=False
    )
    y3_test_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression3_s_targets/synthetic_regression3_s_targets_test.csv", index=False, header=False
    )
    y4_test_subset.to_csv(
        subsample_dataset_bpath + "synthetic_regression4_s_targets/synthetic_regression4_s_targets_test.csv", index=False, header=False
    )



if __name__ == "__main__":
    subsample_MoE_synthetic_dataset()
