import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def create_folds(
    input_csv="FinalProject/train_labels.csv",
    output_csv="FinalProject/splits_kfold.csv",
    n_splits=5,
    random_state=42
):
    print("=" * 60)
    print("CREATING LEAK-FREE STRATIFIED GROUP K-FOLD SPLITS")
    print("=" * 60)
    
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")
    print(f"Total unique writers: {df['writer_id'].nunique()}")
    print(f"Total classes: {df['label'].nunique()} (Labels {df['label'].min()} to {df['label'].max()})")
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, y=df['label'], groups=df['writer_id'])):
        df.loc[val_idx, 'fold'] = fold
        
        train_writers = set(df.loc[train_idx, 'writer_id'])
        val_writers = set(df.loc[val_idx, 'writer_id'])
        overlap = train_writers.intersection(val_writers)
        
        assert len(overlap) == 0, f"Leakage detected in fold {fold}: {overlap}"
        
        val_samples = len(val_idx)
        val_writers_count = len(val_writers)
        val_class_counts = df.loc[val_idx, 'label'].value_counts()
        min_class_count = val_class_counts.min()
        max_class_count = val_class_counts.max()
        
        print(f"Fold {fold}: {val_samples} val images | {val_writers_count} val writers | "
              f"Classes: min={min_class_count}, max={max_class_count} per class | Leakage: ZERO")
        
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully verified and saved splits to: {output_csv}")
    return df

if __name__ == "__main__":
    create_folds()
