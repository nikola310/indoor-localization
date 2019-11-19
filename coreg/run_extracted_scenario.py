from coreg import CoReg
import pandas as pd

if __name__ == "__main__":
    k1 = 3
    k2 = 3
    p1 = 2
    p2 = 5
    max_iters = 100
    pool_size = 1000

    cr = CoReg(k1, k2, p1, p2, max_iters, pool_size, verbose=True)

    cr.set_datasets(pd.read_csv('extracted_features_scaled.csv', index_col=False), pd.read_csv('extracted_features_scaled_unlabeled.csv', index_col=False))

    cr.train()