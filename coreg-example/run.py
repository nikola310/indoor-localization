from coreg import CoReg
import pandas as pd

if __name__ == "__main__":
    k1 = 3
    k2 = 5
    p1 = 2
    p2 = 7
    max_iters = 250
    pool_size = 1000

    cr = CoReg(k1, k2, p1, p2, max_iters, pool_size, verbose=True)

    cr.set_datasets(pd.read_csv('labeled_data_scaled.csv', index_col=False), pd.read_csv('unlabeled_data_scaled.csv', index_col=False))

    cr.train()