import numpy as np
from time import time
from scipy.spatial.distance import minkowski
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from delta_sum import DeltaSum

class CoReg():

    def __init__(self, k_neighbours_1, k_neighbours_2, distance_order_1, distance_order_2, number_of_iterations, pool_size, verbose=False):
        self._k_neighbours_1 = k_neighbours_1
        self._k_neighbours_2 = k_neighbours_2
        self._distance_order_1 = distance_order_1
        self._distance_order_2 = distance_order_2
        self._number_of_iterations = number_of_iterations
        self._pool_size = pool_size
        self._regressor_1 = KNeighborsRegressor(n_neighbors=self._k_neighbours_1, p=self._distance_order_1)
        self._regressor_2 = KNeighborsRegressor(n_neighbors=self._k_neighbours_2, p=self._distance_order_2)
        self._verbose = verbose

    def set_datasets(self, labeled_dataset, unlabeled_dataset):
        self._labeled_dataset = labeled_dataset
        self._unlabeled_dataset = unlabeled_dataset

    def _prepare_data_for_training(self):

        self._unlabeled_pool = self._unlabeled_dataset.sample(self._pool_size)
        self._unlabeled_dataset.drop(self._unlabeled_pool.index.values, inplace=True)

        y = self._labeled_dataset.iloc[:, -2:].values
        x = self._labeled_dataset.iloc[:, :-2].values

        self._train_dataset_x, self._test_dataset_x, self._train_dataset_y, self._test_dataset_y = train_test_split(x, y, test_size = .2, shuffle = False)

        self._labeled_set_1_x = self._train_dataset_x
        self._labeled_set_2_x = self._train_dataset_x
        self._labeled_set_1_y = self._train_dataset_y
        self._labeled_set_2_y = self._train_dataset_y


    def train(self):
        i = 0

        self._prepare_data_for_training()

        list_of_regressors = [self._regressor_1, self._regressor_2]
        labeled_x_list = [self._labeled_set_1_x, self._labeled_set_2_x]
        labeled_y_list = [self._labeled_set_1_y, self._labeled_set_2_y]

        self._fit_regressors(labeled_x_list, labeled_y_list, list_of_regressors)
        labeled_dataset_changes = True
        while i < self._number_of_iterations and labeled_dataset_changes:
            if self._verbose:
                print('Started iteration: ' + str(i+1))

            points = [1, 2]
            for index, regressor in enumerate(list_of_regressors):

                current_labeled_x_set = labeled_x_list[index]
                current_labeled_y_set = labeled_y_list[index]

                new_point = self._find_unlabeled_points_to_add(regressor, current_labeled_x_set, current_labeled_y_set)

                if new_point is not None:
                    
                    points[index] = new_point

                    self._replenish_unlabeled_pool()
                else:
                    points[index] = None

            if points.count(None) == len(points):
                labeled_x_list[0], labeled_x_list[1], labeled_y_list[0], labeled_y_list[1] = self._add_new_points(labeled_x_list, labeled_y_list, points)

                self._fit_regressors(labeled_x_list, labeled_y_list, list_of_regressors)
                self._evaluate_regressors(list_of_regressors, labeled_x_list, labeled_y_list)
            else:
                labeled_dataset_changes = False
            
            if self._verbose:
                print('Regressor results at the end of iteration:', i+1)
                print('Ended iteration:', i+1)
                i += 1

        print('Regressor results at the end of training')
        self._evaluate_regressors(list_of_regressors, labeled_x_list, labeled_y_list)

    def _add_new_points(self, labeled_feature_set_list, labeled_target_set_list, points):
        if points[1][0] is not None:
            x_1 = points[1][0].flatten()
            y_1 = points[1][1].flatten()

            new_set_x_1 = np.vstack((labeled_feature_set_list[0], x_1))
            new_set_y_1 = np.vstack((labeled_target_set_list[0], y_1))

        if points[0][0] is not None:
            x_2 = points[0][0].flatten()
            y_2 = points[0][1].flatten()

            new_set_x_2 = np.vstack((labeled_feature_set_list[1], x_2))
            new_set_y_2 = np.vstack((labeled_target_set_list[1], y_2))

        return new_set_x_1, new_set_x_2, new_set_y_1, new_set_y_2

    def _fit_regressors(self, labeled_feature_set_list, labeled_target_set_list, regressors):
        regressors[0].fit(labeled_feature_set_list[0], labeled_target_set_list[0])
        regressors[1].fit(labeled_feature_set_list[1], labeled_target_set_list[1])

    def _evaluate_regressors(self, list_of_regressors, labeled_x_list, labeled_y_list):
        predictions_on_training_set_1 = list_of_regressors[0].predict(labeled_x_list[0])
        predictions_on_training_set_2 = list_of_regressors[1].predict(labeled_x_list[1])

        predictions_on_test_set_1 = list_of_regressors[0].predict(self._test_dataset_x)
        predictions_on_test_set_2 = list_of_regressors[1].predict(self._test_dataset_x)

        print('MSE of regressor 1 on training set:', mean_squared_error(predictions_on_training_set_1, labeled_y_list[0]))
        print('MSE of regressor 2 on training set:', mean_squared_error(predictions_on_training_set_2, labeled_y_list[1]))
        
        print('MSE of regressor 1 on test set:', mean_squared_error(predictions_on_test_set_1, self._test_dataset_y))
        print('MSE of regressor 2 on test set:', mean_squared_error(predictions_on_test_set_2, self._test_dataset_y))

    def _find_unlabeled_points_to_add(self, regressor, feature_dataset, target_dataset):

        deltas = self._compute_deltas(regressor, feature_dataset, target_dataset)
        
        if len(deltas) > 0:
            deltas.sort(reverse=True)
            
            new_x = self._unlabeled_pool.iloc[deltas[0]._index].values.reshape(1, -1)
            
            new_y = regressor.predict(new_x)
            feature_dataset = np.vstack((feature_dataset, new_x))
            target_dataset = np.vstack((target_dataset, new_y))

            self._unlabeled_pool.drop(self._unlabeled_pool.index[deltas[0]._index], inplace=True)

            return (new_x, new_y)
        else:
            return None

    def _get_new_regressor(self, k_neighbors, distance_order):
        return KNeighborsRegressor(n_neighbors=self._k_neighbours_1, p=self._distance_order_1)

    def _compute_deltas(self, regressor, feature_dataset, target_dataset):
        deltas = []
        real_index = 0
        for _, row in self._unlabeled_pool.iterrows():

            x = row.values.reshape(1, -1)
            prediction = regressor.predict(x)
            x_neighbors_indices = regressor.kneighbors(x, return_distance=False)[0]
            
            new_regressor = self._get_new_regressor(regressor.get_params()['n_neighbors'], regressor.get_params()['p'])
            new_labeled_dataset = np.vstack((feature_dataset, x))
            new_labels = np.vstack((target_dataset, prediction))

            new_regressor.fit(new_labeled_dataset, new_labels)

            delta = self._compute_delta(x_neighbors_indices, feature_dataset, target_dataset, regressor, new_regressor)

            deltas.append(DeltaSum(delta[0][0], delta[0][1], real_index))
            real_index += 1

        return deltas

    def _compute_delta(self, neighbors_indices, feature_dataset, target_dataset, regressor, new_regressor):
        delta = 0
        
        for i in neighbors_indices:
            
            x_value = feature_dataset[i].reshape(1, -1)
            y_value = target_dataset[i]

            regressor_error = y_value - regressor.predict(x_value)

            squared_regressor_error = regressor_error ** 2

            new_regressor_error = y_value - new_regressor.predict(x_value)
            squared_new_regressor_error = new_regressor_error ** 2

            delta += squared_regressor_error - squared_new_regressor_error
        
        return delta

    def _replenish_unlabeled_pool(self):
        new_pool_data = self._unlabeled_dataset.sample(self._pool_size - len(self._unlabeled_pool))

        self._unlabeled_dataset.drop(new_pool_data.index.values, inplace=True)

        self._unlabeled_pool = self._unlabeled_pool.append(new_pool_data)

    def _print_deltas(self, deltas):
        for i in range(len(deltas)): print(deltas[i], sep='\n')