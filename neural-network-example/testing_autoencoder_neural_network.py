import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from pandas import read_csv
from pandas import DataFrame
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def get_euclidean_distances_and_mean(points1, points2):
    x1, y1 = points1
    x2, y2 = points2
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    distance_x = x1 - x2
    distance_y = y1 - y2
    squared_distance_x = distance_x ** 2
    squared_distance_y = distance_y ** 2
    distances = squared_distance_x + squared_distance_y
    euclidean_distances = np.sqrt(distances)
    return euclidean_distances, np.mean(euclidean_distances)

def create_neural_network(input_dimension):
    model = Sequential()
    model.add(Dense(50, input_dim=input_dimension, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='relu'))

    model.compile(loss='mse', optimizer=Adam(.001), metrics=['mse'])
    return model

def prepare_data_for_training(path_to_file):
    data = read_csv(path_to_file, index_col=None)

    y = data.iloc[:, -2:]
    x = data.iloc[:, :-2]

    return train_test_split(x, y, test_size = .2, shuffle = False)

def train_neural_network(neural_network, train_data, validation_data, number_of_epochs, batch_size):
    es = EarlyStopping(monitor = 'val_loss', patience = 100, verbose = 0, mode = 'auto', restore_best_weights = True)
    return neural_network.fit(x = train_data[0], y = train_data[1], validation_data = validation_data, epochs=number_of_epochs, batch_size=batch_size,  verbose=0, callbacks = [es])

def plot_history(history):
    if 'mse' in history.history.keys():
        plt.plot(history.history['mse'])
        plt.plot(history.history['val_mse'])
        plt.title('Mean squared error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
    if 'loss' in history.history.keys():
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    if 'accuracy' in history.history.keys():
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

def plot_cumulative_density_function(distances, save_file_name):
    sorted_distances = np.sort(distances)
    prob_deep = 1. * np.arange(len(sorted_distances))/(len(sorted_distances) - 1)
    
    _, axes = plt.subplots()
    axes.plot(sorted_distances, prob_deep, color='black')
    plt.title('CDF of Euclidean distance error')
    plt.xlabel('Distance (m)')
    plt.ylabel('Probability')
    plt.grid(True)
    gridlines = axes.get_xgridlines() + axes.get_ygridlines()
    
    for line in gridlines:
        line.set_linestyle('-.')
    
    # 'Figure_CDF_error.png'
    plt.savefig(save_file_name, dpi=300)
    plt.show()
    plt.close()

def run_script_on_data(filename):
    train_x, validation_x, train_y, validation_y = prepare_data_for_training(filename)

    neural_network = create_neural_network(train_x.shape[1])

    history = train_neural_network(neural_network, (train_x, train_y), (validation_x, validation_y), 1000, 1000)

    predictions = neural_network.predict(validation_x)
    _, mean_distance = get_euclidean_distances_and_mean((predictions[:, 0], predictions[:, 1]), 
                                                    (validation_y["x"], validation_y["y"]))

    print('Mean distance: ' + str(mean_distance))

    plot_history(history)

def create_autoencoder(input_dimension):
    model = Sequential()
    model.add(Dense(50, input_dim=input_dimension, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(25, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(input_dimension, activation='sigmoid')) #13

    model.compile(optimizer=Adam(.0001, clipnorm = .5, clipvalue = .5),
              loss='mse', metrics=['accuracy'])
    return model

def build_autoencoder_neural_network(autoencoder):
    #predictions = Dense(8, activation='relu')(autoencoder.layers[3].output)
    predictions = Dense(2, activation = 'relu')(autoencoder.layers[3].output)
    neural_network = Model(inputs=autoencoder.input, outputs=predictions)
    return neural_network

def set_trainable_layers(model):
    for layer in model.layers[:-2]:
        layer.trainable = False

def set_new_trainable_layers(model):
    for layer in model.layers[:-2]:
        layer.trainable = True


def run_autoencoder_scenario_on_data(path_to_unlabeled_data, path_to_labeled_data):
    unlabeled_data = read_csv(path_to_unlabeled_data, index_col=None)

    train_unlabeled, validation_unlabeled = train_test_split(unlabeled_data, test_size = .2, shuffle = False)

    autoencoder = create_autoencoder(train_unlabeled.shape[1])
    history = train_neural_network(autoencoder, (train_unlabeled, train_unlabeled), (validation_unlabeled, validation_unlabeled), 1000, 1000)
    plot_history(history)

    neural_network = build_autoencoder_neural_network(autoencoder)
    
    set_trainable_layers(neural_network)
    neural_network.compile(optimizer=Adam(.001, clipnorm = .5, clipvalue = .5),
              loss=rmse, metrics=['accuracy'])

    train_x, validation_x, train_y, validation_y = prepare_data_for_training(path_to_labeled_data)

    history = train_neural_network(neural_network, (train_x, train_y), (validation_x, validation_y), 1000, 50)

    plot_history(history)

    predictions = neural_network.predict(validation_x)
    distances, mean_distance = get_euclidean_distances_and_mean((predictions[:, 0], predictions[:, 1]), 
                                                    (validation_y["x"], validation_y["y"]))

    plot_cumulative_density_function(distances, 'CDF_train_new_decoder.png')
    print('Mean distance: ' + str(mean_distance))

    set_new_trainable_layers(neural_network)
    neural_network.compile(optimizer=Adam(.001, clipnorm = .5, clipvalue = .5),
              loss=rmse, metrics=['accuracy'])

    train_x, validation_x, train_y, validation_y = prepare_data_for_training(path_to_labeled_data)

    history = train_neural_network(neural_network, (train_x, train_y), (validation_x, validation_y), 1000, 50)

    plot_history(history)

    predictions = neural_network.predict(validation_x)
    distances, mean_distance = get_euclidean_distances_and_mean((predictions[:, 0], predictions[:, 1]), 
                                                    (validation_y["x"], validation_y["y"]))

    plot_cumulative_density_function(distances, 'CDF_train_full_neural_network.png')
    print('Mean distance: ' + str(mean_distance))


if __name__ == "__main__":
    #run_script_on_data('data_labeled.csv')
    #run_script_on_data('labeled_data_scaled.csv')
    
    run_autoencoder_scenario_on_data('extracted_features_scaled_unlabeled.csv', 'extracted_features_scaled.csv')
    
    run_autoencoder_scenario_on_data('unlabeled_data_scaled.csv', 'labeled_data_scaled.csv')

