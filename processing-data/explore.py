import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from os.path import exists
from os import mkdir

min_max_ibeacons = False # change to True to see min/max values of the ibeacons
print_location = False # change to True to see location values
print_ibeacon_histograms = False # change to True to see ibeacon histograms
save_class_histograms = False
save_plots = True
plots_directory = './plots/'
    
def get_plots(dataframe):
    if not exists(plots_directory):
        mkdir(plots_directory)
    dataframe.drop(['x', 'y'], axis=1, inplace=True)
    column_names = dataframe.columns

    for index, element in enumerate(dataframe.columns):
        for _, next_element in enumerate(column_names[index+1:]):
            plt.scatter(dataframe[element].values, dataframe[next_element])
            filename = "_".join([element, next_element])

            plt.savefig("".join([plots_directory, filename]))
            plt.clf()
            plt.close()

def ascii_to_coord(loc):
    '''
        Convert ASCII location to numerical coordinate

        Parameters:
            - loc = location as a letter

        Returns:
            - numerical coordinate scaled to A - W
    '''
    return ord(loc.upper()) - 64

def extract_new_features_and_save_them(dataframe, file_name):
    dataframe = dataframe.abs()

    if 'x' in dataframe.columns:
        column_names = list(dataframe.columns.values[:-2])
    else:
        column_names = list(dataframe.columns.values)

    for index, element in enumerate(column_names):
        for _, next_elem in enumerate(column_names[index+1:]):
            dataframe["_".join([element, next_elem])] = dataframe[element] - dataframe[next_elem]
        
    dataframe.drop(column_names, axis=1, inplace=True)

    if 'x' in dataframe.columns:
        x_y = dataframe[['x', 'y']]

        dataframe.drop(['x', 'y'], axis=1, inplace=True)
        dataframe['x'] = x_y['x']
        dataframe['y'] = x_y['y']

    dataframe.to_csv(file_name, sep=',', index=False)

def convert_location_to_x_y(dataframe):
    df['x'] = df['location'].str[0]
    df['y'] = df['location'].str[1:]

    df['x'] = df['x'].apply(ascii_to_coord)
    df['y'] = df['y'].astype(int)

    df.drop(['location'], axis=1, inplace=True)

if __name__ == "__main__":

    df = pd.read_csv('iBeacon_RSSI_Labeled.csv')
    dfU = pd.read_csv('iBeacon_RSSI_Unlabeled.csv')

    # Location for unlabeled dataset is always the same, i.e. ?
    # Date information is useless in this case.    
    dfU.drop(['location', 'date'], axis=1, inplace=True)
    df.drop(['date'], axis=1, inplace=True)

    if min_max_ibeacons:
        print('Minimum values in each column of labeled data: ')
        print(df.min())
        print('Maximum values in each column of labeled data: ')
        print(df.max())

        print('Minimum values in each column of unlabeled data: ')
        print(dfU.min())
        print('Maximum values in each column of unlabeled data: ')
        print(dfU.max())
        # Values of ibeacons is always between -200 and -50

    if print_location:
        print(df.location)

        print(df.location.max())
        # Unlike in the image, values along the x axis go from A to W.
    
    if save_class_histograms:
        sns.countplot(x="location", data=df)
        location_occurences = df['location'].value_counts()
        pd.set_option('display.max_rows', len(location_occurences))
        location_dictionary = location_occurences.to_dict()
        print(location_occurences)
        print(sorted(location_dictionary))

    convert_location_to_x_y(df)
    
    if print_ibeacon_histograms:
        for col in df.columns[0:10]:
            df.hist(column = col)
            plt.show()

    if save_plots:
        get_plots(df.copy(deep=True))


    # Normalize data
    float_labeled_data = df.iloc[:, :-2].values.astype(float)

    min_max_scaler = MinMaxScaler()
    scaled_labeled_data = min_max_scaler.fit_transform(float_labeled_data)
    
    labeled_data_copy = df.copy(deep=True)

    for i in range(13):
        labeled_data_copy.iloc[:, i] = scaled_labeled_data[:, i]

    float_unlabeled_data = dfU.values.astype(float)
    scaled_unlabeled_data = min_max_scaler.fit_transform(float_unlabeled_data)

    unlabeled_data_copy = dfU.copy(deep=True)
    
    for i in range(13):
        unlabeled_data_copy.iloc[:, i] = scaled_unlabeled_data[:, i]

    # Save processed data
    df.to_csv('data_labeled.csv', sep=',', index=False)
    extract_new_features_and_save_them(df.copy(deep=True), 'extracted_features.csv')
    labeled_data_copy.to_csv('labeled_data_scaled.csv', sep=',', index=False)
    extract_new_features_and_save_them(labeled_data_copy.copy(deep=True), 'extracted_features_scaled.csv')
    dfU.to_csv('data_unlabeled.csv', sep=',', index=False)
    extract_new_features_and_save_them(dfU.copy(deep=True), 'extracted_features_unlabeled.csv')
    unlabeled_data_copy.to_csv('unlabeled_data_scaled.csv', sep=',', index=False)
    extract_new_features_and_save_them(unlabeled_data_copy.copy(deep=True), 'extracted_features_scaled_unlabeled.csv')