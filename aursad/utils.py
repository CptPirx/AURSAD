from sklearn import decomposition
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import sys
import tensorflow.keras as k

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest


def binarize_labels(labels):
    """
    Change the labels to binary

    :param labels: np array of digit labels
    :return:
    """
    labels = np.where(labels == 0, labels, 1)

    return labels


def create_window_generator(window, batch_size, train_x, train_y, test_x, test_y):
    """
    Create a TF generator for sliding window

    :param batch_size:
    :param test_y:
    :param test_x:
    :param train_y:
    :param train_x:
    :param window:
    :return:
    """

    # Shift the target samples by one step
    train_y = np.insert(train_y, 0, 0)
    test_y = np.insert(test_y, 0, 0)

    train_y = k.utils.to_categorical(train_y[:-1], num_classes=len(np.unique(train_y)))
    test_y = k.utils.to_categorical(test_y[:-1], num_classes=len(np.unique(test_y)))

    train_generator = k.preprocessing.sequence.TimeseriesGenerator(train_x, train_y,
                                                                   length=window,
                                                                   batch_size=batch_size)

    test_generator = k.preprocessing.sequence.TimeseriesGenerator(test_x, test_y,
                                                                  length=window,
                                                                  batch_size=batch_size)

    return train_generator, test_generator


def delete_padded_rows(data, labels, n_dimensions):
    """
    Delete padded rows from the samples and return continuous data

    :param labels:
    :param data:
    :param n_dimensions:
    :return:
    """
    labels = np.repeat(labels, data.shape[1])
    data = data.reshape(-1, n_dimensions)
    added_rows = np.where(np.all(data == 0, axis=1))
    data = data[~added_rows[0]]
    labels = labels[~added_rows[0]]

    return data, labels


def drop_columns(df):
    """

    :param df:

    :return:
    """
    df = df.reset_index()
    df = df.drop(columns=['timestamp',
                          'output_int_register_25',
                          'output_int_register_26',
                          'output_bit_register_64',
                          'output_bit_register_65',
                          'output_bit_register_66',
                          'output_bit_register_67'], axis=1)

    # Make it multiindex
    df['event'] = df.index
    df = df.set_index(['sample_nr', 'event'])
    df = df.reset_index('event', drop=True)
    df = df.set_index(df.groupby(level=0).cumcount().rename('event'), append=True)
    df = df.sort_index()

    return df


def filter_samples(df, normal_samples, damaged_samples, assembly_samples, missing_samples, damaged_thread_samples,
                   loosening_samples, move_samples):
    """
    Take the requested percentage of each data type

    :param df: df, data
    :param normal_samples: float, percentage of normal samples to take
    :param damaged_samples: float, percentage of damaged samples to take
    :param assembly_samples: float, percentage of assembly samples to take
    :param missing_samples: float, percentage of missing samples to take
    :param damaged_thread_samples: float, percentage of damaged thread hole samples to take
    :param loosening_samples: float, percentage of loosening samples to take
    :param move_samples: float, percentage of movment samples to take
    :return: df, the filtered data
    """
    # Count the sample types
    count_df = df.groupby(['sample_nr'])['label'].median()
    unique, counts = np.unique(count_df, return_counts=True)
    labels_count_dict = {A: B for A, B in zip(unique, counts)}

    # Take only the amount of samples that's needed to fill the requirement
    sampled_list = []
    for label in labels_count_dict:
        subindex = list(np.unique(df.loc[df['label'] == label].index.get_level_values(0)))

        if label == 0:
            to_take = normal_samples * labels_count_dict[0]
        elif label == 1:
            to_take = damaged_samples * labels_count_dict[1]
        elif label == 2:
            to_take = assembly_samples * labels_count_dict[2]
        elif label == 3:
            to_take = missing_samples * labels_count_dict[3]
        elif label == 4:
            to_take = damaged_thread_samples * labels_count_dict[4]
        elif label == 5:
            to_take = loosening_samples * labels_count_dict[5]
        elif label == 6:
            to_take = move_samples * labels_count_dict[6]

        sample_ids = np.random.choice(subindex, int(to_take), replace=False)
        sampled_df = df[df.index.get_level_values(0).isin(sample_ids)]
        sampled_list.append(sampled_df)

    taken_data = pd.concat(sampled_list, ignore_index=False).sort_values(['sample_nr', 'event'])

    # Reset the sample numbers
    taken_data = taken_data.reset_index()
    taken_data['sample_nr'] = (taken_data['sample_nr'] != taken_data['sample_nr'].shift(1)).astype(int).cumsum()
    taken_data['event'] = taken_data.index
    taken_data = taken_data.set_index(['sample_nr', 'event'])
    taken_data = taken_data.reset_index('event', drop=True)
    taken_data = taken_data.set_index(taken_data.groupby(level=0).cumcount().rename('event'), append=True)
    taken_data = taken_data.sort_index()

    return taken_data


def load_dataset(path):
    """
    Load data from the file

    :param: path: path to the data
    :return: pd dataframes, train & test data
    """
    if '.h5' in path:
        dataframe = pd.read_hdf(path)
    elif '.pkl' in path:
        dataframe = pd.read_pickle(path)
    else:
        print('Wrong file')
        sys.exit()

    # Make it multiindex
    dataframe['event'] = dataframe.index
    dataframe = dataframe.set_index(['sample_nr', 'event'])
    dataframe = dataframe.reset_index('event', drop=True)
    dataframe = dataframe.set_index(dataframe.groupby(level=0).cumcount().rename('event'), append=True)

    return dataframe


def pad_df(df):
    """
    Zero pad the samples to have the same length

    :param df: df, input df
    :return: df, padded df
    """
    # 1. compute the sizes of each sample_nr
    sr_sizes = df.groupby(df.index.get_level_values(0)).size()
    # Get the sample label
    labels = df.groupby(df.index.get_level_values(0))['label'].mean()
    # compute max size and #sample_nr
    max_size = sr_sizes.max()
    n_sample_nrs = len(sr_sizes)

    # 2. preallocate the output array and fill
    arr = np.zeros((max_size * n_sample_nrs, len(df.columns)))
    idx_lv0 = df.index.get_level_values(0)  # get sample_nr
    for i in tqdm(range(n_sample_nrs), desc='Padding data'):
        row = i * max_size
        arr[row:row + sr_sizes.iloc[i], :] = df[idx_lv0 == sr_sizes.index[i]].values
        arr[row:row + max_size, -1] = labels[i + 1]

    # 3. convert to dataframe
    df_ans = pd.DataFrame(
            data=arr,
            index=pd.MultiIndex.from_product([sr_sizes.index, range(max_size)]),
            columns=df.columns
    ).rename_axis(df.index.names, axis=0)

    return df_ans


def pd_to_np(df, squeeze):
    """
    Transform pd dataset to np dataset

    :param squeeze:
    :param df: pd dataframe
    :return: np arrays,
        data & labels
    """
    # Extract the labels and create a samples vector out of it
    labels = df.iloc[:, df.columns.get_level_values(0) == 'label']
    labels = labels.droplevel('event')
    if squeeze:
        labels = labels[~labels.index.duplicated(keep='first')]
        labels_np = np.squeeze(labels.values)
    else:
        labels_np = labels.values

    # Drop the labels from data
    dataframe = df.drop('label', axis=1)

    dim_0 = len(dataframe.index.get_level_values(0).unique())
    dim_1 = int(len(dataframe.index.get_level_values(1)) / dim_0)
    dim_2 = dataframe.shape[1]

    dataframe_np = dataframe.values.reshape((dim_0, dim_1, dim_2))

    return dataframe_np, labels_np


def print_info(df):
    """
    Print the info about the collected datset.

    :param df: df, the data
    :return:
    """

    # Data statistics
    # Number of total samples
    print('There are {n_samples} samples in total.'.format(n_samples=len(list(df.index.get_level_values(0).unique()))))

    # Count the different types of labels
    unique = df['label'].unique()
    count = []

    for label in unique:
        count.append(len(df.index.get_level_values(0)[df['label'] == label].unique()))

    count_dict = {unique[i]: count[i] for i in range(len(unique))}
    count_dict_percentage = {
        unique[i]: np.round(count[i] / len(list(df.index.get_level_values(0).unique())), decimals=2)
        for i in range(len(unique))}

    print('The types and counts of different labels : \n {count_dict}'.format(count_dict=count_dict))
    print('The types and counts of different labels as percentage of the total data'
          ' : \n {count_dict}'.format(count_dict=count_dict_percentage))


def reduce_dimensions(df, new_dimensions=60, method='PCA'):
    """

    :param method: string,
        the chosen dimensionality reduction method
    :param df: dataframe,
    :param new_dimensions: int,
        the target dimensionality
    :return:
    """
    # Get the dimensions
    n_samples = len(df.index.get_level_values(0).unique())
    n_events = int(len(df.index.get_level_values(1)) / n_samples)
    n_dimensions = df.shape[1]

    df = df.reset_index()
    labels = df[['label', 'sample_nr', 'event']]

    # Drop the labels, sample_nr and event
    df = df.drop(['label', 'sample_nr', 'event'], axis=1)

    # Turn data to 2d array
    data_arr = df.values

    if method == 'PCA':
        pca = decomposition.PCA(n_components=new_dimensions)
        pca.fit(data_arr)
        transformed_arr = pca.transform(data_arr)

    if method == 'ANOVA':
        labels_arr = np.repeat(labels, n_events)
        data_arr = data_arr.reshape(-1, n_dimensions)

        select_best = SelectKBest(k=new_dimensions)
        transformed_arr = select_best.fit_transform(data_arr, np.argmax(labels_arr, axis=1))

    # Turn the transformed arr to df
    df = pd.DataFrame(transformed_arr)

    # Add the labels, sample_nr and event back
    df = pd.concat([df, labels], axis=1)

    # Make it multiindex
    df = df.set_index(['sample_nr', 'event'])
    df = df.reset_index('event', drop=True)
    df = df.set_index(df.groupby(level=0).cumcount().rename('event'), append=True)

    return df


def relabel_partial(df):
    """
    Relabel the data to partial labeling, where loosening and tightening get different labels.

    :param df: df, data
    :return: df, relabeled data
    """
    df = df.reset_index()

    df['label_shifted'] = df['label'].shift(-1)
    df['label'] = np.where(df['label'] < df['label_shifted'],
                           df['label_shifted'],
                           df['label'])
    df = df.drop(['label_shifted'], axis=1)

    # Make it multiindex
    df['event'] = df.index
    df = df.set_index(['sample_nr', 'event'])
    df = df.reset_index('event', drop=True)
    df = df.set_index(df.groupby(level=0).cumcount().rename('event'), append=True)
    df = df.sort_index()

    return df


def relabel_tighten(df):
    """
    Relabel the data to tighten labeling, where where only the actual tightening process gets labeled as normal or anomaly.

    :param df: df, data
    :return: df, relabeled data
    """
    df = df.reset_index()

    # Create the movement labels
    df['Shift'] = df['output_double_register_25'].diff(periods=1)
    df['Shift_sample'] = df['sample_nr'].diff(periods=1)
    df['Shift_busy'] = df['output_bit_register_70'].diff(periods=2)

    df.loc[(df['Shift'].isin([0.0, 'Nan']) & (df['output_bit_register_70'] == False) |
            (df['Shift_sample'] == 1) &
            (df['Shift_busy'] != 1)),
           'label'] = 6

    # Drop the support columns
    df = df.drop(['Shift', 'Shift_sample', 'Shift_busy'], axis=1)

    # Create the movement sample numbers
    i = 0
    g = df.groupby((df['label'].shift() != df['label']).cumsum())

    new_df_list = []

    for k, v in enumerate(g):
        if (v[1].shape[0]) > 5:
            v[1]['sample_nr'] = i
            new_df_list.append(v[1])

            if v[1]['label'].mean() == 6.0 or v[1].shape[0] >= 100:
                i += 1

    new_df = pd.concat(new_df_list)

    # Make it multiindex
    new_df = new_df.set_index(['sample_nr', 'event'])
    new_df = new_df.reset_index('event', drop=True)
    new_df = new_df.set_index(new_df.groupby(level=0).cumcount().rename('event'), append=True)

    return new_df


def screwdriver_data(df):
    """
    Take only the screwdriver data columns

    :param df: df, the data
    :return: df, screwdriver data
    """
    print('Only screwdriver data taken')

    screwdriver_df = df[['output_double_register_24',
                         'output_double_register_25',
                         'output_double_register_26',
                         'output_double_register_27',
                         'output_bit_register_70',
                         'output_bit_register_71',
                         'output_bit_register_72',
                         'label']]

    return screwdriver_df


def shuffle_dataframe(df):
    """
    Shuffle the dataframe

    :param df: df, the data
    :return: df, shuffled dataframe
    """
    new_order = list(range(df.index.get_level_values(0).max()))
    new_order = [x + 1 for x in new_order]
    random.shuffle(new_order)
    newindex = sorted(df.index, key=lambda x: new_order.index(x[0]))
    dataframe = df.reindex(newindex)

    return dataframe


def split_sequence(sequence, window, horizon):
    """
    Split a univariate sequence into samples

    :param sequence: array, continuous data
    :param window: int, window size
    :param horizon: int, prediction horizon
    :return: x and y arrays
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + window
        out_end_ix = end_ix + horizon
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def subsample(df, freq=2):
    """
    Subsample the original data for reduced size

    :param df: df, dat
    :param freq: int, every freq item will be taken
    :return: df, subsampled df
    """
    df = df.iloc[::freq, :]

    return df


def z_score_std(train, test):
    """
    Apply z-score standardisation to the data

    :param test:
    :param train: np array
    :return: standardised np array
    """
    scalers = {}
    for i, sample in enumerate(train):
        scalers[i] = StandardScaler()
        train[i] = scalers[i].fit_transform(sample)

    for i, sample in enumerate(test):
        test[i] = scalers[i].transform(sample)

    return train, test
