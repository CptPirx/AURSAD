__doc__ = "Data loader for the UR robot dataset"
__author__ = "Błażej Leporowski"
__version__ = "Version 0.1 # 11/01/2021 # Initial release #"

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .utils import load_dataset, reduce_dimensions, subsample, pad_df, pd_to_np, filter_samples, relabel, drop_columns, \
    print_info, create_window_generator


def get_dataset_numpy(path, onehot_labels=True, reduce_dimensionality=False, reduce_method='PCA', n_dimensions=60,
                      subsample_data=True, subsample_freq=2, train_size=0.7, random_state=42, normal_samples=1,
                      damaged_samples=1, assembly_samples=1, missing_samples=1, damaged_thread_samples=0,
                      loosening_samples=1, drop_loosen=True, drop_extra_columns=True, label_full=False):
    """
    Create numpy dataset from input h5 file

    :param path: path to the data
    :param label_full: bool,
        both loosening and tightening are part of one label
    :param drop_extra_columns: bool,
        drop the extra columns as outlined in the paper
    :param drop_loosen: bool,
        drop the loosening columns
    :param missing_samples: float,
        percentage of missing samples to take
    :param assembly_samples: float,
        percentage of extra assembly samples to take
    :param damaged_samples: float,
        percentage of damaged samples to take
    :param normal_samples: float,
        percentage of normal samples to take
    :param loosening_samples: float,
        percentage of loosening samples to take
    :param damaged_thread_samples: float,
        percentage of damaged thread samples to take
    :param random_state: int,
        random state for train_test split
    :param train_size: float,
        percentage of data as training data
    :param subsample_freq: int,
        the frequency of subsampling
    :param subsample_data: bool,
        reduce number of events by taking every subsample_freq event
    :param reduce_dimensionality: bool,
        reduce dimensionality of the dataset
    :param reduce_method: string,
        dimensionality reduction method to be used
    :param n_dimensions: int,
        the target number of dimensions
    :param onehot_labels: bool,
        output onehot encoded labels

    :return: np arrays, train and test data & labels
    """
    data = load_dataset(path=path)

    print('Loaded data')

    if label_full:
        drop_loosen = False
        data = relabel(data)

    data = drop_columns(data, drop_extra_columns=drop_extra_columns, drop_loosen=drop_loosen)

    if subsample_data:
        print('Subsampling data')
        data = subsample(data, subsample_freq)

    if normal_samples < 1 or damaged_samples < 1 or assembly_samples < 1 or missing_samples < 1 or damaged_thread_samples < 1 or loosening_samples < 1:
        print('Filtering samples')
        data = filter_samples(data, normal_samples, damaged_samples, assembly_samples, missing_samples,
                              damaged_thread_samples, loosening_samples)

    print_info(data)

    if reduce_dimensionality:
        print('Reducing dimensionality')
        data = reduce_dimensions(data, method=reduce_method, dimensions=n_dimensions)

    print('Padding data')
    data = pad_df(data)

    data, labels = pd_to_np(data, squeeze=True)

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(data,
                                                        labels,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        stratify=labels)

    if onehot_labels:
        encoder = OneHotEncoder()
        train_y = encoder.fit_transform(X=train_y.reshape(-1, 1)).toarray()
        test_y = encoder.fit_transform(X=test_y.reshape(-1, 1)).toarray()

    return train_x, train_y, test_x, test_y


def get_dataset_generator(path, window_size=100, reduce_dimensionality=False, reduce_method='PCA', n_dimensions=60,
                          subsample_data=True, subsample_freq=2, train_size=0.7, random_state=42, normal_samples=1,
                          damaged_samples=1, assembly_samples=1, missing_samples=1, damaged_thread_samples=0,
                          loosening_samples=1, drop_loosen=True, drop_extra_columns=True, label_full=False,
                          batch_size=256):
    """
    Create Keras sliding window generator from input h5 file

    :param path: path to the data
    :param label_full: bool,
        both loosening and tightening are part of one label
    :param drop_extra_columns: bool,
        drop the extra columns as outlined in the paper
    :param drop_loosen: bool,
        drop the loosening columns
    :param missing_samples: float,
        percentage of missing samples to take
    :param assembly_samples: float,
        percentage of extra assembly samples to take
    :param damaged_samples: float,
        percentage of damaged samples to take
    :param normal_samples: float,
        percentage of normal samples to take
    :param loosening_samples: float,
        percentage of loosening samples to take
    :param damaged_thread_samples: float,
        percentage of damaged thread samples to take
    :param random_state: int,
        random state for train_test split
    :param train_size: float,
        percentage of data as training data
    :param subsample_freq: int,
        the frequency of subsampling
    :param subsample_data: bool,
        reduce number of events by taking every subsample_freq event
    :param reduce_dimensionality: bool,
        reduce dimensionality of the dataset
    :param reduce_method: string,
        dimensionality reduction method to be used
    :param n_dimensions: int,
        the target number of dimensions
    :param window_size: int,
        size of the sliding window
    :param batch_size: int,
        batch size for the sliding window generator

    :return: np arrays,
        full continous data and labels
    :return: keras TimeSeries generators,
        train and test generators
    """
    data = load_dataset(path=path)

    print('Loaded data')

    if label_full:
        drop_loosen = False
        data = relabel(data)

    data = drop_columns(data, drop_extra_columns=drop_extra_columns, drop_loosen=drop_loosen)

    if subsample_data:
        print('Subsampling data')
        data = subsample(data, subsample_freq)

    if normal_samples < 1 or damaged_samples < 1 or assembly_samples < 1 or missing_samples < 1 or damaged_thread_samples < 1 or loosening_samples < 1:
        print('Filtering samples')
        data = filter_samples(data, normal_samples, damaged_samples, assembly_samples, missing_samples,
                              damaged_thread_samples, loosening_samples)

    print_info(data)

    if reduce_dimensionality:
        print('Reducing dimensionality')
        data = reduce_dimensions(data, method=reduce_method, dimensions=n_dimensions)

    print('Padding data')
    data = pad_df(data)

    data, labels = pd_to_np(data, squeeze=False)

    data = data.reshape(-1, n_dimensions)
    added_rows = np.where(np.all(data == 0, axis=1))
    data = data[~added_rows[0]]
    labels = labels[~added_rows[0]]

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(data,
                                                        labels,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        stratify=labels)

    train_generator, test_generator = create_window_generator(train_x=train_x,
                                                              train_y=train_y,
                                                              test_x=test_x,
                                                              test_y=test_y,
                                                              window=window_size,
                                                              batch_size=batch_size)

    return data, labels, train_generator, test_generator
