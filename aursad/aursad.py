__author__ = "Błażej Leporowski"
__doc__ = "Data loader for the UR robot dataset"
__version__ = "Version 0.1 # 01/02/2021 # Initial release #"
__version__ = "Version 0.1.13 # 15/03/2021 # Added onehot_labels to generators #"
__version__ = "Version 0.2 # 30/03/2021 # Added different modes for time series generators #"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

try:
    from .utils import *
except:
    from utils import *


def get_dataset_generator(path, window_size=100, reduce_dimensionality=False, reduce_method='PCA', n_dimensions=60,
                          subsample_data=True, subsample_freq=2, train_size=0.7, random_state=42, normal_samples=1,
                          damaged_samples=1, assembly_samples=1, missing_samples=1, damaged_thread_samples=0,
                          loosening_samples=1, move_samples=1,drop_extra_columns=True, label_type='partial',
                          batch_size=256, binary_labels=False, standardize=False, screwdriver_only=False,
                          onehot_labels=True, prediction_mode=True):
    """
    Create Keras sliding window generator from input h5 file

    :param assembly_samples: float,
        percentage of extra assembly samples to take
    :param batch_size: int,
        batch size for the sliding window generator
    :param binary_labels: bool,
        if True all anomalies are labeled the same
    :param damaged_samples: float,
        percentage of damaged samples to take
    :param damaged_thread_samples: float,
        percentage of damaged thread samples to take
    :param drop_extra_columns: bool,
        drop the extra columns as outlined in the paper
    :param label_type: string,
        'full', 'partial' or 'tighten'
    :param loosening_samples: float,
        percentage of loosening samples to take
    :param missing_samples: float,
        percentage of missing samples to take
    :param move_samples: float,
        percentage of movement samples to take
    :param n_dimensions: int,
        the target number of dimensions
    :param normal_samples: float,
        percentage of normal samples to take
    :param onehot_labels: bool,
        output onehot encoded labels
    :param path: path to the data
    :param prediction_mode: bool,
        if True the targets are shifted, so sample x has label of sample x+1
    :param random_state: int,
        random state for train_test split
    :param reduce_dimensionality: bool,
        reduce dimensionality of the dataset
    :param reduce_method: string,
        dimensionality reduction method to be used
    :param screwdriver_only: bool,
        take only the 4 dimensions from the screwdriver sensors
    :param standardize: bool,
        if True apply z-score standardisation
    :param subsample_data: bool,
        reduce number of events by taking every subsample_freq event
    :param subsample_freq: int,
        the frequency of subsampling
    :param train_size: float,
        percentage of data as training data
    :param window_size: int,
        size of the sliding window

    :return: 4 np arrays,
        train and test data & labels
    :return: keras TimeSeries generators,
        train and test generators
    """
    data = load_dataset(path=path)

    print('Loaded data')

    if screwdriver_only:
        data = screwdriver_data(data)

    if subsample_data:
        data = subsample(data, subsample_freq)

    if label_type == 'tighten':
        print('Relabeling data')
        data = relabel_tighten(data)

    if drop_extra_columns and not screwdriver_only:
        data = drop_columns(data)

    if normal_samples < 1 or damaged_samples < 1 or assembly_samples < 1 or missing_samples < 1 or \
            damaged_thread_samples < 1 or loosening_samples < 1 or move_samples < 1:
        print('Filtering samples')
        data = filter_samples(data, normal_samples, damaged_samples, assembly_samples, missing_samples,
                              damaged_thread_samples, loosening_samples, move_samples)

    print_info(data)

    if reduce_dimensionality and not screwdriver_only:
        print('Reducing dimensionality')
        data = reduce_dimensions(data, method=reduce_method, new_dimensions=n_dimensions)

    data = pad_df(data)

    data, labels = pd_to_np(data, squeeze=True)

    if binary_labels:
        labels = binarize_labels(labels)

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(data,
                                                        labels,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        stratify=labels)

    if standardize:
        print('Standardizing data')
        train_x, test_x = z_score_std(train_x, test_x)

    train_x, train_y = delete_padded_rows(train_x, train_y, data.shape[2])
    test_x, test_y = delete_padded_rows(test_x, test_y, data.shape[2])

    if not prediction_mode:
        # Shift the target samples by one step
        train_y = np.insert(train_y[:-1], 0, 0)
        test_y = np.insert(test_y[:-1], 0, 0)

    if onehot_labels:
        encoder = OneHotEncoder()
        train_y = encoder.fit_transform(X=train_y.reshape(-1, 1)).toarray()
        test_y = encoder.fit_transform(X=test_y.reshape(-1, 1)).toarray()

    train_generator, test_generator = create_window_generator(train_x=train_x,
                                                              train_y=train_y,
                                                              test_x=test_x,
                                                              test_y=test_y,
                                                              window=window_size,
                                                              batch_size=batch_size)

    return train_x, train_y, test_x, test_y, train_generator, test_generator


def get_dataset_numpy(path, onehot_labels=True, reduce_dimensionality=False, reduce_method='PCA', n_dimensions=60,
                      subsample_data=True, subsample_freq=2, train_size=0.7, random_state=42, normal_samples=1,
                      damaged_samples=1, assembly_samples=1, missing_samples=1, damaged_thread_samples=0,
                      loosening_samples=1, move_samples=1, drop_extra_columns=True, pad_data=True,
                      label_type='partial', binary_labels=False, standardize=False, screwdriver_only=False):
    """
    Create numpy dataset from input h5 file

    :param assembly_samples: float,
        percentage of extra assembly samples to take
    :param binary_labels: bool,
        if True all anomalies are labeled the same
    :param damaged_samples: float,
        percentage of damaged samples to take
    :param damaged_thread_samples: float,
        percentage of damaged thread samples to take
    :param drop_extra_columns: bool,
        drop the extra columns as outlined in the paper
    :param label_type: string,
        'full', 'partial' or 'tighten'
    :param loosening_samples: float,
        percentage of loosening samples to take
    :param missing_samples: float,
        percentage of missing samples to take
    :param move_samples: float,
        percentage of movement samples to take
    :param n_dimensions: int,
        the target number of dimensions
    :param normal_samples: float,
        percentage of normal samples to take
    :param onehot_labels: bool,
        output onehot encoded labels
    :param pad_data: bool,
        if True pad data to equal length samples, if False return data in continuous form
    :param path: path to the data
    :param random_state: int,
        random state for train_test split
    :param reduce_dimensionality: bool,
        reduce dimensionality of the dataset
    :param reduce_method: string,
        dimensionality reduction method to be used
    :param screwdriver_only: bool,
        take only the 4 dimensions from the screwdriver sensors
    :param standardize: bool,
        if True apply z-score standardisation
    :param subsample_data: bool,
        reduce number of events by taking every subsample_freq event
    :param subsample_freq: int,
        the frequency of subsampling
    :param train_size: float,
        percentage of data as training data

    :return: 4 np arrays,
        train and test data & labels
    """
    data = load_dataset(path=path)

    print('Loaded data')

    if screwdriver_only:
        data = screwdriver_data(data)

    if subsample_data:
        data = subsample(data, subsample_freq)

    if label_type == 'tighten':
        print('Relabeling data')
        data = relabel_tighten(data)

    if drop_extra_columns and not screwdriver_only:
        data = drop_columns(data)

    if normal_samples < 1 or damaged_samples < 1 or assembly_samples < 1 or missing_samples < 1 or \
            damaged_thread_samples < 1 or loosening_samples < 1 or move_samples < 1:
        print('Filtering samples')
        data = filter_samples(data, normal_samples, damaged_samples, assembly_samples, missing_samples,
                              damaged_thread_samples, loosening_samples, move_samples)

    print_info(data)

    if reduce_dimensionality and not screwdriver_only:
        print('Reducing dimensionality')
        data = reduce_dimensions(data, method=reduce_method, new_dimensions=n_dimensions)

    data = pad_df(data)

    data, labels = pd_to_np(data, squeeze=True)

    if binary_labels:
        labels = binarize_labels(labels)

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(data,
                                                        labels,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        stratify=labels)

    if standardize:
        print('Standardizing data')
        train_x, test_x = z_score_std(train_x, test_x)

    if not pad_data:
        train_x, train_y = delete_padded_rows(train_x, train_y, data.shape[2])
        test_x, test_y = delete_padded_rows(test_x, test_y, data.shape[2])

    if onehot_labels:
        encoder = OneHotEncoder()
        train_y = encoder.fit_transform(X=train_y.reshape(-1, 1)).toarray()
        test_y = encoder.fit_transform(X=test_y.reshape(-1, 1)).toarray()

    return train_x, train_y, test_x, test_y

