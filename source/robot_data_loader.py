import sys
import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from source.utils import load_dataset, reduce_dimensions, subsample, pad_df, pd_to_np

sys.path.append("../")


def download_dataset(foldername):
    """
    Download the dataset.

    :param foldername: folder path,
        folder to which the data will be downloaded to
    :return:
    """
    pass


def get_dataset_numpy(foldername, filename, onehot_labels=True, sliding_window=False, window_size=200,
                      reduce_dimensionality=True, reduce_method='PCA', n_dimensions=60, subsample_data=True,
                      subsample_freq=2, pad_data=True, train_size=0.7, random_state=42, normal_samples=1,
                      damaged_samples=1, assembly_samples=1, missing_samples=1):
    """
    Create a numpy dataset from input dataframe

    :param missing_samples: float, percentage of missing samples to take
    :param assembly_samples: float, percentage of extra assembly samples to take
    :param damaged_samples: float, percentage of damaged samples to take
    :param normal_samples: float, percentage of normal samples to take
    :param random_state: int, random state for train_test split
    :param train_size: float, percentage of data as training data
    :param pad_data: bool, pad data to create even size set of samples
    :param subsample_freq: int
    :param subsample_data: bool, reduce number of events by taking every subsample_freq event
    :param reduce_dimensionality: bool, reduce dimensionality of the dataset
    :param reduce_method: string, dimensionality reduction method to be used
    :param n_dimensions: int, the target number of dimensions
    :param sliding_window: bool, create a sliding window dataset
    :param window_size: int, size of the sliding window
    :param foldername: string, folder name
    :param filename: string, train data name
    :param onehot_labels: bool, output onehot encoded labels
    :return: np arrays, train and test data & labels
    """
    data = load_dataset(foldername=foldername,
                        train_filename=filename)

    if reduce_dimensionality:
        # Reduce dimensionality
        data = reduce_dimensions(data, method=reduce_method, dimensions=n_dimensions)

    if not sliding_window:
        # Subsample the data
        if subsample_data:
            data = subsample(data, subsample_freq)

        # Pad the data
        if pad_data:
            data = pad_df(data)
    else:
        pass

    data, labels = pd_to_np(data)

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(data,
                                                        labels,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        stratify=labels)

    if onehot_labels:
        train_y = to_categorical(train_y, num_classes=len(np.unique(train_y)))
        test_y = to_categorical(test_y, num_classes=len(np.unique(test_y)))

    return train_x, train_y, test_x, test_y
