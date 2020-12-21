from tqdm import tqdm
from sklearn import decomposition
import random

import pandas as pd
import numpy as np


def load_dataset(foldername, train_filename, shuffle=False, drop_loosen=True):
    """
    Load data from the file

    :param drop_loosen: bool,
        drop the loosening parts of the screw motion
    :param shuffle: bool,
        whether to shuffle the data
    :param foldername: string,
        folder name
    :param train_filename: string,
        train data name
    :return: pd dataframes,
        train & test data
    """
    dataframe = pd.read_hdf(foldername + train_filename)

    # Make it multiindex
    dataframe['event'] = dataframe.index
    dataframe = dataframe.set_index(['sample_nr', 'event'])
    dataframe = dataframe.reset_index('event', drop=True)
    dataframe = dataframe.set_index(dataframe.groupby(level=0).cumcount().rename('event'), append=True)

    if drop_loosen:
        dropped_df = dataframe.loc[dataframe['label'].isin([0, 1, 2, 3])]
        dataframe = dropped_df

    # Shuffle the data
    if shuffle:
        new_order = list(range(dataframe.index.get_level_values(0).max()))
        new_order = [x+1 for x in new_order]
        random.shuffle(new_order)
        newindex = sorted(dataframe.index, key=lambda x: new_order.index(x[0]))
        dataframe = dataframe.reindex(newindex)

    return dataframe


def pd_to_np(dataframe):
    """
    Transform pd dataset to np dataset

    :param dataframe: pd dataframe
    :return: np arrays,
        data & labels
    """
    # Extract the labels and create a samples vector out of it
    labels = dataframe.iloc[:, dataframe.columns.get_level_values(0) == 'label']
    labels = labels.droplevel('event')
    labels = labels[~labels.index.duplicated(keep='first')]
    labels_np = np.squeeze(labels.values)

    # Drop the labels from data
    dataframe = dataframe.drop('label', axis=1)

    dim_0 = len(dataframe.index.get_level_values(0).unique())
    dim_1 = int(len(dataframe.index.get_level_values(1)) / dim_0)
    dim_2 = dataframe.shape[1]

    dataframe_np = dataframe.values.reshape((dim_0, dim_1, dim_2))

    return dataframe_np, labels_np


def pad_df(df):
    """
    Zero pad the samples to have the same length

    :param df: df, input df
    :return: df, padded df
    """
    # 1. compute the sizes of each sample_nr
    sr_sizes = df.groupby(df.index.get_level_values(0)).size()
    # compute max size and #sample_nr
    max_size = sr_sizes.max()
    n_sample_nrs = len(sr_sizes)

    # 2. preallocate the output array and fill
    arr = np.zeros((max_size * n_sample_nrs, len(df.columns)))
    idx_lv0 = df.index.get_level_values(0)  # get sample_nr
    for i in tqdm(range(n_sample_nrs), desc='Padding data'):
        row = i*max_size
        arr[row:row + sr_sizes.iloc[i], :] =\
            df[idx_lv0 == sr_sizes.index[i]].values

    # 3. convert to dataframe
    df_ans = pd.DataFrame(
        data=arr,
        index=pd.MultiIndex.from_product([sr_sizes.index, range(max_size)]),
        columns=df.columns
    ).rename_axis(df.index.names, axis=0)

    return df_ans


def subsample(df, freq=2):
    """
    Subsample the original data for reduced size

    :param df: df, dat
    :param freq: int, every freq item will be taken
    :return: df, subsampled df
    """

    groups = df.groupby('sample_nr').cumcount() % freq

    df = df[groups == 1]

    return df


def reduce_dimensions(df, dimensions=60, method='PCA'):
    """

    :param method: string,
        the chosen dimensionality reduction method
    :param df: dataframe,
    :param dimensions: int,
        the target dimensionality
    :return:
    """
    # Copy the labels, sample_nr and event
    df = df.reset_index()
    labels = df[['label', 'sample_nr', 'event']]

    # Drop the labels, sample_nr and event
    df = df.drop(['label', 'sample_nr', 'event'], axis=1)

    # Turn data to 2d array
    data_arr = df.values

    if method == 'PCA':
        pca = decomposition.PCA(n_components=dimensions)
        pca.fit(data_arr)
        transformed_arr = pca.transform(data_arr)

    # Turn the transformed arr to df
    df = pd.DataFrame(transformed_arr)

    # Add the labels, sample_nr and event back
    df = pd.concat([df, labels], axis=1)

    # Make it multiindex
    df = df.set_index(['sample_nr', 'event'])
    df = df.reset_index('event', drop=True)
    df = df.set_index(df.groupby(level=0).cumcount().rename('event'), append=True)

    return df
