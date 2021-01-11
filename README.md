# UR-data
A python library for the UR Screwdriver dataset as described in [link].

The UR-data library contains several useful functionalities for preprocessing the dataset for ML applications:
* Creating numpy training and test datasets from the original file
* Filtering the dataset
* Removing undesired columns as outlined in the paper
* 2 different types of labeling
    * Full sample labeling where loosening and tightening motions are labeled together
    * Separate sample labeling where loosening motion is given its own label
* Dropping the loosening labels
* Subsampling the data
* Dimensionality reduction using PCA
* One-hot label encoding
* Zero padding the samples to equalise their length

### Dataset
The dataset contains 2042 samples in total. The robot was sampled with frequency of 100 Hz, and the resulting dataset 
comes in a single hdf file of ~6 GB.

| Type                     | Label | Samples | %  |
|--------------------------|-------|---------|----|
| Normal operation         | 0     | 1420    | 70 |
| Damaged screw            | 1     | 221     | 11 |
| Extra assembly component | 2     | 183     | 9  |
| Missing screw            | 3     | 218     | 11 |

The dataset can be downloaded from here[link].

## Installation
UR-data has been tested on Windows 10 and Python 3.8.

### PIP installation
To install from pip with required dependencies use:
```bash
pip install ur-data
```
### Source installation
To install latest version from github, clone the source from the project repository and install with setup.py:
```bash
git clone https://github.com/CptPirx/robo-package
cd UR-data
python setup.py install --user
```
## Instructions

The package presents to user a single method get_dataset_numpy(). Its parameters are:
```bash
def get_dataset_numpy(path, onehot_labels=True, sliding_window=False, window_size=200,
                      reduce_dimensionality=False, reduce_method='PCA', n_dimensions=60, subsample_data=True,
                      subsample_freq=5, pad_data=True, train_size=0.7, random_state=42, normal_samples=1,
                      damaged_samples=1, assembly_samples=1, missing_samples=1, damaged_thread_samples=0,
                      loosening_samples=0, drop_loosen=True, drop_extra_columns=True, label_full=False):
    """
    Create a numpy dataset from input dataframe

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
    :param pad_data: bool, 
        pad data to create even size set of samples
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
    :param sliding_window: bool, 
        create a sliding window dataset
    :param window_size: int, 
        size of the sliding window
    :param onehot_labels: bool, 
        output onehot encoded labels
    :return: np arrays, train and test data & labels
    """
```

This method will return 4 numpy arrays: train_x, train_y, test_x and test_y. 