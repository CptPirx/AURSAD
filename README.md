# URdata 
A python library for the UR Screwdriver dataset as described in [link].

The UR-data library contains several useful functionalities for preprocessing the dataset for ML applications:
* Creating numpy training and test datasets for sampled data
* Creating a [Keras TimeSeries generators](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/TimeseriesGenerator) 
  for sliding window data
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
pip install urdata
```
### Source installation
To install latest version from github, clone the source from the project repository and install with setup.py:
```bash
git clone https://github.com/CptPirx/robo-package
cd robo-package
python setup.py install --user
```
## Instructions

The package presents to user two methods: get_dataset_numpy() and get_dataset_generator().

#### Sampling
```bash
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

    :return: 4 np arrays, train and test data & labels
    """
```

Sample usage:
```bash
import urdata.urdata

data_path = 'C:/Users/my_path/robot_data.h5'

train_x, train_y, test_x, test_y = urdata.get_dataset_numpy(data_path)
```

#### Sliding window


```bash
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

    :return: 2 np arrays,
        full continous data and labels
    :return: 2 keras TimeSeries generators,
        train and test generators
    """
```

Sample usage:
```bash
import urdata.urdata

data_path = 'C:/Users/my_path/robot_data.h5'

data, labels, train_generator, test_generator = urdata.get_dataset_generator(data_path)
```