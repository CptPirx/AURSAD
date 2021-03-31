# AURSAD 
<div align="left">

[comment]: <> (  <a href='https://ride.readthedocs.io/en/latest/?badge=latest'>)

[comment]: <> (      <img src='https://readthedocs.org/projects/ride/badge/?version=latest' alt='Documentation Status' height="20"/>)

[comment]: <> (  </a>)

[comment]: <> (  <a href="https://codecov.io/gh/LukasHedegaard/ride">)

[comment]: <> (    <img src="https://codecov.io/gh/LukasHedegaard/ride/branch/main/graph/badge.svg?token=SJ59JOWNAC" height="20"/>)

[comment]: <> (  </a>)
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" height="20">
  </a>
</div>

A python library for the AURSAD dataset. 
Detailed [description](https://arxiv.org/abs/2102.01409) of the dataset and [download](https://zenodo.org/record/4487073).

The library contains several useful functionalities for preprocessing the dataset for ML applications:
* Creating numpy training and test datasets for sampled data
* Creating [Keras TimeSeries generators](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/TimeseriesGenerator) 
  for sliding window data
  * Prediction and classification mode
    * In the default prediction mode, the target label of  window is the label of the next sample. This can be
      used to train a sliding window model that predicts the class of the next sample based on the window.
    * In the classification mode, the target label of a window is the most common label in that window.  
* Filtering the dataset
* Removing undesired columns as outlined in the paper
* 3 different types of labeling
    * Full sample labeling where loosening and tightening motions are labeled together
    * Separate sample labeling where loosening motion is given its own label
    * 'Tighten' sample labeling, when only the tightening parts of the whole process are labeled as normal/anomalies, 
      loosening and movement parts of the motion get its own separate labels
* Binary labels -> every anomaly is given the same label
* Subsampling the data
* Dimensionality reduction using PCA or ANOVA F-values
* One-hot label encoding
* Zero padding the samples to equalise their length
* Z-score standardisation
* Taking data only from screwdriver sensors

### Dataset
The dataset contains 2045 samples in total. The robot was sampled with frequency of 100 Hz.

| Type                     | Label | Samples | %  |
|--------------------------|-------|---------|----|
| Normal operation         | 0     | 1420    | 69 |
| Damaged screw            | 1     | 221     | 11 |
| Extra assembly component | 2     | 183     | 9  |
| Missing screw            | 3     | 218     | 11 |
| Damaged thread samples   | 4     | 3       | 0  |

Additionally, there are 2049 supplementary samples describing the loosening/screw picking motion, labeled 5.

## Installation
AURSAD has been tested on Windows 10 and Python 3.8.

### PIP installation
To install from pip with required dependencies use:
```bash
pip install aursad
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
                      loosening_samples=1, move_samples=1, drop_extra_columns=True, pad_data=True,
                      label_type='partial', binary_labels=False, standardize=False, screwdriver_only=False):
"""
    Create numpy dataset from input file

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
```

Sample usage:
```bash
import aursad

data_path = 'C:/Users/my_path/robot_data.h5'

train_x, train_y, test_x, test_y = aursad.get_dataset_numpy(data_path)
```

#### Sliding window


```bash
def get_dataset_generator(path, window_size=100, reduce_dimensionality=False, reduce_method='PCA', n_dimensions=60,
                          subsample_data=True, subsample_freq=2, train_size=0.7, random_state=42, normal_samples=1,
                          damaged_samples=1, assembly_samples=1, missing_samples=1, damaged_thread_samples=0,
                          loosening_samples=1, move_samples=1,drop_extra_columns=True, label_type='partial',
                          batch_size=256, binary_labels=False, standardize=False, screwdriver_only=False,
                          onehot_labels=True):
    """
    Create Keras sliding window generator from input file

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
        if True the target of a window [x_0, x_100] is label of x_101, if False, the target is the most common label in [x_0, x_100]
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
  ```

Sample usage:
```bash
import aursad

data_path = 'C:/Users/my_path/robot_data.h5'

train_x, train_y, test_x, test_y, train_generator, test_generator = aursad.get_dataset_generator(data_path)
```