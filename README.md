# AURSAD 
A python library for the AURSAD dataset. 
Detailed [description](https://arxiv.org/abs/2102.01409) of the dataset and [download](https://zenodo.org/record/4487073).

The library contains several useful functionalities for preprocessing the dataset for ML applications:
* Creating numpy training and test datasets for sampled data
* Creating [Keras TimeSeries generators](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/TimeseriesGenerator) 
  for sliding window data
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
    Create numpy dataset from input h5 file

    :param path: path to the data
    :param label_type: string,
        'full', 'partial' or 'tighten'
    :param drop_extra_columns: bool,
        drop the extra columns as outlined in the paper
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
    :param move_samples: float,
        percentage of movement samples to take
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
    :param binary_labels: bool,
        if True all anomalies are labeled the same
    :param standardize: bool,
        if True apply z-score standardisation
    :param pad_data: bool,
        if True pad data to equal length samples, if False return data in continuous form
    :param screwdriver_only: bool,
        take only the 4 dimensions from the screwdriver sensors

    :return: 4 np arrays,
        train and test data & labels
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
                          loosening_samples=1, move_samples=1, drop_loosen=True, drop_movement=False,
                          drop_extra_columns=True, label_type='partial', batch_size=256, binary_labels=False,
                          standardize=False, screwdriver_only=False):
    """
    Create Keras sliding window generator from input h5 file

    :param drop_movement: bool,
        drop the the movement samples
    :param path: path to the data
    :param label_type: string,
        'full', 'partial' or 'tighten'
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
    :param move_samples: float,
        percentage of movement samples to take
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
    :param binary_labels: bool,
        if True all anomalies are labeled the same
    :param standardize: bool,
        if True apply z-score standardisation
    :param screwdriver_only: bool,
        take only the 4 dimensions from the screwdriver sensors

    :return: 4 np arrays,
        train and test data & labels
    :return: keras TimeSeries generators,
        train and test generators
  ```

Sample usage:
```bash
import aursad

data_path = 'C:/Users/my_path/robot_data.h5'

train_x, train_y, test_x, test_y, train_generator, test_generator = aursad.get_dataset_generator(data_path)
```