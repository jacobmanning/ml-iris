# ml-iris

## Machine learning for Iris dataset
  - Uses TensorFlow to create a Logistic Regression classifier
  - Full ML pipeline with tensorflow backend

Imports raw data from csv, randomizes data, preprocesses data,
splits data, trains model, tests model, saves model

Uses TensorBoard to visualize the results

## Dependencies
+ TensorFlow
  * Backend for training and model visualization
+ NumPy
  * Data preprocessing and manipulation
+ Pandas
  * Data preprocessing and manipulation
+ Matplotlib
  * Data visualization backend
+ Seaborn
  * Data visualization helper

## Usage
```python
python iris.py [-h] [--visual] [--learning_rate LEARNING_RATE]
               [--filename FILENAME] [--stddev STDDEV]

Train/test iris dataset using logistic regression

optional arguments:
 -h, --help            show this help message and exit
 --load                load model rather than train
 --visual              plot data and features prior to load/test
 --learning_rate LEARNING_RATE
                       learning rate for GradientDescentOptimizer
 --filename FILENAME   file to store/load model to/from
 --stddev STDDEV       standard deviation for random_normal init values
```

#### Train a new model
```python
python iris.py
```

#### Test a model
```python
python iris.py --load
```

#### Custom training
```python
python iris.py --visual --learning_rate 0.1 --filename my_model --stddev 0.5
```

### Changelog
+ 0.2.0
  * Add TensorBoard support
  * Show validation set accuracy during training
  * Implement tf.name_scope for organization of variables

+ 0.1.2
  * Added command line arguments for learning rate and filename
  * Allow save/restore from any file

+ 0.1.1
  * Changed command line argument implementation to use argparse

+ 0.1.0
  * Initial version
