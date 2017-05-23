# ml-iris

## Machine learning for Iris dataset
  - Uses TensorFlow to create a Logistic Regression classifier
  - Full ML pipeline with tensorflow backend

Imports raw data from csv, randomizes data, preprocesses data,
splits data, trains model, tests model, saves model

Uses TensorBoard to visualize the results

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
