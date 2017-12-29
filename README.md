# kaggle-cdiscount-challenge
Cdiscount’s Image Classification Challenge

See the [Jupyter notebook](https://nbviewer.jupyter.org/github/baudm/kaggle-cdiscount-challenge/blob/master/notebooks/submission.ipynb) for the details.

## Project Organization
### input
Contains the sample training data for the demo

### notebooks
Contains the submission notebook and related files

### util
Utility scripts
* `cache-features.py` - the script used for caching the CNN features of the Xception model
* `generate-sample-dist.py` - generates a JSON file containing the number of samples per class
* `shuffle-npz.py` - shuffles the cached features stored in Numpy archives (cached features are stored in batches, so shuffling is useful because 655,360 samples are set aside for validation).

### weights
* `classifier_weights_tf_dim_ordering_tf_kernels.h5` - weights of the trained classifier subnetwork



`train-classifier.py` - used for training the classifier subnetwork

`test.py` - used for generating the predictions for submission to kaggle

`demo-eval.py` - loads and runs predictions on the trained model using `input/train_example.bson` as input
