# Video Captioning with Deep Learning

This projects transforms video data into text descriptions via Deep Learning

### Prerequisites

Please install the TensorFlow package in version 1.12 (and the respective CUDA package if 
training on GPU is required).

Install the required libraries, from the project directory type:
```
pip3 install -r requirements.txt
```

Load the dataset from https://20bn.com/datasets/something-something/v1 .

Set the right directories in /constants/constants.

An example of a  full training and testing process is shown in main.py, which will
create the dataset, train on it, and saves the predicted sentences in a database in 
/sql_models.

Different hyper parameters can be set in /constants/hyperparameters.

### Structure
The most important parts of the model, the implementations of the convolutional 
encoder and multi-head attention are located in /models/vid2sentence_modules and 
/models/vid2sentence_utils.

The word2vec model implementation, as well custom metrics (accuracy, BLEU) are also 
located in /models.

The CNN net implementations (Inception, Mobilenet) found under Models are implementations
by TensorFlow with very small adaptions in order to be usable with the model.

/models/vid2sentence contains controller methods for the TensorFlow training process.

 Methods for data pre processing and data set creation are located in /data/creation.
 
 Everything concerning the SQL-database is under /sql_models, concerning logs is under 
 /logs and tests are located in /test/.
 
 The trained checkpoints are not included on the CD because their size is too big - 
 one single experiment training directory needs 2GB on average.
 