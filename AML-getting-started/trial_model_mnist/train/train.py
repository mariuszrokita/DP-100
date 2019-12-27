
import argparse
import os
import numpy as np
import glob
import time

from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from azureml.core import Run
from azureml.core.model import Model
# from utils import load_data

import gzip
import struct


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


# let user feed in 2 parameters, the dataset to mount or download, 
# and the regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
parser.add_argument('--local', type=str2bool, dest='local', default=False,
                               help='Flag indicating where model training takes place')
args = parser.parse_args()

###
data_folder = os.path.join(args.data_folder, 'data', 'mnist')
print('Data folder:', data_folder)

# load the train and test set into numpy arrays
X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0

#print variable set dimension
print(X_train.shape, X_test.shape, sep = '\n')

y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)

#print the response variable dimension
print( y_train.shape, y_test.shape, sep = '\n')

# get hold of the current run
run = Run.get_context()

print('Train a logistic regression model with regularization rate of', args.reg)
clf = LogisticRegression(C=1.0/args.reg, solver="liblinear", multi_class="auto", random_state=42)
clf.fit(X_train, y_train)

print('Predict the test set')
y_hat = clf.predict(X_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
print('Accuracy is', acc)

run.log('regularization rate', np.float(args.reg))
run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)

# Note, file saved in the outputs folder is automatically uploaded into experiment record.
model_filename = 'outputs/sklearn_mnist_model.pkl'
joblib.dump(value=clf, filename=model_filename)

# Due to latency we need to wait (not more than 60 seconds) until the model file is really uploaded. 
timestamp = datetime.now()
while (model_filename not in run.get_file_names()) or ((datetime.now() - timestamp).seconds <= 60) :
    print("Need to wait...")
    time.sleep(1)  # wait 1 second

print('========================')
print('Files associated with that run:')
print(run.get_file_names())
print('========================')


# register an Azure ML model (it's only possible when training in the AMLS)
local = args.local
print(f"Local: {local}, {type(local)}")

if not local:
    print("Registering model...")   
    model = run.register_model(model_name='sklearn_mnist',
                               model_path=os.path.join('.', 'outputs', 'sklearn_mnist_model.pkl'),
                               tags = {'area': "MNIST", 'type': "sklearn"},
                               description = "identify numbers")
    
    print(model.name, model.id, model.version, sep='\t')
