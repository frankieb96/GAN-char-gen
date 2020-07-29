from scipy.io import loadmat

mat = loadmat("temp_project/matlab/emnist-letters.mat")
data = mat['dataset']
x_train = data['train'][0, 0]['images'][0, 0]
y_train = data['train'][0, 0]['labels'][0, 0]
x_test = data['test'][0, 0]['images'][0, 0]
y_test = data['test'][0, 0]['labels'][0, 0]

x_train = x_train.reshape((x_train.shape[0], 28, 28), order='F')
x_test = x_train.reshape((x_train.shape[0], 28, 28), order='F')

