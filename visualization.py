import os.path
import pickle
from matplotlib import pyplot as plt


path = "ssd512/SSD.Pytorch/loss.pkl"
f = open(path, 'rb')
data = pickle.load(f)
for key in data.keys():
    ax = plt.subplot(len(data), 1, list(data.keys()).index(key)+1)
    plt.plot(data[key])
    ax.set_ylabel(key)
plt.suptitle('Loss for SSD512')
plt.show()

path = "ssd512/SSD.Pytorch/ssd300_120000/test"
for file in os.listdir(path):
    if len(file.split('_')) > 1:  # remove detections.py
        ripeness = file.split('_')[0]  # get classification type
        f = open(path + os.sep + file, 'rb')
        data = pickle.load(f)
        for key in ['rec', 'prec']:
            ax = plt.subplot(2, 1, list(data.keys()).index(key) + 1)
            plt.plot(data[key])
            ax.set_ylabel(key)
        plt.suptitle('Recall, Precision and AP=' + str(data['ap']) + ' for ' + ripeness)
        plt.show()


