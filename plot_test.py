import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(dir_path, 'logs')


def plot(test_name):
    test_path = os.path.join(log_path, test_name)
    folder_names = ['baseline-increment','baseline-random','query-by-committee','uncert-pool','uncert-stream']
    for name in folder_names:
        folder_path = os.path.join(test_path, name)
        if(os.path.exists(folder_path)):
            file_path = os.path.join(folder_path, 'test_data.csv')

            train_acc = []
            train_loss = []
            val_acc = []
            val_loss = []
            samples = []

            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f, delimiter=',')
                line_idx = 0
                for row in reader:
                    # if line_idx == 0:
                    #     pass
                    # else:
                        #print(line_idx)
                        # try:
                    train_acc.append(float(row["train/acc"]))
                    train_loss.append(round(float(row["train/loss"]), 2))
                    val_loss.append(round(float(row["val/loss"]), 2))
                    val_acc.append(round(float(row["val/acc"]), 2))
                    samples.append(int(row["labeled_data_points"]))
                        # except:
                        #     pass
                    line_idx += 1

                plt.subplot(2,2,1)
                plt.plot(samples, train_acc, label=name)
                plt.title('Training')
                plt.ylabel('Accuracy')
                plt.subplot(2,2,2)
                plt.plot(samples, val_acc, label=name)
                plt.title('Validation')
                plt.legend()
                plt.subplot(2,2,3)
                plt.plot(samples, train_loss,label=name)
                #plt.title('Training Loss')
                plt.xlabel('Sample Size')
                plt.ylabel('Loss')
                plt.subplot(2,2,4)
                plt.plot(samples, val_loss, label=name)
                #plt.title('Validation Loss')
                plt.xlabel('Sample Size')
    plt.show()

def main():
    log_dirname = sys.argv[1]
    plot(log_dirname)

if __name__=="__main__":
    main()