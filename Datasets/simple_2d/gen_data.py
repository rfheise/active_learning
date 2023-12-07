import numpy as np
import matplotlib.pyplot as plt
import csv

#sample_size = 2000
sample_sizes = [50, 200, 75, 15]
size_prob = [0.38, 0.4, 0.12, 0.1]
mu_0 = 20
mu = [mu_0]
class_num = 10
for i in range(class_num-1):
    mu.append(mu[i] + 325)
sigma = 35
init_y = 100
class_num = 0
idx = 0

x = []
y = []
data = []
for mu_i in mu:
    sample_size = np.random.choice(sample_sizes,replace=True, p=size_prob)
    dist = np.random.normal(loc=[mu_i, init_y], scale=[2*sigma,0.5*sigma], size=[sample_size, 2])
    for x_i, y_i in dist:
        x_i = round(x_i, 3)
        y_i = round(y_i, 3)
        x.append(x_i)
        y.append(y_i)
        data.append([x_i, y_i, class_num])
        idx += 1
    class_num += 1

data = np.array(data)
np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

"""with open('knn_data.csv', 'w', newline='') as f:
    #['x', 'y','label']
    writer = csv.writer(f, delimiter=',')
    for row in data:
        writer.writerow(row) #['x', 'y','label']"""

fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()
