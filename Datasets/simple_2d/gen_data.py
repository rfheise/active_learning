import numpy as np
import matplotlib.pyplot as plt
import csv

sample_sizes = [50, 200, 75, 15]
size_prob = [0.38, 0.4, 0.12, 0.1]
mu_0 = 20
mu = [mu_0]
class_num = 10
for i in range(class_num-1):
    mu.append(mu[i] + 1000)
sigma = 35
init_y = 10
class_num = 0
idx = 0

x = []
y = []
data = []
for mu_i in mu:
    sample_size = np.random.choice(sample_sizes,replace=True, p=size_prob)
    dist = np.random.normal(loc=[mu_i, init_y], scale=[.5*sigma,0.5*sigma], size=[sample_size, 2])
    for x_i, y_i in dist:
        x_i = round(x_i, 3)
        y_i = round(y_i, 3)
        x.append(x_i)
        y.append(y_i)
        data.append([x_i, y_i, class_num])
        idx += 1
    class_num += 1

data = np.array(data)
# print(data[:,2])
# Randomize data
# np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

# Save data to csv file
with open('knn_data12.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    for row in data:
        writer.writerow(row) #['x', 'y','label']

# Show distribution of data 
fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()

