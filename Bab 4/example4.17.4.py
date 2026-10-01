from matplotlib import pyplot
# retrieve weights from the first hidden layer
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape 
print("Color channels: ", s[0])
print("Filter size: ", s[1],s[2])
print("Total number of filters : ", s[3])
# normalize filter values to 0- 1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
#n_filters, ix = s[3], 1
n_filters, ix = 4, 1
pyplot.figure(figsize=(10,10))
for i in range(n_filters):
 # get the filter
 f = filters[:, :, :, i]
 
 # plot each channel separately
 for j in range(s[0]):
 ax = pyplot.subplot(n_filters, s[0], ix)
 ax.set_xticks([])
 ax.set_yticks([])
 pyplot.imshow(f[j,:, :], cmap='gray')
 ix += 1
# show the figure
pyplot.show()