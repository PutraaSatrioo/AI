def plot_feature_maps(feature_maps):
 # plot all feature maps 
 col = 8
 row = int(feature_maps.shape[3]/col)
 ix = 1
 plt.figure(figsize=(20,20))
 for _ in range(row):
 for _ in range(col):
 # specify subplot and turn of axis
 ax = plt.subplot(row, col, ix)
 ax.set_xticks([])
 ax.set_yticks([])
 # plot filter channel in grayscale
 plt.imshow(feature_maps[0, :, :, ix- 1], cmap='gray')
 ix += 1
 # show the figure
 plt.show()