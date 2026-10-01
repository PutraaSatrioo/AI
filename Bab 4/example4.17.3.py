n=0
for layer in model.layers:
 if 'conv' in layer.name:
 filters, biases = layer.get_weights()
 print(n,layer.name, filters.shape, biases.shape)
 n+=1