from src.utils.dataloader import load_cats_and_dogs
import numpy as np

catsndogs = load_cats_and_dogs()
print(catsndogs['label_names'])
print(catsndogs['labels'])
print(catsndogs['data'][0][0][0])


print([len(i) for i in catsndogs['data']])
print(np.array(catsndogs['data']).shape)