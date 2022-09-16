from src.utils.dataloader import load_cats_and_dogs

catsndogs = load_cats_and_dogs()
print(catsndogs['label_names'])
print(catsndogs['labels'])
print(catsndogs['data'][0][0][0])