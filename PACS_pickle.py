import pickle
import os
import numpy as np
import sys
from PIL import Image
from sklearn.model_selection import train_test_split

rootdir = sys.argv[1]
domains = {'photo', 'sketch', 'cartoon', 'art_painting'}

for test_domain in domains:
    train = []
    test_imgs = []
    test_labels = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files: 
            label = subdir.split("/")
            # Divide data in train and test set
            with Image.open(os.path.join(subdir, file)) as image:
                if test_domain != label[1]:
                    train.append((label[1], np.array(image), label[2]))  
                else:
                    test_imgs.append(np.array(image))
                    test_labels.append(label[2])

    # Split train dataset in train and validation
    train_imgs = []
    train_labels = []
    val_imgs = []
    val_labels = []
    for domain in domains-{test_domain}: # val set constructed with 10% from every domain
        train_domain = [x for x in train if x[0]==domain]
        t, v = train_test_split(train_domain, test_size=0.1)

        t_imgs = [x[1] for x in t]
        t_label = [x[2] for x in t]
        v_imgs = [x[1] for x in v]
        v_label = [x[2] for x in v]

        train_imgs += t_imgs 
        train_labels += t_label
        val_imgs += v_imgs
        val_labels += v_label

    train = {'data':train_imgs, 'labels':train_labels}
    val = {'data':val_imgs, 'labels':val_labels}
    test = {'data':test_imgs, 'labels':test_labels}

    # Save datasets
    name = test_domain.upper()
    with open(f'{name}_train.pickle', 'wb') as handle:
        pickle.dump(train, handle)
    with open(f'{name}_val.pickle', 'wb') as handle:
        pickle.dump(val, handle)
    with open(f'{name}_test.pickle', 'wb') as handle:
        pickle.dump(test, handle)