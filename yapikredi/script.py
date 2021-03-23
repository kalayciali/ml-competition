import os 
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import io
import re
import lzma
import pickle
from dataclasses import make_dataclass
import pandas as pd
from itertools import chain
from concurrent import futures

from scipy import ndimage
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def str2bool(v):
    if v.lower() == "true":
        return True
    else:
        return False


parser = argparse.ArgumentParser()
parser.add_argument("--gen_model", default=False, type=str2bool, help="Run prediction on given data")
parser.add_argument("--predict", default=True, type=str2bool, help="Predict model")
parser.add_argument("--gen_dataset", default=True, type=str2bool, help="Generate image dataset")
parser.add_argument("--pixel_size", default=60, type=int, help="size X size image")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

args = parser.parse_args()

SIZE = args.pixel_size
TEST_PATH = './test/test/'
TRAIN_PATH = './train/'

TEST_FNAME = 'tests.pkl'
TRAIN_FNAME = 'trains.pkl'
MODEL_FNAME = "svc.pkl"

SIGN_REGEX = r"NFI-(?P<person_id>\d\d\d)(?P<sig_id>\d\d).+"
NUM_CLUSTERS = 79


pixels = []
get_pixels = []
for i in range(SIZE * SIZE):
    get_pixels.append(f"pixel{i}")
    pixels.append((f"pixel{i}", int))

Image = make_dataclass("Image", [("person_id", str), ("sign_id", str), *pixels])

def expand_img(img, metadata,
               shift_max=10, shift_min=2, shift_count=5,
               rotations_max=11, rotations_min=2.5, rotations_count=5):
    gen_imgs = []
    flat_img = np.ravel(img.astype(np.uint8))
    gen_imgs.append(Image(*metadata, *flat_img))
    if shift_count > 0 and shift_max > 0:
        shifts = set()
        while len(shifts) < shift_count:
            shift = tuple(np.random.randint(-shift_max, shift_max + 1, 2))
            if shift in shifts or abs(shift[0]) + abs(shift[1]) < shift_min:
                continue
            shifts.add(shift)
            new_img_data = ndimage.shift(img, shift, cval=0)
            new_img_data = np.ravel(new_img_data.astype(np.uint8))
            gen_imgs.append(Image(*metadata, *new_img_data))

    if rotations_count > 0 and rotations_max > 0:
        for _ in range(rotations_count):
            angle = np.random.uniform(-rotations_max, rotations_max)
            if abs(angle) < rotations_min:
                continue
            new_img_data = ndimage.rotate(img, angle, cval=0, reshape=False)
            new_img_data = np.ravel(new_img_data.astype(np.uint8))
            gen_imgs.append(Image(*metadata, *new_img_data))

    return gen_imgs


def bbox(img):
    # border box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img_in, size=SIZE, pad=16):
    ymin, ymax, xmin, xmax = bbox(img_in)
    img = img_in[ymin:ymax, xmin:xmax]

     #remove lo intensity pixels as noise
    img[img < 28] = 0

    num_cols, num_rows = xmax-xmin, ymax-ymin
    l = max(num_rows, num_cols) + pad
    aspect_ratio = [((l-num_rows)//2,), ((l-num_cols)//2, )]
    img = np.pad(img, aspect_ratio, mode='constant')
    return cv2.resize(img, (size, size))


def generate_img_dataset(path, extend):
    print(f"starting creating dataset for {path}")
    images = []
    for img in os.listdir(path):
        print(img)
        metadata = re.search(SIGN_REGEX, img).groups()
        img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img = cv2.bitwise_not(img)
        # normalize each img by it's max val
        pic = img * (255.0 / img.max())
        pic = crop_resize(pic)
        if extend:
            images.extend(expand_img(pic, metadata))
        else:
            pic = np.ravel(pic.astype(np.uint8))
            images.append(Image(*metadata, *pic))

    df = pd.DataFrame(images)
    return df

if args.gen_dataset:
    print("Generating dataset from stratch")

    train_df = generate_img_dataset(TRAIN_PATH, True)
    print("pickling train data")
    train_df.to_pickle(TRAIN_FNAME)

    test_df = generate_img_dataset(TEST_PATH, True)
    print("pickling test data")
    test_df.to_pickle(TEST_FNAME)
else:
    print("Using already built dataset")
    train_df = pd.read_pickle(TRAIN_FNAME)
    test_df = pd.read_pickle(TEST_FNAME)

total_df = train_df.append(test_df)

X_train, X_test, y_train, y_test = train_test_split(total_df[get_pixels], total_df["person_id"], test_size=0.1, random_state=args.seed)

pca = PCA(n_components=32, random_state=args.seed)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


if args.predict:

    print(f'Training on {len(X_train)} images')
    if args.gen_model:
        print("With new model")
#        model = MLPClassifier(
#            hidden_layer_sizes=(800, ),
#            activation='logistic',
#            batch_size=50,
#            alpha=0.00005,
#            learning_rate='invscaling',
#            verbose=True,
#            max_iter=100,
#        )

        #model = GaussianMixture(n_components=NUM_CLUSTERS, random_state=args.seed)
        model = SVC(C=10, kernel='rbf', random_state=args.seed, probability=True)
        model.fit(X_train_pca, y_train)
        print(model.score(X_test_pca, y_test))

        with lzma.open(MODEL_FNAME, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        print("With already built model")
        with lzma.open(MODEL_FNAME, "rb") as model_file:
            model = pickle.load(model_file)
        print(model.predict_proba(X_test_pca))

else:
    print('Datasets are loaded')
    print('Finished without model generation')

