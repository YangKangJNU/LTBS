# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


    

def load_feature(file_dir):
    all_feature = []
    for file in tqdm(file_dir):
        feat = np.load(file, mmap_mode='r').squeeze()
        num_samples = feat.shape[0]
        half_size = num_samples // 2
        indices = np.random.choice(num_samples, half_size, replace=False)
        feat = feat[indices]
        all_feature.append(feat)
    all_features = np.concatenate(all_feature)
    return all_features

def get_file(filelist):
    file_list = []
    root = '/'.join(filelist.split('/')[:-1])
    with open(filelist, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace('_', '/', 1)
            file_list.append(os.path.join(root, 'hubert-large-feats-12th', line+'.npy'))
    return file_list

def learn_kmeans(
    feat_dir,
    km_path,
    n_clusters,
    seed,
    percent,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    np.random.seed(seed)
    feat_dirs = get_file(feat_dir)
    feat = load_feature(feat_dirs)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", type=str, default='/data/sunbomiao/yangkang/Dataset/CMLR/train.csv')
    parser.add_argument("--km_path", type=str, default='/data/sunbomiao/yangkang/LTBS/pretrained_model/km_models/cmlr_200_12th.mdl')
    parser.add_argument("--n_clusters", type=int, default=200)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))
