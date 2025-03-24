# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys

import numpy as np

import joblib
import torch
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_feat_iterator(feat_dir, split, nshard, rank):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)


# def dump_label(feat_dir, split, km_path, nshard, rank, lab_dir):
#     apply_kmeans = ApplyKmeans(km_path)
#     generator, num = get_feat_iterator(feat_dir, split, nshard, rank)
#     iterator = generator()

#     lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.km"
#     os.makedirs(lab_dir, exist_ok=True)
#     with open(lab_path, "w") as f:
#         for feat in tqdm.tqdm(iterator, total=num):
#             # feat = torch.from_numpy(feat).cuda()
#             lab = apply_kmeans(feat).tolist()
#             f.write(" ".join(map(str, lab)) + "\n")
#     logger.info("finished successfully")

def dump_label(root, file_dir, km_path, lab_dir):
    apply_kmeans = ApplyKmeans(km_path)
    with open(lab_dir, 'a') as f:
        for file in tqdm(file_dir):
            feat_file = os.path.join(root, 'hubert-large-feats-12th', file+'.npy')
            feat = np.load(feat_file).squeeze(0)
            lab = apply_kmeans(feat).tolist()
            lab_str = ' '.join(map(str, lab))
            f.write(file + '|' + lab_str + '\n')


def get_filelist(root, split):
    output = []
    with open(os.path.join(root, split)+'.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace('_', '/', 1)
            output.append(line)
    return output


if __name__ == "__main__":
    import argparse
    import math
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='/data/sunbomiao/yangkang/Dataset/CMLR')
    parser.add_argument("--km_path", default='/data/sunbomiao/yangkang/LTBS/pretrained_model/km_models/cmlr_200_12th.mdl')
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--lab_dir", default='/data/sunbomiao/yangkang/LTBS/data')
    args = parser.parse_args()
    logging.info(str(args))
    train_files = get_filelist(args.root, 'train')
    val_files = get_filelist(args.root, 'val')
    test_files = get_filelist(args.root, 'test')   
    all_files = train_files + val_files + test_files
    num_per_shard = math.ceil(len(all_files)/args.nshard)
    start_id, end_id = num_per_shard*args.rank, num_per_shard*(args.rank+1)
    all_files = all_files[start_id:end_id]
    lab_file = os.path.join(args.lab_dir, f'label_{args.rank}.csv')
    print(f"{len(all_files)} files")
    dump_label(args.root, all_files, args.km_path, lab_file)
