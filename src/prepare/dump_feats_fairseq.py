import argparse
import math
import os

from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def postprocess(feats, normalize=False):
    if feats.dim() == 2:
        feats = feats.mean(-1)

    assert feats.dim() == 1, feats.dim()

    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats

def get_filelist(filelist):
    file_list = []
    root = '/'.join(filelist.split('/')[:-1])
    with open(filelist, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace('_', '/', 1)
            file_list.append(os.path.join(root, 'audio-16k', line+'.wav'))
    return file_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default="/data/sunbomiao/yangkang/Dataset/CMLR/test.csv")
    parser.add_argument('--model_path', default="/data/sunbomiao/yangkang/LTBS/pretrained_model/chinese-hubert-large-fairseq-ckpt.pt")
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    args = parser.parse_args()
    fns = get_filelist(args.filelist)
    num_per_shard = math.ceil(len(fns)/args.nshard)
    start_id, end_id = num_per_shard*args.rank, num_per_shard*(args.rank+1)
    fns = fns[start_id:end_id]    
    print(f"{len(fns)} audios")
    model_path = args.model_path
    print("loading model(s) from {}".format(model_path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
    print("loaded model(s) from {}".format(model_path))
    print(f"normalize: {saved_cfg.task.normalize}")
    model = models[0]
    model = model.to(device)
    model = model.half()
    model.eval()
    for file in tqdm(fns):
        wav, sr = sf.read(file)
        feat = torch.from_numpy(wav).float()
        feat = postprocess(feat, normalize=saved_cfg.task.normalize)
        feats = feat.view(1, -1)
        padding_mask = (
            torch.BoolTensor(feats.shape).fill_(False)
        )
        inputs = {
            "source": feats.half().to(device),
            "padding_mask": padding_mask.to(device),
            "output_layer": 12
        }
        with torch.no_grad():
            logits = model.extract_features(**inputs)
        npy_file = file.replace('audio-16k', 'hubert-large-feats-12th').replace('.wav', '.npy')
        os.makedirs(os.path.dirname(npy_file), exist_ok=True) 
        np.save(npy_file, logits[0].cpu().numpy())  