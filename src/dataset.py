from torch.utils.data import Dataset
import numpy as np
import torch
from src.utils import load_wav_to_numpy, STFT, Compose, CenterCrop, RandomCrop, Normalize, RandomErase, HorizontalFlip, TimeMask, load_video, load_wav_to_torch, pitch_norm
import os
# pyin pitch
# audio is extracted by librosa
from librosa.util import normalize

class VADataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.file_paths, self.file_names = self.build_file_list(root, mode)
        if mode == 'train':
            self.video_transform = Compose([
                    Normalize(0.0, 255.0),
                    RandomCrop((88, 88)),
                    HorizontalFlip(0.5),
                    Normalize(0.421, 0.165),
                    TimeMask(),
                    RandomErase(0.5)])
        else:
            self.video_transform = Compose([
                    Normalize(0.0, 255.0),
                    CenterCrop((88, 88)),
                    Normalize(0.421, 0.165)])
        self.build_unit_dict()
        self.stft = STFT(filter_length=1024, hop_length=160, win_length=640, sampling_rate=16000, mel_fmin=55., mel_fmax=7500.)

    def build_unit_dict(self):
        base_fname_batch, quantized_units_batch = [], []
        units_file = ''
        with open(units_file) as f:
            for line in f:
                base_fname, quantized_units_str = line.rstrip().split("|")
                quantized_units = [int(q) for q in quantized_units_str.split(" ")]
                base_fname_batch.append(base_fname)
                quantized_units_batch.append(quantized_units)
        self.unit_dict = dict(zip(base_fname_batch,quantized_units_batch))

    def build_file_list(self, root, mode):
        file_list, paths = [], []
        assert mode in ['train', 'val', 'test']
        with open(f"{root}/{mode}.txt", "r") as f:
            train_data = f.readlines()
        for i in range(len(train_data)):
            file = train_data[i].strip()
            file_list.append(file)
            paths.append(f"{root}/crop/{file}.mp4")
        return paths, file_list 

    def __len__(self):
        return len(self.file_names)

    def get_mel_energy(self, filename):
        audio, _ = load_wav_to_torch(filename)
        audio = audio / 1.1 / audio.abs().max()
        melspectrogram = self.stft.get_mel(audio)
        energy = torch.norm(melspectrogram, p=2, dim=0)
        energy = energy[::4]
        return audio, melspectrogram, energy

    def __getitem__(self, index):
        crop_path = self.file_paths[index]
        f_name = self.file_names[index]
        spk_id = int(f_name.split('/')[0].lstrip('s')) - 1
        video = load_video(crop_path)
        video = np.array(video, dtype=np.float32)
        video = self.video_transform(video)
        video = torch.tensor(video).unsqueeze(0) 
        audio_path = crop_path.replace('/crop/', '/audio/').replace('.mp4', '.wav')
        audio, mel, energy = self.get_mel_energy(audio_path)
        audio = torch.FloatTensor(audio)
        pitch_path = crop_path.replace('/crop/', '/pitch/').replace('.mp4', '.npy')
        pitch = np.load(pitch_path).astype(np.float32)
        pitch = pitch_norm(pitch)
        pitch = torch.FloatTensor(pitch)
        token = torch.tensor(self.unit_dict[f_name], dtype=torch.int32) + 1
        token = token[::2]

        diff = len(token) - video.size(1)
        if diff < 0:
            padding = torch.zeros(-diff, dtype=token.dtype)
            token = torch.cat((token, padding))
        elif diff > 0:
            token = token[:-diff]
        diff = len(pitch) - video.size(1)
        if diff < 0:
            padding = torch.zeros(-diff, dtype=pitch.dtype)
            pitch = torch.cat((pitch, padding))
        elif diff > 0:
            pitch = pitch[:-diff]
        diff = mel.size(-1) - video.size(1) * 4
        if diff < 0:
            padding_mel = torch.zeros(mel.size(0), -diff, dtype=mel.dtype)
            mel = torch.cat((mel, padding_mel), dim=-1)
        elif diff > 0:
            mel = mel[:, :-diff]
        diff = len(energy) - video.size(1)
        if diff < 0:
            padding = torch.zeros(-diff, dtype=energy.dtype)
            energy = torch.cat((energy, padding))
        elif diff > 0:
            energy = energy[:-diff]
        diff = audio.size(0) - video.size(1) * 640
        if diff < 0:
            padding_audio = torch.zeros(-diff, dtype=audio.dtype)
            audio = torch.cat((audio, padding_audio), dim=0)
        elif diff > 0:
            audio = audio[:-diff]

        return  (video, mel, pitch, energy, token, audio, spk_id, f_name)
    
def collate_fn(batch):
    _, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].size(1) for x in batch]),
        dim=0, descending=True)
    
    max_video_len = max([x[0].size(1) for x in batch])
    max_mel_len = max(x[1].size(1) for x in batch)
    max_audio_len = max(x[5].size(0) for x in batch)

    video_lengths = torch.IntTensor(len(batch))
    mel_lengths = torch.IntTensor(len(batch))
    audio_lengths = torch.IntTensor(len(batch))

    video_padded = torch.zeros(len(batch), 1, max_video_len, 88, 88, dtype=torch.float32)
    mel_padded = torch.zeros(len(batch), 80, max_mel_len, dtype=torch.float32)
    pitch_padded = torch.zeros(len(batch), max_video_len, dtype=torch.float32)
    energy_padded = torch.zeros(len(batch), max_video_len, dtype=torch.float32)
    token_padded = torch.zeros(len(batch), max_video_len, dtype=torch.int64)
    audio_padded = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)

    video_padded.zero_()
    mel_padded.zero_()
    pitch_padded.zero_()
    energy_padded.zero_()
    token_padded.zero_()
    audio_padded.zero_()

    spk_ids = []
    f_names = []

    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]    

        video = row[0]
        video_padded[i, :, :video.size(1), :, :] = video
        video_lengths[i] = video.size(1)   

        mel = row[1]
        mel_padded[i, :, :mel.size(1)] = mel
        mel_lengths[i] = mel.size(1)             

        pitch = row[2]
        pitch_padded[i, :pitch.size(0)] = pitch

        energy = row[3]
        energy_padded[i, :energy.size(0)] = energy

        token = row[4]
        token_padded[i, :token.size(0)] = token

        audio = row[5]
        audio_padded[i, :audio.size(0)] = audio
        audio_lengths[i] = audio.size(0)

        spk_ids.append(row[6])
        f_names.append(row[7])

    spk_ids = torch.tensor(spk_ids)

    return dict(
        video = video_padded,
        video_lengths = video_lengths,
        mel = mel_padded,
        mel_lengths = mel_lengths,
        pitch = pitch_padded,
        energy = energy_padded,
        token = token_padded,
        audio = audio_padded,
        audio_lengths = audio_lengths,
        spk_ids = spk_ids,
        file_name = f_names
    )


