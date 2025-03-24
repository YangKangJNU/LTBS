import math
from torch import nn
import torch

from src.models.utils import load_variance_decoder, load_video_encoder, load_mel_decoder, load_flow_postnet




class LTBS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_sqz = cfg.post_net.n_sqz
        self.video_encoder = load_video_encoder(cfg.video_encoder)
        self.linguistic_predictor = load_variance_decoder(cfg.variance_decoder.linguistic)
        self.pitch_predictor = load_variance_decoder(cfg.variance_decoder.pitch)
        self.energy_predictor = load_variance_decoder(cfg.variance_decoder.energy)
        self.mel_decoder = load_mel_decoder(cfg.mel_decoder)
        self.post_net = load_flow_postnet(cfg.post_net)
        self.linguistic_embedding = nn.Embedding(cfg.variance_decoder.linguistic.output_channels + 1, cfg.variance_decoder.linguistic_dim)
        self.pitch_embedding = nn.Conv1d(1, cfg.variance_decoder.pitch_dim, kernel_size=3, stride=1, padding=1)
        self.energy_embedding = nn.Conv1d(1, cfg.variance_decoder.energy_dim, kernel_size=3, stride=1, padding=1)
        self.fusion = nn.Linear(cfg.video_encoder.attention_dim+cfg.variance_decoder.linguistic_dim+cfg.variance_decoder.pitch_dim+cfg.variance_decoder.energy_dim, cfg.video_encoder.attention_dim)
        self.mel_loss = nn.L1Loss()
        self.code_loss = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
        self.pitch_loss = nn.L1Loss()
        self.energy_loss = nn.L1Loss()

    def forward(self, x, x_len, spks_id, code, pitch, energy, mel, p_control=1.0, e_control=1.0):
        encoder_outputs, mask, spk_emb = self.video_encoder(x, x_len, spks_id)
        code_logit, code_embedding = self.get_linguistic_embedding(
            encoder_outputs, code, ~mask
        )
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            encoder_outputs, pitch, ~mask, p_control
        )
        energy_prediction, energy_embedding = self.get_energy_embedding(
            encoder_outputs, energy, ~mask, e_control
        )
        code_ce_loss = self.code_loss(code_logit.permute(0, 2, 1), code-1)
        pitch_l1_loss = self.pitch_loss(pitch_prediction, pitch)
        energy_l1_loss = self.energy_loss(energy_prediction, energy)
        var_loss = code_ce_loss + pitch_l1_loss + energy_l1_loss
        encoder_outputs = torch.cat([encoder_outputs, code_embedding, pitch_embedding.permute(0, 2, 1), energy_embedding.permute(0, 2, 1)], dim=-1)
        encoder_outputs = self.fusion(encoder_outputs)
        encoder_outputs = torch.repeat_interleave(encoder_outputs, repeats=4, dim=1)
        mel_outputs, mel_mask, spk_emb = self.mel_decoder(encoder_outputs, mask, spk_emb)
        mel_loss = self.mel_loss(mel_outputs.permute(0, 2, 1), mel)
        cond = torch.cat([encoder_outputs, mel_outputs], dim=-1)
        mel, mel_lengths, mel_max_length = self.preprocess(mel, x_len*4, (x_len*4).max())
        # mel_fine, logdet = self.post_net(mel_outputs, mel_mask.unsqueeze(1), g=cond.permute(0, 2, 1), reverse=True)
        z, logdet = self.post_net(mel, mel_mask.unsqueeze(1), g=cond.permute(0, 2, 1), reverse=False) 
        z_logs = torch.zeros_like(mel_outputs.permute(0, 2, 1))
        post_loss = mle_loss(z, mel_outputs.permute(0, 2, 1), z_logs, logdet, mel_mask.unsqueeze(1))
        return (mel_loss, var_loss, post_loss, (code_ce_loss, pitch_l1_loss, energy_l1_loss)), (mel_outputs, code_logit, pitch_prediction, energy_prediction)
    
    @torch.no_grad()
    def infer(self, x, x_len, spks_id, p_control=1.0, e_control=1.0, noise_scale=1.):
        encoder_outputs, mask, spk_emb = self.video_encoder(x, x_len, spks_id)
        code_logit, code_embedding = self.get_linguistic_embedding(
            encoder_outputs, None, ~mask
        )
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            encoder_outputs, None, ~mask, p_control
        )
        energy_prediction, energy_embedding = self.get_energy_embedding(
            encoder_outputs, None, ~mask, e_control
        )
        encoder_outputs = torch.cat([encoder_outputs, code_embedding, pitch_embedding.permute(0, 2, 1), energy_embedding.permute(0, 2, 1)], dim=-1)
        encoder_outputs = self.fusion(encoder_outputs)
        encoder_outputs = torch.repeat_interleave(encoder_outputs, repeats=4, dim=1)
        mel_outputs, mel_mask, spk_emb = self.mel_decoder(encoder_outputs, mask, spk_emb)
        cond = torch.cat([encoder_outputs, mel_outputs], dim=-1)
        mel_outputs, mel_lengths, mel_max_length = self.preprocess(mel_outputs, x_len*4, (x_len*4).max())
        z_logs = torch.zeros_like(mel_outputs.permute(0, 2, 1))
        z = (mel_outputs.permute(0, 2, 1) + torch.exp(z_logs) * torch.randn_like(mel_outputs.permute(0, 2, 1)) * noise_scale) * mel_mask.unsqueeze(1)
        y, logdet = self.post_net(z, mel_mask.unsqueeze(1), g=cond.permute(0, 2, 1), reverse=True)
        return y, mel_outputs, code_logit, pitch_prediction, energy_prediction


    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(target.unsqueeze(1))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(prediction.permute(0, 2, 1))
        return prediction.squeeze(-1), embedding
    
    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(target.unsqueeze(1))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(prediction.permute(0, 2, 1))
        return prediction.squeeze(-1), embedding
    
    def get_linguistic_embedding(self, x, target, mask):
        logits = self.linguistic_predictor(x, mask)
        if target is not None:
            embedding = self.linguistic_embedding(target)
        else:
            preds = logits.argmax(-1) + 1
            preds[mask] = 0
            embedding = self.linguistic_embedding(preds)
        return logits, embedding

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:,:,:y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length
    
def mle_loss(z, m, logs, logdet, mask):
  l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m)**2)) # neg normal likelihood w/o the constant term
  l = l - torch.sum(logdet) # log jacobian determinant
  l = l / torch.sum(torch.ones_like(z) * mask) # averaging across batch, channel and time axes
  l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
  return l