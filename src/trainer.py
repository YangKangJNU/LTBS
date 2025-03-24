import os
from loguru import logger
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np
import torch
from tqdm import tqdm
from src.dataset import(
    collate_fn
)
from src.utils import exists, default, plot_spectrogram, plot_curves
from adam_atan2_pytorch.adopt import Adopt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR



class Trainer:
    def __init__(
            self,
            model,
            vocoder=None,
            optimizer = None,
            learning_rate = 1e-4,
            num_warmup_steps = None,
            grad_accumulation_steps = 1,
            checkpoint_path = None,
            log_file = "logs.txt",
            max_grad_norm = 1.0,
            sample_rate = 16000,
            tensorboard_log_dir = 'runs/test1',
            accelerate_kwargs: dict = dict(),
            sample_steps = 100,
            use_switch_ema = False 
    ):
        logger.add(log_file)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
        self.accelerator = Accelerator(
            log_with = "all",
            kwargs_handlers = [ddp_kwargs],
            gradient_accumulation_steps = grad_accumulation_steps,
            mixed_precision='fp16',
            **accelerate_kwargs
        )        
        self.target_sample_rate = sample_rate
        self.model = model
        self.use_switch_ema = use_switch_ema   
        if not exists(optimizer):
            optimizer = Adopt(model.parameters(), lr = learning_rate)
        self.optimizer = optimizer
        self.vocoder = vocoder
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.num_warmup_steps = num_warmup_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_path = default(checkpoint_path, 'model.pth')
    
        self.sample_steps = sample_steps
        self.tensorboard_log_dir = tensorboard_log_dir

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.scheduler = None

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    def save_checkpoint(self, step, ckpt_name):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict = self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict = self.accelerator.unwrap_model(self.optimizer).state_dict(),
                scheduler_state_dict = self.scheduler.state_dict(),
                step = step
            )

            self.accelerator.save(checkpoint, ckpt_name)


    def load_checkpoint(self, ckpt_path):
        if not exists(ckpt_path) or not os.path.exists(ckpt_path):
            return 0

        checkpoint = torch.load(ckpt_path)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step']
    
    def train(self, train_dataset, epochs, batch_size, num_workers, val_dataset=None, ckpt_path=None, log_step=100, val_step=1000):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_dataloder = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True, num_workers=1, pin_memory=True)
        total_steps = len(train_dataloader) * epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[warmup_scheduler, decay_scheduler],
                                      milestones=[self.num_warmup_steps])
        train_dataloader, val_dataloder, self.scheduler = self.accelerator.prepare(train_dataloader, val_dataloder, self.scheduler)
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        start_step = self.load_checkpoint(ckpt_path)
        global_step = start_step
        start_epoch = start_step // len(train_dataloader) - 1
        for epoch in range(start_epoch, epochs):
            self.model.train()
            for batch in tqdm(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    video = batch['video']
                    video_len = batch['video_lengths']
                    mel = batch['mel']
                    mel_lengths = batch['mel_lengths']
                    pitch = batch['pitch']
                    energy = batch['energy']
                    token = batch['token']
                    spk_ids = batch['spk_ids']

                    loss, output = self.model(video, video_len, spk_ids, token, pitch, energy, mel)
                    
                    total_loss = loss[0] + 0.1 * loss[1] + 0.1 * loss[2]

                    self.accelerator.backward(total_loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if global_step % log_step == 0:
                    if self.accelerator.is_local_main_process:
                        mel_loss, var_loss, post_loss = loss[0], loss[1], loss[2]
                        code_ce_loss, pitch_l1_loss, energy_l1_loss = loss[3][0], loss[3][1], loss[3][2]
                        mel_coarse, code_logits, pitch_pred, energy_pred = output[0], output[1], output[2], output[3]
                        acc = (code_logits.argmax(dim=-1) == token-1).sum() / video_len.sum()
                        logger.info(f"step {global_step+1}: acc = {acc:.4f}, pitch = {pitch_l1_loss:.4f}, energy = {energy_l1_loss:.4f}, mel = {mel_loss:.4f}")
                        self.writer.add_scalar('train/total_loss', total_loss.item(), global_step)
                        self.writer.add_scalar('train/mel_loss', mel_loss.item(), global_step)
                        self.writer.add_scalar('train/var_loss', var_loss.item(), global_step)
                        self.writer.add_scalar('train/post_loss', post_loss.item(), global_step)
                        self.writer.add_scalar('train/code_ce_loss', code_ce_loss.item(), global_step)
                        self.writer.add_scalar('train/pitch_l1_loss', pitch_l1_loss.item(), global_step)
                        self.writer.add_scalar('train/energy_l1_loss', energy_l1_loss.item(), global_step)
                        self.writer.add_scalar('train/acc', acc, global_step)
                        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], global_step)
                        self.writer.add_figure("train/mel/target", plot_spectrogram(mel[0].permute(1, 0)), global_step)
                        self.writer.add_figure("train/mel/coarse", plot_spectrogram(mel_coarse[0]), global_step)
                        self.writer.add_figure("train/pitch", plot_curves('pred', pitch_pred[0], 'gt', pitch[0]), global_step)
                        self.writer.add_figure("train/energy", plot_curves('pred', energy_pred[0], 'gt', energy[0]), global_step)

                if global_step % val_step == 0:
                    eval_acc, eval_loss = self.evaluate(val_dataloder, val_dataset, self.sample_steps, global_step)
                    logger.info(f"step {global_step+1}: eval_loss = {eval_loss:.4f}, eval_acc = {eval_acc:.4f}")
                    self.save_checkpoint(global_step, os.path.join(self.tensorboard_log_dir, 'model_{}.pt'.format(global_step)))
                global_step += 1

        self.writer.close()



    def evaluate(self, val_dataloader, val_dataset, sample_steps, global_step, n_timesteps=10, temperature=1.0):
        if self.vocoder is not None:
            self.vocoder = self.accelerator.prepare(self.vocoder)
            self.vocoder.eval()
        eval_step = 0
        ce_losses = []
        mel_losses = []
        pitch_losses = []
        energy_losses = []
        accs = []
        self.model.eval()
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                video = batch['video']
                video_len = batch['video_lengths']
                mel = batch['mel']
                pitch = batch['pitch']
                energy = batch['energy']
                token = batch['token']
                spk_ids = batch['spk_ids']
                mel_fine, mel_coarse, logits, pitch_hat, energy_hat = self.model.infer(video, video_len, spk_ids)
                ce_loss = self.model.code_loss(logits.permute(0, 2, 1), token-1)
                acc = (logits.argmax(dim=-1) == token-1).sum() / video_len.sum()
                mel_loss = self.model.mel_loss(mel_fine, mel)
                pitch_loss = self.model.pitch_loss(pitch_hat, pitch)
                energy_loss = self.model.energy_loss(energy_hat, energy)
                ce_losses.append(ce_loss.item())
                mel_losses.append(mel_loss.item())
                pitch_losses.append(pitch_loss.item())
                energy_losses.append(energy_loss.item())
                accs.append(acc.cpu())
                eval_step += 1
                if eval_step >= sample_steps:
                    break
        
        accs_mean = np.mean(accs).item()
        ce_loss_mean = np.mean(ce_losses).item()
        mel_loss_mean = np.mean(mel_losses).item()
        pitch_loss_mean = np.mean(pitch_losses).item()
        energy_loss_mean = np.mean(energy_losses).item()

        if self.vocoder is not None:
            token_pred = logits.argmax(dim=-1) + 1
            y_pred = self.vocoder(token_pred).squeeze()
            y_vocoder = self.vocoder(token).squeeze()
            terget_lens = video_len * 640

        for _ in range(video.size(0)):
            mel_lengths = video_len * 4
            self.writer.add_scalar(f"eval/ce_loss", ce_loss_mean, global_step)
            self.writer.add_scalar(f"eval/mel_loss", mel_loss_mean, global_step)
            self.writer.add_scalar(f"eval/pitch_loss", pitch_loss_mean, global_step)
            self.writer.add_scalar(f"eval/energy_loss", energy_loss_mean, global_step)
            self.writer.add_scalar(f"eval/acc", accs_mean, global_step)
            self.writer.add_figure(f"eval/mel_{_}/target", plot_spectrogram(mel[_, :, :mel_lengths[_]].permute(1, 0)), global_step)
            self.writer.add_figure(f"eval/mel_{_}/coarse", plot_spectrogram(mel_coarse[_, :mel_lengths[_], :]), global_step)
            self.writer.add_figure(f"eval/mel_{_}/fine", plot_spectrogram(mel_fine[_, :, :mel_lengths[_]].permute(1, 0)), global_step)
            if self.vocoder is not None:
                self.writer.add_audio(f"eval/vocoder_{_}", y_vocoder[_, :terget_lens[_]].cpu(), sample_rate=16000, global_step=global_step)
                self.writer.add_audio(f"eval/pred_{_}", y_pred[_, :terget_lens[_]].cpu(), sample_rate=16000, global_step=global_step)
        self.model.train()
        return accs_mean, mel_loss_mean