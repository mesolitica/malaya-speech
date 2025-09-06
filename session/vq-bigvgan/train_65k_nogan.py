import sys

sys.path.insert(0, '/home/jovyan/BigVGAN')
sys.path.insert(0, '/home/husein/ssd3/BigVGAN')

from meldataset import get_mel_spectrogram
from loss import MultiScaleMelSpectrogramLoss
import soundfile as sf
import librosa
import torch
import json
import bigvgan
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch import nn, optim
from transformers import get_linear_schedule_with_warmup
from accelerate import skip_first_batches
from env import AttrDict
from tqdm import tqdm
import random
import os
import wandb
import time
import numpy as np
from glob import glob

def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def mse_loss_with_mask(input, target, mask):
    loss = torch.nn.functional.mse_loss(input, target, reduction='none')
    loss = loss.mean(dim=-1)
    loss = loss * mask
    return loss.sum() / mask.sum()

def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=0)
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str

class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, centroid_path=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.quantize_vocab_size = num_embeddings
        self.quantize_ema_decay = 0.99
        self.quantize_loss_scale = 10.0
        self.quantize_commit_coefficient = 0.25
        self.codebook = nn.Embedding(self.quantize_vocab_size, embedding_dim)
        if centroid_path is not None:
            init_codes = np.load(centroid_path)
            self.codebook.weight.data.copy_(torch.from_numpy(init_codes))
            print(f'loaded codebook weight from {centroid_path}')
        self.codebook.weight.requires_grad = False

        self.register_buffer("ema_count", torch.ones(self.quantize_vocab_size, dtype=torch.float))
        self.register_buffer("ema_weight", self.codebook.weight.data.clone().float())
        self.quantize_ema_count = 1

        self.quantize_update_interval = 50
        self.quantize_restart_interval = 500

        self.register_buffer("total_code_usage", torch.zeros(self.quantize_vocab_size))

        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))

    def forward(self, z, attention_mask):
        hidden_states = z
        batch_size, seq_len, dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, dim)
        
        # Check for NaN in input
        if torch.isnan(flat_hidden).any():
            print(f"[VQ] NaN detected in input hidden_states!")
            return hidden_states, torch.zeros(batch_size * seq_len, dtype=torch.long, device=z.device)
        
        distances = (
            flat_hidden.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(flat_hidden, self.codebook.weight.t())
            + self.codebook.weight.pow(2).sum(1)
        )

        # Check for NaN in distances
        if torch.isnan(distances).any():
            print(f"[VQ] NaN detected in distances! flat_hidden range: {flat_hidden.min():.4f} to {flat_hidden.max():.4f}")
            print(f"[VQ] Codebook weight range: {self.codebook.weight.min():.4f} to {self.codebook.weight.max():.4f}")
            return hidden_states, torch.zeros(batch_size * seq_len, dtype=torch.long, device=z.device)

        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices).view(batch_size, seq_len, dim)

        if self.training:
            encodings = F.one_hot(indices, self.quantize_vocab_size).float()
            encodings = encodings * attention_mask.reshape(-1, 1)
            n = torch.sum(encodings, dim=0)
            if is_dist_initialized():
                torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)

            # Safety check for perplexity calculation
            n_sum = n.sum()
            if n_sum > 0:
                p = n / n_sum
                # Clamp probabilities to avoid log(0)
                p = torch.clamp(p, min=1e-10)
                self.quantize_perplexity = torch.exp(-torch.sum(p * torch.log(p))).item()
            else:
                self.quantize_perplexity = 0.0
            
            self.num_active_codes = (n > 0).sum().item()
            self.total_code_usage[indices] = 1.0

            hidden_flat = flat_hidden.detach()
            dw = torch.matmul(encodings.t(), hidden_flat)
            if is_dist_initialized():
                torch.distributed.all_reduce(dw, op=torch.distributed.ReduceOp.SUM)

            # Add epsilon to prevent division by zero
            self.ema_count = self.ema_count * self.quantize_ema_decay + (
                1 - self.quantize_ema_decay) * n
            total_count = torch.sum(self.ema_count)
            
            # Safety check for EMA update
            if total_count > 0:
                self.ema_count = (self.ema_count + 1e-5) / (
                    total_count + self.quantize_vocab_size * 1e-5) * total_count
                self.ema_weight = self.ema_weight * self.quantize_ema_decay + (
                    1 - self.quantize_ema_decay) * dw
                
                if self.quantize_ema_count % self.quantize_update_interval == 0:
                    # Check for NaN before updating codebook
                    new_weights = self.ema_weight / self.ema_count.unsqueeze(1)
                    if not torch.isnan(new_weights).any() and not torch.isinf(new_weights).any():
                        self.codebook.weight.data = new_weights
                    else:
                        print(f"[VQ] Skipping codebook update due to NaN/Inf in new weights")
                        
            self.quantize_loss = self.quantize_loss_scale * self.quantize_commit_coefficient * mse_loss_with_mask(
                                hidden_states, quantized.detach(), attention_mask)
            
            # Check quantize_loss for NaN
            if torch.isnan(self.quantize_loss):
                print(f"[VQ] quantize_loss is NaN! Setting to 0")
                self.quantize_loss = torch.tensor(0.0, device=z.device, requires_grad=True)
            
            self._maybe_restart_codes(hidden_flat, attention_mask)
            self.quantize_ema_count += 1

            hidden_states = hidden_states + (quantized - hidden_states).detach()
        else:
            hidden_states = quantized
        
        return hidden_states, indices
    
    def _maybe_restart_codes(self, hidden_flat, attention_mask):
        if self.quantize_restart_interval is None:
            return

        if self.quantize_ema_count % self.quantize_restart_interval != 0:
            return
        
        if not is_dist_initialized():
            return
            
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        segment_vocab_size = self.quantize_vocab_size // world_size
        start_idx = segment_vocab_size * rank
        ema_count_segment = self.ema_count[start_idx: start_idx + segment_vocab_size]
        threshold = self.quantize_ema_decay ** self.quantize_restart_interval
        update_indices = (ema_count_segment < threshold).nonzero()[:, 0] + start_idx
        num_update = update_indices.shape[0]

        print('num_update', num_update)

        if num_update > 0:
            mask_flat = attention_mask.reshape(-1) > 0
            hidden_selected = hidden_flat[mask_flat]
            
            # Safety check
            if len(hidden_selected) == 0:
                print(f"[VQ] No valid hidden states for code restart")
                return
                
            chosen_indices = (
                torch.randperm(len(hidden_selected), device=hidden_selected.device)[:num_update]
                if num_update <= len(hidden_selected)
                else torch.randint(0, len(hidden_selected), (num_update,), device=hidden_selected.device)
            )
            hidden_update = hidden_selected[chosen_indices]
            
            # Check for NaN in hidden_update
            if torch.isnan(hidden_update).any():
                print(f"[VQ] NaN detected in hidden_update, skipping code restart")
                return

            num_update = torch.as_tensor([num_update], dtype=torch.long, device=hidden_flat.device)
            num_update_list = [torch.as_tensor([0], dtype=torch.long, device=hidden_flat.device)
                               for _ in range(world_size)]
            torch.distributed.all_gather(num_update_list, num_update)

            update_indices_list = [
                torch.zeros(num.item(), dtype=torch.long, device=hidden_flat.device)
                for num in num_update_list]
            torch.distributed.all_gather(update_indices_list, update_indices)
            update_indices = torch.cat(update_indices_list)

            hidden_update_list = [
                torch.zeros(num.item(), hidden_flat.shape[-1], dtype=hidden_update.dtype,
                            device=hidden_flat.device) for num in num_update_list]
            torch.distributed.all_gather(hidden_update_list, hidden_update)
            hidden_update = torch.cat(hidden_update_list)

            # Safety check before updating
            if not torch.isnan(hidden_update).any() and not torch.isinf(hidden_update).any():
                self.codebook.weight.data[update_indices] = hidden_update.to(self.codebook.weight.data.dtype)
                self.ema_count[update_indices] = 1
                self.ema_weight[update_indices] = hidden_update.to(self.ema_weight.dtype)

                if rank == 0:
                    print(f"[VQ] Restarted {len(update_indices)} dead codes.")
            else:
                if rank == 0:
                    print(f"[VQ] Skipping code restart due to NaN/Inf values")
    
class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, output_dim, num_layers=12, nhead=8, dropout=0.1, max_len = 3000):
        super().__init__()
        assert model_dim % nhead == 0
        self.nhead = nhead
        self.head_dim = model_dim // nhead
        self.projection = nn.Linear(output_dim, model_dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, model_dim))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, z, attention_mask=None):
        z = self.projection(z)
        B, T, D = z.shape

        pos_emb = self.pos_embedding[:, :T, :]
        pos_emb = pos_emb * attention_mask.unsqueeze(-1)
        
        z = z + pos_emb

        z = self.decoder(z, src_key_padding_mask=~(attention_mask.bool()))
        z = z * attention_mask.unsqueeze(-1)
        return self.output_proj(z)
    
class VQMelTransformer(nn.Module):
    def __init__(
        self, 
        mel_dim=100,
        latent_dim=1536,
        num_layers=12,
        nhead=8,
        num_embeddings=65536,
    ):
        super().__init__()
        self.vq = VectorQuantizerEMA(num_embeddings, mel_dim, centroid_path='centroids-v2-65k.npy')
        self.decoder = TransformerDecoder(latent_dim, mel_dim, num_layers=num_layers, nhead=nhead)

    def forward(self, mel, attention_mask):
        mel = mel.permute(0, 2, 1)
        z_q, indices = self.vq(mel, attention_mask)
        recon = self.decoder(z_q, attention_mask=attention_mask)
        return recon.permute(0, 2, 1), indices

def load_hparams_from_json(path):
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    h = load_hparams_from_json('bigvgan-config.json')

    train_dataset = 'audio-files.json'
    segment_size = (h.sampling_rate * 3) // h.hop_size
    warmup_steps = 2000
    epoch = 2
    log_interval = 5
    save_interval = 500
    mel_ratio = 45
    max_ckpt = 5
    learning_rate_generator = 5e-5
    batch_size = 4
    num_workers = 5
    debug = False

    run_dir = 'checkpoint-65k-nogan-v3'
    os.makedirs(run_dir, exist_ok = True)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, file):
            with open(file) as fopen:
                self.files = json.load(fopen)
        
        def __getitem__(self, idx):
            try:
                wav, _ = librosa.load(self.files[idx], sr=h.sampling_rate, mono=True)
                if (len(wav) / h.sampling_rate) < 1:
                    return
                segment_length = h.sampling_rate * 15
                if len(wav) > segment_length:
                    max_start = len(wav) - segment_length
                    start = random.randint(0, max_start)
                    wav = wav[start:start + segment_length]
                
                return {
                    'wav': torch.FloatTensor(wav),
                    'length': len(wav)
                }
            except Exception as e:
                print(e)
        
        def __len__(self):
            return len(self.files)

    def collator(batch):
        batch = [b for b in batch if b is not None]
        wavs = [b['wav'] for b in batch]
        lengths = [b['length'] for b in batch]

        wavs = torch.nn.utils.rnn.pad_sequence(
            wavs, batch_first=True, padding_value=0.0, padding_side='right')

        mels = []
        for i in range(len(wavs)):
            mel = get_mel_spectrogram(wavs[i][None], h)
            mels.append(mel)
        mels = torch.concat(mels)
        lengths = torch.tensor(lengths) // h.hop_size
        return {'mel': mels, 'lengths': lengths, 'wav': wavs}

    net_g = VQMelTransformer().to(device)
    net_g = torch.nn.parallel.DistributedDataParallel(
        net_g, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=True,
    )

    optim_g = torch.optim.AdamW(
        (p for p in net_g.parameters() if p.requires_grad),
        learning_rate_generator,
    )

    train_dataset = Dataset(train_dataset)
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collator,
    )
    
    total_steps = epoch * len(train_loader)
    scheduler_g = get_linear_schedule_with_warmup(optim_g, warmup_steps, total_steps)

    fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
        sampling_rate=h.sampling_rate
    )

    step = 1
    try:
        ckpts = sorted(glob(os.path.join(run_dir, f"checkpoint_{local_rank}_*.pt")), key=os.path.getmtime)
        ckpt = torch.load(ckpts[-1], map_location=device)
        net_g.load_state_dict(ckpt["net_g"])
        optim_g.load_state_dict(ckpt["optim_g"])
        scheduler_g.load_state_dict(ckpt["scheduler_g"])
        step = ckpt["step"]
        print(f'loaded checkpoint {ckpts[-1]}')
    except Exception as e:
        print(e)

    time.sleep(5.0)

    steps_trained_in_current_epoch = step % len(train_loader)
    train_loader = skip_first_batches(train_loader, steps_trained_in_current_epoch)
    sampler.set_epoch(step // len(train_loader))

    pbar = tqdm(total=total_steps, initial=step)
    iter_train_loader = iter(train_loader)
    
    if dist.get_rank() == 0:
        wandb.init()
    else:
        wandb.init(mode="disabled")

    while step < total_steps:
        try:
            batch = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            batch = next(iter_train_loader)

        mel = batch["mel"].to(device)

        if torch.isnan(mel).any() or torch.isinf(mel).any():
            print("Bad batch detected")
            return
            
        wav = batch["wav"].to(device)
        lengths = batch["lengths"].to(device)
        max_len = lengths.max()
        attention_mask = torch.arange(max_len, device = lengths.device).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
        attention_mask = attention_mask.float()

        with autocast(dtype=torch.bfloat16, enabled=True):

            outputs = net_g(mel, attention_mask)
            
            y_hat_mel, ids_slice = rand_slice_segments(outputs[0], lengths, segment_size)
            y_mel = slice_segments(mel, ids_slice, segment_size)

            if dist.get_rank() == 0:
                print(y_mel.min(), y_mel.max(), y_hat_mel.min(), y_hat_mel.max())

            with autocast(dtype=torch.bfloat16, enabled=False):
                min_len = min(y_mel.shape[-1], y_hat_mel.shape[-1])
                y_mel = y_mel[..., :min_len]
                y_hat_mel = y_hat_mel[..., :min_len]

                loss_mel = F.l1_loss(y_mel, y_hat_mel)
                loss_gen_all = loss_mel
        
        optim_g.zero_grad()
        if torch.isnan(loss_gen_all).any() or torch.isinf(loss_gen_all).any():
            print("NaN in loss!")
            for name, p in net_g.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"NaN in param {name}")

            for name, p in net_g.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    print(f"NaN/Inf gradient in {name}")
                    
            return

        loss_gen_all.backward()
        grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1.0)
        optim_g.step()
        scheduler_g.step()

        if step % log_interval == 0 and dist.get_rank() == 0:
            scalar_dict = {
                "grad_norm_g": grad_norm_g,
                "lr_g": scheduler_g.get_last_lr()[0],
                "quantize_loss": net_g.module.vq.quantize_loss,
                "quantize_perplexity": net_g.module.vq.quantize_perplexity,
                "num_active_codes": net_g.module.vq.num_active_codes,
                "total_code_usage": net_g.module.vq.total_code_usage.sum().item(),
                "loss/g/mel": loss_mel,
                "global_step": step,
            }
            try:
                wandb.log(scalar_dict)
            except:
                pass
        
        if step % save_interval == 0:
            ckpt = {
                "net_g": net_g.state_dict(),
                "optim_g": optim_g.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "step": step,
            }
            path = os.path.join(run_dir, f"checkpoint_{local_rank}_{step}.pt")
            torch.save(ckpt, path)
            print(f'save checkpoint {path}')
            ckpts = sorted(glob(os.path.join(run_dir, "checkpoint_*.pt")), key=os.path.getmtime)
            if len(ckpts) > max_ckpt:
                to_delete = ckpts[0]
                os.remove(to_delete)
                print(f"Deleted old checkpoint: {to_delete}")
        
        step += 1
        pbar.update(1)

if __name__ == "__main__":
    main()