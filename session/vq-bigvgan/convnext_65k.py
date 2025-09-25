import sys

sys.path.insert(0, '/home/jovyan/BigVGAN')
sys.path.insert(0, '/home/ubuntu/BigVGAN')
sys.path.insert(0, '/home/husein/ssd3/BigVGAN')

from meldataset import get_mel_spectrogram
from bigvgan import BigVGAN
from discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiBandDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
    MultiScaleMelSpectrogramLoss,
)
import itertools
import soundfile as sf
import librosa
import torch
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoFeatureExtractor, HubertModel
from x_transformers.x_transformers import RotaryEmbedding
from torch.nn.utils.rnn import pad_sequence
from dit import (
    ConvPositionEmbedding,
    Block
)

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

        # compute distances
        distances = (
            flat_hidden.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(flat_hidden, self.codebook.weight.t())
            + self.codebook.weight.pow(2).sum(1)
        )

        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices).view(batch_size, seq_len, dim)

        if self.training:
            mask_flat = attention_mask.reshape(-1) > 0
            indices_masked = indices[mask_flat]
            hidden_masked = flat_hidden[mask_flat]

            n = torch.bincount(indices_masked,
                            minlength=self.quantize_vocab_size)

            if is_dist_initialized():
                torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)

            p = n.float() / n.sum().clamp(min=1)
            self.quantize_perplexity = torch.exp(
                -torch.sum(p * torch.log(p + 1e-10))
            ).item()
            self.num_active_codes = (n > 0).sum().item()

            self.total_code_usage[indices_masked] = 1.0

            dw = torch.zeros(
                self.quantize_vocab_size,
                dim,
                device=hidden_masked.device,
                dtype=hidden_masked.dtype,
            )
            dw.index_add_(0, indices_masked, hidden_masked)

            if is_dist_initialized():
                torch.distributed.all_reduce(dw, op=torch.distributed.ReduceOp.SUM)

            n = n.to(self.ema_count.dtype)
            dw = dw.to(self.ema_weight.dtype)

            self.ema_count = self.ema_count * self.quantize_ema_decay + (
                1 - self.quantize_ema_decay
            ) * n
            total_count = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + 1e-5) / (
                total_count + self.quantize_vocab_size * 1e-5
            ) * total_count

            self.ema_weight = self.ema_weight * self.quantize_ema_decay + (
                1 - self.quantize_ema_decay
            ) * dw

            if self.quantize_ema_count % self.quantize_update_interval == 0:
                self.codebook.weight.data = self.ema_weight / self.ema_count.unsqueeze(1)

            self.quantize_loss = (
                self.quantize_loss_scale
                * self.quantize_commit_coefficient
                * mse_loss_with_mask(hidden_states, quantized.detach(), attention_mask)
            )
            self._maybe_restart_codes(flat_hidden.detach(), attention_mask)
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
            chosen_indices = (
                torch.randperm(len(hidden_selected), device=hidden_selected.device)[:num_update]
                if num_update <= len(hidden_selected)
                else torch.randint(0, len(hidden_selected), (num_update,), device=hidden_selected.device)
            )
            hidden_update = hidden_selected[chosen_indices]

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

            self.codebook.weight.data[update_indices] = hidden_update.to(self.codebook.weight.data.dtype)
            self.ema_count[update_indices] = 1
            self.ema_weight[update_indices] = hidden_update.to(self.ema_weight.dtype)

            if rank == 0:
                print(f"[VQ] Restarted {len(update_indices)} dead codes.")
    
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.conv_pos_embed(x) + x
        return x

class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        input_dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,     
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()

        self.input_embed = InputEmbedding(input_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.transformer_blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )

        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(
        self,
        x: torch.Tensor, 
        mask: torch.Tensor | None = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        x = self.input_embed(x)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(block, x, mask, rope, use_reentrant=False)
            else:
                x = block(x, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x)
        return self.proj_out(x)

class VQMelTransformer(nn.Module):
    def __init__(
        self, 
        num_embeddings=65536,
    ):
        super().__init__()
        self.dit = DiT(dim=1024, input_dim=768, depth=22, heads=16)
        self.vq = VectorQuantizerEMA(num_embeddings, 768, centroid_path='centroids-v2-65k.npy')

    def forward(self, mel, frame_mask, attention_mask):
        z_q, indices = self.vq(mel, frame_mask)
        # z_q = F.interpolate(z_q.permute(0, 2, 1), scale_factor=1.2554112554112553, mode="linear")
        z_q = F.interpolate(z_q.permute(0, 2, 1), size=attention_mask.shape[1], mode="linear")
        recon = self.dit(z_q.permute(0, 2, 1), mask=attention_mask)
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
    segment_size = (h.sampling_rate * 1) // h.hop_size
    warmup_steps = 2000
    epoch = 5
    log_interval = 5
    save_interval = 1000
    max_ckpt = 5
    learning_rate_generator = 5e-5
    learning_rate = 5e-5
    grad_norm = 1.0
    batch_size = 18
    num_workers = 5
    debug = False

    run_dir = 'convnext_65k'
    os.makedirs(run_dir, exist_ok = True)

    vocoder = BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False).to(torch.bfloat16)
    _ = vocoder.to(device)
    for p in vocoder.parameters():
        p.requires_grad = False

    net_g = VQMelTransformer().to(device)
    hubert = HubertModel.from_pretrained("utter-project/mHuBERT-147").to(torch.bfloat16)
    _ = hubert.to(device)
    for p in hubert.parameters():
        p.requires_grad = False
    processor = AutoFeatureExtractor.from_pretrained("utter-project/mHuBERT-147")

    print(sum(p.numel() for p in net_g.parameters()))
    time.sleep(5.0)
    
    mpd = MultiPeriodDiscriminator(h).to(device)
    mrd = MultiScaleSubbandCQTDiscriminator(h).to(device)

    net_g = torch.nn.parallel.DistributedDataParallel(
        net_g, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=True,
    )
    mpd = torch.nn.parallel.DistributedDataParallel(
        mpd, device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=True,
    )
    mrd = torch.nn.parallel.DistributedDataParallel(
        mrd, device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=True,
    )

    optim_g = torch.optim.AdamW(
        (p for p in net_g.parameters() if p.requires_grad),
        learning_rate_generator)
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        learning_rate)
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, file):
            with open(file) as fopen:
                self.files = json.load(fopen)
        
        def __getitem__(self, idx):
            try:
                wav, _ = librosa.load(self.files[idx], sr=h.sampling_rate, mono=True)
                if (len(wav) / h.sampling_rate) < 2:
                    return

                segment_length = h.sampling_rate * 12
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

        inputs = processor(
            [w.numpy() for w in wavs], 
            sampling_rate=processor.sampling_rate, 
            return_tensors='pt', 
            padding=True,
            return_attention_mask=True,
        )

        wavs = torch.nn.utils.rnn.pad_sequence(
            wavs, batch_first=True, padding_value=0.0, padding_side='right')

        mels = []
        for i in range(len(wavs)):
            mel = get_mel_spectrogram(wavs[i][None], h)
            mels.append(mel)
        mels = torch.concat(mels)
        if mels.shape[-1] % 2 != 0:
            print('not even')
            mels = F.pad(mels, (0, 1, 0, 0, 0, 0))
        
        lengths = torch.tensor(lengths) // h.hop_size
        return {'mel': mels, 'lengths': lengths, 'wav': wavs, **inputs}
    
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
    scheduler_d = get_linear_schedule_with_warmup(optim_d, warmup_steps, total_steps)

    fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
        sampling_rate=h.sampling_rate
    )

    step = 1
    try:
        ckpts = sorted(glob(os.path.join(run_dir, f"checkpoint_{local_rank}_*.pt")), key=os.path.getmtime)
        ckpt = torch.load(ckpts[-1], map_location=device)
        net_g.load_state_dict(ckpt["net_g"])
        mpd.load_state_dict(ckpt["mpd"])
        mrd.load_state_dict(ckpt["mrd"])
        optim_g.load_state_dict(ckpt["optim_g"])
        optim_d.load_state_dict(ckpt["optim_d"])
        scheduler_g.load_state_dict(ckpt["scheduler_g"])
        scheduler_d.load_state_dict(ckpt["scheduler_d"])
        step = ckpt["step"]
        print(f'loaded checkpoint {ckpts[-1]}')
    except Exception as e:
        print(e)

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
        input_values = batch['input_values'].to(device)
        attention_mask_ = batch['attention_mask'].to(device)
        attention_mask = torch.arange(mel.shape[-1], device = lengths.device).expand(lengths.shape[0], mel.shape[-1]) < lengths.unsqueeze(1)

        with autocast(dtype=torch.bfloat16, enabled=True):
            z_q = hubert(input_values=input_values, attention_mask=attention_mask_).last_hidden_state
            frame_mask = F.interpolate(
                attention_mask_.unsqueeze(1).float(),
                size=z_q.shape[1],
                mode="nearest"
            )
            outputs = net_g(z_q, frame_mask, attention_mask)

            y_hat_mel, ids_slice = rand_slice_segments(outputs[0], lengths, segment_size)
            y_mel = slice_segments(mel, ids_slice, segment_size)
            y = slice_segments(wav.unsqueeze(1), ids_slice * h.hop_size, segment_size * h.hop_size)

            print(outputs[0].shape, mel.shape)

            if dist.get_rank() == 0:
                print(y_mel.min(), y_mel.max(), y_hat_mel.min(), y_hat_mel.max())
            
            y_hat = vocoder(y_hat_mel)
            y = y[:,:,:y_hat.shape[-1]]
            print(y.shape, y_hat.shape)
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_hat.detach())
            with autocast(dtype=torch.bfloat16, enabled=False):
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )
            
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_hat.detach())
            with autocast(dtype=torch.bfloat16, enabled=False):
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )
                loss_disc_all = loss_disc_s + loss_disc_f

        optim_d.zero_grad()
        loss_disc_all.backward()
        grad_norm_mpd = torch.nn.utils.clip_grad_norm_(mpd.parameters(), grad_norm)
        grad_norm_mrd = torch.nn.utils.clip_grad_norm_(mrd.parameters(), grad_norm)
        optim_d.step()
        scheduler_d.step()

        with autocast(dtype=torch.bfloat16, enabled=True):
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_hat)
            with autocast(dtype=torch.bfloat16, enabled=False):
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_hat)
            with autocast(dtype=torch.bfloat16, enabled=False):
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                # loss_mel = fn_mel_loss_multiscale(y, y_hat) * 45.0
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45.0
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        optim_g.zero_grad()
        loss_gen_all.backward()
        grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), grad_norm)
        optim_g.step()
        scheduler_g.step()
        
        if step % log_interval == 0 and dist.get_rank() == 0:
            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "lr_g": scheduler_g.get_last_lr()[0],
                "lr_d": scheduler_d.get_last_lr()[0],
                "grad_norm_mpd": grad_norm_mpd,
                "grad_norm_mrd": grad_norm_mrd,
                "grad_norm_g": grad_norm_g,
                "quantize_loss": net_g.module.vq.quantize_loss,
                "quantize_perplexity": net_g.module.vq.quantize_perplexity,
                "num_active_codes": net_g.module.vq.num_active_codes,
                "total_code_usage": net_g.module.vq.total_code_usage.sum().item(),
            }
            scalar_dict.update({"loss/g/fm": loss_fm_s, "loss/g/mel": loss_mel})
            scalar_dict['global_step'] = step 
            wandb.log(scalar_dict)
        
        if step % save_interval == 0:
            ckpt = {
                "net_g": net_g.state_dict(),
                "mpd": mpd.state_dict(),
                "mrd": mrd.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
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