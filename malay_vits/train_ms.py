import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import LoggerType

import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
    TTS_SYMBOLS,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    hps = utils.get_hparams()
    run(hps)


def run(hps):
    global global_step

    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)

    torch.manual_seed(hps.train.seed)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        shuffle=True, num_replicas=1, rank=0)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=4, shuffle=False,
                             batch_size=hps.train.batch_size, pin_memory=True,
                             drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        len(TTS_SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    print(net_g)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(
                hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(
                hps.model_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except BaseException:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(
            epoch, hps, [
                net_g, net_d], [
                optim_g, optim_d], [
                scheduler_g, scheduler_d], [
                    train_loader, eval_loader], logger)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(epoch, hps, nets, optims, schedulers, loaders, logger):
    writer = SummaryWriter(log_dir=hps.model_dir)
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths,
                    speakers) in enumerate(train_loader):
        x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
        spec, spec_lengths = spec.cuda(non_blocking=True), spec_lengths.cuda(non_blocking=True)
        y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)
        speakers = speakers.cuda(non_blocking=True)

        y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)

        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
        y_mel = commons.slice_segments(
            mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )

        y = commons.slice_segments(y, ids_slice *
                                   hps.data.hop_length, hps.train.segment_size)  # slice

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
        optim_d.zero_grad()
        loss_disc_all.backward()
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        optim_d.step()

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        loss_gen_all.backward()
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        optim_g.step()

        if global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]['lr']
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
            logger.info('Train Epoch: {} [{:.0f}%]'.format(
                epoch,
                100. * batch_idx / len(train_loader)))
            logger.info([x.item() for x in losses] + [global_step, lr])

            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g}
            scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel,
                                "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

            scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
            scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
            image_dict = {}
            utils.summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict)

        if global_step % hps.train.eval_interval == 0:
            utils.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(
                    hps.model_dir, "G_{}.pth".format(global_step)))
            utils.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(
                    hps.model_dir, "D_{}.pth".format(global_step)))
        global_step += 1

    logger.info('====> Epoch: {}'.format(epoch))


if __name__ == "__main__":
    main()
