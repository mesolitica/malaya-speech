from malaya_speech.supervised import unet


def hybrid_demucs(**kwargs):
    """
    Load TorchAudio Hybrid Demucs, https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html
    """
    return unet.load_torchaudio(**kwargs)
