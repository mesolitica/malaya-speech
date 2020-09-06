from pysndfx import AudioEffectsChain
from malaya_speech.utils.astype import int_to_float


def sox_enhancer(y):
    y = int_to_float(y)
    apply_audio_effects = (
        AudioEffectsChain()
        .lowshelf(gain = 10.0, frequency = 260, slope = 0.1)
        .reverb(
            reverberance = 25,
            hf_damping = 5,
            room_scale = 5,
            stereo_depth = 50,
            pre_delay = 20,
            wet_gain = 0,
            wet_only = False,
        )
    )
    y_enhanced = apply_audio_effects(y)

    return y_enhanced
