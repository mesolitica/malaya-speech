import tensorflow as tf
from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator


class Discriminator(tf.keras.Model):
    def __init__(self, hp, **kwargs):
        super().__init__(**kwargs)
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)

    def call(self, x):
        return self.MRD(x), self.MPD(x)


if __name__ == '__main__':
    from malaya_boilerplate.train import config
    import malaya_speech.config

    hparams = config(**malaya_speech.config.univnet_config_c16)
    discriminator = Discriminator(hparams)

    x = tf.random.normal(shape=(2, 16000, 1))
    r = discriminator(x)
