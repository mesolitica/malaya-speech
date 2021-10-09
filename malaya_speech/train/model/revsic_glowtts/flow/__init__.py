from typing import Tuple

import tensorflow as tf

from .actnorm import ActNorm
from .coupling import AffineCoupling
from .inv1x1conv import Inv1x1Conv
from ..config import Config
from ..wavenet import WaveNet


class WaveNetFlow(tf.keras.Model):
    """WaveNet based Normalizing flows.
    """

    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: Configuration object.
                flow_block_num: int, the number of the blocks in flow.
                flow_groups: int, size of the groups for invertible 1x1 convolution.
                neck: int, size of the input/output channels.
                channels: int, size of the hidden channels.
                wavenet_block_num: int, the number of the blocks in each wavenet.
                wavenet_cycle: int, the number of the cycles in each wavenet.
                wavenet_kernel_size: int, size of the convolutional kernel in wavenet.
                wavenet_dilation: int, dilation rate for wavenet.
        """
        super().__init__()
        self.flows = []
        for _ in range(config.flow_block_num):
            self.flows.extend([
                ActNorm(config.neck),
                Inv1x1Conv(config.flow_groups),
                AffineCoupling(
                    config.neck,
                    config.channels,
                    WaveNet(
                        config.wavenet_block_num,
                        config.wavenet_cycle,
                        config.channels,
                        config.wavenet_kernel_size,
                        config.wavenet_dilation))])

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute latent from inputs.
        Args:
            inputs: [tf.float32; [B, T, C]], input tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        Returns:
            latent: [tf.float32; [B, T, C]], latent tensor.
            dlogdet: [tf.float32; [B]], likelihood contribution.
        """
        # [B, T, C]
        x, contrib = inputs, []
        for block in self.flows:
            # [B, T, C], [B]
            x, dlogdet = block(x, mask)
            contrib.append(dlogdet)
        # [B, T, C], [B]
        return x, tf.reduce_sum(contrib, axis=0)

    def inverse(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Generate samples from latent.
        Args:
            inputs: [tf.float32; [B, T, C]], latent tensor.
            mask: [tf.float32; [B, T]], binary sequence mask.
        Returns:
            outputs: [tf.float32; [B, T, C]], generated samples.
        """
        # [B, T, C]
        x = inputs
        for block in self.flows[::-1]:
            # [B, T, C]
            x = block.inverse(x, mask)
        # [B, T, C]
        return x
