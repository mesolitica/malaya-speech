from tensor2tensor.data_generators import text_encoder


class ByteTextEncoderWithEos(text_encoder.ByteTextEncoder):
    """Encodes each byte to an id and appends the EOS token."""

    def encode(self, s):
        return super(ByteTextEncoderWithEos, self).encode(s) + [
            text_encoder.EOS_ID
        ]


VOCAB_SIZE = 256
