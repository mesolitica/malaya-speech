import typing as tp

import torch

from config import DataConfig


def build_delay_indices(B: int, T: int, C: int, delay_pattern: tp.List[int]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
    Negative t_idx => BOS; t_idx >= T => PAD.
    """
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)

    t_idx_BxT = torch.broadcast_to(
        torch.arange(T, dtype=torch.int32)[None, :],
        [B, T],
    )
    t_idx_BxTx1 = t_idx_BxT[..., None]
    t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C)

    b_idx_BxTxC = torch.broadcast_to(
        torch.arange(B, dtype=torch.int32).view(B, 1, 1),
        [B, T, C],
    )
    c_idx_BxTxC = torch.broadcast_to(
        torch.arange(C, dtype=torch.int32).view(1, 1, C),
        [B, T, C],
    )

    # We must clamp time indices to [0..T-1] so gather_nd equivalent won't fail
    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)

    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_clamped_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        dim=1,
    ).long()  # Ensure indices are long type for indexing

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    bos_value: int,
    precomp: tp.Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Applies the delay pattern to batched audio tokens using precomputed indices,
    inserting BOS where t_idx < 0 and PAD where t_idx >= T.

    Args:
        audio_BxTxC: [B, T, C] int16 audio tokens (or int32/float)
        pad_value: the padding token
        bos_value: the BOS token
        precomp:  (t_idx_BxTxC, indices_BTCx3) from build_delay_indices

    Returns:
        result_BxTxC: [B, T, C] delayed audio tokens
    """
    device = audio_BxTxC.device  # Get device from input tensor
    t_idx_BxTxC, indices_BTCx3 = precomp
    t_idx_BxTxC = t_idx_BxTxC.to(device)  # Move precomputed indices to device
    indices_BTCx3 = indices_BTCx3.to(device)

    # Equivalent of tf.gather_nd using advanced indexing
    # Ensure indices are long type if not already (build_delay_indices should handle this)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)

    # Create masks on the correct device
    mask_bos = t_idx_BxTxC < 0  # => place bos_value
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]  # => place pad_value

    # Create scalar tensors on the correct device
    bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype)
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype)

    # If mask_bos, BOS; else if mask_pad, PAD; else original gather
    # All tensors should now be on the same device
    result_BxTxC = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))

    return result_BxTxC


@torch.no_grad()
@torch.inference_mode()
def audio_to_codebook(
    model,
    input_values,
    data_config: DataConfig,
    padding_mask=None,
    sample_rate=44100,
):
    """
    Encodes the input audio waveform into discrete codes.

    Args:
        model: The model to use for encoding.
        input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
            Float values of the input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
            Padding mask used to pad the `input_values`.
        sample_rate (`int`, *optional*) :
            Signal sampling_rate

    Returns:
        A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
        factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
        `codebook` of shape `[batch_size, num_codebooks, frames]`.
        Scale is not used here.

    """
    audio_data = model.preprocess(input_values, sample_rate)

    if padding_mask is None:
        padding_mask = torch.ones_like(input_values).bool()

    _, encoded_frame, _, _, _ = model.encode(audio_data, n_quantizers=None)  # 1, C, T
    seq_length = encoded_frame.shape[2]

    t_idx_BxTxC, indices_BTCx3 = build_delay_indices(
        B=1,
        T=seq_length,
        C=data_config.channels,
        delay_pattern=data_config.delay_pattern,
    )

    encoded_frame = apply_audio_delay(
        audio_BxTxC=encoded_frame.transpose(1, 2),  # 1, T, C
        pad_value=data_config.audio_pad_value,
        bos_value=data_config.audio_bos_value,
        precomp=(t_idx_BxTxC, indices_BTCx3),
    )

    return encoded_frame


def build_revert_indices(B: int, T: int, C: int, delay_pattern: tp.List[int]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute indices for the revert operation using PyTorch.

    Returns:
        A tuple (t_idx_BxTxC, indices_BTCx3) where:
            - t_idx_BxTxC is a tensor of shape [B, T, C] computed as time indices plus the delay.
            - indices_BTCx3 is a tensor of shape [B*T*C, 3] used for gathering, computed from:
                batch indices, clamped time indices, and channel indices.
    """

    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)

    t_idx_BT1 = torch.broadcast_to(torch.arange(T).unsqueeze(0), [B, T])
    t_idx_BT1 = t_idx_BT1.unsqueeze(-1)

    t_idx_BxTxC = torch.minimum(
        t_idx_BT1 + delay_arr.view(1, 1, C),
        torch.tensor(T - 1),
    )
    b_idx_BxTxC = torch.broadcast_to(torch.arange(B).view(B, 1, 1), [B, T, C])
    c_idx_BxTxC = torch.broadcast_to(torch.arange(C).view(1, 1, C), [B, T, C])

    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_idx_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        axis=1,
    ).long()  # Ensure indices are long type

    return t_idx_BxTxC, indices_BTCx3


def revert_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    precomp: tp.Tuple[torch.Tensor, torch.Tensor],
    T: int,
) -> torch.Tensor:
    """
    Reverts a delay pattern from batched audio tokens using precomputed indices (PyTorch version).

    Args:
        audio_BxTxC: Input delayed audio tensor
        pad_value: Padding value for out-of-bounds indices
        precomp: Precomputed revert indices tuple containing:
            - t_idx_BxTxC: Time offset indices tensor
            - indices_BTCx3: Gather indices tensor for original audio
        T: Original sequence length before padding

    Returns:
        Reverted audio tensor with same shape as input
    """
    t_idx_BxTxC, indices_BTCx3 = precomp
    device = audio_BxTxC.device  # Get device from input tensor

    # Move precomputed indices to the same device as audio_BxTxC if they aren't already
    t_idx_BxTxC = t_idx_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)

    # Using PyTorch advanced indexing (equivalent to tf.gather_nd or np equivalent)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.size())  # Use .size() for robust reshaping

    # Create pad_tensor on the correct device
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype)
    # Create T tensor on the correct device for comparison
    T_tensor = torch.tensor(T)

    result_BxTxC = torch.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)  # Changed np.where to torch.where

    return result_BxTxC


@torch.no_grad()
@torch.inference_mode()
def decode(
    model,
    audio_codes,
):
    """
    Decodes the given frames into an output audio waveform
    """
    if len(audio_codes) != 1:
        raise ValueError(f"Expected one frame, got {len(audio_codes)}")

    try:
        audio_values = model.quantizer.from_codes(audio_codes)
        audio_values = model.decode(audio_values[0])

        return audio_values
    except Exception as e:
        print(f"Error in decode method: {str(e)}")
        raise


def codebook_to_audio(generated_codes: torch.Tensor, model, delay_pattern, B=1, T=2600, C=9):
    """Process a single codebook file to generate audio"""
    # Remove BOS token
    generated_codes = generated_codes[:, 1:]

    if generated_codes.shape[1] > T:
        generated_codes = generated_codes[:, :T]

    seq_length = generated_codes.shape[1]

    # Build revert indices
    t_idx_BxTxC, indices_BTCx3 = build_revert_indices(B=B, T=seq_length, C=C, delay_pattern=delay_pattern)

    # Transpose and add batch dimension
    audio_BxTxC = generated_codes.transpose(1, 0).unsqueeze(0)
    reverted_codebook = revert_audio_delay(
        audio_BxTxC=audio_BxTxC,
        pad_value=0,
        precomp=(t_idx_BxTxC, indices_BTCx3),
        T=seq_length,
    )
    reverted_codebook = reverted_codebook[:, :-30, :]

    codebook = reverted_codebook.transpose(1, 2)

    min_valid_index = 0
    max_valid_index = 1023
    invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)

    num_invalid = torch.sum(invalid_mask).item()
    if num_invalid > 0:
        print(f"Warning: Clamping {num_invalid} indices outside range [{min_valid_index}, {max_valid_index}] to 0.")

    # Set invalid values to 0 (modify the tensor in-place)
    codebook[invalid_mask] = 0
    audio_array = decode(model, codebook)

    return audio_array