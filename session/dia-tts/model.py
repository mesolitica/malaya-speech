import dac
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from audio import audio_to_codebook, codebook_to_audio
from config import DiaConfig
from layers import DiaModel, KVCache


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    use_cfg_filter: bool,
    cfg_filter_top_k: int | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature
    if use_cfg_filter and cfg_filter_top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        # Calculate indices to remove based on top_p
        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        # Shift the mask to the right to keep the first token above the threshold
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[..., :-1].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0  # Always keep the most probable token

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV.scatter_(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


class Dia:
    def __init__(self, config: DiaConfig, device: torch.device | None = None):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = config
        self.device = device if device is not None else get_default_device()
        self.model = DiaModel(config)
        self.dac_model = None

    @classmethod
    def from_local(cls, config_path: str, checkpoint_path: str, device: torch.device | None = None) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.pth) file.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config, device)

        try:
            state_dict = torch.load(checkpoint_path, map_location=dia.device)
            dia.model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}") from e

        dia.model.to(dia.device)
        dia.model.eval()
        dia._load_dac_model()
        return dia

    @classmethod
    def from_pretrained(
        cls, model_name: str = "nari-labs/Dia-1.6B", device: torch.device | None = None
    ) -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository.

        Downloads the configuration and checkpoint files from the specified
        repository ID and then loads the model.

        Args:
            model_name: The Hugging Face Hub repository ID (e.g., "NariLabs/Dia-1.6B").
            device: The device to load the model onto. If None, will automatically select the best available device.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        checkpoint_path = hf_hub_download(repo_id=model_name, filename="dia-v0_1.pth")
        return cls.from_local(config_path, checkpoint_path, device)

    def _load_dac_model(self):
        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    def _create_attn_mask(
        self,
        q_padding_mask_1d: torch.Tensor,
        k_padding_mask_1d: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Creates the attention mask (self or cross) mimicking JAX segment ID logic.
        """
        B1, Tq = q_padding_mask_1d.shape
        B2, Tk = k_padding_mask_1d.shape
        assert B1 == B2, "Query and key batch dimensions must match"

        p_mask_q = q_padding_mask_1d.unsqueeze(2)  # Shape [B, Tq, 1]
        p_mask_k = k_padding_mask_1d.unsqueeze(1)  # Shape [B, 1, Tk]

        # Condition A: Non-padding query attends to non-padding key
        non_pad_attends_non_pad = p_mask_q & p_mask_k  # Shape [B, Tq, Tk]

        # Condition B: Padding query attends to padding key
        pad_attends_pad = (~p_mask_q) & (~p_mask_k)  # Shape [B, Tq, Tk]

        # Combine: True if padding status is compatible (both non-pad OR both pad)
        # This implementation follows Jax TPU splash attention kernel
        mask = non_pad_attends_non_pad | pad_attends_pad  # Shape [B, Tq, Tk]

        if is_causal:
            # Ensure causality for self-attention (Tq == Tk)
            assert Tq == Tk, "Causal mask requires query and key sequence lengths to be equal"
            # Standard lower-triangular causal mask (True means allow)
            causal_mask_2d = torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=self.device))  # Shape [Tq, Tk]
            causal_mask = mask & causal_mask_2d  # Shape [B, Tq, Tk]
            return causal_mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk] for broadcasting across heads
        else:
            # For cross-attention or non-causal self-attention
            return mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk] for broadcasting across heads

    def _prepare_text_input(self, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes text prompt, pads, and creates attention mask and positions."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length

        byte_text = text.encode("utf-8")
        
        
        replaced_bytes = byte_text

        LANG2BYTE = {
            "en": 3,
            "de": 4,
            "fr": 5,
            "es": 6,
            "it": 7,
            "nl": 14,
            "pl": 15,
            "pt": 16,
            "tr": 17,
            "hu": 18,
        }

        for lang, byte_val in LANG2BYTE.items():
            tag = f"[{lang}]".encode("ascii")        # e.g. b"[de]"
            code = bytes([byte_val])                 # e.g. b"\x04"
            replaced_bytes = replaced_bytes.replace(tag, code)
        text_tokens = list(replaced_bytes)

        current_len = len(text_tokens)
        padding_needed = max_len - current_len
        if padding_needed <= 0:
            text_tokens = text_tokens[:max_len]
            padded_text_np = np.array(text_tokens, dtype=np.uint8)
        else:
            padded_text_np = np.pad(
                text_tokens,
                (0, padding_needed),
                mode="constant",
                constant_values=text_pad_value,
            ).astype(np.uint8)

        src_tokens = torch.from_numpy(padded_text_np).to(torch.long).to(self.device).unsqueeze(0)  # [1, S]
        src_positions = torch.arange(max_len, device=self.device).to(torch.long).unsqueeze(0)  # [1, S]

        src_padding_mask = (src_tokens != text_pad_value).to(self.device)  # [1, S]

        enc_self_attn_mask = self._create_attn_mask(src_padding_mask, src_padding_mask, is_causal=False)  # [1, S, S]

        return src_tokens, src_positions, src_padding_mask, enc_self_attn_mask

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_tokens: int | None = None,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        use_cfg_filter: bool = True,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 35,
        audio_prompt_path: str | None = None,
    ) -> np.ndarray:
        """
        Generates audio from a text prompt (and optional audio prompt) using the Nari model.

        Returns:
            A tensor of generated audio codes (shape: [max_tokens, num_channels]).
        """
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_tokens = self.config.data.audio_length if max_tokens is None else max_tokens
        delay_tensor = torch.tensor(delay_pattern, dtype=torch.long, device=self.device)
        max_delay_pattern = max(delay_pattern)
        self.model.eval()

        (
            cond_src_BxS,
            cond_src_positions_BxS,
            cond_src_padding_mask_BxS,
            cond_enc_self_attn_mask_Bx1xSxS,
        ) = self._prepare_text_input(text)

        unc_src_BxS = torch.zeros_like(cond_src_BxS)
        src_BxS = torch.cat([unc_src_BxS, cond_src_BxS], dim=0)
        src_positions_BxS = cond_src_positions_BxS.expand(2, -1)
        src_padding_mask_BxS = cond_src_padding_mask_BxS.expand(2, -1)
        enc_self_attn_mask_Bx1xSxS = cond_enc_self_attn_mask_Bx1xSxS.expand(2, -1, -1, -1)

        # 2. Encoder Pass
        # with torch.autocast(device_type="cuda", dtype=forward_dtype):
        encoder_out = self.model.encoder(
            x_ids=src_BxS,
            src_positions=src_positions_BxS,
            deterministic=True,
            attn_mask=enc_self_attn_mask_Bx1xSxS,
        )  # Shape: (B, S, E)

        # 3. Prepare Decoder Inputs
        # 3-1. Allocate KV Cache (Static)
        decoder_cross_attention_cache: list[KVCache] = self.model.decoder.precompute_cross_attention_kv(
            max_tokens, encoder_out, src_positions_BxS
        )

        decoder_self_attention_cache: list[KVCache] = []
        for _ in range(self.model.decoder.num_layers):
            decoder_self_attention_cache.append(
                KVCache(
                    self.config.model.decoder.gqa_query_heads,
                    max_tokens,
                    self.config.model.decoder.gqa_head_dim,
                    self.device,
                )
            )

        # 3-2. Initialize Decoder Inputs
        generated_BxTxC = torch.full(
            (2, 1, num_channels),
            fill_value=audio_bos_value,
            dtype=torch.long,
            device=self.device,
        )

        current_step = 0
        prompt_len_inc_bos = 1  # Start with BOS length

        # 3-3. Load Audio Prompt (if provided)
        if audio_prompt_path is not None:
            audio_prompt, sr = torchaudio.load(audio_prompt_path, channels_first=True)  # C, T
            if sr != 44100:  # Resample to 44.1kHz
                audio_prompt = torchaudio.functional.resample(audio_prompt, sr, 44100)
            audio_prompt = audio_prompt.to(self.device).unsqueeze(0)  # 1, C, T
            audio_prompt = audio_to_codebook(self.dac_model, audio_prompt, data_config=self.config.data)
            generated_BxTxC = torch.cat([generated_BxTxC, audio_prompt.expand(2, -1, -1)], dim=1)

            prefill_len = generated_BxTxC.shape[1]
            prompt_len_inc_bos = prefill_len
            prefill_tgt_pos = torch.arange(prefill_len, device=self.device).unsqueeze(0).expand(2, -1)
            prefill_tgt_padding_mask = (generated_BxTxC != audio_pad_value).any(dim=2)

            prefill_self_attn_mask = self._create_attn_mask(
                prefill_tgt_padding_mask,
                prefill_tgt_padding_mask,
                is_causal=True,
            )
            prefill_cross_attn_mask = self._create_attn_mask(
                prefill_tgt_padding_mask,
                src_padding_mask_BxS,
                is_causal=False,
            )

            _ = self.model.decoder.forward(
                tgt_ids_BxTxC=generated_BxTxC,
                encoder_out=encoder_out,
                tgt_positions=prefill_tgt_pos,
                src_positions=src_positions_BxS,
                deterministic=True,
                self_attn_mask=prefill_self_attn_mask,
                cross_attn_mask=prefill_cross_attn_mask,
                self_attention_cache=decoder_self_attention_cache,
                cross_attention_cache=decoder_cross_attention_cache,
            )

            current_step = prefill_len - 1

        # 4. Autoregressive Generation Loop
        eos_detected_channel_0 = False
        eos_countdown = -1
        extra_steps_after_eos = 30
        # Make generated_BxTxC a fixed size tensor
        # Length is either 1 + max tokens or 1 + prompt len + max tokens
        generated_BxTxC = torch.cat(
            [
                generated_BxTxC,
                torch.full(
                    (2, max_tokens, num_channels),
                    fill_value=-1,
                    dtype=torch.long,
                    device=self.device,
                ),
            ],
            dim=1,
        )

        decode_step = self.model.decoder.decode_step
        if use_torch_compile:
            decode_step = torch.compile(
                self.model.decoder.decode_step,
                mode="default",
            )

        tgt_padding_mask = (
            (generated_BxTxC[:, -1, :].unsqueeze(1) != audio_pad_value).any(dim=2).to(self.device)
        )  # [B, 1]
        # Generated tokens are never PAD, so we use fixed mask
        decoder_cross_attn_mask = self._create_attn_mask(
            tgt_padding_mask,  # Query mask [B, 1]
            src_padding_mask_BxS,  # Key mask [B, S]
            is_causal=False,
        )  # [B, 1, 1, S]

        for step in range(current_step, current_step + max_tokens):
            tgt_ids_Bx1xC = generated_BxTxC[:, step, :].unsqueeze(1)
            tgt_pos_Bx1 = torch.full(
                (2, 1),
                fill_value=step,
                dtype=torch.long,
                device=self.device,
            )

            logits_Bx1xCxV, new_cache = decode_step(
                tgt_ids_Bx1xC=tgt_ids_Bx1xC,
                tgt_pos_Bx1=tgt_pos_Bx1,
                encoder_out=encoder_out,
                self_attn_mask=None,
                cross_attn_mask=decoder_cross_attn_mask,
                self_attention_cache=decoder_self_attention_cache,
                cross_attention_cache=decoder_cross_attention_cache,
            )

            for i, layer_cache in enumerate(decoder_self_attention_cache):
                layer_cache.update_cache(new_cache[i][0], new_cache[i][1])

            V = self.config.model.tgt_vocab_size
            logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :]  # B, C, V
            uncond_logits_CxV = logits_last_BxCxV[0, :, :]
            cond_logits_CxV = logits_last_BxCxV[1, :, :]

            cfg_logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV)

            logits_CxV = cfg_logits_CxV.reshape((-1, V))  # C, V
            logits_CxV[:, 1025:] = -torch.inf

            # Sample next token
            pred_C = _sample_next_token(
                logits_CxV.float(),
                temperature=temperature,
                top_p=top_p,
                use_cfg_filter=use_cfg_filter,
                cfg_filter_top_k=cfg_filter_top_k,
            )

            generation_step_index = step - current_step
            if audio_prompt_path is None:
                pred_C = torch.where(
                    generation_step_index >= delay_tensor,
                    pred_C,
                    audio_bos_value,
                )

            generated_BxTxC[:, step + 1, :] = pred_C.unsqueeze(0).expand(2, -1)

            if not eos_detected_channel_0 and pred_C[0] == audio_eos_value:
                eos_detected_channel_0 = True
                eos_countdown = extra_steps_after_eos

            if eos_countdown > 0:
                step_after_eos = max_delay_pattern - eos_countdown
                for i, d in enumerate(delay_pattern):
                    if step_after_eos == d:
                        generated_BxTxC[:, step + 1, i] = audio_eos_value
                    elif step_after_eos > d:
                        generated_BxTxC[:, step + 1, i] = audio_pad_value
                eos_countdown -= 1
                if eos_countdown == 0:
                    break

            generation_step_index = step - current_step + 1

        output_codes = generated_BxTxC[:, prompt_len_inc_bos : step + 1, :]

        generated_codes = output_codes[0]

        audio = codebook_to_audio(
            generated_codes.transpose(1, 0), self.dac_model, delay_pattern, B=1, T=max_tokens, C=num_channels
        )
        return audio.squeeze().cpu().numpy()