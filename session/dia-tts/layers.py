from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import RMSNorm
from torch.utils.checkpoint import checkpoint
from config import DiaConfig


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _str_to_dtype(dtype_str: str) -> torch.dtype | None:
    # Allow None for default behavior
    if dtype_str is None or dtype_str.lower() == "none":
        return None
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")


class DenseGeneral(nn.Module):
    """
    PyTorch equivalent of flax.linen.DenseGeneral with shapes defined at init.

    Stores weights (`kernel`) in the same layout as Jax and uses torch.tensordot
    for the generalized matrix multiplication. Weight/bias shapes are calculated
    and parameters created during initialization based on config.
    `load_weights` validates shapes and copies data.

    Attributes:
        axis (Tuple[int, ...]): Input axis or axes to contract.
        in_shapes (Tuple[int, ...]): Sizes of the input dimensions specified by `axis`.
        out_features (Tuple[int, ...]): Shape of the output features (non-contracted dims).
        use_bias (bool): Whether to add a bias term.
        weight (nn.Parameter): The kernel parameter.
        bias (Optional[nn.Parameter]): The bias parameter (if use_bias=True).
    """

    def __init__(
        self,
        in_shapes: tuple[int, ...],
        out_features: tuple[int, ...],
        axis: tuple[int, ...] = (-1,),
        dtype: torch.dtype | None = None,
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis
        self.dtype = dtype
        self.kernel_shape = self.in_shapes + self.out_features

        factory_kwargs = {"device": device, "dtype": weight_dtype}
        self.weight = nn.Parameter(torch.empty(self.kernel_shape, **factory_kwargs))
        self.register_parameter("bias", None)

    def forward(self, inputs: Tensor) -> Tensor:
        norm_axis = _normalize_axes(self.axis, inputs.ndim)
        kernel_contract_axes = tuple(range(len(norm_axis)))

        output = torch.tensordot(
            inputs.float(),
            self.weight.float(),
            dims=(norm_axis, kernel_contract_axes),
        ).to(inputs.dtype)
        return output


def get_activation_fn(activation_string: str) -> nn.Module:  # Return Module instance
    """Maps activation string to PyTorch activation function module."""
    if activation_string == "gelu":
        return nn.GELU()
    elif activation_string == "relu":
        return nn.ReLU()
    elif activation_string == "silu" or activation_string == "swish":
        return nn.SiLU()
    elif activation_string == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation_string}")


class MlpBlock(nn.Module):
    """MLP block using DenseGeneral."""

    def __init__(
        self,
        config: DiaConfig,
        embed_dim: int,
        intermediate_dim: int,
        dropout_rate: float,
        activations: list[str] = ["silu", "linear"],
        use_pre_norm: bool = False,
    ):
        super().__init__()
        self.use_pre_norm = use_pre_norm
        num_activations = len(activations)
        compute_dtype = _str_to_dtype(config.training.dtype)
        weight_dtype = _str_to_dtype(config.model.weight_dtype)
        self.dtype = compute_dtype
        # Assume default device for now, could be passed in config

        if use_pre_norm:
            self.pre_norm = RMSNorm(
                embed_dim,
                eps=config.model.normalization_layer_epsilon,
                dtype=torch.float32,
            )

        self.wi_fused = DenseGeneral(
            in_shapes=(embed_dim,),
            out_features=(
                num_activations,
                intermediate_dim,
            ),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )

        self.activation_fn_0 = get_activation_fn(activations[0])  # silu
        self.activation_fn_1 = get_activation_fn(activations[1])  # linear

        self.dropout = nn.Dropout(dropout_rate)

        # Output layer using DenseGeneral
        self.wo = DenseGeneral(
            in_shapes=(intermediate_dim,),
            out_features=(embed_dim,),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )

    def forward(self, x: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Forward pass."""
        if self.use_pre_norm and hasattr(self, "pre_norm"):
            x = self.pre_norm(x)

        fused_x = self.wi_fused(x)

        gate_input = fused_x[..., 0, :]
        up_input = fused_x[..., 1, :]

        gate = self.activation_fn_0(gate_input)
        up = self.activation_fn_1(up_input)
        hidden = torch.mul(gate, up).to(self.dtype)

        if not deterministic:
            hidden = self.dropout(hidden)

        output = self.wo(hidden)
        return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in PyTorch."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.dtype = dtype

        half_embedding_dim = embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / embedding_dims
        self.register_buffer(
            "timescale",
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction,
            persistent=False,
        )

    def extra_repr(self) -> str:
        s = f"{self.timescale.shape}"
        return s

    def forward(self, inputs: torch.Tensor, position: torch.Tensor):
        """Applies RoPE."""
        position = position.unsqueeze(-1).unsqueeze(-1)
        timescale = self.timescale.to(inputs.device)
        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp).to(inputs.dtype)
        cos = torch.cos(sinusoid_inp).to(inputs.dtype)
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat((first_part, second_part), dim=-1)


class KVCache:
    def __init__(self, batch_size, num_heads, max_len, head_dim, device, k=None, v=None):
        self.k = torch.zeros((batch_size, num_heads, max_len, head_dim), device=device) if k is None else k
        self.v = torch.zeros((batch_size, num_heads, max_len, head_dim), device=device) if v is None else v
        self.current_idx = 0
        self.max_len = max_len

    def get_kv_for_attention(self, current_k, current_v):
        if self.current_idx == 0:
            return current_k, current_v
        else:
            past_k = self.k[:, :, : self.current_idx, :]
            past_v = self.v[:, :, : self.current_idx, :]
            attn_k = torch.cat((past_k, current_k), dim=2)
            attn_v = torch.cat((past_v, current_v), dim=2)
            return attn_k, attn_v

    def update_cache(self, k, v):
        assert self.current_idx < self.max_len
        self.k[:, :, self.current_idx : self.current_idx + 1, :] = k
        self.v[:, :, self.current_idx : self.current_idx + 1, :] = v
        self.current_idx += 1

    def prefill_kv(self, k, v):
        prefill_len = k.shape[2]
        assert prefill_len <= self.max_len
        self.k[:, :, :prefill_len, :] = k
        self.v[:, :, :prefill_len, :] = v
        self.current_idx = prefill_len


class Attention(nn.Module):
    """Attention using DenseGeneral."""

    def __init__(
        self,
        config: DiaConfig,
        q_embed_dim: int,
        kv_embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_rate: float,
        is_cross_attn: bool = False,
        out_embed_dim: int | None = None,
    ):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_cross_attn = is_cross_attn
        self.dropout_rate = dropout_rate
        compute_dtype = _str_to_dtype(config.training.dtype)
        weight_dtype = _str_to_dtype(config.model.weight_dtype)
        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        self.num_gqa_groups = num_query_heads // num_kv_heads

        # --- Projection Layers using DenseGeneral ---
        self.q_proj = DenseGeneral(
            in_shapes=(q_embed_dim,),
            out_features=(num_query_heads, head_dim),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )
        self.k_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )
        self.v_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )
        self.o_proj = DenseGeneral(
            in_shapes=(num_query_heads, head_dim),
            out_features=(self.output_dim,),
            axis=(-2, -1),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )

        # --- Rotary Embedding ---
        self.rotary_emb = RotaryEmbedding(
            embedding_dims=self.head_dim,
            min_timescale=config.model.rope_min_timescale,
            max_timescale=config.model.rope_max_timescale,
            dtype=compute_dtype,
        )

    def forward(
        self,
        Xq: torch.Tensor,  # (B, T, D) T = 1 in AR generation
        Xkv: torch.Tensor,  # (B, S, E) S = 1 in AR generation
        q_positions: torch.Tensor,  # (B, T)
        kv_positions: torch.Tensor | None = None,  # (B, S)
        deterministic: bool = True,
        attn_mask: torch.Tensor | None = None,  # None in Decoder Self Attention, Valid mask in Others
        cache: KVCache | None = None,  # None in Encoder, KVCache in Decoder
        prefill: bool = False,  # True only when prefilling KV Cache
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Performs attention calculation with optional KV caching.

        Args:
            Xq: Query tensor (B, T, D). T=1 during single-step decoding.
            Xkv: Key/Value source tensor (B, S, E). S=1 during single-step decoding for self-attn.
            q_positions: Positions for queries (B, T).
            kv_positions: Positions for keys/values (B, S). If None, uses q_positions.
            deterministic: If True, disable dropout.
            attn_mask: Attention mask.
            cache: KVCache.
            prefill: If True, use prefill mode.

        Returns:
            A tuple containing:
            - output: The attention output tensor (B, T, output_dim).
            - present_kv: The K/V state to be cached for the next step ((B, N, S_new, H), (B, N, S_new, H)). For self-attn, S_new = S_past + S. For cross-attn, S_new = S_kv.
        """
        if kv_positions is None:
            kv_positions = q_positions
        original_dtype = Xq.dtype

        Xq_BxTxNxH = self.q_proj(Xq)
        Xq_BxTxNxH = self.rotary_emb(Xq_BxTxNxH, position=q_positions)
        Xq_BxNxTxH = Xq_BxTxNxH.transpose(1, 2)

        # Input values into attention calculation
        attn_k: torch.Tensor | None = None
        attn_v: torch.Tensor | None = None
        new_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        # Decoder Cross Attention
        if self.is_cross_attn:
            # Directly use cache (no need to check index)
            attn_k, attn_v = cache.k, cache.v
            if attn_k.shape[1] != self.num_query_heads or attn_v.shape[1] != self.num_query_heads:
                raise ValueError(
                    f"Cross-attention cache head dimension ({attn_k.shape[1]}) "
                    f"does not match num_query_heads ({self.num_query_heads}). "
                    "Cache should be pre-repeated for GQA."
                )
        # Self Attention
        else:
            Xk_BxSxKxH = self.k_proj(Xkv)  # (B, S, K, H)
            Xv_BxSxKxH = self.v_proj(Xkv)  # (B, S, K, H)
            Xk_BxSxKxH = self.rotary_emb(Xk_BxSxKxH, position=kv_positions)  # (B, S, K, H)

            Xk_BxKxSxH = Xk_BxSxKxH.transpose(1, 2)  # (B, K, S, H)
            Xv_BxKxSxH = Xv_BxSxKxH.transpose(1, 2)  # (B, K, S, H)
            # S=1 for Decode Step

            if self.num_gqa_groups > 1:
                Xk_BxNxSxH = Xk_BxKxSxH.repeat_interleave(self.num_gqa_groups, dim=1)
                Xv_BxNxSxH = Xv_BxKxSxH.repeat_interleave(self.num_gqa_groups, dim=1)
            else:
                Xk_BxNxSxH = Xk_BxKxSxH
                Xv_BxNxSxH = Xv_BxKxSxH

            # Encoder Self Attention
            if cache is None:
                attn_k = Xk_BxNxSxH
                attn_v = Xv_BxNxSxH
            # Decoder Self Attention
            else:
                # In prefill mode, we fill in cache until prefill length
                if prefill:
                    attn_k, attn_v = Xk_BxNxSxH, Xv_BxNxSxH
                    cache.prefill_kv(attn_k, attn_v)
                # In decode step, we add current K/V to cache step by step
                else:
                    new_kv_cache = Xk_BxNxSxH, Xv_BxNxSxH
                    attn_k, attn_v = cache.get_kv_for_attention(Xk_BxNxSxH, Xv_BxNxSxH)

        attn_output = F.scaled_dot_product_attention(
            Xq_BxNxTxH,
            attn_k,
            attn_v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_rate if not deterministic else 0.0,
            scale=1.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, N, H)
        output = self.o_proj(attn_output)

        return output.to(original_dtype), new_kv_cache


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        embed_dim = enc_config.n_embd

        self.pre_sa_norm = RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.self_attention = Attention(
            config=config,
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            num_query_heads=enc_config.n_head,
            num_kv_heads=enc_config.n_head,
            head_dim=enc_config.head_dim,
            dropout_rate=model_config.dropout,
            is_cross_attn=False,
            out_embed_dim=embed_dim,
        )
        self.post_sa_norm = RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.mlp = MlpBlock(
            config=config,
            embed_dim=embed_dim,
            intermediate_dim=enc_config.n_hidden,
            activations=enc_config.mlp_activations,
            dropout_rate=model_config.dropout,
            use_pre_norm=enc_config.use_pre_norm,
        )
        self.dropout = nn.Dropout(model_config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_positions: torch.Tensor | None = None,
        deterministic: bool = True,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.pre_sa_norm(x)

        sa_out, _ = self.self_attention(
            Xq=x_norm,
            Xkv=x_norm,
            q_positions=src_positions,
            kv_positions=src_positions,
            deterministic=deterministic,
            attn_mask=attn_mask,
        )
        x = residual + sa_out

        residual = x
        x_norm = self.post_sa_norm(x)
        mlp_out = self.mlp(x_norm, deterministic=deterministic)
        x = residual + mlp_out

        if not deterministic:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """Transformer Encoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        compute_dtype = _str_to_dtype(config.training.dtype)

        self.embedding = nn.Embedding(
            model_config.src_vocab_size,
            enc_config.n_embd,
            dtype=compute_dtype,
        )
        self.dropout = nn.Dropout(model_config.dropout)
        self.layers = nn.ModuleList([EncoderLayer(config=config) for _ in range(enc_config.n_layer)])
        self.norm = RMSNorm(
            enc_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )

    def forward(
        self,
        x_ids: torch.Tensor,
        src_positions: torch.Tensor | None = None,
        deterministic: bool = True,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(x_ids)

        if not deterministic:
            x = self.dropout(x)

        for layer in self.layers:
            x = layer(
                x,
                src_positions=src_positions,
                deterministic=deterministic,
                attn_mask=attn_mask,
            )
        x = self.norm(x)
        if not deterministic:
            x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        enc_config = config.model.encoder
        dec_embed_dim = dec_config.n_embd
        enc_embed_dim = enc_config.n_embd

        # Norms
        self.pre_sa_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.pre_ca_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.pre_mlp_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )

        # Self-Attention (GQA) with Causal Masking
        self.self_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=dec_embed_dim,
            num_query_heads=dec_config.gqa_query_heads,
            num_kv_heads=dec_config.kv_heads,
            head_dim=dec_config.gqa_head_dim,
            dropout_rate=model_config.dropout,
            is_cross_attn=False,
            out_embed_dim=dec_embed_dim,
        )
        # Cross-Attention (MHA)
        self.cross_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=enc_embed_dim,  # Note kv_embed_dim
            num_query_heads=dec_config.cross_query_heads,
            num_kv_heads=dec_config.cross_query_heads,
            head_dim=dec_config.cross_head_dim,
            dropout_rate=model_config.dropout,
            is_cross_attn=True,
            out_embed_dim=dec_embed_dim,
        )
        # MLP
        self.mlp = MlpBlock(
            config=config,
            embed_dim=dec_embed_dim,
            intermediate_dim=dec_config.n_hidden,
            activations=dec_config.mlp_activations,
            dropout_rate=model_config.dropout,
            use_pre_norm=dec_config.use_pre_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_positions: torch.Tensor,
        src_positions: torch.Tensor | None,
        deterministic: bool,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
        self_attn_cache: KVCache,
        cross_attn_cache: KVCache,
        prefill: bool = False,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.pre_sa_norm(x)

        sa_out, new_kv_cache = self.self_attention(
            Xq=x_norm,  # (2, 1, D)
            Xkv=x_norm,  # (2, 1, D)
            q_positions=tgt_positions,  # (2, 1)
            kv_positions=tgt_positions,  # (2, 1)
            deterministic=deterministic,
            attn_mask=self_attn_mask,  # (2, 1, 1, S_max)
            cache=self_attn_cache,
            prefill=prefill,
        )

        x = residual + sa_out

        # 2. Cross-Attention
        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out, _ = self.cross_attention(
            Xq=x_norm,
            Xkv=encoder_out,
            q_positions=tgt_positions,
            kv_positions=src_positions,
            deterministic=deterministic,
            attn_mask=cross_attn_mask,
            cache=cross_attn_cache,
        )
        x = residual + ca_out

        # 3. MLP
        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm, deterministic=deterministic)
        x = residual + mlp_out

        return x, new_kv_cache


class Decoder(nn.Module):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        train_config = config.training
        data_config = config.data
        compute_dtype = _str_to_dtype(config.training.dtype)
        weight_dtype = _str_to_dtype(config.model.weight_dtype)
        self.num_channels = data_config.channels
        self.num_layers = dec_config.n_layer

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(model_config.tgt_vocab_size, dec_config.n_embd, dtype=compute_dtype)
                for _ in range(self.num_channels)
            ]
        )
        self.dropout = nn.Dropout(model_config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(config=config) for _ in range(self.num_layers)])
        self.norm = RMSNorm(
            dec_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )

        # Final Logits Projection using DenseGeneral
        self.logits_dense = DenseGeneral(
            in_shapes=(dec_config.n_embd,),
            out_features=(self.num_channels, model_config.tgt_vocab_size),
            axis=(-1,),
            dtype=(torch.float32 if train_config.logits_dot_in_fp32 else compute_dtype),
            weight_dtype=weight_dtype,
        )
        self.logits_in_fp32 = train_config.logits_dot_in_fp32
        self.use_gradient_checkpointing = False

    def precompute_cross_attention_kv(
        self,
        max_len: int,
        encoder_out: torch.Tensor,  # (B, S, E)
        src_positions: torch.Tensor | None,  # (B, S)
    ) -> list[KVCache]:
        """
        Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
        """
        per_layer_kv_cache: list[KVCache] = []

        for layer in self.layers:
            cross_attn_module = layer.cross_attention
            k_proj = cross_attn_module.k_proj(encoder_out)
            v_proj = cross_attn_module.v_proj(encoder_out)

            k_proj = cross_attn_module.rotary_emb(k_proj, position=src_positions)
            k = k_proj.transpose(1, 2)
            v = v_proj.transpose(1, 2)

            per_layer_kv_cache.append(
                KVCache(
                    k.shape[0],
                    cross_attn_module.num_kv_heads,
                    max_len,
                    cross_attn_module.head_dim,
                    k.device,
                    k=k,
                    v=v,
                )
            )

        return per_layer_kv_cache

    def decode_step(
        self,
        tgt_ids_Bx1xC: torch.Tensor,  # [B, 1, C]
        tgt_pos_Bx1: torch.Tensor,  # [B, 1]
        encoder_out: torch.Tensor,  # [B, S, E]
        self_attn_mask: Any,  # None
        cross_attn_mask: torch.Tensor,  # [B, 1, 1, S]
        self_attention_cache: list[KVCache],
        cross_attention_cache: list[KVCache],
    ) -> torch.Tensor:
        """
        Performs a single decoding step, managing KV caches layer by layer.

        Returns:
            A tuple containing:
            - logits_Bx1xCV: The final output logits for the current step (B, 1, C*V), cast to float32.
        """
        assert self_attn_mask is None, "Self-attention mask should be None, kept for pattern"

        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        new_cache = []

        for i, layer in enumerate(self.layers):
            self_cache = self_attention_cache[i]
            cross_cache = cross_attention_cache[i]
            x, new_kv_cache = layer(
                x,  # (2, 1, D)
                encoder_out,  # (2, S, E)
                src_positions=None,  # CA KV is already computed
                tgt_positions=tgt_pos_Bx1,  # (2, 1)
                deterministic=True,
                self_attn_mask=None,
                cross_attn_mask=cross_attn_mask,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
            )
            new_cache.append(new_kv_cache)

        x = self.norm(x)
        logits_Bx1xCxV = self.logits_dense(x)

        return logits_Bx1xCxV.to(torch.float32), new_cache

    def forward(
        self,
        tgt_ids_BxTxC: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_positions: torch.Tensor,
        src_positions: torch.Tensor,
        deterministic: bool,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
        self_attention_cache: list[KVCache],
        cross_attention_cache: list[KVCache],
    ) -> torch.Tensor:
        """
        Forward pass for the Decoder stack, managing KV caches.

        Args:
            tgt_ids_BxTxC: Target token IDs (B, T, C).
            encoder_out: Output from the encoder (B, S, E).
            tgt_positions: Positions for target sequence (B, T).
            src_positions: Positions for source sequence (B, S).
            deterministic: Disable dropout if True.
            self_attn_mask: Mask for self-attention.
            cross_attn_mask: Mask for cross-attention.
            past_key_values: List containing the self-attention KV cache for each layer
                             from the previous decoding step. `len(past_key_values)` should
                             equal `num_layers`.
            precomputed_cross_attn_kv: A single tuple containing the pre-computed K/V cache
                                      derived from `encoder_out`. This is passed identically
                                      to all layers.

        Returns:
            A tuple containing:
            - logits: The final output logits (B, T, C * V), cast to float32.
            - present_key_values: A list containing the updated self-attention KV cache
                                 for each layer for the *current* decoding step.
        """
        _, _, num_channels_in = tgt_ids_BxTxC.shape
        assert num_channels_in == self.num_channels, "Input channels mismatch"

        # Embeddings
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        if not deterministic:
            x = self.dropout(x)

        """
        def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_positions: torch.Tensor,
        src_positions: torch.Tensor | None,
        deterministic: bool,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
        self_attn_cache: KVCache,
        cross_attn_cache: KVCache,
        prefill: bool = False
        """

        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    
                    return custom_forward
                
                x, _ = checkpoint(
                    create_custom_forward(layer),
                    x,
                    encoder_out,
                    tgt_positions,
                    src_positions,
                    deterministic,
                    self_attn_mask,
                    cross_attn_mask,
                    self_attention_cache[i],
                    cross_attention_cache[i],
                    True,
                    use_reentrant=False,
                )
            else:
                x, _ = layer(
                    x,
                    encoder_out,
                    tgt_positions=tgt_positions,
                    src_positions=src_positions,
                    deterministic=deterministic,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    self_attn_cache=self_attention_cache[i],
                    cross_attn_cache=cross_attention_cache[i],
                    prefill=True,
                )

        # Final Norm
        x = self.norm(x)
        # logits_BxTxCxV = self.logits_dense(x)

        return x


class DiaModel(nn.Module):
    """PyTorch Dia Model using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        #self._init_weights()

    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, torch.nn.LayerNorm) or isinstance(module, torch.nn.modules.normalization.RMSNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        src_BxS: torch.Tensor,
        tgt_BxTxC: torch.Tensor,
        src_positions: torch.Tensor | None = None,
        tgt_positions: torch.Tensor | None = None,
        enc_self_attn_mask: torch.Tensor | None = None,
        dec_self_attn_mask: torch.Tensor | None = None,
        dec_cross_attn_mask: torch.Tensor | None = None,
        enable_dropout: bool = True,
    ):
        deterministic = not enable_dropout

        # --- Encoder Pass ---
        encoder_out = self.encoder(
            x_ids=src_BxS,
            src_positions=src_positions,
            deterministic=deterministic,
            attn_mask=enc_self_attn_mask,
        )

        B, T, C = tgt_BxTxC.shape  # Batch size, target sequence length, channels
        device = tgt_BxTxC.device

        self_attention_cache = [
            KVCache(
                num_heads=self.decoder.layers[i].self_attention.num_query_heads,  # ✅ FIXED: use query heads!
                max_len=T,
                head_dim=self.decoder.layers[i].self_attention.head_dim,
                device=device,
            )
            for i in range(self.decoder.num_layers)
        ]

        cross_attention_cache = self.decoder.precompute_cross_attention_kv(
            max_len=encoder_out.shape[1],
            encoder_out=encoder_out,
            src_positions=src_positions,
        )

        # --- Decoder Pass ---
        logits = self.decoder(
            tgt_ids_BxTxC=tgt_BxTxC,
            encoder_out=encoder_out,
            tgt_positions=tgt_positions,
            src_positions=src_positions,
            deterministic=deterministic,
            self_attn_mask=dec_self_attn_mask,
            cross_attn_mask=dec_cross_attn_mask,
            self_attention_cache=self_attention_cache,
            cross_attention_cache=cross_attention_cache
        )

        return logits