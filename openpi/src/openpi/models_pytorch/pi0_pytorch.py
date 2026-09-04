import dataclasses
import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.lora_pytorch as _lora_pytorch
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        if config.paligemma_depth is not None:
            paligemma_config = dataclasses.replace(paligemma_config, depth=config.paligemma_depth)
        if config.action_expert_depth is not None:
            action_expert_config = dataclasses.replace(action_expert_config, depth=config.action_expert_depth)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
            vision_depth=config.vision_depth,
        )
        paligemma_lora_layers = _lora_pytorch.apply_gemma_lora(
            self.paligemma_with_expert.paligemma.language_model,
            paligemma_config,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )
        action_lora_layers = _lora_pytorch.apply_gemma_lora(
            self.paligemma_with_expert.gemma_expert.model,
            action_expert_config,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )
        vision_lora_layers = _lora_pytorch.apply_vision_lora(
            self.paligemma_with_expert.paligemma.vision_tower,
            rank=config.vision_lora_rank,
            alpha=config.vision_lora_alpha,
        )
        if paligemma_lora_layers or action_lora_layers:
            logging.info(
                "Applied PyTorch LoRA to PI0Pytorch: paligemma=%d linear layers, action_expert=%d linear layers",
                paligemma_lora_layers,
                action_lora_layers,
            )
        if vision_lora_layers:
            logging.info("Applied PyTorch Vision-LoRA to PI0Pytorch: vision=%d linear layers", vision_lora_layers)

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = (
            "transformers_replace is not installed correctly. Please install it with "
            "`uv pip install transformers==4.53.2` and "
            "`cp -r ./src/openpi/models_pytorch/transformers_replace/* "
            ".venv/lib/python3.11/site-packages/transformers/`."
        )
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def configure_trainable_parameters(self):
        """Freeze base VLM/expert weights for PyTorch LoRA finetuning configs."""
        has_lora = (
            "lora" in self.config.paligemma_variant
            or "lora" in self.config.action_expert_variant
            or self.config.vision_lora_rank is not None
        )
        if not has_lora:
            _lora_pytorch.log_trainable_parameters(self, prefix="PI0Pytorch full-finetune ")
            return

        trainable_substrings = (
            "lora_",
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp",
            "time_mlp",
        )
        _lora_pytorch.set_requires_grad_by_name(self, trainable_substrings=trainable_substrings)
        if self.config.train_vision_encoder:
            for name, param in self.named_parameters():
                if "vision_tower" in name or "vision_model" in name:
                    param.requires_grad = True
        _lora_pytorch.log_trainable_parameters(self, prefix="PI0Pytorch LoRA ")

    def load_pretrained_weights(self, weight_path: str, *, strict: bool = False):
        return _lora_pytorch.load_state_dict_with_lora_remap(self, weight_path, strict=strict)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            state = state.to(dtype=self.state_proj.weight.dtype)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions.to(dtype=self.action_in_proj.weight.dtype))

        if not self.pi05:
            time_emb = time_emb.to(dtype=action_emb.dtype)[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)
            action_time_emb = action_time_emb.to(dtype=self.action_time_mlp_in.weight.dtype)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = time_emb.to(dtype=self.time_mlp_in.weight.dtype)
            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def predict_velocity(self, observation, actions, noise=None, time=None) -> Tensor:
        """Predict the flow-matching velocity for a shared noise/time sample."""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        return self._apply_checkpoint(action_out_proj_func, suffix_out)

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        u_t = noise - actions
        v_t = self.predict_velocity(observation, actions, noise=noise, time=time)

        return F.mse_loss(u_t, v_t, reduction="none")

    def _make_rtc_weights(
        self,
        n: int,
        inference_delay: int,
        schedule: str,
        device,
        dtype,
    ) -> Tensor:
        if n <= 0:
            return torch.empty((0,), dtype=dtype, device=device)

        i = torch.arange(n, dtype=torch.float32, device=device)
        delay = max(0, min(int(inference_delay), n))
        if schedule == "ones":
            weights = torch.ones((n,), dtype=torch.float32, device=device)
        elif schedule == "zeros":
            weights = (i < delay).to(torch.float32)
        else:
            if delay >= n:
                weights = torch.ones((n,), dtype=torch.float32, device=device)
            else:
                span = max(n - delay, 1)
                remaining = torch.clamp((n - i) / float(span + 1), min=0.0, max=1.0)
                if schedule == "linear":
                    decay = remaining
                else:
                    decay = (torch.exp(remaining) - 1.0) / (math.e - 1.0)
                weights = torch.where(i < delay, torch.ones_like(decay), decay)
        return weights.to(dtype=dtype)

    def _prepare_rtc_guidance(
        self,
        rtc_prev_actions,
        *,
        rtc_inference_delay: int,
        rtc_execution_horizon: int,
        rtc_schedule: str,
        rtc_guidance_dims: int | None,
        device,
        dtype,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if rtc_prev_actions is None or rtc_execution_horizon <= 0:
            return None, None, None
        if not torch.is_tensor(rtc_prev_actions):
            rtc_prev_actions = torch.as_tensor(rtc_prev_actions, device=device)
        rtc_prev_actions = rtc_prev_actions.to(device=device, dtype=dtype)
        if rtc_prev_actions.ndim == 2:
            rtc_prev_actions = rtc_prev_actions.unsqueeze(0)
        if rtc_prev_actions.ndim != 3 or rtc_prev_actions.shape[1] == 0:
            return None, None, None

        execution_horizon = max(0, min(int(rtc_execution_horizon), self.config.action_horizon))
        overlap_horizon = self.config.action_horizon - execution_horizon
        n = min(overlap_horizon, rtc_prev_actions.shape[1], self.config.action_horizon)
        if n <= 0:
            return None, None, None
        target = rtc_prev_actions[:, :n, : self.config.action_dim].detach()
        weights = self._make_rtc_weights(
            n,
            rtc_inference_delay,
            rtc_schedule,
            device,
            dtype,
        ).view(1, n, 1)
        dims = self.config.action_dim if rtc_guidance_dims is None else max(0, min(int(rtc_guidance_dims), self.config.action_dim))
        dim_mask = torch.zeros((1, 1, self.config.action_dim), dtype=dtype, device=device)
        if dims > 0:
            dim_mask[..., :dims] = 1.0
        return target, weights, dim_mask

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        rtc_prev_actions=None,
        rtc_inference_delay: int = 0,
        rtc_execution_horizon: int = 0,
        rtc_max_guidance_weight: float = 5.0,
        rtc_schedule: str = "exp",
        rtc_guidance_dims: int | None = None,
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        rtc_target, rtc_weights, rtc_dim_mask = self._prepare_rtc_guidance(
            rtc_prev_actions,
            rtc_inference_delay=rtc_inference_delay,
            rtc_execution_horizon=rtc_execution_horizon,
            rtc_schedule=rtc_schedule,
            rtc_guidance_dims=rtc_guidance_dims,
            device=device,
            dtype=noise.dtype,
        )
        use_rtc = (
            rtc_target is not None
            and rtc_weights is not None
            and rtc_dim_mask is not None
            and float(rtc_max_guidance_weight) > 0.0
        )

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            if use_rtc:
                with torch.enable_grad():
                    x_t = x_t.detach().requires_grad_(True)
                    v_t = self.denoise_step(
                        state,
                        prefix_pad_masks,
                        past_key_values,
                        x_t,
                        expanded_time,
                    )
                    n = rtc_target.shape[1]
                    clean_action_estimate = x_t - time.to(dtype=x_t.dtype) * v_t
                    overlap = clean_action_estimate[:, :n, :]
                    cotangent = ((rtc_target - overlap) * rtc_weights * rtc_dim_mask).detach()
                    rtc_grad = torch.autograd.grad(
                        (overlap * cotangent).sum(),
                        x_t,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    r_t = time * time / (time * time + (1.0 - time) * (1.0 - time))
                    guidance_scale = torch.clamp(
                        time / ((1.0 - time) * r_t * r_t + 1e-6),
                        max=float(rtc_max_guidance_weight),
                    ).to(dtype=v_t.dtype)
                    v_t = v_t - guidance_scale * rtc_grad
            else:
                v_t = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = (x_t + dt * v_t).detach()
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
