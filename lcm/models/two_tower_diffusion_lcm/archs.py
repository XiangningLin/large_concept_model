# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from lcm.models.two_tower_diffusion_lcm.builder import (
    DenoiserConfig,
    EncoderFrontendConfig,
    TransformerConfig,
    TwoTowerDiffusionLCModelConfig,
    lcm_arch,
)
from lcm.nn.projection import ProjectionConfig
from lcm.nn.schedulers import DDIMSchedulerConfig


@lcm_arch("toy_two_tower_diffusion_lcm")
def toy_lcm() -> TwoTowerDiffusionLCModelConfig:
    return TwoTowerDiffusionLCModelConfig(
        context_encoder=TransformerConfig(num_layers=2),
        denoiser=DenoiserConfig(num_layers=2),
        # TODO change normalizer name to align with the normalizer instructions
        sonar_normalizer_name="dummy_sonar_normalizer",
    )


@lcm_arch("two_tower_diffusion_lcm_60M")
def two_tower_diffusion_lcm_60M() -> TwoTowerDiffusionLCModelConfig:
    """1-layer encoder / 8-layer denoiser / model dim 512
    Parameter Size: ~60M"""
    model_dim: int = 512
    num_attn_heads: int = 8
    return TwoTowerDiffusionLCModelConfig(
        model_dim=model_dim,
        max_seq_len=4096,
        frontend=EncoderFrontendConfig(),
        context_encoder=TransformerConfig(
            num_layers=1,
            ffn_inner_dim=4 * model_dim,
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pos_embedding_style="rope",
        ),
        denoiser=DenoiserConfig(
            num_layers=8,
            timestep_embed_dim=model_dim,
            ffn_inner_dim=4 * model_dim,
            pos_embedding_style="none",
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pre_denoiser=ProjectionConfig(),
            post_denoiser=ProjectionConfig(),
        ),
        sonar_normalizer_name="dummy_sonar_normalizer",
        trained_with_cf_guidance=True,
        noise_scheduler=DDIMSchedulerConfig(num_diffusion_train_steps=100),
    )


@lcm_arch("two_tower_diffusion_lcm_130M")
def two_tower_diffusion_lcm_130M() -> TwoTowerDiffusionLCModelConfig:
    """1-layer encoder / 8-layer denoiser / model dim 768
    Parameter Size: ~128M"""
    model_dim: int = 768
    num_attn_heads: int = 12
    return TwoTowerDiffusionLCModelConfig(
        model_dim=model_dim,
        max_seq_len=4096,
        frontend=EncoderFrontendConfig(),
        context_encoder=TransformerConfig(
            num_layers=1,
            ffn_inner_dim=4 * model_dim,
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pos_embedding_style="rope",
        ),
        denoiser=DenoiserConfig(
            num_layers=8,
            timestep_embed_dim=model_dim,
            ffn_inner_dim=4 * model_dim,
            pos_embedding_style="none",
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pre_denoiser=ProjectionConfig(),
            post_denoiser=ProjectionConfig(),
        ),
        sonar_normalizer_name="dummy_sonar_normalizer",
        trained_with_cf_guidance=True,
        noise_scheduler=DDIMSchedulerConfig(num_diffusion_train_steps=100),
    )


@lcm_arch("two_tower_diffusion_lcm_250M")
def two_tower_diffusion_lcm_250M() -> TwoTowerDiffusionLCModelConfig:
    """2-layer encoder / 8-layer denoiser / model dim 1024
    Parameter Size: ~242M"""
    model_dim: int = 1024
    num_attn_heads: int = 16
    return TwoTowerDiffusionLCModelConfig(
        model_dim=model_dim,
        max_seq_len=4096,
        frontend=EncoderFrontendConfig(),
        context_encoder=TransformerConfig(
            num_layers=2,
            ffn_inner_dim=4 * model_dim,
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pos_embedding_style="rope",
        ),
        denoiser=DenoiserConfig(
            num_layers=8,
            timestep_embed_dim=model_dim,
            ffn_inner_dim=4 * model_dim,
            pos_embedding_style="none",
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pre_denoiser=ProjectionConfig(),
            post_denoiser=ProjectionConfig(),
        ),
        sonar_normalizer_name="dummy_sonar_normalizer",
        trained_with_cf_guidance=True,
        noise_scheduler=DDIMSchedulerConfig(num_diffusion_train_steps=100),
    )


@lcm_arch("two_tower_diffusion_lcm_370M")
def two_tower_diffusion_lcm_370M() -> TwoTowerDiffusionLCModelConfig:
    """3-layer encoder / 12-layer denoiser / model dim 1024
    Parameter Size: ~361M"""
    model_dim: int = 1024
    num_attn_heads: int = 16
    return TwoTowerDiffusionLCModelConfig(
        model_dim=model_dim,
        max_seq_len=4096,
        frontend=EncoderFrontendConfig(),
        context_encoder=TransformerConfig(
            num_layers=3,
            ffn_inner_dim=4 * model_dim,
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pos_embedding_style="rope",
        ),
        denoiser=DenoiserConfig(
            num_layers=12,
            timestep_embed_dim=model_dim,
            ffn_inner_dim=4 * model_dim,
            pos_embedding_style="none",
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pre_denoiser=ProjectionConfig(),
            post_denoiser=ProjectionConfig(),
        ),
        sonar_normalizer_name="dummy_sonar_normalizer",
        trained_with_cf_guidance=True,
        noise_scheduler=DDIMSchedulerConfig(num_diffusion_train_steps=100),
    )


@lcm_arch("two_tower_diffusion_lcm_780M")
def two_tower_diffusion_lcm_780M() -> TwoTowerDiffusionLCModelConfig:
    """2-layer encoder / 12-layer denoiser / model dim 1536
    Parameter Size: ~772M"""
    model_dim: int = 1536
    num_attn_heads: int = 16
    return TwoTowerDiffusionLCModelConfig(
        model_dim=model_dim,
        max_seq_len=4096,
        frontend=EncoderFrontendConfig(),
        context_encoder=TransformerConfig(
            num_layers=2,
            ffn_inner_dim=4 * model_dim,
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pos_embedding_style="rope",
        ),
        denoiser=DenoiserConfig(
            num_layers=12,
            timestep_embed_dim=model_dim,
            ffn_inner_dim=4 * model_dim,
            pos_embedding_style="none",
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pre_denoiser=ProjectionConfig(),
            post_denoiser=ProjectionConfig(),
        ),
        sonar_normalizer_name="dummy_sonar_normalizer",
        trained_with_cf_guidance=True,
        noise_scheduler=DDIMSchedulerConfig(num_diffusion_train_steps=100),
    )


@lcm_arch("two_tower_diffusion_lcm_1_6B")
def two_tower_diffusion_lcm_1_6B() -> TwoTowerDiffusionLCModelConfig:
    """5-layer encoder / 13-layer denoiser / model dim 2048
    Parameter Size: 1,635,101,696"""
    model_dim: int = 2048
    num_attn_heads: int = 16
    return TwoTowerDiffusionLCModelConfig(
        model_dim=model_dim,
        max_seq_len=4096,
        frontend=EncoderFrontendConfig(),
        context_encoder=TransformerConfig(
            num_layers=5,
            ffn_inner_dim=4 * model_dim,
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pos_embedding_style="rope",
        ),
        denoiser=DenoiserConfig(
            num_layers=13,
            timestep_embed_dim=model_dim,
            ffn_inner_dim=4 * model_dim,
            pos_embedding_style="none",
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pre_denoiser=ProjectionConfig(),
            post_denoiser=ProjectionConfig(),
        ),
        sonar_normalizer_name="dummy_sonar_normalizer",
        trained_with_cf_guidance=True,
        noise_scheduler=DDIMSchedulerConfig(num_diffusion_train_steps=100),
    )


@lcm_arch("two_tower_diffusion_lcm_7B")
def two_tower_diffusion_lcm_7B() -> TwoTowerDiffusionLCModelConfig:
    # 5-layer encoder / 14-layer denoiser / model dim 4096
    # Parameter Size: 6,930,781,696
    model_dim: int = 4096
    num_attn_heads: int = 32
    return TwoTowerDiffusionLCModelConfig(
        model_dim=model_dim,
        max_seq_len=4096,
        frontend=EncoderFrontendConfig(),
        context_encoder=TransformerConfig(
            num_layers=5,
            ffn_inner_dim=4 * model_dim,
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pos_embedding_style="rope",
        ),
        denoiser=DenoiserConfig(
            num_layers=14,
            timestep_embed_dim=model_dim,
            ffn_inner_dim=4 * model_dim,
            pos_embedding_style="none",
            num_attn_heads=num_attn_heads,
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            use_swiglu=True,
            layer_normalization_style="rms",
            pre_denoiser=ProjectionConfig(),
            post_denoiser=ProjectionConfig(),
        ),
        sonar_normalizer_name="dummy_sonar_normalizer",
        trained_with_cf_guidance=True,
        noise_scheduler=DDIMSchedulerConfig(num_diffusion_train_steps=100),
    )
