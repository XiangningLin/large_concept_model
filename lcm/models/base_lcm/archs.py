# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from lcm.models.base_lcm.builder import (
    BaseLCModelConfig,
    LCMFrontendConfig,
    ProjectionConfig,
    TransformerConfig,
    lcm_arch,
)


# Every model must register a toy_{model_family}
@lcm_arch("toy_base_lcm")
def toy_base_lcm() -> BaseLCModelConfig:
    return BaseLCModelConfig(
        lcm=TransformerConfig(num_layers=2),
    )


@lcm_arch("base_lcm_60M")
def base_lcm_60M() -> BaseLCModelConfig:
    """Base 60M model
    Approximate Parameter Size: ~62M
    Configuration: model_dim=640, num_layers=12, num_heads=8
    """
    model_dim: int = 640
    num_attn_heads: int = 8
    return BaseLCModelConfig(
        max_seq_len=4096,
        model_dim=model_dim,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,  # 2560
            num_attn_heads=num_attn_heads,
            num_layers=12,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )


@lcm_arch("base_lcm_780M")
def base_lcm_780M() -> BaseLCModelConfig:
    """Base 780M model
    Parameter Size: ~780M (approximately half of 1.6B)
    """
    model_dim: int = 2048
    num_attn_heads: int = 16
    return BaseLCModelConfig(
        max_seq_len=4096,
        model_dim=model_dim,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,
            num_attn_heads=num_attn_heads,
            num_layers=16,  # Reduced from 32 to approximately halve parameters
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )


@lcm_arch("base_lcm_1_6B")
def base_lcm_1_6B() -> BaseLCModelConfig:
    """Base 1.6B model
    Parameter Size: 1,647,635,456
    """
    model_dim: int = 2048
    num_attn_heads: int = 16
    return BaseLCModelConfig(
        max_seq_len=4096,
        model_dim=model_dim,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,
            num_attn_heads=num_attn_heads,
            num_layers=32,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )


@lcm_arch("base_lcm_130M")
def base_lcm_130M() -> BaseLCModelConfig:
    """Base 130M model
    Approximate Parameter Size: ~127M
    Configuration: model_dim=768, num_layers=18, num_heads=12
    """
    model_dim: int = 768
    num_attn_heads: int = 12
    return BaseLCModelConfig(
        max_seq_len=4096,
        model_dim=model_dim,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,  # 3072
            num_attn_heads=num_attn_heads,
            num_layers=18,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )


@lcm_arch("base_lcm_250M")
def base_lcm_250M() -> BaseLCModelConfig:
    """Base 250M model
    Approximate Parameter Size: ~246M
    Configuration: model_dim=1024, num_layers=19, num_heads=16
    """
    model_dim: int = 1024
    num_attn_heads: int = 16
    return BaseLCModelConfig(
        max_seq_len=4096,
        model_dim=model_dim,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,  # 4096
            num_attn_heads=num_attn_heads,
            num_layers=19,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )


@lcm_arch("base_lcm_370M")
def base_lcm_370M() -> BaseLCModelConfig:
    """Base 370M model
    Approximate Parameter Size: ~376M
    Configuration: model_dim=1280, num_layers=19, num_heads=16
    """
    model_dim: int = 1280
    num_attn_heads: int = 16
    return BaseLCModelConfig(
        max_seq_len=4096,
        model_dim=model_dim,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,  # 5120
            num_attn_heads=num_attn_heads,
            num_layers=19,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )


@lcm_arch("base_lcm_780M_v2")
def base_lcm_780M_v2() -> BaseLCModelConfig:
    """Base 780M model (variant 2)
    Approximate Parameter Size: ~793M
    Configuration: model_dim=1536, num_layers=28, num_heads=16
    Alternative configuration with different depth-width tradeoff
    """
    model_dim: int = 1536
    num_attn_heads: int = 16
    return BaseLCModelConfig(
        max_seq_len=4096,
        model_dim=model_dim,
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=TransformerConfig(
            final_dropout_p=0.0,
            attention_dropout_p=0.0,
            dropout_p=0.1,
            mha_output_proj_bias=True,
            ffn_inner_dim=model_dim * 4,  # 6144
            num_attn_heads=num_attn_heads,
            num_layers=28,
            pos_embedding_style="rope",
            use_swiglu=True,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )
