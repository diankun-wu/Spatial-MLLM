from .mlp_add_connector import MLPAddConnector

def get_connector(config):
    """
    Factory function to get the connector class based on the type.
    Args:
        connector_type (str): The type of connector to retrieve.
    Returns:
        Connector class corresponding to the specified type.
    """
    if config.connector_config["connector_type"] == "mlp_add":
        return MLPAddConnector(
            vggt_dim=config.spatial_config.embed_dim,
            language_dim=config.hidden_size,
            spatial_embeds_layer_idx=config.connector_config["spatial_embeds_layer_idx"],
            visual_temporal_merge_size=config.vision_config.temporal_patch_size,
            visual_spatial_merge_size=config.vision_config.spatial_merge_size,
        )
    # elif config.connector_config["connector_type"] == "mlp_cat":
    #     return MLPCatConnector(
    #         vggt_dim=config.spatial_config.embed_dim,
    #         language_dim=config.hidden_size,
    #         spatial_embeds_layer_idx=config.connector_config["spatial_embeds_layer_idx"],
    #         visual_temporal_merge_size=config.vision_config.temporal_patch_size,
    #         visual_spatial_merge_size=config.vision_config.spatial_merge_size,
    #     )
    # elif config.connector_config["connector_type"] == "cross_attn":
    #     return SpatialMLLMCrossAttnConnector(
    #         clip_dim=config.vision_config.out_hidden_size,
    #         vggt_dim=config.spatial_config.embed_dim,
    #         language_dim=config.hidden_size,
    #         spatial_embeds_layer_idx=config.connector_config["spatial_embeds_layer_idx"],
    #         num_heads=config.connector_config.get("num_heads", 8),
    #         attention_dropout=config.connector_config.get("attention_dropout", 0.0),
    #         mlp_ratio=config.connector_config.get("mlp_ratio", 4),
    #         hidden_act=config.connector_config.get("hidden_act", "gelu"),
    #         bias=config.connector_config.get("bias", False),
    #     )