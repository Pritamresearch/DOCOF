import torch
import torch.nn as nn

from models.temporal_encoder_v2 import TemporalEncoderV2
from models.intent_encoder_v2 import IntentEncoderV2
from models.forecast_decoder_v2 import ForecastDecoderV2
from models.object_pooling_v2 import ObjectRegionPooling


class IntentConditionedModelV2(nn.Module):

    def __init__(self, d_model=128, num_intent_classes=5):
        super().__init__()

        self.temporal_encoder = TemporalEncoderV2(
            in_channels=1,
            d_model=d_model
        )

        self.object_pool = ObjectRegionPooling()

        self.intent_encoder = IntentEncoderV2(
            d_model=d_model,
            num_classes=num_intent_classes
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True
        )

        self.decoder = ForecastDecoderV2(d_model=d_model, future_frames=3)

    def pad_object_tokens(self, object_tokens_list):
        """
        Pads variable-length object token lists to batch tensor.
        Returns:
            padded_tokens: [B, N_max, D]
            valid_mask: [B, N_max] (1 = valid, 0 = padded)
        """
        B = len(object_tokens_list)
        D = object_tokens_list[0].shape[1]

        max_objects = max([t.shape[0] for t in object_tokens_list])

        padded_tokens = []
        valid_mask = []

        for tokens in object_tokens_list:

            n = tokens.shape[0]
            pad = max_objects - n

            if pad > 0:
                padding = torch.zeros(pad, D, device=tokens.device)
                tokens = torch.cat([tokens, padding], dim=0)

            padded_tokens.append(tokens)

            mask = torch.zeros(max_objects, device=tokens.device)
            mask[:n] = 1
            valid_mask.append(mask)

        padded_tokens = torch.stack(padded_tokens)  # [B,N_max,D]
        valid_mask = torch.stack(valid_mask)        # [B,N_max]

        return padded_tokens, valid_mask

    def forward(self, bev_sequence, boxes):
        """
        bev_sequence: [B, T, C, H, W]
        boxes: list of length B,
               each element tensor [N_i, 4] in grid coords
        """

        # -----------------------------------
        # 1️⃣ Temporal Encoding
        # -----------------------------------

        bev_tokens, H, W = self.temporal_encoder(bev_sequence)
        B, HW, D = bev_tokens.shape

        # Convert tokens back to feature map
        bev_feature_map = bev_tokens.permute(0, 2, 1).view(B, D, H, W)

        # -----------------------------------
        # 2️⃣ Object Region Pooling
        # -----------------------------------

        object_tokens_list = self.object_pool(bev_feature_map, boxes)

        # If no objects in entire batch
        if all([t.shape[0] == 0 for t in object_tokens_list]):
            # Skip conditioning entirely
            future_bev = self.decoder(bev_tokens, H, W)
            dummy_logits = torch.zeros(B, 1, 5, device=bev_tokens.device)
            return future_bev, dummy_logits, None

        # Pad object tokens
        object_tokens, valid_mask = self.pad_object_tokens(object_tokens_list)

        # -----------------------------------
        # 3️⃣ Intent Prediction
        # -----------------------------------

        intent_logits, intent_features = self.intent_encoder(object_tokens)
        # intent_logits: [B, N_max, C]

        # -----------------------------------
        # 4️⃣ Cross-Attention Conditioning
        # -----------------------------------

        conditioned_tokens, _ = self.cross_attn(
            query=bev_tokens,          # [B, HW, D]
            key=intent_features,       # [B, N_max, D]
            value=intent_features
        )

        # -----------------------------------
        # 5️⃣ Forecast Decoder
        # -----------------------------------

        future_bev = self.decoder(conditioned_tokens, H, W)

        return future_bev, intent_logits, valid_mask