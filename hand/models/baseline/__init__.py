"""
Baseline HAND — the architecture behind every checkpoint in `models/`.

Verified against the released state dicts (docs/model_card.md):
    encoder  FCN_Encoder, 1.706 M parameters, 84 tensors
    decoder  GlobalHTADecoder, 8 layers, 5.30-5.33 M parameters
    total    7.003-7.034 M parameters
"""
from hand.models.baseline.fcn_encoder import FCN_Encoder, DepthSepConv2D, MixDropout  # noqa: F401
from hand.models.baseline.dan_decoder import GlobalHTADecoder  # noqa: F401
