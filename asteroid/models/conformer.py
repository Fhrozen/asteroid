import logging
import numpy as np

from ..filterbanks import make_enc_dec

from .base_models import BaseEncoderMaskerDecoder
from .conformer_utils.encoder import Encoder


class Conformer(BaseEncoderMaskerDecoder):
    """Conformer
    """

    def __init__(
        self,
        n_src,
        out_chan=None,
        n_blocks=8,
        dropout=0,
        adim=512,
        aheads=2,
        eunits=2048,
        chunk_size=512,
        hop_size=128,
        cnn_module_kernel=31,
        in_chan=None,
        fb_name="free",
        kernel_size=16,
        n_filters=512,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        # Update in_chan
        masker = Encoder(
            idim=n_feats,
            n_src=n_src,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=n_blocks,
            input_layer="linear",
            chunk_size=chunk_size,
            hop_size=hop_size,
            dropout_rate=dropout,
            positional_dropout_rate=dropout,
            attention_dropout_rate=dropout,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            activation_type="swish",
            macaron_style=False,
            use_cnn_module=True,
            cnn_module_kernel=cnn_module_kernel,
        )
        n_params = 0
        for p in list(masker.parameters()):
            n_params += np.prod(list(p.size()))
        logging.info(n_params)
        logging.info(masker)
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)
