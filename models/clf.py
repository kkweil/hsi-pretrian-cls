# sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from torch import nn
from timm.models.vision_transformer import Block
from models.PixelEmbed import PixelEmbed, PosCNN
from models.LinearComb import Linear_comb
from common.datautils import *


class Classfier(nn.Module):
    def __init__(self, name=None, in_chans=None, class_num=10,
                 encoder_embed_dim=512, encoder_depth=16, encoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.name = name
        # TODO: encoder
        self.patch_embed = PixelEmbed(in_channels=in_chans, embed_dim=encoder_embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = PosCNN(in_chans=encoder_embed_dim, embed_dim=encoder_embed_dim)

        self.blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(encoder_embed_dim)
        # --------------------------------------------------------------------------------------------------------------
        # TODO: clf partition
        self.linear_comb = Linear_comb(embed_dim=encoder_embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(encoder_embed_dim, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),

        )

        self.head = nn.Linear(1024, class_num)
        self.initialize_weights()

    def initialize_weights(self):

        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear, nn.Conv and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if isinstance(m, nn.Conv3d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encoder_forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.linear_comb(x)
        return x

    def forward(self, inputs):
        x = self.encoder_forward(inputs)
        # token = self.linear_comb(latent)

        # token = latent[:, 0, :]

        x = self.fc(x)
        if self.head is not None:
            x = self.head(x)
        return x


if __name__ == '__main__':
    pass
