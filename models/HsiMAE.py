import torch
from torch import nn
from timm.models.vision_transformer import Block, Mlp
from models.PixelEmbed import PixelEmbed, PosCNN
import numpy as np


# from common.LinearComb import linear_comb


class HsiPatchMaskedAutoEncoder(nn.Module):
    def __init__(self, name=None, in_chans=None,
                 encoder_embed_dim=512, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.name = name

        # self.max_len = max_size ** 2
        # TODO: encoder
        self.patch_embed = PixelEmbed(in_channels=in_chans, embed_dim=encoder_embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = PosCNN(in_chans=encoder_embed_dim, embed_dim=encoder_embed_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len+1, encoder_embed_dim),
        #                               requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(encoder_embed_dim)
        # --------------------------------------------------------------------------------------------------------------
        # TODO: modify decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = PosCNN(in_chans=decoder_embed_dim, embed_dim=decoder_embed_dim)

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.max_len+1, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans, bias=True)
        # --------------------------------------------------------------------------------------------------------------
        mlp_hidden_dim = int(in_chans * mlp_ratio)
        self.mlp1 = Mlp(in_features=in_chans, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.mlp2 = Mlp(in_features=in_chans, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.max_len ** .5),
        #                                     cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        #
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.max_len ** .5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.conv2d_1.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # w = self.patch_embed.conv3d.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # w = self.patch_embed.conv2d_2.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def encoder_forward(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def decoder_forward(self, x, ids_restore):
        x = self.decoder_embed(x)
        '''
        线性embedding后，应该将其和可学习的mask tokens堆叠到一起，形成完整的token。
        然后进行unshuffle
        接着将cls token加到开头，输入trm网络
        最后进行一次层归一化和线性层输出结果
        '''
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)  # learnable vector
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x_ = self.decoder_pos_embed(x_)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        y = x[:, 1:, :]
        y_center = x[:, :1, :]
        return y, y_center

    def forward_loss(self, imgs, pred, pred_center, mask=None):
        """
        imgs: [N,C, H, W]
        pred: [N, H*W, C]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = imgs.squeeze().reshape(imgs.shape[0], imgs.shape[1], -1).transpose(-1, -2)  # N,L,D
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss_r = (pred - target) ** 2
        loss_r = loss_r.sum(-1).mean()  # [N, L], mean loss per pixel

        pred_center = self.mlp1(pred_center.squeeze())
        target_center = self.mlp2(target[:, target.shape[1] // 2, :])
        loss_center = (pred_center - target_center) ** 2
        loss_center = loss_center.sum(-1).mean()
        # index = pred.shape[1]//2
        # loss_center = (pred[:, index] - target[:, index])**2
        # loss_center = loss_center.mean(dim=-1, keepdim=True)

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss_all = loss_r + loss_center
        return loss_all, loss_r.detach(), loss_center.detach()

    # def latent_comb(self, imgs, mask_ratio):
    #     imgs = imgs.unsqueeze(1)
    #     latent, _, _ = self.encoder_forward(imgs, mask_ratio)
    #     latent = linear_comb(latent)
    #     return latent
    #
    # def cluster(self, imgs, mask_ratio, clusters):
    #     imgs = imgs.unsqueeze(1)
    #     latent, _, _ = self.encoder_forward(imgs, mask_ratio)
    #     latent = linear_comb(latent)
    #
    #     return latent

    def forward(self, imgs, mask_ratio):
        latent, mask, ids_restore = self.encoder_forward(imgs, mask_ratio)
        pred, pred_center = self.decoder_forward(latent, ids_restore)
        loss_all, loss_r, loss_center = self.forward_loss(imgs, pred, pred_center)
        pred = pred.reshape(pred.shape[0], int(np.sqrt(pred.shape[1])), int(np.sqrt(pred.shape[1])), -1)
        pred = torch.einsum('nhwc->nchw', pred)
        return pred, mask, loss_all, loss_r, loss_center


hsimae_15p_204c_base_model = HsiPatchMaskedAutoEncoder(name='hsimae_15p_204c_base_model', in_chans=297,
                                                       encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
                                                       decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                                       mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)

hsimae_15p_204c_tiny_model = HsiPatchMaskedAutoEncoder(name='hsimae_15p_204c_tiny_model', in_chans=297,
                                                       encoder_embed_dim=512, encoder_depth=16, encoder_num_heads=8,
                                                       decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
                                                       mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)

hsimae_15p_204c_stiny_model = HsiPatchMaskedAutoEncoder(name='hsimae_15p_204c_stiny_model', in_chans=297,
                                                        encoder_embed_dim=256, encoder_depth=16, encoder_num_heads=8,
                                                        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=4,
                                                        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)

hsimae_15p_204c_sstiny_model = HsiPatchMaskedAutoEncoder(name='hsimae_15p_204c_sstiny_model', in_chans=297,
                                                         encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=8,
                                                         decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,
                                                         mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
hsimae_15p_204c_sstiny_model_64 = HsiPatchMaskedAutoEncoder(name='hsimae_15p_204c_sstiny_model', in_chans=297,
                                                         encoder_embed_dim=64, encoder_depth=4, encoder_num_heads=8,
                                                         decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4,
                                                         mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
if __name__ == '__main__':
    inputs = torch.randn((5, 297, 15, 15))
    # hsimae_15p_204c_base_model
    pred, mask, loss_all, loss_r, loss_center = hsimae_15p_204c_base_model(inputs, 0.5)
    a = 0
