from functools import partial

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from pwvdswin_ViT import *
from utils.pos_embed import get_2d_sincos_pos_embed


class SwinMAE(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True,mask_type='mixmask'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type=mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim==384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim ==768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, x_c: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')
        x_c = rearrange(x_c, 'B H W C -> B (H W) C')
        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            x_masked_c = torch.clone(x_c)
            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
                x_masked_c[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            x_masked_c = rearrange(x_masked_c, 'B (H W) C -> B H W C', H=int(x_masked_c.shape[1] ** 0.5))
            return x_masked,x_masked_c, mask

    def window_masking_shannon(self, x: torch.Tensor, x_c: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False,beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W,C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')
        x_c = rearrange(x_c, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W),device=x.device)
        mask_size = int(H * W * (1-self.mask_ratio) )
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row  * W)[:top_bottom_size//2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2 ] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask=1-mask
        mask=rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)
        x_masked_c = torch.clone(x_c)
        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
            x_masked_c[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
        x_masked_c = rearrange(x_masked_c, 'B (H W) C -> B H W C', H=int(x_masked_c.shape[1] ** 0.5))
        return x_masked,x_masked_c, mask

    def random_masking(self, x: torch.Tensor, x_c: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio=self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')
        x_c = rearrange(x_c, 'B H W C -> B (H W) C')
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)
        x_masked_c = torch.clone(x_c)
        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
            x_masked_c[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
        x_masked_c = rearrange(x_masked_c, 'B (H W) C -> B H W C', H=int(x_masked_c.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked,x_masked_c, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise,clean,mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type=='mixmask':
            if np.random.rand()<0.5:
                x_noise,x_clean, mask =self.random_masking(x_noise,x_clean)
            else:
                x_noise,x_clean, mask = self.window_masking(x_noise,x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_noise, x_clean, mask = self.window_masking(x_noise, x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_noise, x_clean, mask = self.random_masking(x_noise, x_clean)
        elif mask_type=='shannon':
            x_noise, x_clean, mask = self.window_masking_shannon(x_noise, x_clean)
        for i,layer in enumerate(self.layers):
            if i==0:
                x_noise = layer(x_noise,x_clean)
            else:
                x_noise = layer(x_noise, x_noise)

        return x_noise, mask

    def forward_decoder(self, x):

        x = self.first_patch_expanding(x)

        for layer in self.layers_up:
            x = layer(x)

        x = self.norm_up(x)

        x = rearrange(x, 'B H W C -> B (H W) C')

        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, noise,clean):
        latent, mask = self.forward_encoder(noise,clean,self.mask_type)
        pred = self.forward_decoder(latent)
        # loss = self.forward_loss(noise, pred, mask)
        target = self.patchify(clean)
        return target, pred, mask

class SwinMAE_fc(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True,mask_type='mixmask',numclass=10):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type=mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.linear_n=nn.Sequential(nn.Linear(64, 32),
                                          nn.Dropout(0.2),
                                          nn.Linear(32, numclass))
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.norm1 = nn.LayerNorm(64)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()



        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise,mask_type):
        x_noise = self.patch_embed(noise)
        if self.training:
            if mask_type == 'mixmask':
                if np.random.rand() < 0.5:
                    x_noise, mask = self.random_masking(x_noise)
                else:
                    x_noise, mask = self.window_masking(x_noise, remove=False, mask_len_sparse=False)
            elif mask_type == 'window':
                x_noise, mask = self.window_masking(x_noise, remove=False, mask_len_sparse=False)
            elif mask_type == 'suiji':
                x_noise, mask = self.random_masking(x_noise)
            elif mask_type == 'shannon':
                x_noise, mask = self.window_masking_shannon(x_noise)
        for i,layer in enumerate(self.layers):
                x_noise = layer(x_noise, x_noise)

        return x_noise


    def forward(self, noise,clean):
        latent = self.forward_encoder(noise,self.mask_type)
        latent=latent.reshape(latent.shape[0], latent.shape[1]*latent.shape[2],latent.shape[3])
        latent=self.norm1(latent.transpose(1, 2))
        latent = self.avgpool(latent.transpose(1, 2))  # B C 1
        latent = torch.flatten(latent, 1)
        out=self.linear_n(latent)
        # pred = self.forward_decoder(latent)
        # # loss = self.forward_loss(noise, pred, mask)
        # target = self.patchify(clean)
        return out

class SwinMAE_fc_old(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True,mask_type='mixmask',numclass=10):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type=mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.linear_n=nn.Sequential(nn.Linear(decoder_embed_dim, 128),
                                          nn.Dropout(0.2),
                                          nn.Linear(128, numclass))
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.norm1 = nn.LayerNorm(decoder_embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()



        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise,mask_type):
        x_noise = self.patch_embed(noise)
        if self.training:
            if mask_type == 'mixmask':
                if np.random.rand() < 0.5:
                    x_noise, mask = self.random_masking(x_noise)
                else:
                    x_noise, mask = self.window_masking(x_noise, remove=False, mask_len_sparse=False)
            elif mask_type == 'window':
                x_noise, mask = self.window_masking(x_noise, remove=False, mask_len_sparse=False)
            elif mask_type == 'suiji':
                x_noise, mask = self.random_masking(x_noise)
            elif mask_type == 'shannon':
                x_noise, mask = self.window_masking_shannon(x_noise)
        for i,layer in enumerate(self.layers):
                x_noise = layer(x_noise, x_noise)

        return x_noise


    def forward(self, noise,clean):
        latent = self.forward_encoder(noise,self.mask_type)
        latent=latent.reshape(latent.shape[0], latent.shape[1]*latent.shape[2],latent.shape[3])
        latent=self.norm1(latent)
        latent = self.avgpool(latent.transpose(1, 2))  # B C 1
        latent = torch.flatten(latent, 1)
        out=self.linear_n(latent)
        # pred = self.forward_decoder(latent)
        # # loss = self.forward_loss(noise, pred, mask)
        # target = self.patchify(clean)
        return out



class SwinMAE_con(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x_noise = layer(x_noise, x_noise)
                x_clean = layer(x_clean, x_clean)
            else:
                x_noise = layer(x_noise, x_noise)
                x_clean = layer(x_clean, x_clean)

        return x_noise, x_clean, mask

    def forward_decoder(self, x):

        x = self.first_patch_expanding(x)

        for layer in self.layers_up:
            x = layer(x)

        x = self.norm_up(x)

        x = rearrange(x, 'B H W C -> B (H W) C')

        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, noise, clean):
        latent_noise, latent_clean, mask = self.forward_encoder(noise, clean, self.mask_type)
        pred = self.forward_decoder(latent_noise)
        # loss = self.forward_loss(noise, pred, mask)
        latent_noise = latent_noise.reshape(latent_noise.shape[0], -1)
        latent_clean = latent_clean.reshape(latent_clean.shape[0], -1)
        return latent_noise, latent_clean, pred, mask

#在CON2中，将CLEAN也输入dncoder进行解码
class SwinMAE_con2(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x_noise = layer(x_noise, x_noise)
                x_clean = layer(x_clean, x_clean)
            else:
                x_noise = layer(x_noise, x_noise)
                x_clean = layer(x_clean, x_clean)

        return x_noise, x_clean, mask

    def forward_decoder(self, x):

        x = self.first_patch_expanding(x)

        for layer in self.layers_up:
            x = layer(x)

        x = self.norm_up(x)

        x = rearrange(x, 'B H W C -> B (H W) C')

        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, noise, clean):
        latent_noise, latent_clean, mask = self.forward_encoder(noise, clean, self.mask_type)
        pred_noise = self.forward_decoder(latent_noise)
        pred_clean = self.forward_decoder(latent_clean)
        # loss = self.forward_loss(noise, pred, mask)
        latent_noise = latent_noise.reshape(latent_noise.shape[0], -1)
        latent_clean = latent_clean.reshape(latent_clean.shape[0], -1)
        return latent_noise, latent_clean, pred_noise,pred_clean, mask

class SwinMAE_con_hec(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        x_noise_out = []
        x_clean_out = []
        for i, layer in enumerate(self.layers):
            x_noise = layer(x_noise, x_noise)
            x_clean = layer(x_clean, x_clean)
            x_noise_out.append(x_noise)
            x_clean_out.append(x_clean)

        return x_noise_out, x_clean_out, mask


    def forward_decoder(self, input):

        input[1] = self.first_patch_expanding(input[1])
        input[2] = self.first_patch_expanding(input[2])
        output=[]
        for x in input:
            for layer in self.layers_up:
                x = layer(x)

            x = self.norm_up(x)

            x = rearrange(x, 'B H W C -> B (H W) C')

            x = self.decoder_pred(x)
            output.append(x)

        return output

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, noise, clean):
        latent_noise_out, latent_clean_out, mask = self.forward_encoder(noise, clean, self.mask_type)
        pred_noise = self.forward_decoder(latent_noise_out)
        pred_clean = self.forward_decoder(latent_clean_out)
        # loss = self.forward_loss(noise, pred, mask)
        latent_noise=latent_noise_out[2]
        latent_clean=latent_clean_out[2]
        latent_noise = latent_noise.reshape(latent_noise.shape[0], -1)
        latent_clean = latent_clean.reshape(latent_clean.shape[0], -1)
        return latent_noise, latent_clean, pred_noise,pred_clean, mask

class SwinMAE_con_hec_predplot(nn.Module):
    """
    这是专门为plot函数写的，返回所以层的输出量
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask',return_type='B_L_L_D->B_L*L*D'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.return_type=return_type

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        x_noise_out = []
        x_clean_out = []
        for i, layer in enumerate(self.layers):
            x_noise = layer(x_noise, x_noise)
            x_clean = layer(x_clean, x_clean)
            x_noise_out.append(x_noise)
            x_clean_out.append(x_clean)

        return x_noise_out, x_clean_out, mask


    def forward_decoder(self, input):

        input[1] = self.first_patch_expanding(input[1])
        input[2] = self.first_patch_expanding(input[2])
        output=[]
        for x in input:
            for layer in self.layers_up:
                x = layer(x)

            x = self.norm_up(x)

            x = rearrange(x, 'B H W C -> B (H W) C')

            x = self.decoder_pred(x)
            output.append(x)

        return output

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, noise, clean):
        latent_noise_out, latent_clean_out, mask = self.forward_encoder(noise, clean, self.mask_type)
        latent_noise_out1 =[t.clone() for t in latent_noise_out]
        latent_clean_out1 =[t.clone() for t in latent_clean_out]
        pred_noise = self.forward_decoder(latent_noise_out)
        pred_clean = self.forward_decoder(latent_clean_out)

        if self.return_type=='B_L_L_D->B_L*L*D':
            for i in range(3):
                latent_noise_out[i] = latent_noise_out[i].reshape(latent_noise_out[i].shape[0], -1)
                latent_clean_out[i] = latent_clean_out[i].reshape(latent_clean_out[i].shape[0], -1)

        elif self.return_type=='B_L_L_D->B_L*L_D':
            for i in range(3):
                Ba,L,D=latent_noise_out1[i].shape[0],latent_noise_out1[i].shape[1],latent_noise_out1[i].shape[3]
                latent_noise_out1[i] = latent_noise_out1[i].reshape(Ba,L*L,D)
                latent_clean_out1[i] = latent_clean_out1[i].reshape(Ba,L*L,D)
        return latent_noise_out1, latent_clean_out1, pred_noise,pred_clean, mask


class SwinMAE_con_hecmean(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        x_noise_out = []
        x_clean_out = []
        for i, layer in enumerate(self.layers):
            x_noise = layer(x_noise, x_noise)
            x_clean = layer(x_clean, x_clean)
            x_noise_out.append(x_noise)
            x_clean_out.append(x_clean)

        return x_noise_out, x_clean_out, mask


    def forward_decoder(self, input):

        input[1] = self.first_patch_expanding(input[1])
        input[2] = self.first_patch_expanding(input[2])
        output=[]
        for x in input:
            for layer in self.layers_up:
                x = layer(x)

            x = self.norm_up(x)

            x = rearrange(x, 'B H W C -> B (H W) C')

            x = self.decoder_pred(x)
            output.append(x)

        return output

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, noise, clean):
        latent_noise_out, latent_clean_out, mask = self.forward_encoder(noise, clean, self.mask_type)
        latent_noise=latent_noise_out[2]
        latent_clean=latent_clean_out[2]
        pred_noise = self.forward_decoder(latent_noise_out)
        pred_clean = self.forward_decoder(latent_clean_out)
        # loss = self.forward_loss(noise, pred, mask)
        latent_clean =torch.mean(latent_clean,-1)
        latent_noise = torch.mean(latent_noise, -1)
        latent_noise = latent_noise.reshape(latent_noise.shape[0], -1)
        latent_clean = latent_clean.reshape(latent_clean.shape[0], -1)
        return latent_noise, latent_clean, pred_noise,pred_clean, mask

class SwinMAE_con_hecmean_moco(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, clean, mask_type):
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)

        x_clean_out = []
        for i, layer in enumerate(self.layers):
            x_clean = layer(x_clean, x_clean)
            x_clean_out.append(x_clean)

        return x_clean_out, mask


    def forward_decoder(self, input):

        input[1] = self.first_patch_expanding(input[1])
        input[2] = self.first_patch_expanding(input[2])
        output=[]
        for x in input:
            for layer in self.layers_up:
                x = layer(x)

            x = self.norm_up(x)

            x = rearrange(x, 'B H W C -> B (H W) C')

            x = self.decoder_pred(x)
            output.append(x)

        return output

    def forward(self, clean):
        latent_clean_out, mask = self.forward_encoder( clean, self.mask_type)
        latent_clean=latent_clean_out[2]
        pred_clean = self.forward_decoder(latent_clean_out)
        # loss = self.forward_loss(noise, pred, mask)
        latent_clean =torch.mean(latent_clean,-1)
        latent_clean = latent_clean.reshape(latent_clean.shape[0], -1)
        return latent_clean,pred_clean, mask

class SwinMAE_con_hecmean_sk(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask',m=0.99):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        self.m=m
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
            self.norm1 = nn.LayerNorm(decoder_embed_dim)
            self.norm2 = nn.LayerNorm(decoder_embed_dim)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            # self.linear_c=nn.Sequential(nn.Linear(decoder_embed_dim,128),
            #                             nn.Dropout(0.2),
            #                             nn.Linear(128,10))
            # self.linear_n = nn.Sequential(nn.Linear(decoder_embed_dim, 128),
            #                               nn.Dropout(0.2),
            #                               nn.Linear(128, 10))
            self.linear_c=nn.Sequential(nn.Linear(64,32),
                                        nn.Dropout(0.2),
                                        nn.Linear(32,10))
            self.linear_n = nn.Sequential(nn.Linear(64,32),
                                        nn.Dropout(0.2),
                                        nn.Linear(32,10))
            for param_q, param_k in zip(self.linear_c.parameters(), self.linear_n.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.linear_c.parameters(), self.linear_n.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        x_noise_out = []
        x_clean_out = []
        for i, layer in enumerate(self.layers):
            x_noise = layer(x_noise, x_noise)
            x_clean = layer(x_clean, x_clean)
            x_noise_out.append(x_noise)
            x_clean_out.append(x_clean)

        return x_noise_out, x_clean_out, mask


    def forward_decoder(self, input):

        input[1] = self.first_patch_expanding(input[1])
        input[2] = self.first_patch_expanding(input[2])
        output=[]
        for x in input:
            for layer in self.layers_up:
                x = layer(x)

            x = self.norm_up(x)

            x = rearrange(x, 'B H W C -> B (H W) C')

            x = self.decoder_pred(x)
            output.append(x)

        return output


    def forward(self, noise, clean):
        latent_noise_out, latent_clean_out, mask = self.forward_encoder(noise, clean, self.mask_type)
        latent_noise=latent_noise_out[2]
        latent_clean=latent_clean_out[2]
        latent_clean1 = torch.mean(latent_clean, -1)
        latent_noise1 = torch.mean(latent_noise, -1)
        latent_noise1 = latent_noise1.reshape(latent_noise1.shape[0], -1)
        latent_clean1 = latent_clean1.reshape(latent_clean1.shape[0], -1)
        pred_noise = self.forward_decoder(latent_noise_out)
        pred_clean = self.forward_decoder(latent_clean_out)
        # loss = self.forward_loss(noise, pred, mask)
        latent_clean = latent_clean.reshape(latent_clean.shape[0], latent_clean.shape[1] * latent_clean.shape[2], latent_clean.shape[3])
        # latent_clean = self.norm1(latent_clean)
        latent_clean = self.avgpool(latent_clean)  # B C 1
        latent_clean = torch.flatten(latent_clean, 1)
        
        latent_noise = latent_noise.reshape(latent_noise.shape[0], latent_noise.shape[1] * latent_noise.shape[2],
                                            latent_noise.shape[3])
        # latent_noise = self.norm2(latent_noise)
        latent_noise = self.avgpool(latent_noise)  # B C 1
        latent_noise = torch.flatten(latent_noise, 1)
        outc=self.linear_c(latent_clean)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            outn = self.linear_n(latent_noise)
        # latent_clean = torch.mean(latent_clean, -1)
        # latent_noise = torch.mean(latent_noise, -1)
        # latent_noise = latent_noise.reshape(latent_noise.shape[0], -1)
        # latent_clean = latent_clean.reshape(latent_clean.shape[0], -1)
        return latent_noise1, latent_clean1, pred_noise,pred_clean, mask,outc,outn

class SwinMAE_con_hecmean_sk2(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask', m=0.99):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        self.m = m
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
            self.norm1 = nn.LayerNorm(decoder_embed_dim)
            self.norm2 = nn.LayerNorm(decoder_embed_dim)
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            # self.linear_c=nn.Sequential(nn.Linear(decoder_embed_dim,128),
            #                             nn.Dropout(0.2),
            #                             nn.Linear(128,10))
            # self.linear_n = nn.Sequential(nn.Linear(decoder_embed_dim, 128),
            #                               nn.Dropout(0.2),
            #                               nn.Linear(128, 10))
            self.linear_c = nn.Sequential(nn.Linear(64, 32),
                                          nn.Dropout(0.2),
                                          nn.Linear(32, 10))
            self.linear_n = nn.Sequential(nn.Linear(64, 32),
                                          nn.Dropout(0.2),
                                          nn.Linear(32, 10))
            for param_q, param_k in zip(self.linear_c.parameters(), self.linear_n.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.linear_c.parameters(), self.linear_n.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        x_noise_out = []
        x_clean_out = []
        for i, layer in enumerate(self.layers):
            x_noise = layer(x_noise, x_noise)
            x_clean = layer(x_clean, x_clean)
            x_noise_out.append(x_noise)
            x_clean_out.append(x_clean)

        return x_noise_out, x_clean_out, mask

    def forward_decoder(self, input):

        input[1] = self.first_patch_expanding(input[1])
        input[2] = self.first_patch_expanding(input[2])
        output = []
        for x in input:
            for layer in self.layers_up:
                x = layer(x)

            x = self.norm_up(x)

            x = rearrange(x, 'B H W C -> B (H W) C')

            x = self.decoder_pred(x)
            output.append(x)

        return output

    def forward(self, noise, clean):
        latent_noise_out, latent_clean_out, mask = self.forward_encoder(noise, clean, self.mask_type)
        latent_noise = latent_noise_out[2]
        latent_clean = latent_clean_out[2]
        
        pred_noise = self.forward_decoder(latent_noise_out)
        pred_clean = self.forward_decoder(latent_clean_out)
        # loss = self.forward_loss(noise, pred, mask)
        latent_clean = latent_clean.reshape(latent_clean.shape[0], latent_clean.shape[1] * latent_clean.shape[2],
                                            latent_clean.shape[3])
        latent_noise = latent_noise.reshape(latent_noise.shape[0], latent_noise.shape[1] * latent_noise.shape[2],
                                            latent_noise.shape[3])
        
        latent_clean1 = torch.mean(latent_clean.transpose(1, 2), -1)
        latent_noise1 = torch.mean(latent_noise.transpose(1, 2), -1)
        latent_noise1 = latent_noise1.reshape(latent_noise1.shape[0], -1)
        latent_clean1 = latent_clean1.reshape(latent_clean1.shape[0], -1)
        
         # latent_clean = self.norm1(latent_clean)
        latent_clean = self.avgpool(latent_clean)  # B C 1
        latent_clean = torch.flatten(latent_clean, 1)
        # latent_noise = self.norm2(latent_noise)
        latent_noise = self.avgpool(latent_noise)  # B C 1
        latent_noise = torch.flatten(latent_noise, 1)
        outc = self.linear_c(latent_clean)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            outn = self.linear_n(latent_noise)
        # latent_clean = torch.mean(latent_clean, -1)
        # latent_noise = torch.mean(latent_noise, -1)
        # latent_noise = latent_noise.reshape(latent_noise.shape[0], -1)
        # latent_clean = latent_clean.reshape(latent_clean.shape[0], -1)
        return latent_noise1, latent_clean1, pred_noise, pred_clean, mask, outc, outn
    
class SwinMAE_con_hecmean_outall(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        x_noise_out = []
        x_clean_out = []
        for i, layer in enumerate(self.layers):
            x_noise = layer(x_noise, x_noise)
            x_clean = layer(x_clean, x_clean)
            x_noise_out.append(x_noise)
            x_clean_out.append(x_clean)

        return x_noise_out, x_clean_out, mask


    def forward_decoder(self, input):

        input[1] = self.first_patch_expanding(input[1])
        input[2] = self.first_patch_expanding(input[2])
        output=[]
        for x in input:
            for layer in self.layers_up:
                x = layer(x)

            x = self.norm_up(x)

            x = rearrange(x, 'B H W C -> B (H W) C')

            x = self.decoder_pred(x)
            output.append(x)

        return output

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, noise, clean):
        latent_noise_out, latent_clean_out, mask = self.forward_encoder(noise, clean, self.mask_type)
        latent_noise=latent_noise_out
        latent_clean=latent_clean_out
        pred_noise = self.forward_decoder(latent_noise_out)
        pred_clean = self.forward_decoder(latent_clean_out)
        # loss = self.forward_loss(noise, pred, mask)
        for i in range(len(latent_clean)):
            latent_clean[i] =torch.mean(latent_clean[i],-1)
            latent_noise[i] = torch.mean(latent_noise[i], -1)
            latent_noise[i] = latent_noise[i].reshape(latent_noise[i].shape[0], -1)
            latent_clean[i] = latent_clean[i].reshape(latent_clean[i].shape[0], -1)
        return latent_noise, latent_clean, pred_noise,pred_clean, mask

'''
SwinMAE_con_hecmean_outall_conv
是不使用全连接层进行Encoderd 的PatchMerging，而是使用残差卷积替代PatchMerging
'''

class SwinMAE_con_hecmean_outall_conv(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock_conv(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        x_noise_out = []
        x_clean_out = []
        for i, layer in enumerate(self.layers):
            x_noise = layer(x_noise, x_noise)
            x_clean = layer(x_clean, x_clean)
            x_noise_out.append(x_noise)
            x_clean_out.append(x_clean)

        return x_noise_out, x_clean_out, mask


    def forward_decoder(self, input):

        input[1] = self.first_patch_expanding(input[1])
        input[2] = self.first_patch_expanding(input[2])
        output=[]
        for x in input:
            for layer in self.layers_up:
                x = layer(x)

            x = self.norm_up(x)

            x = rearrange(x, 'B H W C -> B (H W) C')

            x = self.decoder_pred(x)
            output.append(x)

        return output

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, noise, clean):
        latent_noise_out, latent_clean_out, mask = self.forward_encoder(noise, clean, self.mask_type)
        latent_noise=latent_noise_out
        latent_clean=latent_clean_out
        pred_noise = self.forward_decoder(latent_noise_out)
        pred_clean = self.forward_decoder(latent_clean_out)
        # loss = self.forward_loss(noise, pred, mask)
        for i in range(len(latent_clean)):
            latent_clean[i] =torch.mean(latent_clean[i],-1)
            latent_noise[i] = torch.mean(latent_noise[i], -1)
            latent_noise[i] = latent_noise[i].reshape(latent_noise[i].shape[0], -1)
            latent_clean[i] = latent_clean[i].reshape(latent_clean[i].shape[0], -1)
        return latent_noise, latent_clean, pred_noise,pred_clean, mask


class SwinMAE_con_hec2(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask'):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.layers_up1 = self.build_layers_up()
        self.layers_up2 = self.build_layers_up()
        self.layers_up3 = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        if decoder_embed_dim == 384:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 4, patch_size ** 2 * in_chans, bias=True)
        elif decoder_embed_dim == 768:
            self.decoder_pred = nn.Linear(decoder_embed_dim // 8, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4,
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        x = rearrange(x, 'B H W C -> B (H W) C')

        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - self.mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)

            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

            return x_masked, mask

    def window_masking_shannon(self, x: torch.Tensor, r: int = 4,
                               remove: bool = False, mask_len_sparse: bool = False, beta=0.85):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        B, H, W, C = x.shape
        x = rearrange(x, 'B H W C -> B (H W) C')

        mask = torch.zeros((B, H, W), device=x.device)
        mask_size = int(H * W * (1 - self.mask_ratio))
        top_bottom_size = int(mask_size * beta)
        middle_size = mask_size - top_bottom_size
        top_bottom_row = 7
        for b in range(B):
            top_bottom_index1 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            top_bottom_index2 = torch.randperm(top_bottom_row * W)[:top_bottom_size // 2]
            middle_index = torch.randperm((H - top_bottom_row * 2) * W)[:middle_size]
            mask[b, :top_bottom_row].view(-1)[top_bottom_index1] = 1
            mask[b, -top_bottom_row:].view(-1)[top_bottom_index2] = 1
            mask[b, top_bottom_row:-top_bottom_row].view(-1)[middle_index] = 1
        mask = 1 - mask
        mask = rearrange(mask, 'B H W  -> B (H W) ')
        max_length = int(mask.sum(dim=1).max())
        index_mask = torch.zeros((B, max_length), dtype=torch.long)
        for b in range(B):
            index = (mask[b] == 1).nonzero(as_tuple=True)[0]
            index_mask[b, :len(index)] = index
        x_masked = torch.clone(x)

        for i in range(B):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        return x_masked, mask

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        x = rearrange(x, 'B H W C -> B (H W) C')

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index_all = np.expand_dims(range(L), axis=0).repeat(N, axis=0)
        index_mask = np.zeros([N, int(L - ids_keep.shape[-1])], dtype=np.int64)
        for i in range(N):
            index_mask[i] = np.setdiff1d(index_all[i], ids_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        x_masked = torch.clone(x)

        for i in range(N):
            x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token

        x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, noise, clean, mask_type):
        x_noise = self.patch_embed(noise)
        x_clean = self.patch_embed(clean)
        if mask_type == 'mixmask':
            if np.random.rand() < 0.5:
                x_clean, mask = self.random_masking(x_clean)
            else:
                x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'window':
            x_clean, mask = self.window_masking(x_clean, remove=False, mask_len_sparse=False)
        elif mask_type == 'suiji':
            x_clean, mask = self.random_masking(x_clean)
        elif mask_type == 'shannon':
            x_clean, mask = self.window_masking_shannon(x_clean)
        x_noise_out = []
        x_clean_out = []
        for i, layer in enumerate(self.layers):
            x_noise = layer(x_noise, x_noise)
            x_clean = layer(x_clean, x_clean)
            x_noise_out.append(x_noise)
            x_clean_out.append(x_clean)

        return x_noise_out, x_clean_out, mask



    def forward_decoder(self, input):

        input[1] = self.first_patch_expanding(input[1])
        input[2] = self.first_patch_expanding(input[2])
        output=[]
        for i,x in enumerate(input):
            self.layers_up=getattr(self, f'layers_up{i+1}')
            for layer in self.layers_up:
                x = layer(x)

            x = self.norm_up(x)

            x = rearrange(x, 'B H W C -> B (H W) C')

            x = self.decoder_pred(x)
            output.append(x)

        return output

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, noise, clean):
        latent_noise_out, latent_clean_out, mask = self.forward_encoder(noise, clean, self.mask_type)
        pred_noise = self.forward_decoder(latent_noise_out)
        pred_clean = self.forward_decoder(latent_clean_out)
        # loss = self.forward_loss(noise, pred, mask)
        latent_noise=latent_noise_out[-1]
        latent_clean=latent_clean_out[-1]
        latent_noise = latent_noise.reshape(latent_noise.shape[0], -1)
        latent_clean = latent_clean.reshape(latent_clean.shape[0], -1)
        return latent_noise, latent_clean, pred_noise,pred_clean, mask

# def swin_mae(**kwargs):
#     model = SwinMAE(
#         img_size=224, patch_size=4, in_chans=3,
#         decoder_embed_dim=768,
#         depths=(2, 2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
#         window_size=7, qkv_bias=True, mlp_ratio=4,
#         drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# img_size=128
# model = SwinMAE_con_hecmean_outall_conv(
#     img_size=128, patch_size=4, in_chans=1,
#     decoder_embed_dim=384,
#     depths=(2, 2 , 2), embed_dim=96, num_heads=(3, 6, 12, 24),
#     window_size=2, qkv_bias=True, mlp_ratio=4,
#     drop_path_rate=0.1, drop_rate=0.1, attn_drop_rate=0.1,
#     norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75,mask_type='suiji')
# noise = torch.randn((2, 1, img_size, img_size))
# clean = torch.randn((2, 1, img_size, img_size))
# model(noise,clean)
