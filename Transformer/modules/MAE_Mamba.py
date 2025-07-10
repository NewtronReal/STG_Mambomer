from modules.Encoder import Encoder
import torch
import torch.nn as nn
from modules.GraphFeatures import GraphFeatures


class MAE_Mamba(nn.Module):
    def __init__(
        self,
        C=1,
        C_out=1,
        N=307,
        S=50,
        T=12,
        hno=16,
        d_e=1024,
        enc_layers = 24,
        d_d=512,
        dec_layers = 8,
        dropout=0.1,
        norm_layer = nn.LayerNorm,
        graph:GraphFeatures=None
    ):
        super().__init__()
        self.C = C
        self.C_out = C_out
        self.d_e=d_e
        self.T=T
        self.N = N
        self.activation = nn.GELU()
        self.graph_token_embed = nn.Parameter(torch.zeros(1,1,1,d_e))
        self.decoder_graph_token_embed = nn.Parameter(torch.zeros(1,1,1,d_d))
        self.graph_token_virtual_distance = nn.Embedding(1,hno)
        self.blocks = Encoder(
            C,N,graph,enc_layers,d_e,d_e*4,hno,dropout,S
        )
        self.norm = norm_layer(d_e)
        self.mask_token = nn.Parameter(torch.zeros(1,1,d_d))
        self.detodd = nn.Linear(d_e,d_d)
        self.decoder_blocks = Encoder(C,N,graph,dec_layers,d_d,d_d*4,hno,dropout,S)
        self.dec_norm = norm_layer(d_d)
        self.dec_conv_1 = nn.Conv2d(
            in_channels=d_d,
            out_channels=d_d//2,
            kernel_size=(1,1),
            bias=False
        )
        self.dec_b_norm = nn.BatchNorm2d(d_d//2)
        self.dec_conv_2=nn.Conv2d(
            in_channels=d_d//2,
            out_channels=C_out,
            kernel_size=(1,1),
            bias=True
        )
        self.pos_embed_time_enc = nn.Parameter(
            torch.zeros(1,T,d_e)
        )
        self.pos_embed_space_enc = nn.Parameter(
            torch.zeros(1,N,d_e)
        )
        self.pos_embed_time_dec = nn.Parameter(
            torch.zeros(1,T,d_d)
        )
        self.pos_embed_space_dec = nn.Parameter(
            torch.zeros(1,N,d_d)
        )
        self.pos_embed_cls_enc = nn.Parameter(torch.zeros(1,1,d_e))
        self.pos_embed_cls_dec = nn.Parameter(torch.zeros(1,1,d_d))
    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.pos_embed_time_enc,std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_space_enc,std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_time_dec,std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_space_dec,std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_cls_enc,std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_cls_dec,std=0.02)
        torch.nn.init.trunc_normal_(
                self.blocks.gnf.c_enc.z_in.weight,
                mean=0.0,
                std=0.02
        )
        torch.nn.init.trunc_normal_(
                self.blocks.gnf.c_enc.z_out.weight,
                mean=0.0,
                std=0.02
        )
        torch.nn.init.trunc_normal_(
                self.blocks.sp_bias.sp_enc.weight,
                mean=0.0,
                std=0.02
        )
        torch.nn.init.trunc_normal_(self.graph_token_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.graph_token_virtual_distance.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_graph_token_embed, std=0.02)
        torch.nn.init.normal_(self.mask_token,std=0.02)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def shuffle_attn_bias(self,attn_bias,ids_keep,x_shape,n_token):
        B,T,N,_ = x_shape
        _,hno,num_nodes,_ = attn_bias.shape
        ids_keep_ = ids_keep.detach().clone()
        ids_keep_ = ids_keep_ +1
        ids_keep_ = torch.cat([torch.zeros(B,T,1,dtype = ids_keep_.dtype,device=ids_keep_.device),ids_keep_],dim=2)
        assert ids_keep_.shape[1]*ids_keep_.shape[2] == n_token
        attn_bias = attn_bias.unsqueeze(2).expand(-1,-1,T,-1,-1).clone()
        attn_bias = torch.gather(
            attn_bias,
            dim=3,
            index=ids_keep_.unsqueeze(-1).unsqueeze(1).expand(-1, hno, -1, -1, N + 1)
        )
        attn_bias = attn_bias.unsqueeze(-2).expand(-1, -1, -1, -1, T, -1).clone()
        attn_bias = torch.gather(
            attn_bias,
            dim=-1,
            index=ids_keep_.unsqueeze(1).unsqueeze(2).unsqueeze(1).
            expand(-1, hno, T, ids_keep_.shape[2], -1, -1)
        )
        return attn_bias.contiguous().view(B, hno, n_token, n_token)
        
    def random_masking(self,x,mask_ratio:float=.5):
        B,T,N,C = x.shape
        len_keep = int(N*(1-mask_ratio))
        noise = torch.rand(B,T,N,device = x.device)
        ids_shuffle = torch.argsort(
            noise,dim=2
        )
        ids_restore = torch.argsort(ids_shuffle,dim=2)
        ids_keep = ids_shuffle[:,:,:len_keep]
        x_masked = torch.gather(x,dim=2,index=ids_keep.unsqueeze(-1).expand(-1,-1,-1,C))
        mask = torch.ones([B,T,N],device=x.device)
        mask[:,:,:len_keep]=0
        mask = torch.gather(mask,dim=2,index=ids_restore)
        del noise, ids_shuffle
        return x_masked, mask,ids_restore,ids_keep
    def add_token_distance(self,attn_bias:torch.Tensor,device:torch.device):
        B,hno,L,_ = attn_bias.shape#L is tokens length
        t= self.graph_token_virtual_distance.weight.repeat(B,1).view(-1,hno,1)
        attn_bias[:,:,1:,0] = attn_bias[:,:,1:,0].clone()+t
        attn_bias[:, :, 0, :] = attn_bias[:, :, 0, :].clone() + t
        return attn_bias
    def get_graph_rep(self,x,x_shape):
        B,T,N,d_e = x_shape
        x=x.contiguous().view(B,T,-1,d_e)
        graph_rep = x[:,:,:1,:]
        x=x[:,:,1:,:]
        return graph_rep,x
    def forward_encoder(self,x:torch.Tensor,attn_bias:torch.Tensor,mask_ratio:float=0.5):
        x_shape = list(x.shape)
        B,T,N,d_e = x_shape
        x,mask,ids_restore,ids_keep = self.random_masking(x,mask_ratio)
        graph_token_feature = self.graph_token_embed.expand(B,T,-1,-1)
        x=torch.cat([graph_token_feature,x],dim=2)
        x=x.contiguous().view(B,-1,d_e)
        n_tokens = x.shape[1]
        assert n_tokens == T*(int(N*(1-mask_ratio))+1),\
            f'{n_tokens} does not match {T}*{N}*(1-{mask_ratio}+1)'
        pos_embed = self.pos_embed_space_enc.repeat(1,T,1)+\
            torch.repeat_interleave(self.pos_embed_time_enc,N,dim=1)
        pos_embed = pos_embed.contiguous().expand(B,-1,-1).view(B,T,N,d_e)
        pos_embed_token = self.pos_embed_cls_enc.expand(B,T,-1,-1)
        pos_embed_x = torch.gather(
            pos_embed,dim=2,index = ids_keep.unsqueeze(-1).expand(-1,-1,-1,d_e)
        )
        pos_embed = torch.cat([pos_embed_token,pos_embed_x],dim=2)
        x=x+pos_embed.contiguous().view(B,-1,d_e)
        attn_bias = self.add_token_distance(attn_bias,x.device)
        attn_bias = self.shuffle_attn_bias(attn_bias,ids_keep,x_shape,n_tokens)
        x, attn = self.blocks(x,attn_bias)
        x=x.contiguous().transpose(0,1)
        x=self.norm(x)
        graph_rep,x = self.get_graph_rep(x,x_shape)
        x = x.contiguous().view(B,-1,d_e)
        return x, graph_rep,mask,ids_restore,x_shape
    def forward_decoder(self,x,ids_restore,x_shape,attn_bias:torch.Tensor):
        B,T,N,d_d = x_shape
        x=self.detodd(x)
        d_d = x.shape[-1]
        mask_tokens = self.mask_token.repeat(B,T*N-x.shape[1],1)
        x=torch.cat([x[:,:,:],mask_tokens],dim=1)
        
        assert ids_restore.shape[1]*ids_restore.shape[2] == T*N
        x=x.contiguous().view(B,T,N,d_d)
        x=torch.gather(x,dim=2,index=ids_restore.unsqueeze(-1).expand(-1,-1,-1,d_d))
        graph_token_feature = self.decoder_graph_token_embed.expand(B,T,-1,-1)
        x=torch.cat([graph_token_feature,x],dim=2)
        x=x.contiguous().view(B,-1,d_d)
        attn_bias = self.add_token_distance(attn_bias,x.device)
        attn_bias = attn_bias.repeat(1,1,T,T)
        decoder_pos_embed = self.pos_embed_space_dec.repeat(1,T,1)+self.pos_embed_time_dec.repeat_interleave(N,dim=1)
        decoder_pos_embed = decoder_pos_embed.expand(B,-1,-1)
        decoder_pos_embed_token=self.pos_embed_cls_dec.expand(B,T,-1,-1)
        decoder_pos_embed = torch.cat(
            [
                decoder_pos_embed_token,
                decoder_pos_embed.contiguous().view(B,T,N,d_d)
            ],dim=2
        )
        deco_pos = decoder_pos_embed.contiguous().view(B,-1,d_d)
        x=x+decoder_pos_embed.contiguous().view(B,-1,d_d)
        x,attn = self.decoder_blocks(x,attn_bias)
        x=x.contiguous().transpose(0,1)
        x=self.dec_norm(x)
        
        _,x = self.get_graph_rep(x,(B,T,N,d_d))
        x=x.contiguous().view(B,T,N,-1).transpose(1,3)
        x=self.dec_conv_1(x)
        x=self.dec_b_norm(x)
        x=self.activation(x)
        x=self.dec_conv_2(x)
        x=x.transpose(1,3)
        x=x.contiguous().view(B,-1,self.C_out)
        return x
    def forward_loss(self,orig_x,pred,mask):
        B,T,N,C = orig_x.shape
        orig_x = orig_x.contiguous().view(B,-1,self.C_out)
        loss = (pred-orig_x)**2
        loss = loss.mean(dim=-1)
        mask = mask.view(loss.shape)
        loss = (loss*mask).sum()/mask.sum()
        return loss
    def forward(self,x,mask_ratio):
        x_orig = x.clone()
        x,attn_bias=self.blocks.compute_mods(x)
        latent,_,mask,ids_restore,x_shape = self.forward_encoder(x,attn_bias.clone(),mask_ratio)
        pred = self.forward_decoder(latent,ids_restore,x_shape,attn_bias)
        loss = self.forward_loss(x_orig,pred,mask)
        return loss,pred,mask