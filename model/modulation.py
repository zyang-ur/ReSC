from collections import OrderedDict
import math
import random
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models
from torch.nn.init import kaiming_normal, kaiming_uniform
from .darknet import ConvBatchNormReLU, ConvBatchNormReLU_3d

def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            init_params(m.weight)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        # gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        # betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas

def mask_softmax(attn_score, word_mask, tempuature=10., clssep=False, lstm=False):
    if len(attn_score.shape)!=2:
        attn_score = attn_score.squeeze(2).squeeze(2)
    word_mask_cp = word_mask[:,:attn_score.shape[1]].clone()
    score = F.softmax(attn_score*tempuature, dim=1)
    if not clssep:
        for ii in range(word_mask_cp.shape[0]):
            if lstm:
                word_mask_cp[ii,word_mask_cp[ii,:].sum()-1]=0
            else:
                word_mask_cp[ii,0]=0
                word_mask_cp[ii,word_mask_cp[ii,:].sum()]=0 ## set one to 0 already
    mask_score = score * word_mask_cp.float()
    mask_score = mask_score/(mask_score.sum(1)+1e-8).view(mask_score.size(0), 1).expand(mask_score.size(0), mask_score.size(1))
    return mask_score

class FiLMedConvBlock_context(nn.Module):
    def __init__(self, with_residual=True, with_batchnorm=True,
                             with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
                             with_input_proj=1, num_cond_maps=8, kernel_size=1, batchnorm_affine=False,
                             num_layers=1, condition_method='bn-film', debug_every=float('inf'),
                             textdim=768,visudim=512,contextdim=512,emb_size=512,fusion='prod',cont_map=False,
                             lstm=False,baseline=False):
        super(FiLMedConvBlock_context, self).__init__()

        self.cont_map = cont_map    ## mapping context with language feature
        self.lstm = lstm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.fusion = fusion
        self.baseline = baseline
        self.film = FiLM()

        if self.cont_map:
            self.sent_map = nn.Linear(768, emb_size)
            self.context_map = nn.Linear(emb_size, emb_size)
        if self.fusion == 'cat':
            self.attn_map = nn.Conv1d(textdim+visudim, emb_size//2, kernel_size=1)
        elif self.fusion == 'prod':
            assert(textdim==visudim) ## if product fusion
            self.attn_map = nn.Conv1d(visudim, emb_size//2, kernel_size=1)

        self.attn_score = nn.Conv1d(emb_size//2, 1, kernel_size=1)
        if self.baseline:
            self.fusion_layer = ConvBatchNormReLU(visudim+textdim+8, emb_size, 1, 1, 0, 1)
        else:
            self.gamme_decode = nn.Linear(textdim, 2 * emb_size)
            self.conv1 = nn.Conv2d(visudim+8, emb_size, kernel_size=1)
            # self.bn1 = nn.BatchNorm2d(emb_size)
            self.bn1 = nn.InstanceNorm2d(emb_size)
        init_modules(self.modules())


    def forward(self, fvisu, fword, context_score, fcoord, textattn=None,weight=None,fsent=None,word_mask=None):
        fword = fword.permute(0, 2, 1)
        B, Dvisu, H, W = fvisu.size()
        B, Dlang, N = fword.size()
        B, N = context_score.size()
        assert(Dvisu==Dlang)

        if self.cont_map and fsent is not None:
            fsent = F.normalize(F.relu(self.sent_map(fsent)), p=2, dim=1)
            fcont = torch.matmul(context_score.view(B,1,N),fword.permute(0,2,1)).squeeze(1)
            fcontext = F.relu(self.context_map(fsent*fcont)).unsqueeze(2).repeat(1,1,N)
            ## word attention
            tile_visu = torch.mean(fvisu.view(B, Dvisu, -1),dim=2,keepdim=True).repeat(1,1,N)
            if self.fusion == 'cat':
                context_tile = torch.cat([tile_visu,\
                    fword, fcontext], dim=1)
            elif self.fusion == 'prod':
                context_tile = tile_visu * \
                    fword * fcontext
        else:
            ## word attention
            tile_visu = torch.mean(fvisu.view(B, Dvisu, -1),dim=2,keepdim=True).repeat(1,1,N)
            if self.fusion == 'cat':
                context_tile = torch.cat([tile_visu,\
                    fword * context_score.view(B, 1, N).repeat(1, Dlang, 1,)], dim=1)
            elif self.fusion == 'prod':
                context_tile = tile_visu * \
                    fword * context_score.view(B, 1, N).repeat(1, Dlang, 1,)

        attn_feat = F.tanh(self.attn_map(context_tile))
        attn_score = self.attn_score(attn_feat).squeeze(1)
        mask_score = mask_softmax(attn_score,word_mask,lstm=self.lstm)
        attn_lang = torch.matmul(mask_score.view(B,1,N),fword.permute(0,2,1))
        attn_lang = attn_lang.view(B,Dlang).squeeze(1)

        if self.baseline:
            fmodu = self.fusion_layer(torch.cat([fvisu,\
                attn_lang.unsqueeze(2).unsqueeze(2).repeat(1,1,fvisu.shape[-1],fvisu.shape[-1]),fcoord],dim=1))
        else:
            ## lang-> gamma, beta
            film_param = self.gamme_decode(attn_lang)
            film_param = film_param.view(B,2*self.emb_size,1,1).repeat(1,1,H,W)
            gammas, betas = torch.split(film_param, self.emb_size, dim=1)
            gammas, betas = F.tanh(gammas), F.tanh(betas)

            ## modulate visu feature
            fmodu = self.bn1(self.conv1(torch.cat([fvisu,fcoord],dim=1)))
            fmodu = self.film(fmodu, gammas, betas)
            fmodu = F.relu(fmodu)
        if self.with_residual:
            if weight is None:
                fmodu = fvisu + fmodu
            else:
                weight = weight.view(B,1,1,1).repeat(1, Dvisu, H, W)
                fmodu = (1-weight)*fvisu + weight*fmodu
        return fmodu, attn_lang, attn_score

class FiLMedConvBlock_multihop(nn.Module):
    def __init__(self, NFilm=2, with_residual=True, with_batchnorm=True,
                             with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
                             with_input_proj=1, num_cond_maps=8, kernel_size=1, batchnorm_affine=False,
                             num_layers=1, condition_method='bn-film', debug_every=float('inf'),
                             textdim=768,visudim=512,emb_size=512,fusion='cat',intmd=False,lstm=False,erasing=0.):
        super(FiLMedConvBlock_multihop, self).__init__()

        self.NFilm = NFilm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.cont_size = emb_size
        self.fusion = fusion
        self.intmd = intmd
        self.lstm = lstm
        self.erasing = erasing
        if self.fusion=='cat':
            self.cont_size = emb_size*2

        self.modulesdict = nn.ModuleDict()
        modules = OrderedDict()
        modules["film0"] = FiLMedConvBlock_context(textdim=textdim,visudim=emb_size,contextdim=emb_size,emb_size=emb_size,fusion=fusion,lstm=self.lstm)
        for n in range(1,NFilm):
            modules["conv%d"%n] = ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1)
            modules["film%d"%n] = FiLMedConvBlock_context(textdim=textdim,visudim=emb_size,contextdim=self.cont_size,emb_size=emb_size,fusion=fusion,lstm=self.lstm)
        self.modulesdict.update(modules)

    def forward(self, fvisu, fword, fcoord, weight=None,fsent=None,word_mask=None):
        B, Dvisu, H, W = fvisu.size()
        B, N, Dlang = fword.size()
        intmd_feat, attnscore_list = [], []

        x, _, attn_score = self.modulesdict["film0"](fvisu, fword, Variable(torch.ones(B,N).cuda()), fcoord, fsent=fsent,word_mask=word_mask)
        attnscore_list.append(attn_score.view(B,N,1,1))
        if self.intmd:
            intmd_feat.append(x)
        if self.NFilm==1:
            intmd_feat = [x]
        for n in range(1,self.NFilm):
            score_list = [mask_softmax(score.squeeze(2).squeeze(2),word_mask,lstm=self.lstm) for score in attnscore_list]

            score = torch.clamp(torch.max(torch.stack(score_list, dim=1), dim=1, keepdim=False)[0],min=0.,max=1.)
            x = self.modulesdict["conv%d"%n](x)
            x, _, attn_score = self.modulesdict["film%d"%n](x, fword, (1-score), fcoord, fsent=fsent,word_mask=word_mask)
            attnscore_list.append(attn_score.view(B,N,1,1)) ## format match div loss in main func
            if self.intmd:
                intmd_feat.append(x)
            elif n==self.NFilm-1:
                intmd_feat = [x]
        return intmd_feat, attnscore_list