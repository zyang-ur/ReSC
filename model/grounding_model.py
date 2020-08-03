from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .darknet import *
from .convlstm import *
from .modulation import *

import argparse
import collections
import logging
import json
import re
import time
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

class grounding_model_multihop(nn.Module):
    def __init__(self, corpus=None, emb_size=256, jemb_drop_out=0.1, bert_model='bert-base-uncased', \
        NFilm=2, fusion='prod', intmd=False, mstage=False, convlstm=False, \
        coordmap=True, leaky=False, dataset=None, bert_emb=False, tunebert=False):
        super(grounding_model_multihop, self).__init__()
        self.coordmap = coordmap
        self.emb_size = emb_size
        self.NFilm = NFilm
        self.intmd = intmd
        self.mstage = mstage
        self.convlstm = convlstm
        self.tunebert = tunebert
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024
        ## Visual model
        self.visumodel = Darknet(config_path='./model/yolov3.cfg')
        self.visumodel.load_weights('./saved_models/yolov3.weights')
        ## Text model
        self.textmodel = BertModel.from_pretrained(bert_model)

        ## Mapping module
        self.mapping_visu = ConvBatchNormReLU(512 if self.convlstm else 256, emb_size, 1, 1, 0, 1, leaky=leaky)

        self.mapping_lang = torch.nn.Sequential(
          nn.Linear(self.textdim, emb_size),
          nn.ReLU(),
          nn.Dropout(jemb_drop_out),
          nn.Linear(emb_size, emb_size),
          nn.ReLU(),)

        textdim=emb_size
        self.film = FiLMedConvBlock_multihop(NFilm=NFilm,textdim=textdim,visudim=emb_size,\
            emb_size=emb_size,fusion=fusion,intmd=(intmd or mstage or convlstm))

        ## output head
        output_emb = emb_size
        if self.mstage:
            self.fcn_out = nn.ModuleDict()
            modules = OrderedDict()
            for n in range(0,NFilm):
                modules["out%d"%n] = torch.nn.Sequential(
                    ConvBatchNormReLU(output_emb, output_emb//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(output_emb//2, 9*5, kernel_size=1))
            self.fcn_out.update(modules)
        else:
            if self.intmd: 
                output_emb = emb_size*NFilm
            if self.convlstm:
                output_emb = emb_size
                self.global_out = ConvLSTM(input_size=(32, 32),
                     input_dim=emb_size,
                     hidden_dim=[emb_size],
                     kernel_size=(1, 1),
                     num_layers=1,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False)
            self.fcn_out = torch.nn.Sequential(
                    ConvBatchNormReLU(output_emb, output_emb//2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(output_emb//2, 9*5, kernel_size=1))

    def forward(self, image, word_id, word_mask):
        ## Visual Module
        batch_size = image.size(0)
        raw_fvisu = self.visumodel(image)
        if self.convlstm:
            raw_fvisu = raw_fvisu[1]
        else:
            raw_fvisu = raw_fvisu[2]
        fvisu = self.mapping_visu(raw_fvisu)
        raw_fvisu = F.normalize(fvisu, p=2, dim=1)

        ## Language Module
        all_encoder_layers, _ = self.textmodel(word_id, \
            token_type_ids=None, attention_mask=word_mask)
        ## Sentence feature at the first position [cls]
        raw_flang = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:]\
             + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
        raw_fword = (all_encoder_layers[-1] + all_encoder_layers[-2]\
             + all_encoder_layers[-3] + all_encoder_layers[-4])/4
        if not self.tunebert:
            ## fix bert during training
            # raw_flang = raw_flang.detach()
            hidden = raw_flang.detach()
            raw_fword = raw_fword.detach()

        fword = Variable(torch.zeros(raw_fword.shape[0],raw_fword.shape[1],self.emb_size).cuda())
        for ii in range(raw_fword.shape[0]):
            ntoken = (word_mask[ii] != 0).sum()
            fword[ii,:ntoken,:] = F.normalize(self.mapping_lang(raw_fword[ii,:ntoken,:]), p=2, dim=1)
            ## [CLS], [SEP]
            # fword[ii,1:ntoken-1,:] = F.normalize(self.mapping_lang(raw_fword[ii,1:ntoken-1,:].view(-1,self.textdim)), p=2, dim=1)
        raw_fword = fword

        coord = generate_coord(batch_size, raw_fvisu.size(2), raw_fvisu.size(3))
        x, attnscore_list = self.film(raw_fvisu, raw_fword, coord,fsent=None,word_mask=word_mask)
        if self.mstage:
            outbox = []
            for film_ii in range(len(x)):
                outbox.append(self.fcn_out["out%d"%film_ii](x[film_ii]))
        elif self.convlstm:
            x = torch.stack(x, dim=1)
            output, state = self.global_out(x)
            output, hidden, cell = output[-1], state[-1][0], state[-1][1]
            outbox = [self.fcn_out(hidden)]
        else:
            x = torch.stack(x, dim=1).view(batch_size,-1,raw_fvisu.size(2), raw_fvisu.size(3))
            outbox = [self.fcn_out(x)]
        return outbox, attnscore_list   ## list of (B,N,H,W)