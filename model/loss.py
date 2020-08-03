import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model.modulation import mask_softmax
from utils.utils import bbox_iou

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def lr_cos(base_lr, iter, max_iter, warm_up=0.05):
    warm_up_epoch = int(max_iter*warm_up)
    if iter<=warm_up_epoch:
        lr = base_lr*(0.8*iter/warm_up_epoch+0.2)
    else:
        lr = 0.5*base_lr*(1+math.cos(math.pi*(iter-warm_up_epoch)/(max_iter-warm_up_epoch)))
    return lr

def adjust_learning_rate(args, optimizer, i_iter):
    # print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
    if args.power==-1:
        lr = lr_cos(args.lr, i_iter, args.nb_epoch)
    elif args.power!=0.:
        lr = lr_poly(args.lr, i_iter, args.nb_epoch, args.power)
    else:
        # lr = args.lr*((0.1)**(i_iter//(args.nb_epoch//4)))
        lr = args.lr*((0.5)**(i_iter//(args.nb_epoch//10)))
    print(lr)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
      optimizer.param_groups[1]['lr'] = lr / 10
    if len(optimizer.param_groups) > 2:
      optimizer.param_groups[2]['lr'] = lr / 10

def yolo_loss(input, target, gi, gj, best_n_list, w_coord=5., w_neg=1./5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    batch = input.size(0)

    pred_bbox = Variable(torch.zeros(batch,4).cuda())
    gt_bbox = Variable(torch.zeros(batch,4).cuda())
    for ii in range(batch):
        pred_bbox[ii, 0:2] = F.sigmoid(input[ii,best_n_list[ii],0:2,gj[ii],gi[ii]])
        pred_bbox[ii, 2:4] = input[ii,best_n_list[ii],2:4,gj[ii],gi[ii]]
        gt_bbox[ii, :] = target[ii,best_n_list[ii],:4,gj[ii],gi[ii]]
    loss_x = mseloss(pred_bbox[:,0], gt_bbox[:,0])
    loss_y = mseloss(pred_bbox[:,1], gt_bbox[:,1])
    loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
    loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])

    pred_conf_list, gt_conf_list = [], []
    pred_conf_list.append(input[:,:,4,:,:].contiguous().view(batch,-1))
    gt_conf_list.append(target[:,:,4,:,:].contiguous().view(batch,-1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x+loss_y+loss_w+loss_h)*w_coord + loss_conf

def diverse_loss(score_list, word_mask, m=-1, coverage_reg=True):
    score_matrix = torch.stack([mask_softmax(score,word_mask) for score in score_list], dim=1)    ## (B,Nfilm,N,H,W)
    cov_matrix = torch.bmm(score_matrix,score_matrix.permute(0,2,1))    ## (BHW,Nfilm,Nfilm)
    id_matrix = Variable(torch.eye(cov_matrix.shape[1]).unsqueeze(0).repeat(cov_matrix.shape[0],1,1).cuda())
    if m==-1.:
        div_reg = torch.sum(((cov_matrix*(1-id_matrix))**2).view(-1))/cov_matrix.shape[0]
    else:
        div_reg = torch.sum(((cov_matrix-m*id_matrix)**2).view(-1))/cov_matrix.shape[0]
    if coverage_reg:
        word_mask_cp = word_mask.clone()
        for ii in range(word_mask_cp.shape[0]):
            word_mask_cp[ii,0]=0
            word_mask_cp[ii,word_mask_cp[ii,:].sum()]=0 ## set one to 0 already
        cover_matrix = 1.-torch.clamp(torch.sum(score_matrix, dim=1, keepdim=False),min=0.,max=1.)
        cover_reg = torch.sum((cover_matrix*word_mask_cp.float()).view(-1))/cov_matrix.shape[0]
        div_reg += cover_reg
    return div_reg

def build_target(raw_coord, pred, anchors_full, args):
    coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
    batch, grid = raw_coord.size(0), args.size//args.gsize
    coord[:,0] = (raw_coord[:,0] + raw_coord[:,2])/(2*args.size)
    coord[:,1] = (raw_coord[:,1] + raw_coord[:,3])/(2*args.size)
    coord[:,2] = (raw_coord[:,2] - raw_coord[:,0])/(args.size)
    coord[:,3] = (raw_coord[:,3] - raw_coord[:,1])/(args.size)
    coord = coord * grid
    bbox=torch.zeros(coord.size(0),9,5,grid, grid)

    best_n_list, best_gi, best_gj = [],[],[]

    for ii in range(batch):
        batch, grid = raw_coord.size(0), args.size//args.gsize
        gi = coord[ii,0].long()
        gj = coord[ii,1].long()
        tx = coord[ii,0] - gi.float()
        ty = coord[ii,1] - gj.float()
        gw = coord[ii,2]
        gh = coord[ii,3]

        anchor_idxs = range(9)
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]

        ## Get shape of gt box
        gt_box = torch.FloatTensor(np.array([0, 0, gw, gh],dtype=np.float32)).unsqueeze(0)
        ## Get shape of anchor box
        anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
        ## Calculate iou between gt and anchor shapes
        # anch_ious = list(bbox_iou(gt_box, anchor_shapes))
        anch_ious = list(bbox_iou(gt_box, anchor_shapes,x1y1x2y2=False))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))

        tw = torch.log(gw / scaled_anchors[best_n][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n][1] + 1e-16)

        bbox[ii, best_n, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    bbox = Variable(bbox.cuda())
    return bbox, best_gi, best_gj, best_n_list