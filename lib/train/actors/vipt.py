import pdb
from . import BaseActor
from HOPL.lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from HOPL.lib.train.admin import multigpu
from os.path import join, isdir, abspath, dirname
from HOPL.lib.utils.ce_utils import generate_mask_cond
from HOPL.lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from HOPL.lib.train.admin import multigpu
from HOPL.lib.test.utils.hann import hann2d
import numpy as np
import torch

import cv2
prj = join(dirname(__file__), '..')

# sam = sam_model_registry["default"](checkpoint="/home/zl/sam_vit_h_4b8939.pth")
# device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# sam.to(device=device1)
# predictor = SamPredictor(sam)
class Frame:
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(prev, curr):
    return abs(curr - prev) / prev if prev != 0 else 0


def smooth(x, window_len=13):
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


def extract_keyframe_difference(frame11, frame22, len_window=50, use_thresh=True, thresh=0.87):
    frame1 = frame11
    frame1 = frame1.to('cpu')
    frame2 = frame22
    frame2= frame2.to('cpu')
    luv1 = frame1.numpy()
    luv2 = frame2.numpy()
    diff = cv2.absdiff(luv1, luv2)
    diff_sum = np.sum(diff)
    diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
    return diff_sum_mean

def _get_jittered_box(box):
    """ Jitter the input box
    args:
        box - input bounding box
        mode - string 'template' or 'search' indicating template or search data

    returns:
        torch.Tensor - jittered box
    """
    scale_jitter_factor=0.25
    center_jitter_factor=3
    jittered_size = box[2:4] * torch.exp(torch.randn(2) * scale_jitter_factor)
    max_offset = (jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def map_box_back(state, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * 256/ resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
def show_box1(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2))

def cal_bbox(score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // 16
        idx_x = idx % 16

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / 16,
                          (idx_y.to(torch.float) + offset[:, 1:]) / 16,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox
class ViPTActor(BaseActor):
    """ Actor for training ViPT models """


    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.output_window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()


    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()



    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)
        # bbox = out_dict['pred_boxes']
        # out = bbox.tolist()
        # for i in range(0,8):
        #     out1 = out[i][0]
        #     x1 = out1[0]
        #     y1 = out1[1]
        #     w1 = out1[2]
        #     h1 = out1[3]
        #     wc = w1 * 256
        #     hc = h1 * 256
        #     xc = x1 * 256
        #     yc = y1 * 256
        #     x11 = xc - wc / 2
        #     y11 = yc - hc / 2
        #     x22 = xc + wc / 2
        #     y22 = yc + hc / 2
        #     x11 = max(x11, 0)
        #     y11 = max(y11, 0)
        #
        #     image = data['search_images']
        #     img = image[:, i:i + 1, 0:3, :, :]
        #     img2 = image[:, i:i + 1, 6:9, :, :]
        #     samimg1 = img.squeeze()
        #     samimg2 = img2.squeeze()
        #     im1=samimg1.permute(1,2,0)
        #     im2=samimg2.permute(1,2,0)
        #     zjc=extract_keyframe_difference(im1,im2)
        #     print(zjc)
        #
        #     # mean = [0.485, 0.456, 0.406]
        #     # std = [0.229, 0.224, 0.225]
        #     # mean = torch.as_tensor(mean)
        #     # std = torch.as_tensor(std)
        #     # mean = mean.view(-1, 1, 1)
        #     # std = std.view(-1, 1, 1)
        #     # mean = mean.to('cuda:0')
        #     # std = std.to('cuda:0')
        #     # original = samimg1 * std
        #     # original = original + mean
        #     # brightness = 1.0195254015709299
        #     # showimg = original.mul(255.0 / brightness).clamp(0.255)
        #     # showimg = showimg.permute(1, 2, 0)
        #     # showimg1 = showimg
        #     # showimg1 = showimg1.to('cpu')
        #     # showimg1 = showimg1.numpy()
        #     # showimg1 = np.uint8(showimg1)
        #     # predictor.set_image(showimg1)
        #     #
        #     # input_box = np.array([x11, y11, x22, y22])
        #     # masks, f1, f2 = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :],
        #     #                                 multimask_output=False)
        #     # coords = np.argwhere(masks[0])
        #     # samw = x22 - x11
        #     # samh = y22 - y11
        #     # samx = x11
        #     # samy = y11
        #     # if len(coords) > 0:
        #     #     min_coords = np.min(coords, axis=0)
        #     #     max_coords = np.max(coords, axis=0)
        #     #     samw = max_coords[1] - min_coords[1] + 1
        #     #     samh = max_coords[0] - min_coords[0] + 1
        #     #     samx = min_coords[1]
        #     #     samy = min_coords[0]
        #     # samwc = samw / 2
        #     # samhc = samh / 2
        #     # samx = samx + samwc
        #     # samy = samy + samhc
        #     # samx = samx / 256
        #     # samy = samy / 256
        #     # samw = samw / 256
        #     # samh = samh / 256
        #     # samxc = (samx + x1) / 2
        #     # samyc = (samy + y1) / 2
        #     # samw = (samw + w1) / 2
        #     # samh = (samh + h1) / 2
        #     # bbox[i, 0, 0] = samxc
        #     # bbox[i, 0, 1] = samyc
        #     # bbox[i, 0, 2] = samw
        #     # bbox[i, 0, 3] = samh
        #out_dict['pred_boxes'] = bbox
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
            # ce_keep_rate = 0.7

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss