# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from first_dsnet_all.span_utils import generalized_temporal_iou, span_cxw_to_xx

from first_dsnet_all.matcher import build_matcher
from first_dsnet_all.transformer import build_transformer
from first_dsnet_all.position_encoding import build_position_encoding
from first_dsnet_all.misc import accuracy

from dsnet.src.anchor_based import anchor_helper
from dsnet.src.helpers import bbox_helper
from dsnet.src.modules.models import build_base_model

# class MomentDETR(nn.Module):
#     """ This is the Moment-DETR module that performs moment localization. """

#     def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
#                  num_queries, input_dropout, base_model_vid, base_model_txt, num_feature, num_hidden, anchor_scales, num_head,
#                  aux_loss=False, contrastive_align_loss=False, contrastive_hdim=64,
#                  max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2):
#         """ Initializes the model.
#         Parameters:
#             transformer: torch module of the transformer architecture. See transformer.py
#             position_embed: torch module of the position_embedding, See position_encoding.py
#             txt_position_embed: position_embedding for text
#             txt_dim: int, text query input dimension
#             vid_dim: int, video feature input dimension
#             num_queries: number of object queries, ie detection slot. This is the maximal number of objects
#                          Moment-DETR can detect in a single video.
#             aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
#             contrastive_align_loss: If true, perform span - tokens contrastive learning
#             contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
#             max_v_l: int, maximum #clips in videos
#             span_loss_type: str, one of [l1, ce]
#                 l1: (center-x, width) regression.
#                 ce: (st_idx, ed_idx) classification.
#             # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
#             # background_thd: float, intersection over prediction <= background_thd: labeled background
#         """
#         super().__init__()
#         self.num_queries = num_queries
#         self.transformer = transformer
#         self.position_embed = position_embed
#         self.txt_position_embed = txt_position_embed
#         hidden_dim = transformer.d_model
#         self.span_loss_type = span_loss_type
#         self.max_v_l = max_v_l
#         span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
#         self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
#         self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
#         self.use_txt_pos = use_txt_pos
#         self.n_input_proj = n_input_proj
#         # self.foreground_thd = foreground_thd
#         # self.background_thd = background_thd
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)

#         # dsnet
        
#         self.anchor_scales = anchor_scales
#         self.num_scales = len(anchor_scales)
#         self.base_model_vid = build_base_model(base_model_vid, vid_dim, num_head)
#         self.base_model_txt = build_base_model(base_model_txt, txt_dim, num_head)
        
#         self.semantics_self_attention = SemanticSelfAttention(hidden_dim, num_head)
#         self.cross_attention = CrossAttention(hidden_dim, num_head)

#         self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
#                              for scale in anchor_scales]

#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.fc1 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Dropout(0.5),
#             nn.LayerNorm(hidden_dim)
#         )
#         self.fc_cls = nn.Linear(hidden_dim, 1) # classification
#         self.fc_loc = nn.Linear(hidden_dim, 2) # location
        


        
#         relu_args = [True] * 3
#         relu_args[n_input_proj-1] = False
#         self.input_txt_proj = nn.Sequential(*[
#             LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
#             LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
#             LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
#         ][:n_input_proj])
#         self.input_vid_proj = nn.Sequential(*[
#             LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
#             LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
#             LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
#         ][:n_input_proj])
#         self.contrastive_align_loss = contrastive_align_loss
#         if contrastive_align_loss:
#             self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
#             self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
#             self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

#         self.saliency_proj = nn.Linear(hidden_dim, 1)
#         self.aux_loss = aux_loss



        

#     def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, picks, picks_mask,  ground_truth_span, ground_truth_span_mask):
#         """The forward expects two tensors:
#                - src_txt: [batch_size, L_txt, D_txt]
#                - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
#                     will convert to 1 as padding later for transformer
#                - src_vid: [batch_size, L_vid, D_vid]
#                - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
#                     will convert to 1 as padding later for transformer

#             It returns a dict with the following elements:
#                - "pred_spans": The normalized boxes coordinates for all queries, represented as
#                                (center_x, width). These values are normalized in [0, 1],
#                                relative to the size of each individual image (disregarding possible padding).
#                                See PostProcess for information on how to retrieve the unnormalized bounding box.
#                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
#                                 dictionnaries containing the two above keys for each decoder layer.
#         """
#         src_vid = self.input_vid_proj(src_vid)
#         src_txt = self.input_txt_proj(src_txt)
#         src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
#         mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
#         # TODO should we remove or use different positional embeddings to the src_txt?
#         pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
#         pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
#         # pos_txt = torch.zeros_like(src_txt)
#         # pad zeros for txt positions
#         pos = torch.cat([pos_vid, pos_txt], dim=1)
#         # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
#         hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
#         # import ipdb;ipdb.set_trace()
#         outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
#         outputs_coord = self.span_embed(hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
#         if self.span_loss_type == "l1":
#             outputs_coord = outputs_coord.sigmoid()
#         # out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
#         txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
#         vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
#         if self.contrastive_align_loss:
#             proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
#             proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
#             proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
#             out.update(dict(
#                 proj_queries=proj_queries[-1],
#                 proj_txt_mem=proj_txt_mem,
#                 proj_vid_mem=proj_vid_mem
#             ))

#         # out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)  # (bsz, L_vid)

#         # if self.aux_loss:
#         #     # assert proj_queries and proj_txt_mem
#         #     out['aux_outputs'] = [
#         #         {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
#         #     if self.contrastive_align_loss:
#         #         assert proj_queries is not None
#         #         for idx, d in enumerate(proj_queries[:-1]):
#         #             out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))


#         # dsnet 
#         bs, _, _ = src_vid.shape
#         # import ipdb; ipdb.set_trace()   
#         video_features = src_vid.transpose(0, 1) # torch.Size([75, 32, 256])
#         semantics_features = src_txt.transpose(0, 1) # torch.Size([20, 32, 256])
        
#         semantics_out = self.semantics_self_attention(semantics_features) # torch.Size([20, 32, 256])
#         cross_out = self.cross_attention(video_features, semantics_out) # torch.Size([75, 32, 256])
        
#         cross_out = cross_out + video_features # torch.Size([75, 32, 256])  
#         cross_out = self.layer_norm(cross_out) # torch.Size([75, 32, 256])
        
#         cross_out = cross_out.transpose(2, 1)
#         # import ipdb; ipdb.set_trace()
#         # pool_results = []
#         # for roi_pooling in self.roi_poolings:
#         #     pool = roi_pooling(cross_out)
#         #     pool_results.append(pool)
#         pool_results = [roi_pooling(cross_out) for roi_pooling in self.roi_poolings]
#         cross_out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]
#         cross_out = self.fc1(cross_out)
#         # pred_cls = self.fc_cls(cross_out).sigmoid()
#         # pred_loc = self.fc_loc(cross_out)
#         pred_cls = self.fc_cls(cross_out).sigmoid().view(bs, -1, self.num_scales)
#         pred_loc = self.fc_loc(cross_out).view(bs, -1, self.num_scales, 2)
#         print('pred_cls',pred_cls)
#         out = {}
#         out['pred_cls'] = pred_cls
#         out['pred_loc'] = pred_loc
        
#         return out

#     # @torch.jit.unused
#     # def _set_aux_loss(self, outputs_class, outputs_coord):
#     #     # this is a workaround to make torchscript happy, as torchscript
#     #     # doesn't support dictionary with non-homogeneous values, such
#     #     # as a dict having both a Tensor and a list.
#     #     return [{'pred_logits': a, 'pred_spans': b}
#     #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

#     def predict(self, seq):
#         seq_len = seq.shape[1]
#         pred_cls, pred_loc = self(seq)

#         pred_cls = pred_cls.cpu().numpy().reshape(-1)
#         pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))

#         anchors = anchor_helper.get_anchors(seq_len, self.anchor_scales)
#         anchors = anchors.reshape((-1, 2))

#         pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
#         pred_bboxes = bbox_helper.cw2lr(pred_bboxes)

#         return pred_cls, pred_bboxes


class DsnetDETR(nn.Module):
    """ This is the Moment-DETR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, base_model_vid, base_model_txt, num_feature, num_hidden, anchor_scales, num_head,
                 aux_loss=False, contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        # hidden_dim = 256
        # print('transformer.d_model',transformer.d_model)
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd
        self.query_embed = nn.Embedding(num_queries, hidden_dim)


        # dsnet
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model_vid = build_base_model(base_model_vid, vid_dim, num_head)
        self.base_model_txt = build_base_model(base_model_txt, txt_dim, num_head)
        
        self.semantics_self_attention = SemanticSelfAttention(hidden_dim, num_head)
        self.cross_attention = CrossAttention(hidden_dim, num_head)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.LayerNorm(hidden_dim)
        )
        self.fc_cls = nn.Linear(hidden_dim, 1) # classification
        self.fc_loc = nn.Linear(hidden_dim, 2) # location
        


        
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.aux_loss = aux_loss



        

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, picks, picks_mask,  ground_truth_span, ground_truth_span_mask):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
        hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        # import ipdb;ipdb.set_trace()
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))

        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)  # (bsz, L_vid)

        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))


        # dsnet 
        bs, _, _ = src_vid.shape
        # import ipdb; ipdb.set_trace()   
        # video_features = src_vid.transpose(0, 1) # torch.Size([75, 32, 256])
        # semantics_features = src_txt.transpose(0, 1) # torch.Size([20, 32, 256])
        
        video_features = memory[:, :src_vid.shape[1]].transpose(0, 1)
        semantics_features = memory[:, src_vid.shape[1]:].transpose(0, 1)
        semantics_out = self.semantics_self_attention(semantics_features) # torch.Size([20, 32, 256])
        cross_out = self.cross_attention(video_features, semantics_out) # torch.Size([75, 32, 256])
        
        cross_out = cross_out + video_features # torch.Size([75, 32, 256])  
        # cross_out = video_features


        # cross_out = self.layer_norm(cross_out) # torch.Size([75, 32, 256])
        cross_out = cross_out.transpose(0, 1)
        cross_out = cross_out.transpose(2, 1)
        # import ipdb; ipdb.set_trace()
        # pool_results = []
        # for roi_pooling in self.roi_poolings:
        #     pool = roi_pooling(cross_out)
        #     pool_results.append(pool)
        pool_results = [roi_pooling(cross_out) for roi_pooling in self.roi_poolings]
        cross_out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]
        cross_out = self.fc1(cross_out)
        # pred_cls = self.fc_cls(cross_out).sigmoid()
        # pred_loc = self.fc_loc(cross_out)
        pred_cls = self.fc_cls(cross_out).sigmoid().view(bs, -1, self.num_scales)
        pred_loc = self.fc_loc(cross_out).view(bs, -1, self.num_scales, 2)
        # print('pred_cls',pred_cls)
        # out = {}
        out['pred_cls'] = pred_cls
        out['pred_loc'] = pred_loc
        
        return out

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_spans': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        # import ipdb; ipdb.set_trace()
        assert 'pred_spans' in outputs
        import ipdb; ipdb.set_trace()
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_loc(self, outputs, targets, use_smooth: bool = True):
        """Compute location regression loss only on positive samples.

        :param pred_loc: Predicted bbox offsets. Sized [N, S, 2].
        :param test_loc: Ground truth bbox offsets. Sized [N, S, 2].
        :param cls_label: Class labels where the 1 marks the positive samples. Sized
            [N, S].
        :param use_smooth: If true, use smooth L1 loss. Otherwise, use L1 loss.
        :return: Scalar loss value.
        """
        pred_loc = outputs['pred_loc']
        local_label = targets['loc_label']
        test_loc = local_label
        cls_label = targets['cls_label']
        
        pos_idx = cls_label.eq(1).unsqueeze(-1).repeat((1, 1, 1, 2))

        pred_loc = pred_loc[pos_idx]
        test_loc = test_loc[pos_idx]

        if use_smooth:
            loc_loss = F.smooth_l1_loss(pred_loc, test_loc)
        else:
            loc_loss = (pred_loc - test_loc).abs().mean()
    
        losses = {"loss_loc": loc_loss}
        return losses

    def loss_cls(self, outputs, targets):
        """Compute classification loss.

        :param pred: Predicted confidence (0-1). Sized [N, S].
        :param test: Class label where 1 marks positive, -1 marks negative, and 0
            marks ignored. Sized [N, S].
        :return: Scalar loss value.
        """
        # import ipdb; ipdb.set_trace()
        pred = outputs['pred_cls']
        test = targets['cls_label']
        
        pred = pred.view(-1)
        test = test.view(-1)

        pos_idx = test.eq(1).nonzero().squeeze(-1)
        pred_pos = pred[pos_idx].unsqueeze(-1)
        pred_pos = torch.cat([1 - pred_pos, pred_pos], dim=-1)
        gt_pos = torch.ones(pred_pos.shape[0], dtype=torch.long, device=pred.device)
        loss_pos = F.nll_loss(pred_pos.log(), gt_pos)

        neg_idx = test.eq(-1).nonzero().squeeze(-1)
        pred_neg = pred[neg_idx].unsqueeze(-1)
        pred_neg = torch.cat([1 - pred_neg, pred_neg], dim=-1)
        gt_neg = torch.zeros(pred_neg.shape[0], dtype=torch.long,
                            device=pred.device)
        loss_neg = F.nll_loss(pred_neg.log(), gt_neg)

        loss = (loss_pos + loss_neg) * 0.5
        
        losses = {"loss_cls": loss}
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # import ipdb; ipdb.set_trace()
        # src_like_list = []
        # for i,(src,_) in enumerate(indices):
        #     src_like = torch.full_like(src, i)
        #     src_like_list.append(src_like)
        # batch_idx = torch.cat(src_like_list)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            # "spans": self.loss_spans,
            # "labels": self.loss_labels,
            # "contrastive_align": self.loss_contrastive_align,
            # "saliency": self.loss_saliency,
            "loc": self.loss_loc,
            "cls": self.loss_cls
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def get_loss_dsnet(self, loss, outputs, targets, **kwargs):
        loss_map = {
            # "spans": self.loss_spans,
            # "labels": self.loss_labels,
            # "contrastive_align": self.loss_contrastive_align,
            # "saliency": self.loss_saliency,
            "loc": self.loss_loc,
            "cls": self.loss_cls
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        # indices = self.matcher(outputs_without_aux, targets)
        # Compute all the requested losses
        # losses = {}
        # for loss in self.losses:
        #     import ipdb;ipdb.set_trace()
        #     test_loss = self.get_loss(loss, outputs, targets, indices)
            
        #     losses.update(test_loss)

# ------------------dsnet   --------------------
        losses = {}
        for loss in self.losses:
            # import ipdb;ipdb.set_trace()    
            test_loss = self.get_loss_dsnet(loss, outputs, targets)
            losses.update(test_loss)

            # losses.update(self.get_loss(loss, outputs, targets, indices))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if 'aux_outputs' in outputs:
        #     for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #         indices = self.matcher(aux_outputs, targets)
        #         for loss in self.losses:
        #             if "saliency" == loss:  # skip as it is only in the top layer
        #                 continue
        #             if loss == "cls" or loss == "loc":
        #                 continue
        #             kwargs = {}
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
        #             l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #             losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class SemanticSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SemanticSelfAttention, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
    def forward(self, semantic_features):
        # 输入需要是 [seq_len, batch_size, embed_dim] 的形式
        attn_output, _ = self.self_attention(semantic_features, semantic_features, semantic_features)
        return attn_output

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    def forward(self, semantic_features, visual_features):
        # 语义特征作为查询，视觉特征作为键和值
        attn_output, _ = self.cross_attention(semantic_features, visual_features, visual_features)
        return attn_output



def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = DsnetDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,

        base_model_vid=args.base_model, 
        base_model_txt=args.base_model, 
        num_feature=args.num_feature,
        num_hidden=args.num_hidden, 
        anchor_scales=args.anchor_scales,
        num_head=args.num_head
    )

    matcher = build_matcher(args)
    weight_dict = {
        # "loss_span": args.span_loss_coef,
                #    "loss_giou": args.giou_loss_coef,
                #    "loss_label": args.label_loss_coef,
                #    "loss_saliency": args.lw_saliency,
                   "loss_cls": args.cls_loss_coef,
                   "loss_loc":  args.loc_loss_coef
                   }
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    # losses = ['spans', 'labels', 'saliency', 'cls', 'loc']
    losses = ['cls', 'loc']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin
        
    )
    criterion.to(device)
    return model, criterion
