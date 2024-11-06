import os, sys
sys.path.append(os.getcwd())
import pprint
from tqdm import tqdm, trange
import numpy as np
import os
from collections import OrderedDict, defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from first_dsnet_all.config import TestOptions
from first_dsnet_all.model import build_model
from first_dsnet_all.span_utils import span_cxw_to_xx
from first_dsnet_all.start_end_dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from first_dsnet_all.postprocessing_moment_detr import PostProcessorDETR
from standalone_eval.eval_dsnet import eval_submission
from utils.basic_utils import save_jsonl, save_json
from utils.temporal_nms import temporal_nms

# dsnet

from dsnet.src.anchor_based.losses import calc_cls_loss, calc_loc_loss

from dsnet.src.anchor_based import anchor_helper
# from dsnet.src.helpers import data_helper, vsumm_helper, bbox_helper
from dsnet.src.helpers import bbox_helper, data_helper

import copy
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    # import ipdb; ipdb.set_trace()
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    # import ipdb; ipdb.set_trace()
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val", "test"]:  # since test_public has no GT
        print('gt_data',len(gt_data))
        metrics = eval_submission(
            submission, gt_data,
            verbose=opt.debug, match_number=not opt.debug
        )
        
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug, match_number=not opt.debug
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    # mr_res = []
    bool_eval = True
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        # -----------------------------dsnet 添加方法 --------------------------------
        # build anchors 
        picks = model_inputs['picks']
        picks_mask = model_inputs['picks_mask']
        picks_data = picks.masked_fill(picks_mask == 0,-1)
        # picks_shape = list(model_inputs['picks'].shape)
        picks_shape = list(picks_data.shape)
        
        # ground_truth_span = model_inputs['ground_truth_span'].cpu().numpy()
        ground_truth_span = model_inputs['ground_truth_span']
        ground_truth_span_mask = model_inputs['ground_truth_span_mask']
        ground_truth_span_data = ground_truth_span.masked_fill(ground_truth_span_mask == 0,-1)
        ground_truth_span_data = ground_truth_span_data.cpu().numpy()
        # import ipdb; ipdb.set_trace()
        # picks_shape = [model_inputs['picks'].shape[0], model_inputs['picks'].shape[1]]
        
        # picks_shapes = [model_inputs['picks'].shape[0], model_inputs['picks'].shape[1]]
        anchors = anchor_helper.get_anchors(picks, picks_shape, opt.anchor_scales) # (bs, center, anchor_scales, 2)
        target_bboxes = bbox_helper.seq2bbox(ground_truth_span_data) # (bs, left, right)
        target_bboxes = bbox_helper.lr2cw(target_bboxes) # (bs, center, length)
        # Get class and location label for positive samples
        cls_label, loc_label, adapted_sample_dict= anchor_helper.get_pos_label(anchors, target_bboxes, opt.pos_iou_thresh)
         
        # Get negative samples
        num_pos = np.sum(cls_label, axis=(1,2))
        num_neg = np.array(num_pos * opt.neg_sample_ratio).astype(int)
        cls_label_neg, _, _ = anchor_helper.get_pos_label(anchors, target_bboxes, opt.neg_iou_thresh)
        cls_label_neg = anchor_helper.get_neg_label(cls_label_neg, num_neg)
        
        # Get incomplete samples
        incomplete_sample_ratio = np.array(num_pos * opt.incomplete_sample_ratio).astype(int)
        
        cls_label_incomplete, _, _ = anchor_helper.get_pos_label(anchors, target_bboxes, opt.incomplete_iou_thresh, adapted_sample_dict)
        cls_label_incomplete[cls_label_neg != 1] = 1
        cls_label_incomplete = anchor_helper.get_neg_label(cls_label_incomplete, incomplete_sample_ratio)

        cls_label[cls_label_neg == -1] = -1
        cls_label[cls_label_incomplete == -1] = -1

        cls_label = torch.tensor(cls_label, dtype=torch.float32).to(opt.device) # torch.Size([32, 75, 5])
        loc_label = torch.tensor(loc_label, dtype=torch.float32).to(opt.device) # torch.Size([32, 75, 5, 2])

        targets['cls_label'] = cls_label
        targets['loc_label'] = loc_label

        
        outputs = model(**model_inputs)
        
        # dsnet predict 部分
        stats = data_helper.AverageMeter('fscore', 'diversity')
        # import ipdb; ipdb.set_trace()
        pred_cls = outputs['pred_cls'].cpu().numpy()
        pred_cls= pred_cls.reshape(len(pred_cls),-1)
        pred_loc = outputs['pred_loc'].cpu().numpy()
        pred_loc = pred_loc.reshape((len(pred_loc), -1, 2))  

        anchors = anchor_helper.get_anchors(picks, picks_shape, opt.anchor_scales)
        anchors = anchors.reshape((len(batch[0]),-1, 2))

        pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
        
        for i in range(len(pred_bboxes)):    
            pred_bboxes[i] = bbox_helper.cw2lr(pred_bboxes[i])
        # post-process
        pred_bboxes = np.clip(pred_bboxes, 0, query_meta[0]["duration"]).round().astype(np.int32)
        pred_cls_nms_list = []
        pred_bboxes_nms_list = []
        for i in range(len(pred_bboxes)):
            pred_cls_nms, pred_bboxes_nms = bbox_helper.nms(pred_cls[i], pred_bboxes[i], opt.nms_thresh)
            pred_cls_nms_list.append(pred_cls_nms)
            pred_bboxes_nms_list.append(pred_bboxes_nms)
            # pred_summ = vsumm_helper.bbox2summary_v2(query_meta[i]['duration'], pred_cls_nms, pred_bboxes_nms)
        pred_bboxes_cls_nms = [np.concatenate((arr1, arr2.reshape(-1,1)),axis=1) for arr1, arr2 in zip(pred_bboxes_nms_list, pred_cls_nms_list)]
        # import ipdb; ipdb.set_trace()
        
        # pred_summ = vsumm_helper.bbox2summary(seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

        # pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, opt.nms_thresh)
        # import ipdb; ipdb.set_trace()
        # loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label)
        # cls_loss = calc_cls_loss(pred_cls, cls_label)
        
        # loss = cls_loss + opt.lambda_reg * loc_loss

#         -----------------------------dsnet predict 部分结束 -
    #     prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
    #     if opt.span_loss_type == "l1":
    #         scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take ict
    #         pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)
    #         _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
    #         saliency_scores = []
    #         valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
    #         for j in range(len(valid_vid_lengths)):
    #             saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
    #     else:
    #         bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
    #         pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
    #         # TODO use more advanced decoding method with st_ed product
    #         pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
    #         scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
    #         pred_spans[:, 1] += 1
    #         pred_spans *= opt.clip_length
    #     # compose predictions
    #     # import ipdb; ipdb.set_trace()
    #     mr_res = []
    #     for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
    #         if opt.span_loss_type == "l1":
    #             spans = span_cxw_to_xx(spans) * meta["duration"]
    #         # # (#queries, 3), [st(float), ed(float), score(float)]
    #         cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
    #         if not opt.no_sort_results:
    #             cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
    #         cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
    #         cur_query_pred = dict(
    #             qid=meta["qid"],
    #             query=meta["query"],
    #             vid=meta["vid"],
    #             pred_relevant_windows=cur_ranked_preds,
    #             pred_saliency_scores=saliency_scores[idx]
    #         )
    #         mr_res.append(cur_query_pred)
    #     if criterion:
    #         loss_dict = criterion(outputs, targets)
    #         weight_dict = criterion.weight_dict
    #         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    #         loss_dict["loss_overall"] = float(losses)  # for logging only
    #         for k, v in loss_dict.items():
    #             loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    #     if opt.debug:
    #         break

    # if write_tb and criterion:
    #     for k, v in loss_meters.items():
    #         tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)
    # post_processor = PostProcessorDETR(
    #     clip_length=2, min_ts_val=0, max_ts_val=150,
    #     min_w_l=2, max_w_l=150, move_window_method="left",
    #     process_func_names=("clip_ts", "round_multiple")
    # )
    # mr_res = post_processor(mr_res)
    # mr_res_dsnet = copy.deepcopy(mr_res)
    # for i in range(len(mr_res)):
    #     print(i)
    #     print('mr_res',len(mr_res))
    #     print('pred_bboxes_nms_list',len(pred_bboxes_nms_list))
    #     line =mr_res_dsnet[i]
    #     line['pred_relevant_windows_dsnet'] = pred_bboxes_nms_list[i].tolist()
    #     # line['pred_relevant_windows_all'] = mr_res[i]['pred_relevant_windows'] + pred_bboxes_nms_list[i]
    #     line['pred_relevant_windows_score_dsnet'] = pred_cls_nms_list[i].tolist()
    #     line['pred_relevant_windows_score_all'] = pred_bboxes_cls_nms[i].tolist()

#  改写interence部分
        # prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
        # if opt.span_loss_type == "l1":
        #     scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take ict
        #     pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)
        #     _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        #     saliency_scores = []
        #     valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        #     for j in range(len(valid_vid_lengths)):
        #         saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
        # else:
        #     bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
        #     pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
        #     # TODO use more advanced decoding method with st_ed product
        #     pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
        #     scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
        #     pred_spans[:, 1] += 1
        #     pred_spans *= opt.clip_length
        # compose predictions
        # import ipdb; ipdb.set_trace()
        mr_res_revise = []
        for idx, meta in enumerate(query_meta):
            # if opt.span_loss_type == "l1":
            #     spans = span_cxw_to_xx(spans) * meta["duration"]
            # # (#queries, 3), [st(float), ed(float), score(float)]
            # cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            # if not opt.no_sort_results:
            #     cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            # cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                # pred_relevant_windows=cur_ranked_preds,
                # pred_saliency_scores=saliency_scores[idx]
            )
            mr_res_revise.append(cur_query_pred)
        mr_res_dsnet = copy.deepcopy(mr_res_revise)
        for i in range(len(mr_res_revise)):
            # print(i)
            # print('mr_res_revise',len(mr_res_revise))
            # print('pred_bboxes_nms_list',len(pred_bboxes_nms_list))
            line =mr_res_dsnet[i]
            line['pred_relevant_windows_dsnet'] = pred_bboxes_nms_list[i].tolist()
            # line['pred_relevant_windows_all'] = mr_res_revise[i]['pred_relevant_windows'] + pred_bboxes_nms_list[i]
            line['pred_relevant_windows_score_dsnet'] = pred_cls_nms_list[i].tolist()
            line['pred_relevant_windows_score_all'] = pred_bboxes_cls_nms[i].tolist()

        # import ipdb; ipdb.set_trace()
        if criterion:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        if opt.debug:
            break

    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)
    post_processor = PostProcessorDETR(
        clip_length=2, min_ts_val=0, max_ts_val=150,
        min_w_l=2, max_w_l=150, move_window_method="left",
        process_func_names=("clip_ts", "round_multiple")
    )
    # mr_res_revise = post_processor(mr_res_revise)
    # import ipdb; ipdb.set_trace()

    return mr_res_revise, mr_res_dsnet, loss_meters


def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    # eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)  # list(dict)
    eval_res, eval_res_dsnet, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)  # list(dict)
    return eval_res, eval_res_dsnet, eval_loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    submission, submission_dsnet, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)
    if opt.no_sort_results:
        save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
    metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
        submission_dsnet, opt, eval_dataset.data, save_submission_filename)
    return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)
        
    # import ipdb; ipdb.set_trace()
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    # optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    optimizer = torch.optim.Adam(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, criterion, optimizer, lr_scheduler


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0
    )

    # import ipdb; ipdb.set_trace()
    model, criterion, _, _ = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_preds.jsonl".format(
        opt.dset_name, opt.eval_split_name, opt.eval_id)
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion=criterion)
    logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))


if __name__ == '__main__':
    start_inference()
