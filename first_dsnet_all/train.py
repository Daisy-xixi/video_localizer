import os, sys
sys.path.append(os.getcwd())
# print(os.getcwd())
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from first_dsnet_all.config import BaseOptions
from first_dsnet_all.start_end_dataset import \
    StartEndDataset, start_end_collate, prepare_batch_inputs
from first_dsnet_all.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters

# dsnet
from dsnet.src.anchor_based.losses import calc_cls_loss, calc_loc_loss

from dsnet.src.anchor_based import anchor_helper
# from dsnet.src.helpers import data_helper, vsumm_helper, bbox_helper
from dsnet.src.helpers import bbox_helper


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        max_length = [data['duration'] for data in batch[0]]
        # import ipdb; ipdb.set_trace()
        
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        # import ipdb; ipdb.set_trace()
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        timer_start = time.time()
        # import ipdb; ipdb.set_trace()
        # build anchors 待定 感觉还是要该点

        # import ipdb; ipdb.set_trace()
        picks = model_inputs['picks']
        picks_mask = model_inputs['picks_mask']
        picks_data = picks.masked_fill(picks_mask == 0, -1)
        # picks_shape = list(model_inputs['picks'].shape)
        picks_shape = list(picks_data.shape)
        
        # ground_truth_span = model_inputs['ground_truth_span'].cpu().numpy()
        ground_truth_span = model_inputs['ground_truth_span']
        ground_truth_span_mask = model_inputs['ground_truth_span_mask']
        ground_truth_span_data = ground_truth_span.masked_fill(ground_truth_span_mask == 0, -1)
        ground_truth_span_data = ground_truth_span_data.cpu().numpy()
        # import ipdb; ipdb.set_trace()
        # picks_shape = [model_inputs['picks'].shape[0], model_inputs['picks'].shape[1]]
        
        # picks_shapes = [model_inputs['picks'].shape[0], model_inputs['picks'].shape[1]]
        anchors = anchor_helper.get_anchors(picks, picks_shape, opt.anchor_scales) # (bs, center, anchor_scales, 2) (32, 75, 5, 2)
        preprocess_anchors = anchor_helper.preprocess_anchors(anchors, max_length)
        
        target_bboxes = bbox_helper.seq2bbox(ground_truth_span_data) # (bs, left, right) 左右端点
        target_bboxes = bbox_helper.lr2cw(target_bboxes) # (bs, center, length)
        # Get class and location label for positive samples
        # duration = model_inputs['duration']
        cls_label, loc_label, adapted_sample_dict = anchor_helper.get_pos_label(preprocess_anchors, target_bboxes, opt.pos_iou_thresh)
         
        # Get negative samples
        num_pos = np.sum(cls_label, axis=(1,2))
        num_neg = np.array(num_pos * opt.neg_sample_ratio).astype(int)
        cls_label_neg, _, _ = anchor_helper.get_pos_label(preprocess_anchors, target_bboxes, opt.neg_iou_thresh)
        # import ipdb; ipdb.set_trace()
        
        cls_label_neg = anchor_helper.get_neg_label(cls_label_neg, num_neg)
        # num_neg = np.sum(cls_label_neg, axis=(1,2))
        # import ipdb; ipdb.set_trace()
        
        # Get incomplete samples
        incomplete_sample_ratio = np.array(num_pos * opt.incomplete_sample_ratio).astype(int)
        
        cls_label_incomplete, _, _ = anchor_helper.get_pos_label(preprocess_anchors, target_bboxes, opt.incomplete_iou_thresh, adapted_sample_dict)
        cls_label_incomplete[cls_label_neg != 1] = 1
        cls_label_inconmplete = anchor_helper.get_neg_label(cls_label_incomplete, incomplete_sample_ratio)
        # num_incomplete = np.sum(cls_label_incomplete, axis=(1,2))

        cls_label[cls_label_neg == -1] = -1
        cls_label[cls_label_incomplete == -1] = -1

        cls_label = torch.tensor(cls_label, dtype=torch.float32).to(opt.device) # torch.Size([32, 75, 5])
        loc_label = torch.tensor(loc_label, dtype=torch.float32).to(opt.device) # torch.Size([32, 75, 5, 2])

        targets['cls_label'] = cls_label
        targets['loc_label'] = loc_label
        # import ipdb; ipdb.set_trace()
        outputs = model(**model_inputs) # pred_cls torch.Size([32, 75, 5]) pred_loc torch.Size([32, 75, 5, 2])
        # outputs中包含了, pred_cls, pred_loc
        # dsnet loss
        pred_cls = outputs['pred_cls']
        pred_loc = outputs['pred_loc']
        # import ipdb; ipdb.set_trace()
        loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label)
        cls_loss = calc_cls_loss(pred_cls, cls_label)
        
        loss = cls_loss + opt.lambda_reg * loc_loss
        # import ipdb;ipdb.set_trace()
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # import ipdb; ipdb.set_trace()
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        # losses.backward()

        loss.backward()

        # print("cls_loss:",cls_loss)
        # print("loc_loss:",loc_loss)
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 1
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                # import ipdb; ipdb.set_trace()
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)
            # import ipdb; ipdb.set_trace()
            stop_score = metrics["brief"]["MR-full-mAP"]
            # stop_score = 5.
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()


def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
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
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio
    )

    dataset_config["data_path"] = opt.train_path
    train_dataset = StartEndDataset(**dataset_config)

    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        eval_dataset = StartEndDataset(**dataset_config)
    else:
        eval_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug


if __name__ == '__main__':
    best_ckpt_path, eval_split_name, eval_path, debug = start_training()
    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference()
