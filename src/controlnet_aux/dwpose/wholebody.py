# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import warnings

try:
    import mmcv
except ImportError:
    warnings.warn(
        "The module 'mmcv' is not installed. The package will have limited functionality. Please install it using the command: mim install 'mmcv>=2.0.1'"
    )

try:
    from mmpose.apis import inference_topdown
    from mmpose.apis import init_model as init_pose_estimator
    from mmpose.evaluation.functional import nms
    from mmpose.utils import adapt_mmdet_pipeline
    from mmpose.structures import merge_data_samples
except ImportError:
    warnings.warn(
        "The module 'mmpose' is not installed. The package will have limited functionality. Please install it using the command: mim install 'mmpose>=1.1.0'"
    )

try:
    from mmdet.apis import inference_detector, init_detector
except ImportError:
    warnings.warn(
        "The module 'mmdet' is not installed. The package will have limited functionality. Please install it using the command: mim install 'mmdet>=3.1.0'"
    )


class Wholebody:
    def __init__(
        self,
        det_config=None,
        det_ckpt=None,
        pose_config=None,
        pose_ckpt=None,
        device="cpu",
    ):
        if det_config is None:
            det_config = os.path.join(
                os.path.dirname(__file__), "yolox_config/yolox_l_8xb8-300e_coco.py"
            )

        if pose_config is None:
            pose_config = os.path.join(
                os.path.dirname(__file__), "dwpose_config/dwpose-l_384x288.py"
            )

        if det_ckpt is None:
            det_ckpt = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"

        if pose_ckpt is None:
            pose_ckpt = "https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth"

        # build detector
        self.detector = init_detector(det_config, det_ckpt, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # build pose estimator
        self.pose_estimator = init_pose_estimator(pose_config, pose_ckpt, device=device)

    def to(self, device):
        self.detector.to(device)
        self.pose_estimator.to(device)
        return self

    def __call__(self, oriImg):
        # predict bbox
        det_result = inference_detector(self.detector, oriImg)

        if isinstance(oriImg, list):
            assert isinstance(det_result, list)
            all_bboxes = []
            for det_result_ in det_result:
                pred_instance = det_result_.pred_instances.cpu().numpy()
                bboxes = np.concatenate(
                    (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
                )
                bboxes = bboxes[
                    np.logical_and(
                        pred_instance.labels == 0, pred_instance.scores > 0.5
                    )
                ]

                # set NMS threshold
                bboxes = bboxes[nms(bboxes, 0.7), :4]
                all_bboxes.append(bboxes)
            bboxes = all_bboxes
        else:
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
            )
            bboxes = bboxes[
                np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.5)
            ]

            # set NMS threshold
            bboxes = bboxes[nms(bboxes, 0.7), :4]

        pose_results = inference_topdown_(self.pose_estimator, oriImg, bboxes)

        if isinstance(oriImg, list):
            pose_results_idx = 0
            all_pose_results = []
            for bboxes_ in bboxes:
                some_pose_results = []
                for _ in range(len(bboxes_) if len(bboxes_) > 0 else 1): # inference_topdown_ always adds a single bounding box if none was detected
                    some_pose_results.append(pose_results[pose_results_idx])
                    pose_results_idx += 1
                all_pose_results.append(some_pose_results)
        else:
            all_pose_results = [pose_results]

        all_keypoints = []
        all_scores = []

        for pose_results in all_pose_results:
            if len(pose_results) == 0:
                all_keypoints.append(None)
                all_scores.append(None)
                continue

            preds = merge_data_samples(pose_results)
            preds = preds.pred_instances

            # preds = pose_results[0].pred_instances
            keypoints = preds.get("transformed_keypoints", preds.keypoints)
            if "keypoint_scores" in preds:
                scores = preds.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if "keypoints_visible" in preds:
                visible = preds.keypoints_visible
            else:
                visible = np.ones(keypoints.shape[:-1])
            keypoints_info = np.concatenate(
                (keypoints, scores[..., None], visible[..., None]), axis=-1
            )
            # compute neck joint
            neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
            # neck score when visualizing pred
            neck[:, 2:4] = np.logical_and(
                keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
            ).astype(int)
            new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
            mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
            openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
            keypoints_info = new_keypoints_info

            keypoints, scores, visible = (
                keypoints_info[..., :2],
                keypoints_info[..., 2],
                keypoints_info[..., 3],
            )

            all_keypoints.append(keypoints)
            all_scores.append(scores)

        return all_keypoints, all_scores


from torch import nn
from typing import Union, Optional, List
from mmpose.structures import PoseDataSample
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from PIL import Image

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.builder import build_pose_estimator
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy


def inference_topdown_(
    model: nn.Module,
    imgs: List[np.ndarray],
    bboxes: Optional[Union[List, np.ndarray]] = None,
    bbox_format: str = "xyxy",
) -> List[PoseDataSample]:
    """Inference image with a top-down pose estimator.

    Args:
        model (nn.Module): The top-down pose estimator
        img (np.ndarray | str): The loaded image or image file to inference
        bboxes (np.ndarray, optional): The bboxes in shape (N, 4), each row
            represents a bbox. If not given, the entire image will be regarded
            as a single bbox area. Defaults to ``None``
        bbox_format (str): The bbox format indicator. Options are ``'xywh'``
            and ``'xyxy'``. Defaults to ``'xyxy'``

    Returns:
        List[:obj:`PoseDataSample`]: The inference results. Specifically, the
        predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints`` and
        ``data_sample.pred_instances.keypoint_scores``.
    """
    scope = model.cfg.get("default_scope", "mmpose")
    if scope is not None:
        init_default_scope(scope)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # construct batch data samples
    data_list = []
    for img, bboxes_ in zip(imgs, bboxes):
        if bboxes_ is None or len(bboxes_) == 0:
            h, w = imgs[0].shape[:2]
            bboxes_ = np.array([[0, 0, w, h]], dtype=np.float32)
        else:
            if isinstance(bboxes_, list):
                bboxes_ = np.array(bboxes_)

            assert bbox_format in {
                "xyxy",
                "xywh",
            }, f'Invalid bbox_format "{bbox_format}".'

            if bbox_format == "xywh":
                bboxes_ = bbox_xywh2xyxy(bboxes_)

        for bbox in bboxes_:
            data_info = dict(img=img)
            data_info["bbox"] = bbox[None]  # shape (1, 4)
            data_info["bbox_score"] = np.ones(1, dtype=np.float32)  # shape (1,)
            data_info.update(model.dataset_meta)
            data_list.append(pipeline(data_info))

    if data_list:
        # collate data list into a batch, which is a dict with following keys:
        # batch['inputs']: a list of input images
        # batch['data_samples']: a list of :obj:`PoseDataSample`
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
    else:
        results = []

    return results
