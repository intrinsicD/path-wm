import numpy as np
import torch
from world_model.curriculum.encoder_masks import transform_mask,decode_union,mask_statistics,masked_bce


def test_rectangular_mask_follows_original_field_of_view_and_crop():
    # 128x64 source scales1:1 then removes32columns each side.
    x=np.zeros((64,128),dtype=np.uint8);x[10:20,40:56]=1
    out=transform_mask(x)
    expected=np.zeros((64,64),dtype=np.uint8);expected[10:20,8:24]=1
    assert np.array_equal(out,expected)
    # A2x larger source follows the same geometry and keeps binary labels.
    assert np.array_equal(transform_mask(np.repeat(np.repeat(x,2,axis=0),2,axis=1)),expected)


def test_instance_union_and_crowd_ignore_do_not_create_positive_targets():
    annotations=[{'segmentation':[[2,2,6,2,6,6,2,6]],'iscrowd':0},
                 {'segmentation':[[4,4,8,4,8,8,4,8]],'iscrowd':1}]
    mask,valid=decode_union(annotations,10,10)
    assert mask[3,3] and valid[3,3]
    assert not valid[5,5] and not valid[7,7]
    assert not mask[7,7] and not mask[0,0]


def test_mask_metrics_ignore_crowd_and_define_empty_case():
    target=torch.zeros(2,1,4,4);valid=torch.ones_like(target);pred=target.clone()
    target[0,0,:2,:2]=1;pred[0,0,:2,:2]=1
    valid[:,:,3,3]=0;pred[:,:,3,3]=1
    stats=mask_statistics(pred,target,valid)
    np.testing.assert_allclose(stats['iou'],[1,1]);np.testing.assert_allclose(stats['dice'],[1,1])
    logits=torch.zeros_like(target,requires_grad=True)
    masked_bce(logits,target,valid).backward()
    assert not logits.grad[:,:,3,3].any()
    assert torch.isfinite(logits.grad).all()
