_base_ = ['../../../../_base_/datasets/cats.py'] # DO NOT MODIFY THIS LINE
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam', # You may choose other optimizers
    lr=1e-5, # Tune the base learning rate
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict( # You may choose another learning rate scheduler
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[10, 15])
total_epochs = 20
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict( # DO NOT MODIFY THIS LINE
    num_output_channels=17, # DO NOT MODIFY THIS LINE
    dataset_joints=17, # DO NOT MODIFY THIS LINE
    dataset_channel=[ # DO NOT MODIFY THIS LINE
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[  # DO NOT MODIFY THIS LINE
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet50', # This is an ImageNet-pretrained model
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[128, 128], # You may also use higher input resolution, e.g. 256x256
    heatmap_size=[32, 32], # heatmap_size should be 1/4 of the image_size
    num_output_channels=channel_cfg['num_output_channels'], # DO NOT MODIFY THIS LINE
    num_joints=channel_cfg['dataset_joints'], # DO NOT MODIFY THIS LINE
    dataset_channel=channel_cfg['dataset_channel'], # DO NOT MODIFY THIS LINE
    inference_channel=channel_cfg['inference_channel'], # DO NOT MODIFY THIS LINE
    soft_nms=False, # DO NOT MODIFY THIS LINE
    nms_thr=1.0, # DO NOT MODIFY THIS LINE
    oks_thr=0.9, # DO NOT MODIFY THIS LINE
    vis_thr=0.2, # DO NOT MODIFY THIS LINE
    use_gt_bbox=True, # DO NOT MODIFY THIS LINE
    det_bbox_thr=0.0, # DO NOT MODIFY THIS LINE
    bbox_file='', # DO NOT MODIFY THIS LINE
)

train_pipeline = [
    dict(type='LoadImageFromFile'), # DO NOT MODIFY THIS LINE
    dict(type='TopDownRandomFlip', flip_prob=0.5), # It is suggested to perform random flipping.
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=10, scale_factor=0.2), # You may tune these hyper-parameters
    # You may also add more data augmentation pipelines here.
    dict(type='TopDownAffine'), # DO NOT MODIFY THIS LINE
    dict(type='ToTensor'), # DO NOT MODIFY THIS LINE
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406], # This is the mean/var of the ImageNet dataset.
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=1), # Sigma normally increases with the input size.
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'), # DO NOT MODIFY THIS LINE
    dict(type='TopDownAffine'), # DO NOT MODIFY THIS LINE
    dict(type='ToTensor'), # DO NOT MODIFY THIS LINE
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406], # Should be the same as that of training pipeline
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect', # DO NOT MODIFY THIS LINE
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/ap10k'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='AnimalCatsDataset',
        ann_file=f'{data_root}/annotations/train.json',
        img_prefix=f'{data_root}/data/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='AnimalCatsDataset',
        ann_file=f'{data_root}/annotations/val.json',
        img_prefix=f'{data_root}/data/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='AnimalCatsDataset',
        ann_file=f'{data_root}/annotations/test_info.json',
        img_prefix=f'{data_root}/data/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
