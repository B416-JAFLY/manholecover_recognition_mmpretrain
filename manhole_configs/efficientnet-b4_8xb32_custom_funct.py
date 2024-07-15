import os

# 数据集准备
dataset_path="/home/u2021213565/jupyterlab/images_dataset_resnet_train/1"
# 统计数据集中的类的数量
train_path = dataset_path+"/train"
num_classes = len([f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))])
print(f"Number of classes: {num_classes}")

_base_ = [
    '/home/u2021213565/jupyterlab/manholecover_recognition_mmpretrain/mmpretrain/configs/_base_/schedules/imagenet_bs256.py',
    '/home/u2021213565/jupyterlab/manholecover_recognition_mmpretrain/mmpretrain/configs/_base_/default_runtime.py',
]
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNet', 
        arch='b4',
        frozen_stages=6,
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty-ra-noisystudent_in1k_20221103-16ba8a2d.pth', prefix='backbone.')
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=1792,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
    ))

# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=380),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=380),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_path,
        ann_file='',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_path,
        ann_file='',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
