import os

# 数据集准备
dataset_path="/home/u2021213565/jupyterlab/images_dataset_resnet_train/1"
# 统计数据集中的类的数量
train_path = dataset_path+"/train"
num_classes = len([f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))])
print(f"Number of classes: {num_classes}")

_base_ = [
    '../mmpretrain/configs/_base_/models/resnet50.py',           # 模型设置
    '../mmpretrain/configs/_base_/datasets/imagenet_bs32.py',    # 数据设置
    '../mmpretrain/configs/_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '../mmpretrain/configs/_base_/default_runtime.py',           # 运行设置
]

# 模型设置
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=num_classes,topk=(1,),  ),
)

# 数据设置
data_root = dataset_path

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='train',
        pipeline=train_pipeline,
        _delete_=True,
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',
        data_prefix='val',
        pipeline=test_pipeline,
        _delete_=True,
    ))

val_evaluator = dict(type='Accuracy',topk=(1,))

test_dataloader = val_dataloader

# 训练策略设置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)