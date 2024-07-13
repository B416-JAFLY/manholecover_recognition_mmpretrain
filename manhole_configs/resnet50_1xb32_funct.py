import os

_base_ = [                                    # 此配置文件将继承所有 `_base_` 中的配置
               # 模型配置
    '../mmpretrain/configs/_base_/schedules/imagenet_bs256.py',  # 训练策略配置
    '../mmpretrain/configs/_base_/default_runtime.py'            # 默认运行设置
]

# 数据集准备
dataset_path="/home/u2021213565/jupyterlab/images_dataset_resnet_train/1"
# 统计数据集中的类的数量
train_path = dataset_path+"/train"
num_classes = len([f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))])
print(f"Number of classes: {num_classes}")

# 模型配置
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,  # 修改类别数
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),       # 只计算top-1准确率
    ))

# 数据集加载配置
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
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_path,
        data_prefix='train',
        pipeline=train_pipeline,
        ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_path,
        data_prefix='val',
        pipeline=test_pipeline,
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy',topk=(1,))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator