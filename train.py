import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights


from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

def train(args):
    dataset_path = "./data"

    train_transform = v2.Compose([
        v2.ToImage(), 
        v2.Resize(size=(128, 128)),  # Resize images to a fixed size
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_transform2 = v2.Compose([
        v2.ToImage(), 
        v2.Resize(size=(128, 128)),  # Resize masks to a fixed size
        v2.ToDtype(torch.long, scale=False)
    ])
    
    val_transform = v2.Compose([
        v2.ToImage(), 
        v2.Resize(size=(128, 128)),  # Resize images to a fixed size
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform2 = v2.Compose([
        v2.ToImage(), 
        v2.Resize(size=(128, 128)),  # Resize masks to a fixed size
        v2.ToDtype(torch.long, scale=False)
    ])

    train_data = OxfordPetsCustom(root=dataset_path, 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

    val_data = OxfordPetsCustom(root=dataset_path, 
                            split="test",
                            target_types='segmentation', 
                            transform=val_transform,
                            target_transform=val_transform2,
                            download=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU if available, otherwise use CPU

    if args.pretrained:
        model = fcn_resnet50(weights_backbone=FCN_ResNet50_Weights.DEFAULT)  # FCN ResNet50 model with pre-trained weights (backbone)
    else:
        model = fcn_resnet50(weights_backbone=None)  # Empty model training from scratch

    model.classifier[4] = torch.nn.Conv2d(512, len(train_data.classes_seg), kernel_size=(1, 1))
    model = DeepSegmenter(model)
    model = model.to(device)
    
    # Initialize the AdamW optimizer with AMSGrad, CrossEntropyLoss and learning rate
    optimizer = AdamW(model.parameters(), lr=0.01, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = ExponentialLR(optimizer, gamma=0.98)
    
    # Initialize segmentation metrics for training and validation data
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 1

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    trainer = ImgSemSegTrainer(model,
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=32,
                    val_frequency=val_frequency)
    trainer.train()  # Start the training process

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose()  # Release any resources held by the trainer

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('--pretrained', action='store_true', help='Pre-trained weights for the backbone')
    args.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')

    if not isinstance(args, tuple):
        args = args.parse_args()

    train(args)
