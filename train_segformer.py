
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os
import torch.nn as nn

from dlvc.models.segformer import  SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer


def train(args):
    dataset_oxford_path = "./data"
    dataset_city_path = "./data/cityscapes_assg2"
    if args.dataset == "city":
        num_calsses = 19
    else:
        num_calsses = 3

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    if args.dataset == "oxford":
        train_data = OxfordPetsCustom(root=dataset_oxford_path, 
                                split="trainval",
                                target_types='segmentation', 
                                transform=train_transform,
                                target_transform=train_transform2,
                                download=True)

        val_data = OxfordPetsCustom(root=dataset_oxford_path, 
                                split="test",
                                target_types='segmentation', 
                                transform=val_transform,
                                target_transform=val_transform2,
                                download=True)
    if args.dataset == "city":
        train_data = CityscapesCustom(root=dataset_city_path, 
                                split="train",
                                mode="fine",
                                target_type='semantic', 
                                transform=train_transform,
                                target_transform=train_transform2)
        val_data = CityscapesCustom(root=dataset_city_path, 
                                split="val",
                                mode="fine",
                                target_type='semantic', 
                                transform=val_transform,
                                target_transform=val_transform2)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU if available, otherwise use CPU

    # not sure if model initialization is correct

    segformer_model = SegFormer(num_classes=num_calsses)
    model = DeepSegmenter(net=segformer_model)
    # If you are in the fine-tuning phase:
    path_models = Path("./saved_models")
    if args.dataset == 'oxford' and any(path_models.iterdir()):
        ##TODO update the encoder weights of the model with the loaded weights of the pretrained model
        # e.g. load pretrained weights with: state_dict = torch.load("path to model", map_location='cpu')
        # read first model in the directory
        state_dict = torch.load(os.path.join(path_models, os.listdir(path_models)[0]), map_location='cpu')
        encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder')}
        model.net.encoder.load_state_dict(encoder_state_dict)

        # Option to freeze the encoder and only train the decoder
        if args.freeze_encoder:
            for param in model.net.encoder.parameters():
                param.requires_grad = False

    model.to(device)
    # init optimizer, loss_fn, lr_scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=254) # class 255 has index 254 in Cityscapes datset
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2 # for 

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
                    batch_size=64,
                    val_frequency = val_frequency)
    trainer.train()
    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str, help='index of which GPU to use')
    args.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
    args.add_argument('--freeze_encoder', action='store_true', help='Freeze the encoder during fine-tuning')
    args.add_argument('--dataset', type=str, required=True, choices=['oxford', 'city'], help='Dataset to use')

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    train(args)