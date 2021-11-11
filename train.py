from os.path import join
import logging

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.sintel import SINTELDataset
from models.fastflownet import FastFlowNet
import utils.parsers

# retrieves line arguments
args = utils.parsers.parse_train_args()

# loads the datasets
train_dataset, val_dataset = None, None
pretrained_model_path, is_finetuning = None, False
if args.dataset in {"sintel_final", "sintel_clean"}:
    train_dataset = SINTELDataset(path=join(args.data_path, "sintel"),
                                  db_type="clean" if args.dataset == "sintel_clean" else "final",
                                  split="train", random_crop=True, zoom=False,
                                  horizontal_flip=False, rotation=False)
    val_dataset = train_dataset
    pretrained_model_path = join(args.weights_path, "fastflownet_ft_sintel.pth")
    is_finetuning = True
    logging.info(f"Loaded {args.dataset}")
else:
    raise Exception(f"Unknown dataset {args.dataset}")

# builds the dataloaders
train_dataloader, val_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers), \
                                   DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)

# builds the model
model = FastFlowNet(device=args.device, learning_rate=args.learning_rate, is_finetuning=is_finetuning)
if pretrained_model_path is not None:
    model.load_state_dict(torch.load(pretrained_model_path), strict=False)
    logging.info(f"Loaded pretrained model from {pretrained_model_path}")

# trains the model
trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
