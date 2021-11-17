from os.path import join
import logging

import torch
from torch import autograd
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from data.kitty_2012 import KITTI2012FlowDataset
from data.kitty_2015 import KITTI2015FlowDataset
from data.sintel import SINTELDataset
from models.decoder import FasterDecoder
from models.fastflownet import FastFlowNet
import utils.parsers
torch.cuda.empty_cache()
# retrieves line arguments
args = utils.parsers.parse_train_args()

# loads the datasets
train_dataset, val_dataset = None, None
pretrained_model_path, is_finetuning = None, True
if args.dataset in {"sintel_final", "sintel_clean"}:
    train_dataset_sintel = SINTELDataset(path=join(args.data_path, "sintel"),
                                  db_type="both", #if args.dataset == "sintel_clean" else "final",
                                  split="train", random_crop=True, zoom=False,
                                  horizontal_flip=False, rotation=False,photometric_augmentations=True)
    train_dataset_kitti2012 = KITTI2012FlowDataset(path=join(args.data_path, "kitty_2012/training"),
                                   random_crop=True, zoom=False,
                                  horizontal_flip=False, rotation=False, photometric_augmentations=True)
    train_dataset_kitti2015 = KITTI2015FlowDataset(path=join(args.data_path, "kitty_2015/training"),
                                                   random_crop=True, zoom=False,
                                                   horizontal_flip=False, rotation=False,
                                                   photometric_augmentations=True)
    train_dataset = ConcatDataset((train_dataset_sintel, train_dataset_kitti2012,train_dataset_kitti2015))

    val_dataset = SINTELDataset(path=join(args.data_path, "sintel"),
                                  db_type="clean" if args.dataset == "sintel_clean" else "final",
                                  split="train", center_crop=True, random_crop=False,zoom=False,
                                  horizontal_flip=False, rotation=False,photometric_augmentations=False)
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
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
     monitor='val_loss',
    dirpath='.',
    filename='sample-mnist-{epoch:02d}-{val_loss:.2f}'
)
#trainer.test(model,val_dataloader)

trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = False
model.decoder6= FasterDecoder(87,model.groups,90)
for param in model.decoder6.parameters():
		param.requires_grad = True

with autograd.detect_anomaly():
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = True
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = False
model.decoder5= FasterDecoder(87,model.groups,105)
for param in model.decoder5.parameters():
		param.requires_grad = True
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = True
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = False
model.decoder4= FasterDecoder(87,model.groups,120)
for param in model.decoder4.parameters():
		param.requires_grad = True
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = True
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = False
model.decoder3= FasterDecoder(87,model.groups,135)
for param in model.decoder3.parameters():
		param.requires_grad = True
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = True
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = False
model.decoder2= FasterDecoder(87,model.groups,150)
for param in model.decoder2.parameters():
		param.requires_grad = True
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, precision=args.precision,weights_save_path="/",callbacks=[checkpoint_callback])
for param in model.parameters():
		param.requires_grad = True
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
torch.save(model,"ciao2.modelfinal")
