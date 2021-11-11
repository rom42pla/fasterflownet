from os.path import join
import logging

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.sintel import SINTELDataset
from models.fastflownet import FastFlowNet
import utils.parsers

# retrieves line arguments
args = utils.parsers.parse_test_args()

# loads the datasets
test_dataset = None
pretrained_model_path = None
if args.dataset in {"sintel_final", "sintel_clean"}:
    test_dataset = SINTELDataset(path=join(args.data_path, "sintel"),
                                 db_type="clean" if args.dataset == "sintel_clean" else "final",
                                 split="train", random_crop=True, zoom=False,
                                 horizontal_flip=False, rotation=False)
    pretrained_model_path = join(args.weights_path, "fastflownet_ft_sintel.pth")
    logging.info(f"Loaded {args.dataset}")
else:
    raise Exception(f"Unknown dataset {args.dataset}")

# builds the dataloaders
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

# builds the model
model = FastFlowNet(device=args.device)
if pretrained_model_path is not None:
    model.load_state_dict(torch.load(pretrained_model_path), strict=False)
    logging.info(f"Loaded pretrained model from {pretrained_model_path}")

# trains the model
trainer = pl.Trainer(gpus=1, precision=args.precision)
trainer.test(model, dataloaders=test_dataloader)
