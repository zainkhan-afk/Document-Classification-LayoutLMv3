from utils import *
from DocumentDataset import DocumentDataset

from torch.utils.data import DataLoader
from model import LayoutLMv3Wrapper
import torch
from trainer import ModelTrainer
import config as cfg

device = get_device()
train_paths, train_labels, val_paths, val_labels, label_map = load_paths_and_split(cfg.DATA_DIR, val_split = 0.2)

print("Loading Model")
model = LayoutLMv3Wrapper(num_classes = cfg.NUM_CLASSES, device = device)
model.set_labels(label_map)

train_dataset = DocumentDataset(train_paths, train_labels, device = device)
val_dataset = DocumentDataset(val_paths, val_labels, device = device)

train_loader = DataLoader(
						train_dataset,
						batch_size = cfg.BATCH_SIZE,
						shuffle = True
					)

val_loader = DataLoader(
						val_dataset,
						batch_size = cfg.BATCH_SIZE,
						shuffle = False
					)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
trainer = ModelTrainer(optimizer, checkpoints_dir = "checkpoints")
trainer.train(model, cfg.EPOCHS, train_loader, val_loader)