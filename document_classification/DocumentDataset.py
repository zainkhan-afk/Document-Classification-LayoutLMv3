from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from Processor import Processor

class DocumentDataset(Dataset):
	def __init__(self, image_paths, image_labels, device):
		self.image_paths = image_paths
		self.image_labels = image_labels
		self.device = device
		
		self.processor = Processor()

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		image_label = self.image_labels[idx]

		img = Image.open(image_path).convert("RGB")
		
		encoding = self.processor(img)

		out_data = {
					"input_ids":encoding["input_ids"].flatten().to(self.device),
					"attention_mask":encoding["attention_mask"].flatten().to(self.device),
					"bbox":encoding["bbox"].flatten(end_dim = 1).to(self.device),
					"pixel_values":encoding["pixel_values"].flatten(end_dim = 1).to(self.device),
					"labels":torch.tensor(image_label).to(self.device)
		}
		# 'input_ids', 'attention_mask', 'bbox', 'pixel_values']
		return out_data