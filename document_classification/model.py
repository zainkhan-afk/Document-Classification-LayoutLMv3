import torch
import torch.nn as nn
from transformers import LayoutLMv3ForSequenceClassification

class LayoutLMv3Wrapper(nn.Module):
	def __init__(self, num_classes, device, model_path = None):
		super(LayoutLMv3Wrapper, self).__init__()
		self.device = device
		if model_path is None:
			self.model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels = num_classes).to(self.device)
		else:
			self.model = LayoutLMv3ForSequenceClassification.from_pretrained(model_path).to(self.device)

	def forward(self, x):
		if 'labels' in x:
			pred = self.model(x['input_ids'], 
							attention_mask=x['attention_mask'], 
							bbox=x['bbox'], 
							pixel_values=x['pixel_values'], 
							labels=x['labels'])
		else:
			pred = self.model(x['input_ids'], 
							attention_mask=x['attention_mask'], 
							bbox=x['bbox'], 
							pixel_values=x['pixel_values'] 
							)

		return pred


	def save(self, path):
		self.model.save_pretrained(path, from_pt=True) 
		print(f"Model saved at {path}")

	def get_label(self, idx):
		return self.model.config.id2label[idx]

	def set_labels(self, label_map):
		self.model.config.id2label = label_map