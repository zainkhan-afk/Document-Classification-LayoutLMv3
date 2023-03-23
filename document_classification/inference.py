from utils import *
from model import LayoutLMv3Wrapper
import torch
import config as cfg
from Processor import Processor
from PIL import Image
import os


class Inference:
	def __init__(self, model_path, device):
		self.model_path = model_path
		self.device = device
		
		self.processor = Processor()
		
		self.model = LayoutLMv3Wrapper(num_classes = cfg.NUM_CLASSES, device = self.device, model_path = model_path)
		# print(self.model.parameters())
		# self.model.load(self.model_path)
		# print()


	def infer(self, img_path):
		img = Image.open(img_path).convert("RGB")
		encoding = self.processor(img)
		encoding_dict = {
							"input_ids":encoding["input_ids"].to(self.device),
							"attention_mask":encoding["attention_mask"].to(self.device),
							"bbox":encoding["bbox"].to(self.device),
							"pixel_values":encoding["pixel_values"].to(self.device)
					}

		with torch.inference_mode():
			pred = self.model(encoding_dict)
			class_idx = pred.logits.argmax()
			pred_label = self.model.get_label(class_idx.item())


		return pred_label



if __name__ == "__main__":
	img_name = os.environ["TEST_IMG_NAME"]
	if len(img_name) == 0:
		print("Provide image name as environment variable 'TEST_IMG_NAME' for inference")
		exit()

	device = get_device()
	inference = Inference(model_path = "checkpoints", device = device)
	
	path = f"/images_dir/{img_name}"

	if not os.path.exists(path):
		print(f"The image does not exist at {path}")
		exit()

	pred_label = inference.infer(path)

	print(f"\n\nPredicted Class: {pred_label}")