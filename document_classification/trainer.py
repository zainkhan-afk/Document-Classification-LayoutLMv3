import os
import json
import torch
import numpy as np
import torch.nn.functional as F

os

class ModelTrainer:
	def __init__(self, optimizer, checkpoints_dir = None, log_interval = 10):
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		self.optimizer = optimizer
		self.best_val_acc = 0
		self.checkpoints_dir = checkpoints_dir
		self.log_interval = log_interval

		self.stats_file_path = os.path.join("stats.json")

		self.stats = {}
		self.stats['current_epoch'] = 0
		self.stats['total_epoch'] = 0
		self.stats['train_loss'] = []
		self.stats['train_acc'] = []
		self.stats['val_loss'] = []
		self.stats['val_acc'] = []

		if self.checkpoints_dir is not None:
			if not os.path.exists(self.checkpoints_dir):
				os.mkdir(self.checkpoints_dir)


	def save_best_model(self, model, val_acc):
		if val_acc>self.best_val_acc:
			print(f"Model accuracy increased from {self.best_val_acc} to {val_acc}.")
			self.best_val_acc = val_acc
			model_name = f"best_model.pt"
			model_path = os.path.join(self.checkpoints_dir, model_name)
			model.save(self.checkpoints_dir)

	def compute_acc(self, pred, labels):
		pred = F.softmax(pred, dim = 1)
		pred = torch.argmax(pred, dim = 1)

		return (pred == labels).sum()/(len(pred))

	def inference_step(self, model, data):
		pred = model(data)
		acc = self.compute_acc(pred.logits, data['labels'])
		return pred.loss, acc

	def training_loop(self, model, train_loader):
		loss_values = []
		acc_values = []
		step = 1
		for processed_batch in train_loader:
			self.optimizer.zero_grad()
			loss, acc = self.inference_step(model, processed_batch)
			loss.backward()
			self.optimizer.step()

			loss_values.append(loss.item())
			acc_values.append(acc.item())

			if step%self.log_interval == 0:
				print(f"Training: {step}/{len(train_loader)}: Loss: {np.mean(loss_values)}, Acc: {np.mean(acc_values)}")
				# return np.mean(loss_values), np.mean(acc_values)
			
			step += 1

		print(f"Training Loss: {np.mean(loss_values)}, Training Acc: {np.mean(acc_values)}")

		return np.mean(loss_values), np.mean(acc_values)

	def validation_loop(self, model, val_loader):
		loss_values = []
		acc_values = []
		step = 1
		with torch.no_grad():
			for processed_batch in val_loader:
				loss, acc = self.inference_step(model, processed_batch)

				if step%self.log_interval == 0:
					print(f"Validation: {step}/{len(val_loader)}: Loss: {np.mean(loss_values)}, Acc: {np.mean(acc_values)}")	
					# return np.mean(loss_values), np.mean(acc_values)

				step += 1
		
		print(f"Validation Loss: {np.mean(loss_values)}, Validation Acc: {np.mean(acc_values)}")
		return np.mean(loss_values), np.mean(acc_values)



	def train(self, model, epochs, train_loader, val_loader):
		for epoch in range(1, epochs + 1):
			print(f"\n\nEpoch {epoch}/{epochs}")
			train_loss, train_acc = self.training_loop(model, train_loader)
			val_loss, val_acc = self.training_loop(model, val_loader)

			self.stats['current_epoch'] = epoch
			self.stats['total_epoch'] = epochs
			self.stats['train_loss'].append(train_loss)
			self.stats['train_acc'].append(train_acc)
			self.stats['val_loss'].append(val_loss)
			self.stats['val_acc'].append(val_acc)

			with open(self.stats_file_path, 'w', encoding='utf-8') as f:
				json.dump(self.stats, f, ensure_ascii=False, indent=4)

			if self.checkpoints_dir is not None:
				self.save_best_model(model, val_acc)
			