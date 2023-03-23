import json
import matplotlib.pyplot as plt

f = open('stats.json')
data = json.load(f)
f.close()

print(data.keys())
'train_loss', 'train_acc', 'val_loss', 'val_acc'

plt.figure()
plt.plot(data['train_loss'], label = "Training Loss")
plt.plot(data['val_loss'], label = "Validation Loss" )
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.savefig("Loss.png")


plt.figure()
plt.plot(data['train_acc'], label = "Training Accuracy")
plt.plot(data['val_acc'], label = "Validation Accuracy" )
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.savefig("Accuracy.png")

plt.show()