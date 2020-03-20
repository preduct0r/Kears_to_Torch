import torch
import os
from prepare_datasets import get_iemocap_raw
from torch_cnn_model import  Config, torch_model, EarlyStopping, My_Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score
from torch.utils.data import DataLoader


[x_train, x_val, x_test, y_train, y_val, y_test] = get_iemocap_raw()

net = torch.load(os.path.join(r'C:\Users\kotov-d\Documents\TASKS\keras_to_torch', 'net_unbalanced_3cls.pb'))
batcher_test = DataLoader(My_Dataset(x_val, y_val), batch_size=512)

############################
# Validate
############################
loss = 0.0
correct = 0
iterations = 0
f_scores = 0

criterion = torch.nn.CrossEntropyLoss()
net.eval()
torch.manual_seed(7)

for i, (items, classes) in enumerate(batcher_test):
    items = items.to('cuda')
    classes = classes.to('cuda')

    outputs = net(items)
    loss += criterion(outputs, classes.long()).item()

    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == classes.data.long()).sum()

    f_scores += f1_score(predicted.cpu().numpy(), classes.data.cpu().numpy(), average='macro')

    iterations += 1

print(loss / iterations)
print(f_scores / iterations)

# Epoch 10/300, Tr Loss: 0.5235, Tr Fscore: 0.5086, Val Loss: 1.0344, Val Fscore: 0.3485