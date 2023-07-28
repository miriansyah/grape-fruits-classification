import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

#train and test data directory
data_dir = "D:/Proyek/Oprek_YoloV5/Program_SS/Dataset_grape_fruits/Training"
test_data_dir = "D:/Proyek/Oprek_YoloV5/Program_SS/Dataset_grape_fruits/Testing"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 3 input channels (for RGB images), 16 output channels, kernel size 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: 16 input channels, 32 output channels, kernel size 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 32 output channels * 8x8 spatial size after pooling
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: (batch_size, channels, height, width)

        # Convolutional layer 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Convolutional layer 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layer 1
        x = F.relu(self.fc1(x))

        # Fully connected layer 2 (output layer)
        x = self.fc2(x)

        # Output: (batch_size, num_classes)
        return x
def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (5,10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=5).permute(1,2,0))
        plt.show()
        break
def plot_acccuracy():
    plt.plot(train_accu,'-o')
    plt.plot(eval_accu,'-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy')
    plt.show()
    
def plot_losses():
    plt.plot(train_losses,'-o')
    plt.plot(eval_losses,'-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')
    plt.show()

def conf_matrix():
    y_true = []
    y_pred = []

    for data in tqdm(val_dl):
        images,labels=data[0].to(device),data[1]  
        y_true.extend(labels.numpy())

        outputs=model(images)

        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())

    cf_matrix = confusion_matrix(y_true, y_pred)
    class_names = ('good_grape','bad_grape')
   
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt="d")

    plt.title("Confusion Matrix"), plt.tight_layout()

    plt.ylabel("True Class"), 
    plt.xlabel("Predicted Class")
    plt.show()

#load the train and test data
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((32,32)),transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((32,32)),transforms.ToTensor()
]))

# Define relevant variables for the ML task
num_classes = 2
learning_rate = 0.001
num_epochs = 20
batch_size = 5


#load the train and validation into batches.
train_dl = DataLoader(dataset, batch_size, shuffle = True)
val_dl = DataLoader(test_dataset, batch_size,shuffle = False)

# Device will determine whether to run the training on GPU or CPU.
model = SimpleCNN(num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

total_step = len(train_dl)

train_losses=[]
train_accu=[]

def train(epoch):
  print('\nEpoch : %d'%epoch)
   
  model.train()
 
  running_loss=0
  correct=0
  total=0
 
  for data in tqdm(train_dl):
     
    inputs,labels=data[0].to(device),data[1].to(device)
     
    optimizer.zero_grad()
    outputs=model(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()
 
    running_loss += loss.item()
     
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
       
  train_loss=running_loss/len(train_dl)
  accu=100.*correct/total
   
  train_accu.append(accu)
  train_losses.append(train_loss)
  print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

eval_losses=[]
eval_accu=[]
 
def test(epoch):
  model.eval()
 
  running_loss=0
  correct=0
  total=0
 
  with torch.no_grad():
    for data in tqdm(val_dl):
      images,labels=data[0].to(device),data[1].to(device)
       
      outputs=model(images)
 
      loss= criterion(outputs,labels)
      running_loss+=loss.item()
       
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
   
  test_loss=running_loss/len(val_dl)
  accu=100.*correct/total
 
  eval_losses.append(test_loss)
  eval_accu.append(accu)
 
  print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 

for epoch in range(1,num_epochs+1): 
  train(epoch)
  test(epoch)

plot_acccuracy()
plot_losses()
conf_matrix()
