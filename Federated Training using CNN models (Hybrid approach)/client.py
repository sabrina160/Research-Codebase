# client.py
import os, gc, random, numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms
import flwr as fl

#  SETTINGS 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMG_SIZE = 224
N, K, Q = 10, 3, 4
EPISODES = 1000   # Few-shot episodes
TOTAL_ROUNDS = 10
SAVE_DIR = "C:/Users/MSC055/Desktop/Sabrina/CSE711/Using_Federated_Learning_Framework/for_CNN/output_pth_file"
os.makedirs(SAVE_DIR, exist_ok=True)

#  DATASET 
class ProtoDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = np.array(paths)
        self.labels = np.array(labels)
        self.transform = transform
        self.classes = np.unique(labels)
        self.class_to_indices = {c: np.where(labels == c)[0] for c in self.classes}

    def sample_episode(self):
        sx, sy, qx, qy = [], [], [], []
        classes = np.random.choice(self.classes, N, replace=False)
        for i, c in enumerate(classes):
            idxs = np.random.choice(self.class_to_indices[c], K + Q, replace=False)
            for idx in idxs[:K]:
                sx.append(self.transform(Image.open(self.paths[idx]).convert("RGB")))
                sy.append(i)
            for idx in idxs[K:]:
                qx.append(self.transform(Image.open(self.paths[idx]).convert("RGB")))
                qy.append(i)
        return torch.stack(sx), torch.tensor(sy), torch.stack(qx), torch.tensor(qy)

#  PROTONET 
class ProtoNetwork(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        backbone_func = getattr(models, backbone_name)
        backbone = backbone_func(pretrained=True)
        if hasattr(backbone, "fc"): backbone.fc = nn.Identity()
        if hasattr(backbone, "classifier"): backbone.classifier = nn.Identity()
        if hasattr(backbone, "head"): backbone.head = nn.Identity()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

class ProtoLoss(nn.Module):
    def forward(self, sx, sy, qx, qy, model):
        s = model(sx)
        q = model(qx)
        protos = torch.stack([s[sy==c].mean(0) for c in torch.unique(sy)])
        logits = -torch.cdist(q, protos)
        return nn.CrossEntropyLoss()(logits, qy)

#  FLOWER CLIENT 
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, models, dataset):
        self.models = models  
        self.dataset = dataset
        self.loss_fn = ProtoLoss()
        self.opts = {name: torch.optim.Adam(m.parameters(), lr=1e-4) for name, m in models.items()}

    def get_parameters(self, config=None):
        params = []
        for m in self.models.values():
            params += [p.detach().cpu().numpy() for p in m.parameters()]
        return params

    def set_parameters(self, parameters):
        i = 0
        for m in self.models.values():
            for p in m.parameters():
                p.data = torch.tensor(parameters[i]).to(DEVICE)
                i += 1

    def fit(self, parameters, config):
        round_num = config["server_round"]
        self.set_parameters(parameters)
        for name, model in self.models.items():
            print(f"\n[Client] Starting training backbone: {name} for Round {round_num}.")
            model.train()
            opt = self.opts[name]
            for episode in range(EPISODES):
                sx, sy, qx, qy = self.dataset.sample_episode()
                sx, sy, qx, qy = sx.to(DEVICE), sy.to(DEVICE), qx.to(DEVICE), qy.to(DEVICE)
                loss = self.loss_fn(sx, sy, qx, qy, model)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if episode % 50 == 0:
                    print(f"[Client | {name}] Episode {episode}/{EPISODES} | Prototypical Loss: {loss.item():.4f}")
            print(f"[Client] Finished training backbone: {name}.")
            
            # SAVE ONLY AFTER FINAL ROUND
            if round_num == TOTAL_ROUNDS:
                save_path = os.path.join(SAVE_DIR, f"global_{name}.pth")
                torch.save(model.state_dict(), save_path)
                print(f" Saved global model: {save_path}")
    
        print(f"[Client] Completed all backbones for Round {round_num}.\n")

        return self.get_parameters(), len(self.dataset.paths), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.dataset.paths), {}

#  START CLIENT 
def start_client(data_path):
    classes = sorted(os.listdir(data_path))
    paths, labels = [], []
    for i, c in enumerate(classes):
        for f in os.listdir(os.path.join(data_path, c)):
            if f.lower().endswith((".jpg",".png",".jpeg")):
                paths.append(os.path.join(data_path, c, f))
                labels.append(i)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    backbones = ["resnet50","densenet121","mobilenet_v2","efficientnet_b0","vgg16"]
    models_dict = {name: ProtoNetwork(name).to(DEVICE) for name in backbones}
    dataset = ProtoDataset(paths, labels, transform)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(models_dict, dataset)
    )

if __name__ == "__main__":
    start_client("C:/Users/MSC055/Desktop/Sabrina/CSE711/Eye Image Dataset")
