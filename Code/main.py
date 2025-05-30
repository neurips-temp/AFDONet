
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import DataLoader, Dataset
from load_data import load_or_generate_data, NavierStokesDataset
from model import AFDONet
from training import train_epoch
from visualization import evaluate, visualize_prediction
start_time = time.time()
np.random.seed(42)
#torch.manual_seed(42)

        

data = load_or_generate_data("NS_equation_autoreg_dataset.npz")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = NavierStokesDataset(train_data)
test_dataset = NavierStokesDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

rhos = [0.1, 0.2, 0.3, 0.4, 0.5]
model = AFDONet(input_dim=2*64*64, inter_dim=256, latent_dim=10, rhos=rhos).to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = next(model.parameters()).device

for epoch in range(10):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    test_loss = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.9f}, Test Loss: {test_loss:.9f}")
torch.save(model.state_dict(), "vae_fno_model.pth")
visualize_prediction(model, test_loader, device)
end_time = time.time()
print(f"Inference Time: {end_time - start_time:.6f} seconds")
