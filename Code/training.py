import torch.nn.functional as F
import torch

def homolorphic_loss(u_pred, u_true, h=1.0 / 64):
    loss = F.mse_loss(u_pred, u_true, reduction='mean')
    grads = torch.autograd.grad(outputs=u_pred, inputs=u_pred, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    grad_loss = torch.norm(grads, p=2) ** 2
    return loss + h * grad_loss

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x = x.to(device).requires_grad_()
        y = y.to(device).requires_grad_()
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        mse = F.mse_loss(recon, y)
        homo_l=homolorphic_loss(recon, y)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + 1e-8 * kl + 1e-8 * homo_l
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
