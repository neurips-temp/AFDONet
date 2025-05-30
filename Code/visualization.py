import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def compute_errors(pred, true):
    mae = torch.mean(torch.abs(pred - true)).item()
    l2_rel = torch.norm(pred - true) / (torch.norm(true) + 1e-8)
    return mae, l2_rel.item()



def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            recon, _, _ = model(x)
            loss = F.mse_loss(recon, y)
            total_loss += loss.item()
    return total_loss / len(loader)

def visualize_prediction(model, loader, device):
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        recon, _, _ = model(x)
    mae, l2_rel = compute_errors(recon, y)
    print(f"MAE: {mae:.6f}\nL2 Relative Error: {l2_rel:.6f}")

    u_true = y[0, 0, 0].cpu().numpy()
    u_pred = recon[0, 0, 0].cpu().numpy()
    v_true = y[0, 0, 1].cpu().numpy()
    v_pred = recon[0, 0, 1].cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].imshow(u_true, cmap='viridis'); axs[0, 0].set_title('component u (ground truth)')
    axs[0, 1].imshow(u_pred, cmap='viridis'); axs[0, 1].set_title('component v (ground truth)')
    axs[1, 0].imshow(v_true, cmap='viridis'); axs[1, 0].set_title('component u (AFDONet)')
    axs[1, 1].imshow(v_pred, cmap='viridis'); axs[1, 1].set_title('component v (AFDONet)')
    plt.tight_layout(); plt.show()
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ims = [
        axs[0, 0].imshow(u_true, cmap='viridis'),
        axs[0, 1].imshow(u_pred, cmap='viridis'),
        axs[1, 0].imshow(v_true, cmap='viridis'),
        axs[1, 1].imshow(v_pred, cmap='viridis')
    ]
    titles = ["component u (ground truth)", "component v (ground truth)", "component u (AFDONet)", "component v (AFDONet)"]
    for ax, title, im in zip(axs.ravel(), titles, ims):
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    theta = np.linspace(0, 2 * np.pi, u_true.shape[0])
    phi = np.linspace(0, 2 * np.pi, u_true.shape[1])
    theta, phi = np.meshgrid(theta, phi)
    R, r = 2, 1
    X = (R + r * np.cos(theta)) * np.cos(phi)
    Y = (R + r * np.cos(theta)) * np.sin(phi)
    Z = r * np.sin(theta)

    def norm(f): return (f - np.min(f)) / (np.max(f) - np.min(f))

    fig = plt.figure(figsize=(12, 10))
    for i, (val, title) in enumerate(zip([u_true, u_pred, v_true, v_pred],
                                         ['component u (ground truth)', 'component v (ground truth)', 'component u (AFDONet)', 'component v (AFDONet)'])):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(norm(val)), rstride=1, cstride=1, linewidth=0, shade=False)
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()