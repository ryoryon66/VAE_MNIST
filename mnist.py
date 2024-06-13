from cv2 import log
from numpy import size
import torch
import torchvision
from torchvision import transforms
from vae import VAE
from torch.utils.tensorboard import SummaryWriter   
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_trained = False
model_path = "./vae40_save.pth"
log_dir = "./logs/exp1"


def train(train_loader,num_epochs = 20):
    vae = VAE(latent_dim=2)
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    vae.to(device)
    
    summary_writer = SummaryWriter(log_dir=log_dir)


    for epoch in range(num_epochs):
        loss_total = 0.0
        reconst_loss_total = 0.0
        kl_loss_total = 0.0
        train_total = 0
        for _, (x, _) in enumerate(train_loader):
            x = x.to(device)

            loss,reconst_loss,kl_loss = vae.loss(x)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_total += loss.item()
            reconst_loss_total += reconst_loss.item()
            kl_loss_total += kl_loss.item()
            train_total += x.size(0)
            
        print("=====================================")
        print(f"Epoch:{epoch+1}")
        print(f"Loss:{loss_total/train_total}")
        print(f"Reconst Loss:{reconst_loss_total/train_total}")
        print(f"KL Loss:{kl_loss_total/train_total}")
        
        summary_writer.add_scalar("Loss", loss_total/train_total, epoch)
        summary_writer.add_scalar("Reconst Loss", reconst_loss_total/train_total, epoch)
        summary_writer.add_scalar("KL Loss", kl_loss_total/train_total, epoch)
        
        if (epoch+1) % 10 == 0:
            torch.save(vae.state_dict(), f"./vae{epoch+1}.pth")

    torch.save(vae.state_dict(), f"./vae{num_epochs}.pth")
    summary_writer.close()

    return vae


if __name__ == "__main__":
    
    torch.autograd.set_detect_anomaly(True)
    
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
            # transforms.Lambda(lambda x: x / 255.)
        ]
    )

    mnist_train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=trans, download=True
    )

    # データを少なくして実験
    # mnist_train_dataset = torch.utils.data.Subset(mnist_train_dataset,range(10000))

    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_train_dataset, batch_size=512, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=mnist_test_dataset, batch_size=128, shuffle=False
    )

    if use_trained:
        vae = VAE(latent_dim=2)
        vae.load_state_dict(torch.load(model_path))
        vae.to(device)
    else:
        vae = train(train_loader)
    
    vae.eval()



    # テストデータで再構成画像を生成
    print("Generating reconst images...")
    import matplotlib.pyplot as plt
    import numpy as np

    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            x_reconst, _, _ = vae(x)
            break

    x = x.view(-1, 28, 28).cpu().numpy()
    x_reconst = x_reconst.view(-1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        axes[0, i].imshow(x[i], cmap="gray")
        axes[1, i].imshow(x_reconst[i], cmap="gray")

    plt.suptitle("Original (top row) vs Reconst (bottom row)")
    plt.savefig("reconst.png")
    
    # visualize latent space
    print("Visualizing latent space...")
    points_list = []
    labels_list = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            mu, _ = vae.encode(x)
            points_list.append(mu.cpu().numpy())
            labels_list.append(y.numpy())
    
    points = np.concatenate(points_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.title("Latent Space")
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=8,marker=".", alpha=0.8)
    plt.colorbar()
    plt.savefig("latent.png")
    
    
    # visualize generated images from latent space
    print("Generating images from latent space...")
    grids = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
    
    points_to_img = {}
    
    with torch.no_grad():
        for i, x in enumerate(grids[0].ravel()):
            for j, y in enumerate(grids[1].ravel()):
                z = torch.tensor([[x, y]]).float().to(device)
                img = vae.decode(z).view(28, 28).cpu().numpy()
                points_to_img[(i, j)] = img

    
    fig, axes = plt.subplots(20, 20, figsize=(20, 20))
    for i in range(20):
        for j in range(20):
            axes[i, j].imshow(points_to_img[(i, j)], cmap="gray")
            axes[i, j].axis("off")
    

    plt.savefig("latent_space.png")
    
