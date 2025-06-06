import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image

# Defining hyperparameters
latent_dim = 100
image_size = 28 * 28
batch_size = 64
num_epochs = 50
lr = 2e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory to save generated images
os.makedirs("generated_images", exist_ok=True)

# Loading data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Scale images to [-1, 1]
])

train_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Developing Generator Model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Output values in [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# Developing Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

# Initializingg models
generator = Generator(latent_dim, image_size).to(device)
discriminator = Discriminator(image_size).to(device)

# Defining loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Specifying the training loop
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.view(-1, image_size).to(device)
        batch_size_curr = real_images.size(0)

        # Labels
        real_labels = torch.ones(batch_size_curr, 1).to(device)
        fake_labels = torch.zeros(batch_size_curr, 1).to(device)

        # ---------------------
        #  Training Discriminator
        # ---------------------
        # Real images
        outputs_real = discriminator(real_images)
        d_loss_real = criterion(outputs_real, real_labels)

        # Synthetic images
        z = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_images = generator(z)
        outputs_fake = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        # Computing total discriminator loss
        d_loss = d_loss_real + d_loss_fake

        # Performing backpropagation and optimization
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Training Generator
        # -----------------
        z = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # Generator tries to fool discriminator

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Printing training stats
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    # Saving generated images for visualization
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)
        generated = generator(z).view(-1, 1, 28, 28)
        save_image(generated, f"generated_images/epoch_{epoch+1}.png", normalize=True)