import argparse
from pathlib import Path

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper
import os
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def plot_current_losses(save_path, staring_epoch, epoch, losses):
    plt.figure(figsize=(10, 5))
    x = np.arange(staring_epoch, epoch).astype(int)
    lws = []
    for k, l in losses.items():
        if "total" in k.lower():
            lw = 0.5
        else:
            lw = 0.25
        plt.plot(x, l, linewidth=lw, label=k)
        lws.append(lw)
    
    leg = plt.legend(loc='upper left')
    leg_lines = leg.get_lines()
    for i, lw in enumerate(lws):
        plt.setp(leg_lines[i], linewidth=lw*10)
    leg_texts = leg.get_texts()
    plt.setp(leg_texts, fontsize=12)

    plt.xlabel("Epochs")
    plt.yscale("log")
    plt.title("epoch {}".format(epoch))
    plt.savefig(save_path, dpi=150)
    plt.close()

def visualize_train(model, epoch):
    c = model.content.detach().cpu().numpy()
    s = model.style.detach().cpu().numpy()
    g_t = model.g_t.detach().cpu().numpy()
    
    c = np.transpose(c, (0, 3, 2, 1))
    s = np.transpose(s, (0, 3, 2, 1))
    g_t = np.transpose(g_t, (0, 3, 2, 1))
    g_t = (g_t-g_t.min()) / (g_t.max()-g_t.min())

    save_path = os.path.join("output", "tr_{:05}.jpg".format(epoch))
    suptitle = "Train | epoch {}".format(epoch)
    (B, _, H, W) = c.shape

    fs = 20
    w = 2
    n_rows = B
    n_cols = 3
    fig = plt.figure(figsize=(w*(n_cols), w*(n_rows)+2))
    for i in range(B):
        i0 = n_cols*i
        # A
        ax = fig.add_subplot(n_rows, n_cols, i0 + 1)
        ax.imshow(c[i].squeeze())
        ax.set_title("A", fontsize=fs)

        # B
        ax = fig.add_subplot(n_rows, n_cols, i0 + 2)
        ax.imshow(s[i].squeeze())
        ax.set_title("B", fontsize=fs)

        # AB
        ax = fig.add_subplot(n_rows, n_cols, i0 + 3)
        ax.imshow(g_t[i].squeeze())
        ax.set_title("A -> B", fontsize=fs)

    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle(suptitle, fontsize=fs)
    plt.savefig(save_path, dpi=150)
    plt.close()

class FlatFolderDataset(data.Dataset):
    def __init__(self, root):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))

    def __getitem__(self, index):
        path = self.paths[index]
        img = torchvision.transforms.ToTensor()(Image.open(str(path)).convert('RGB'))
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default="data/synth",
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default="data/labels_gray",
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--plot_interval', type=int, default=500)
args = parser.parse_args()
os.makedirs("output", exist_ok=True)
device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
# writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)


content_dataset = FlatFolderDataset(args.content_dir)
style_dataset = FlatFolderDataset(args.style_dir)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

losses = {'c': [], 's': []}
for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses['c'].append(loss_c.item())
    losses['s'].append(loss_s.item())

    # writer.add_scalar('loss_content', loss_c.item(), i + 1)
    # writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth.tar'.format(i + 1))
    if i > 0 and (i + 1) % args.plot_interval == 0:
        visualize_train(network, i+1)
        save_path = 'output/loss.jpg'
        plot_current_losses(save_path, staring_epoch=0, epoch=i+1, losses=losses)

# writer.close()
