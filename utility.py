import os
import torch
import pickle
import imageio
import itertools
import matplotlib.pyplot as plt
from torch.autograd import Variable


def save_results(epoch, gen=None, show=False, save=False, path='result/output/output_{}.png', device='cuda'):
    if not os.path.exists('result/output'):
        os.makedirs('result/output')

    fixed_noise = Variable(torch.randn((5 * 5, 100)).view(-1, 100, 1, 1).to(device))

    gen.eval()
    test_images = gen(fixed_noise)
    gen.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path.format(epoch))
    if show:
        plt.show()
    else:
        plt.close()


def save_checkpoint(gen=None, disc=None):
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(gen, "model/gen.pt")
    torch.save(disc, "model/disc.pt")


def create_gif(path='result/output'):
    images = []
    for files in os.listdir(path):
        images.append(imageio.imread(os.path.join(path, files)))
    imageio.mimsave('result/progress.gif', images, fps=5)


def plot(show=False, save=False, path='result/loss_plot.png'):
    if not os.path.exists('result/losses.pkl'):
        print('Loss pickle not found')
    else:
        with open('result/losses.pkl', 'rb') as f:
            losses = pickle.load(f)

        if not os.path.exists('result'):
            os.makedirs('result')

        x = range(len(losses['disc_loss']))
        y1 = losses['gen_loss']
        y2 = losses['disc_loss']
        plt.plot(x, y1, label='gen_loss')
        plt.plot(x, y2, label='disc_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        if save:
            plt.savefig(path)
        if show:
            plt.show()
        else:
            plt.close()
