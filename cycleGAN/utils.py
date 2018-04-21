import os
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.imgs = []

    def query(self, img):
        if self.pool_size == 0:
            return img

        if len(self.imgs) < self.pool_size:
            self.imgs.append(img)
            return img
        else:
            if random.random() > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp_img = self.imgs[random_id].copy()
                self.imgs[random_id] = img.copy()
                return tmp_img
            else:
                return img


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()


def plots(imgs, iter_time, image_size, save_file):
    n_cols, n_rows, cell_size, margin = 2, imgs[0].shape[0], 3, 0.05
    fig = plt.figure(figsize=(cell_size * n_cols, cell_size * n_rows))  # (column, row)
    gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
    gs.update(wspace=margin, hspace=margin)

    imgs = [inverse_transform(imgs[idx]) for idx in range(len(imgs))]

    # save more bigger image
    for col_index in range(n_cols):
        for row_index in range(n_rows):
            ax = plt.subplot(gs[row_index * n_cols + col_index])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow((imgs[col_index][row_index]).reshape(
                image_size[0], image_size[1], image_size[2]), cmap='Greys_r')

    plt.savefig(save_file + '/XtoY_{}.png'.format(str(iter_time)), bbox_inches='tight')
    plt.close(fig)


def inverse_transform(img):
    return (img + 1.) / 2.


