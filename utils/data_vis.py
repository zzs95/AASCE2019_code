import matplotlib.pyplot as plt
import numpy as np

def plot_img_and_mask(img, mask):
    plt.figure()
    # a = fig.add_subplot(1, 2, 1)
    # a.set_title('Input image')
    plt.imshow(img)

    # b = fig.add_subplot(1, 2, 2)
    # b.set_title('Output mask')
    for i in range(mask.shape[0]):
        plt.figure()
        plt.imshow(mask[i], plt.cm.gray)
    plt.show()

def plot_img_and_label(image, labels, plot_num=False):
    plt.figure()
    plt.imshow(image, 'gray')
    labels = np.array(labels)
    if plot_num:
        for i, label in enumerate(labels):
            if i%2 == 0:
                plt.text(label[0], label[1], s=str(i), fontsize=10, color='b')
            if i%2 == 1:
                plt.text(label[0], label[1], s=str(i), fontsize=10, color='b')
    else:
        for i, label in enumerate(labels):
            if i%2 == 0:
                plt.scatter(label[0], label[1], color='c', s=10)
            if i%2 == 1:
                plt.scatter(label[0], label[1], color='c', s=10)

    plt.show()