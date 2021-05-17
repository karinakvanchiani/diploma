from matplotlib import pyplot as plt


def plot_images(LR, HR):
    plt.figure(figsize=(18, 6))

    for i in range(6):
        plt.subplot(2, 6, i + 1)
        plt.axis('off')
        plt.title('HR', fontdict={'fontsize': 15})
        plt.imshow(HR[i])

        plt.subplot(2, 6, i + 7)
        plt.axis('off')
        plt.title('LR', fontdict={'fontsize': 15})
        plt.imshow(LR[i])

    plt.show()


def plot_results(LR_images, HR_gen, HR_gt, num):

    plt.figure(figsize=(42, 14))
    images = [LR_images[num], HR_gen[num], HR_gt[num]]
    names = ['LR', 'HR', 'GT']

    for i in range(len(images)):
        plt.subplot(2, 6, i + 1)
        plt.axis('off')
        plt.title(names[i], fontdict={'fontsize': 15})
        plt.imshow(images[i])

    plt.show()