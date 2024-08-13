import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    try:
        img = img.detach().numpy().transpose((1, 2, 0))
    except:
        pass
    try:
        mask = mask.detach().numpy()
        mask = mask.transpose((1,2,0))
    except:
        pass
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    plt.xticks([]), plt.yticks([])
    ax[1].set_title(f'Mask')
    ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()