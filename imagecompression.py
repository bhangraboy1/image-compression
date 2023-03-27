import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

vmin = 0
vmax = 1
image_bias = 1  # sometimes 1


def plot_svd(A):
    n = len(A)
    imshow(image_bias - A, cmap='gray', vmin=vmin, vmax=vmax)
#    plt.show()
    U, S, V = svd(A)

    imgs = []
    for i in range(n):
        imgs.append(S[i] * np.outer(U[:, i], V[i]))

    combined_imgs = []
    for i in range(n):
        img = sum(imgs[:i + 1])
        combined_imgs.append(img)

    fig, axes = plt.subplots(figsize=(n * n, n), nrows=2, ncols=n, sharex=True, sharey=True)
    for num, ax in zip(range(n), axes.flat):
        ax.imshow(image_bias - imgs[num], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title("Component #" + str(num) + " of SVD", fontsize=8)
#    plt.show()

#    fig, axes = plt.subplots(figsize=(n * n, n), nrows=1, ncols=n, sharex=True, sharey=True)
    for num, ax in zip(range(n), axes.flat[n:]):
        ax.imshow(image_bias - combined_imgs[num], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title("Combined Image (#" + str(num) + ")", fontsize=8)
    fig.suptitle("This is an Illustration of the Components (SVD) and the Combined Images")
    plt.show()

    return U, S, V
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("hi")
    D = np.array([[0, 1, 1, 0, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  ])
    U, S, V = plot_svd(D)
    np.set_printoptions(precision=2)
    np.set_printoptions(formatter={'float_kind':'{:5.2f}'.format})
    print("Original Matrix", D)
    print("SVD Matrix", S)
    print("U", U)
    print("V", V)
