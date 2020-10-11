import numpy as np
import cv2
import math


def gaussian_filter():
    img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

    # Applying Gaussian filters
    sigma = 1 / (2 * np.log(2))
    blur_img = cv2.GaussianBlur(img, (3, 3), sigma)
    cv2.imwrite('3_1-gaussian.png', blur_img)


def edge_detection():
    height = width = 512
    img_orig = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    img_orig = cv2.copyMakeBorder(img_orig, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    img_gaussian = cv2.imread('3_1-gaussian.png', cv2.IMREAD_GRAYSCALE)
    img_gaussian = cv2.copyMakeBorder(img_gaussian, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    
    kx = ky = [-1, 0, 1]

    # Draw Ix and Iy
    Ix = np.zeros((height, width), dtype=int)
    Iy = np.zeros((height, width), dtype=int)
    Ix_g = np.zeros((height, width), dtype=int)
    Iy_g = np.zeros((height, width), dtype=int)
    
    for h in range(1, height+1):
        for w in range(1, width+1):
            Ix[h-1, w-1] = (int(img_orig[h, w+1]) - int(img_orig[h, w-1])) / 2
            Iy[h-1, w-1] = (int(img_orig[h+1, w]) - int(img_orig[h-1, w])) / 2
            Ix_g[h-1, w-1] = (int(img_gaussian[h, w+1]) - int(img_gaussian[h, w-1])) // 2
            Iy_g[h-1, w-1] = (int(img_gaussian[h+1, w]) - int(img_gaussian[h-1, w])) // 2

    cv2.imwrite('3_2-Ix.png', Ix)
    cv2.imwrite('3_2-Iy.png', Iy)

    # Draw gradient magnitude image
    output_orig = np.zeros((height, width), dtype=int)
    output_gaussian = np.zeros((height, width), dtype=int)

    for h in range(height):
        for w in range(width):
            output_orig[h, w] = math.sqrt(Ix[h, w]**2 + Iy[h, w]**2)
            output_gaussian[h, w] = math.sqrt(Ix_g[h, w]**2 + Iy_g[h, w]**2)

    cv2.imwrite('3_3-gm-original.png', output_orig)
    cv2.imwrite('3_3-gm-gaussian.png', output_gaussian)

def main():
    # gaussian_filter()
    edge_detection()


if __name__ == "__main__":
    main()

