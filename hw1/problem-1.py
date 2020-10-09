import numpy as np
import cv2
import random


def kmeans(img, k):
    means = []
    for i in range(0, k):
        x, y = random.randint(0, 64), random.randint(0, 64)
        means.append((x, y))
    print(means)



def main():
    img = cv2.imread('bird.jpg')
    kmeans(np.copy(img), 2)    


if __name__ == "__main__":
    main()