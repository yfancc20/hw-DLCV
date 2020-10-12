import numpy as np
import time
import cv2

def kmeans_rgb(img, k):
    print('K =', k, '...')
    start_time = time.time()

    # Reshape the data to 1-D
    data = img.reshape((-1, 3))
    data = np.float32(data)

    # Define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Apply K-means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reshape original image and write image
    center = np.uint8(center)
    result = center[label.flatten()].reshape((img.shape))
    cv2.imwrite('1_1-k' + str(k) + '.jpg', result)

    # Print execution time
    print('<<< %.1s seconds >>>\n' % (time.time() - start_time))


def kmeans_rgbloc(img, k):
    print('K =', k, '...')
    start_time = time.time()

    # Reshape the data to 1-D
    data = img.reshape((-1, 3))
    location = np.mgrid[0:1024, 0:1024]
    lx, ly = location[0].reshape(-1, 1), location[1].reshape(-1, 1)
    data = np.concatenate((data, lx, ly), axis=1)
    data = np.float32(data)

    # Define criteria and apply k-menas
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape original image and write image
    center = np.uint8(center)
    center = center[:, :3] # remove attributes other than rgb
    result = center[label.flatten()].reshape((img.shape))
    cv2.imwrite('1_2-k' + str(k) + '.jpg', result)

    # Print execution time
    print('<<< %.1s seconds >>>\n' % (time.time() - start_time))


def kmeans_optimize(img, k):
    print('K =', k, '...')

    # Reshape the data to 1-D
    data = img.reshape((-1, 3))
    location = np.mgrid[0:1024, 0:1024]
    lx, ly = location[0].reshape(-1, 1), location[1].reshape(-1, 1)

    # normalization
    scalar = (256 / 1024) * 0.3
    lx = lx * scalar
    ly = ly * scalar
    data = np.concatenate((data, lx, ly), axis=1)
    data = np.float32(data)

    # Define criteria and apply k-menas
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape original image and write image
    center = np.uint8(center)
    center = center[:, :3] # remove attributes other than rgb
    result = center[label.flatten()].reshape((img.shape))
    cv2.imwrite('1_3-k' + str(k) + '.jpg', result)



def problem_1_1():
    print('----- Problem 1-1 -----')
    img = cv2.imread('bird.jpg')
    kmeans_rgb(img, 2)
    kmeans_rgb(img, 4)
    kmeans_rgb(img, 8)
    kmeans_rgb(img, 16)
    kmeans_rgb(img, 32)


def problem_1_2():
    print('----- Problem 1-2 -----')
    img = cv2.imread('bird.jpg')
    kmeans_rgbloc(img, 2)
    kmeans_rgbloc(img, 4)
    kmeans_rgbloc(img, 8)
    kmeans_rgbloc(img, 16)
    kmeans_rgbloc(img, 32)


def problem_1_3():
    print('----- Problem 1-3 -----')
    img = cv2.imread('bird.jpg')
    kmeans_optimize(img, 32)


def main():
    print('Applying K-Means on bird.jpg')
    print('-----------------------------')

    problem_1_1()
    problem_1_2()
    problem_1_3()


if __name__ == "__main__":
    main()