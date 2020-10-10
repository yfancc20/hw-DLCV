import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score


random_state = 330
train_X = np.zeros((360, 2576))
train_Y = np.zeros(360)
test_X = np.zeros((40, 2576)) 
test_Y = np.zeros(40)

def cv2normalize(x):
    return cv2.normalize(x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def load_dataset():
    # There are 40 persons and 10 images for each person.
    # Each grayscale image's size is 56 x 46 (2576 pixels).
    # 
    # Need to split the data to:
    #   - training set: 9 * 40 = 360 images
    #       Size: (360, 2576)
    #   - testing set: 1 * 40 = 40 images
    #       Size: (40, 2576)

    for i in range(40):
        for j in range(10):
            imgname = 'p2_data/' + str(i+1) + '_' + str(j+1) + '.png'
            img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
            if j < 9:
                train_X[i*9 + j] = np.reshape(img, (1, -1))
                train_Y[i*9 + j] = i + 1
            else:
                test_X[i] = np.reshape(img, (1, -1))
                test_Y[i] = i + 1

    print('Training set:', train_X.shape)
    print('Testing set:', test_X.shape)
    print('-' * 20)


def pca_trainset():
    # Draw mean face
    mean = np.mean(train_X, axis=0).reshape(1, -1)
    mean_img = mean.reshape(56, 46)

    # cv2.imwrite('2_1-meanface.png', mean_img)
    print('Drawing mean face done.\n')

    # Draw the eigenfaces. 
    #   - The training set is (n_samples, n_features)
    pca = PCA(random_state=random_state)
    L = pca.fit(train_X - mean)
    for i in range(0, 4):
        eigenface = L.components_[i].reshape(56, 46)
        eigenface = cv2normalize(eigenface)
        cv2.imwrite('2_1-v' + str(i+1) + '.png', eigenface)
    print('Drawing first 4 eigenfaces done.\n')

    # Reconstruct person2 image1 and project it onto the PCA eigenspace
    n = [3, 50, 170, 240, 345]
    input_img = cv2.imread('p2_data/2_1.png', cv2.IMREAD_GRAYSCALE).reshape(1, -1)
    project = pca.transform(input_img - mean)
    print('Projected:', project.shape)
    print('Eigenspace:', L.components_.shape)
    print('Mean:', mean.shape, '\n')

    print('Reconstructing Person 2 Image 1...')
    for i, j in enumerate(n):
        recon_img = (project[:, :j] @ L.components_[:j]) + mean         
        mse = np.mean((input_img - recon_img) ** 2)
        print('n = %3d,' % j, 'MSE =', round(mse, 4))
        cv2.imwrite('2_2-p2i1-re-by-top-' + str(j) + '.png', cv2normalize(recon_img).reshape(56, 46))

    return pca, mean

def crossvalidation(pca, mean):
    print('\nPerforming 3-fold cross validation and selecting best k, n...')
    candidate_k = [1, 3, 5]
    candidate_n = [3, 50, 170]
    k = 1
    n = 3
    max_value = 0
    train_X_reduced = pca.transform(train_X - mean)

    for i in candidate_k:
        for j in candidate_n:
            knn = KNeighborsClassifier(n_neighbors=i)
            scores = cross_validate(knn, train_X_reduced[:, :j], train_Y, cv=3, scoring='accuracy')
            mean_score = round(np.mean(scores['test_score']), 5)
            if (mean_score > max_value):
                k = i
                n = j
                max_value = mean_score
            print('k = %d, n = %3d, scores =' % (i, j), mean_score) 
    
    return k, n


def predict(k, n, pca, mean):
    test_X_reduced = pca.transform(test_X - mean)
    train_X_reduced = pca.transform(train_X - mean)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_X_reduced[:, :n], train_Y)
    pred_Y = knn.predict(test_X_reduced[:, :n])
    accuracy = accuracy_score(y_pred=pred_Y, y_true=test_Y)
    print('\nAccuracy on testing set:', accuracy)


def main():
    load_dataset()
    pca, mean = pca_trainset()
    k, n = crossvalidation(pca, mean)
    predict(k, n, pca, mean)


if __name__ == "__main__":
    main()