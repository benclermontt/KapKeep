import glob
import cv2
import numpy as np
import timeit
import time
from skimage import draw
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split


def normalize_vector(vector):
    try:
        y = np.sqrt(sum(i ** 2 for i in vector))
        x = vector / y
    except RuntimeWarning as ex:
        print()
    return x


def histogram_of_nxn_cells(direction, magnitude, cell_size=8):
    """
    @author: Ben Clermont, Nicholas Nordstrom
    returns a list of 2d array of cells containing histograms for each cell
    NOTE: assumes that image is divisible by cell size

    :param magnitude: magnitude array of image (same shape as image)
    :param direction: direction array of image (same shape as image)
    :param cell_size: size of the cells to break the image into
    :return: row-major list of bins for each nxn cell
    """

    num_bins = 9
    num_cells_vertically = int(direction.shape[1] / cell_size)
    num_cells_horizontally = int(direction.shape[0] / cell_size)
    binz = np.zeros((num_cells_horizontally, num_cells_vertically, num_bins))
    bin_inc = 20

    # non-vectorized solution for simplicity. may revisit for efficiency
    for c_vert in range(num_cells_vertically):
        for c_hor in range(num_cells_horizontally):
            for i in range(cell_size):
                for j in range(cell_size):
                    d = direction[c_hor * cell_size + i][c_vert * cell_size + j]
                    m = magnitude[c_hor * cell_size + i][c_vert * cell_size + j]

                    direction_lower_bin = int(d // bin_inc)
                    direction_upper_bin = int((d // bin_inc) + 1) % num_bins
                    direction_split = (d / num_bins) % 1

                    binz[c_hor][c_vert][direction_lower_bin] += m * direction_split
                    binz[c_hor][c_vert][direction_upper_bin] += m * (1 - direction_split)

    return binz


def normalize_bins(binz, block_size=2):
    """
    @author: Ben Clermont, Nicholas Nordstrom
    normalize lighting in blocks of block_size x block_size cells/bins
    create feature vectors of HOG
    :param block_size: number of cells to combine into one block to normalize
    :param binz: Histogram of cells
    :return: normalized bin matrix
    """

    cells_in_row = binz.shape[0]
    cells_in_col = binz.shape[1]
    normalized_binz = np.zeros((cells_in_row - 1, cells_in_col - 1, binz.shape[2]))

    for curr_row in range(cells_in_row - 1):
        for curr_col in range(cells_in_col - 1):
            vector = binz[curr_row][curr_col]
            np.append(vector, binz[curr_row + 1][curr_col])
            np.append(vector, binz[curr_row][curr_col + 1])
            np.append(vector, binz[curr_row + 1][curr_col + 1])
            vector = normalize_vector(vector)
            normalized_binz[curr_row][curr_col] = vector

    return normalized_binz


def normalize_gamma(image, gamma=1.0):
    """
    Normalizes the gamma values of the passed frame, making brighter areas darker and darker areas brighter.
    It does this by creating an array of gamma values that maps the input pixel values to an output value.

    For Ex: The table could say that if the input gamma is 75, the output gamma will be 90 (brighter).

    :param image: The img matrix
    :param gamma: The gamma threshold to normalize with
    :return A look-up table transform of the image matrix using the created gamma table
    """
    inverse_gamma = 1 / gamma

    gamma_table = np.array([((i / 255.0) ** inverse_gamma) * 255
                            for i in np.arange(0, 256)]).astype('uint8')

    return cv2.LUT(image, gamma_table)


def visualize_vectors(image, binz, cell_size=8, length=4):
    """
    @author: Ben Clermont, Nicholas Nordstrom

    Overlays vector lines on an image to visualize magnitude and direction vectors

    NOTE: length MUST BE LESS THAN OR EQUAL TO cell_size / 2
    NOTE: cell_size must be the same cell size used to make the histogram

    :param image: image to overlay lines on
    :param cell_size: square chunk of image to draw a vector line for
    :param length: maximum length of vector lines (NOTE: MUST BE LESS THAN OR EQUAL TO cell_size / 2)
    :return: image with vectors drawn on it
    """
    s_row = image.shape[0]
    s_col = image.shape[1]
    radius = cell_size / 2
    num_cells_row = binz.shape[0]
    num_cells_col = binz.shape[1]

    orientations_arr = np.arange(9)
    orientations_bin_midpoint = (np.pi * (orientations_arr + .5) / 8)
    dr_arr = radius * np.sin(orientations_bin_midpoint)
    dc_arr = radius * np.cos(orientations_bin_midpoint)
    hog_image = np.zeros((s_row, s_col))

    for r in range(num_cells_row):
        for c in range(num_cells_col):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                center = tuple([r * cell_size + cell_size // 2,
                                c * cell_size + cell_size // 2])

                rr, cc = draw.line(int(center[0] - dc),
                                   int(center[1] + dr),
                                   int(center[0] + dc),
                                   int(center[1] - dr))

                x = binz[r][c][o]
                hog_image[rr, cc] += x
                image[rr, cc] = (image[rr, cc] + [0, 0, int(x * 50)])

    """
    Displays soloed vectors of the histogram on a white background
    Great for actually visualizing where the edges are.
    """

    # fig = plt.imshow(hog_image, cmap=plt.cm.binary)
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    # plt.show()

    return image


def process_frame(frame):
    adjusted_frame = normalize_gamma(frame, 1.25)

    # Calculate Gradients

    adjusted_frame = np.float32(adjusted_frame) / 255.0

    """
    We might want to use an alternative to a Sobel in the future
    
    However the research paper said they got the best results with a simple 1x3 mask
    In this case the Sobel is creating a 1x3/3x1 mask (depending on x or y derivative) the contains
    
    [-1, 0, 1]
    
    We may also want to explore calculating the gradient invididually for each color channel to see which gives
    the lowest normal value and use that each frame. This depends on how computationally heavy later steps will be.
    """

    gradient_x = cv2.Sobel(adjusted_frame, cv2.CV_32F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(adjusted_frame, cv2.CV_32F, 0, 1, ksize=1)

    """
    To find magnitude and direction of the gradients
    Convert from cartesian to Polar with OpenCV function
    
    This may be a formula we implement ourselves later to look cool
    """
    magnitude, direction = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    avg_magnitude = np.amax(magnitude, axis=2)
    avg_direction = np.amax(direction, axis=2)
    avg_direction = avg_direction // 2

    binz = histogram_of_nxn_cells(avg_direction, avg_magnitude)

    # visualized_image = visualize_vectors(frame, binz)
    binz = normalize_bins(binz)

    return binz.ravel()


def train_SVC(X_train, y_train):
    """
        Function to train an svm.
    """
    svc = svm.LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return svc


def test_classifier(svc, X_test, y_test):
    """
        Funtion to test the classifier.
    """
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    pred = svc.predict(X_test[0:n_predict])
    actual = y_test[0:n_predict]
    print('My SVC predicts: ', pred)
    print('For these',n_predict, 'labels: ', actual)
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


def setup_train_data():

    a = timeit.default_timer()
    t_start = timeit.default_timer()
    peds = [process_frame(cv2.resize(cv2.imread(im), (32, 64))) for im in glob.glob('../Dataset/data_jpg/1_*.jpg', recursive=True)]

    nopeds = [process_frame(cv2.resize(cv2.imread(im), (32, 64))) for im in glob.glob('../Dataset/data_jpg/0_*.jpg', recursive=True)]

    t_end = timeit.default_timer()
    print(f'Histogram Creation took: {(t_end - t_start)} Seconds')
    # Peds should now contain a list ravelled histograms for each image

    stack = np.vstack((peds, nopeds)).astype(np.float64)

    stack_scaler = StandardScaler().fit(stack)

    scaled_stack = stack_scaler.transform(stack)

    y_train = np.hstack((np.ones(len(peds)), np.zeros(len(nopeds))))

    ped_train, ped_test, y_train, y_test = train_test_split(scaled_stack, y_train, test_size=0.2, random_state=2)

    svc = train_SVC(ped_train, y_train)

    test_classifier(svc, ped_test, y_test)

    b = timeit.default_timer()
    print("entire operation took", round(b-a, 5), "seconds")

    unshaped_image = process_frame(cv2.resize(cv2.imread('../Dataset/1_289.jpg'), (32, 64)))
    prediction_image = unshaped_image.reshape(1, -1)

    test = svc.predict(prediction_image)
    print(test)


def setup_train_data_prebuilt():
    """
    @author Nicholas Nordstrom

    sends data to a prebuilt HOG to compare accuracies and timing to other implementations with the same dataset

    :return: None
    """

    start = timeit.default_timer()
    hog_1 = [hog(cv2.resize(cv2.imread(im), (32, 64))) for im in glob.glob('../Dataset/data_jpg/1_*.jpg', recursive=True)]
    hog_0 = [hog(cv2.resize(cv2.imread(im), (32, 64))) for im in glob.glob('../Dataset/data_jpg/0_*.jpg', recursive=True)]
    end = timeit.default_timer()
    print("file io and prebuilt HOG calculations took", round(end - start, 3), "seconds")

    x = hog_1 + hog_0
    y = [1]*len(hog_1) + [0]*len(hog_0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    clf = train_SVC(x_train, y_train)
    test_classifier(clf, x_test, y_test)

    end = timeit.default_timer()
    print("entire operation took", round(end-start, 5), "seconds")

    unshaped_image = hog(cv2.resize(cv2.imread('../Dataset/1_289.jpg'), (32, 64)))
    prediction_image = unshaped_image.reshape(1, -1)

    test = clf.predict(prediction_image)
    print(test)


def main():
    print("OUR IMPLEMENTATION:")
    setup_train_data()

    print("\n\nPREBUILT:")
    setup_train_data_prebuilt()


if __name__ == '__main__':
    main()
