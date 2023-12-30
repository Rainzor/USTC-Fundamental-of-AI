import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_image(filepath='./data/ustc-cow.png'):
    img = cv2.imread(filepath) # Replace with the actual path to your image
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class KMeans:
    def __init__(self, k=4, max_iter=10):
        self.k = k
        self.max_iter = max_iter
    

    # # Randomly initialize the centers
    # def initialize_centers(self, points):
    #     '''
    #     points: (n_samples, n_dims,)
    #     '''
    #     n, d = points.shape

    #     centers = np.zeros((self.k, d))
    #     for k in range(self.k):
    #         # use more random points to initialize centers, make kmeans more stable
    #         random_index = np.random.choice(n, size=32, replace=False)
    #         centers[k] = points[random_index].mean(axis=0)

    #     return centers

    def initialize_centers(self, points):
        '''
        points: (n_samples, n_dims,)
        '''
        n, d = points.shape

        # Choose the first center uniformly at random from the data points
        centers = np.zeros((self.k, d))
        first_center_index = np.random.choice(n)
        centers[0] = points[first_center_index]

        # Compute distances from the first center chosen to all the other data points
        distances = np.linalg.norm(points - centers[0], axis=1)
        # Choose the remaining k-1 centers
        for k in range(1, self.k):
            # Select the next center with probability proportional to the distance squared
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            center_index = np.random.choice(n, p=probabilities)

            # Update the centers and distances
            centers[k] = points[center_index]
            distance = np.linalg.norm(points - centers[k], axis=1)
        return centers


    # Assign each point to the closest center
    def assign_points(self, centers, points):
        '''
        centers: (n_clusters, n_dims,)
        points: (n_samples, n_dims,)
        return labels: (n_samples, )
        '''
        n_samples, n_dims = points.shape
        labels = np.zeros(n_samples)
        # TODO: Compute the distance between each point and each center
        # and Assign each point to the closest center
        # L2 distance

        dist = np.zeros((n_samples, self.k))
        for i in range(self.k):
            dist[:,i] = np.sum(np.square(points - centers[i]), axis=1)
        labels = np.argmin(dist, axis=1)    
        return labels
    

    # Update the centers based on the new assignment of points
    def update_centers(self, centers, labels, points):
        '''
        centers: (n_clusters, n_dims,)
        labels: (n_samples, )
        points: (n_samples, n_dims,)
        return centers: (n_clusters, n_dims,)
        '''
        # TODO: Update the centers based on the new assignment of points
        for i in range(self.k):
            cluster_points = points[labels == i]
            if len(cluster_points) > 0:
                centers[i] = cluster_points.mean(axis=0)
        return centers

    # k-means clustering
    def fit(self, points):
        '''
        points: (n_samples, n_dims,)
        return centers: (n_clusters, n_dims,)
        '''
        # TODO: Implement k-means clustering
        centers = self.initialize_centers(points)
        for i in range(self.max_iter):
            labels = self.assign_points(centers, points)
            centers = self.update_centers(centers, labels, points)
        return centers

    def compress(self, img):
        '''
        img: (width, height, 3)
        return compressed img: (width, height, 3)
        '''
        # flatten the image pixels
        points = img.reshape((-1, img.shape[-1]))
        # TODO: fit the points and Replace each pixel value with its nearby center value
        centers = self.fit(points)
        labels = self.assign_points(centers, points)
        for i in range(self.k):
            points[labels == i] = centers[i]
        return points.reshape(img.shape)


if __name__ == '__main__':

    k_list = [2, 4, 8, 16, 32, 64]
    
    plt.figure(figsize=(15, 7.5))

    for index, k in enumerate(k_list):
        kmeans = KMeans(k=k, max_iter=10)
        img = read_image(filepath='../data/ustc-cow.png')
        compressed_img = kmeans.compress(img).round().astype(np.uint8)
        
        plt.subplot(2, 3, index + 1)
        plt.imshow(compressed_img)
        plt.title('k={}'.format(k))
        plt.axis('off')

    plt.savefig('../output/compressed_images.png')
    plt.show()