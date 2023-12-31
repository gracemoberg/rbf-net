'''kmeans.py
Performs K-Means clustering
Grace Moberg
CS 252: Mathematical Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        self.num_samps = data.shape[0]
        self.num_features = data.shape[1]

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        dist = np.sqrt(np.sum((pt_1 - pt_2)**2))
        return dist

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        distances = np.sqrt(np.sum((pt - centroids)**2, axis=1))
        return distances

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''

        nRows = self.data.shape[0]
        self.num_features = self.data.shape[1]
        rand_indices = np.random.choice(nRows, size=k)
        arr = self.data[rand_indices, :].copy()
        #self.centroids = arr
        return arr

    def initialize_plusplus(self, k):
        '''Initializes K-means by setting the initial centroids (means) according to the K-means++
        algorithm

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        TODO:
        - Set initial centroid (i = 0) to a random data sample.
        - To pick the i-th centroid (i > 0)
            - Compute the distance between all data samples and i-1 centroids already initialized.
            - Create the distance-based probability distribution (see notebook for equation).
            - Select the i-th centroid by randomly choosing a data sample according to the probability
            distribution.
        '''

        #for the first, intialize randomly
        centroidinds = np.random.choice(np.arange(len(self.data)), size=1, replace=False)
        labels = self.update_labels(self.data[centroidinds])

        #for the rest of the centroids
        for i in range(k - 1):
            #calculate the probability
            num = np.power([self.dist_pt_to_pt(self.data[i], self.data[centroidinds[labels[i]], :])for i in range(len(self.data))],2)
            denom = np.sum(np.power([self.dist_pt_to_pt(self.data[i], self.data[centroidinds[labels[i]], :])for i in range(len(self.data))], 2))
            probs = num/denom

            #pick a new one based on probability
            index = np.random.choice(np.arange(len(self.data)), size = 1, replace = False, p=probs)
            centroidinds = np.append(centroidinds, index, axis = 0)
            #assign labels so we can get the closest centroid for each 
            labels = self.update_labels(self.data[centroidinds])
        
        #get the centroids based on ids
        self.centroids = self.data[centroidinds]
        return self.centroids        


    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, init_method = 'random'):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        iter = 0
        self.k = k
    
        if init_method == 'random':
            self.centroids = self.initialize(k)
        elif init_method == 'kmeans++':
            self.centroids = self.initialize_plusplus(k)

        centroid_diff = 1000

        while iter < max_iter and centroid_diff > tol:
            iter = iter + 1
            labels = self.update_labels(self.get_centroids())
            cents = self.get_centroids()
            new_centroids, centroid_diff = self.update_centroids(k, labels, cents)
            centroid_diff = np.max(np.abs(centroid_diff))
            self.centroids = new_centroids
            

        # self.centroids = new_centroids
        self.data_centroid_labels = np.array(labels)
        self.inertia = self.compute_inertia()
        
        return self.inertia, iter

    def cluster_batch(self, k=2, n_iter=10, verbose=False, init_method='random'):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        _________

        avg_iters, the average number of iterations it takes the algorithm to converge
        '''
        
        best = np.inf
        iterations = []
       
        for j in range(n_iter):
            curr_inertia, iters = self.cluster(k, init_method=init_method)
            iterations.append(iters)
            if curr_inertia < best:
                best = curr_inertia
                self.centroids = self.get_centroids()
                self.data_centroid_labels = self.get_data_centroid_labels()
                self.inertia = best

        avg_iters = np.sum(iterations)/len(iterations)

        return avg_iters
        

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        indices = []
        for i in range(self.num_samps):
            distances = self.dist_pt_to_centroids(self.data[i,:], centroids)
            index = int(np.argmin(distances))
            indices.append(index)
        arr = np.array(indices)
        return arr

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
    
        new_centroids = np.zeros(prev_centroids.shape) #prev_centroids.copy()

        for i in range(k):
            vals = self.data[data_centroid_labels == i, :]
            if vals.size == 0:
                new_centroid = self.data[np.random.randint(0, self.data.shape[0]), :]
            else: 
                new_centroid = np.mean(vals, axis=0)
            new_centroids[i] = new_centroid

        self.centroids = new_centroids

        centroid_diff = new_centroids - prev_centroids

        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        
        distances = []

        for i in range(self.data.shape[0]):
            dist = self.dist_pt_to_pt(self.data[i, :], self.centroids[self.data_centroid_labels[i], :])
            distances.append(dist**2)

        inertia = np.mean(distances)

        return inertia

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''

        brewer_colors = cartocolors.qualitative.Safe_10.mpl_colors

        for i in range(self.k):
            dat = self.data[self.data_centroid_labels == i]
            plt.scatter(dat[:,0], dat[:,1], color = brewer_colors[i])
            plt.scatter(self.centroids[i,0], self.centroids[i,1], marker = '*', color='black', s=200)

        plt.xlabel('X')
        plt.ylabel('Y')

    def elbow_plot(self, max_k, n_iter):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: number of iterations

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        
        inertia = []
        iterations = [i for i in range(1, max_k + 1)]
        for i in range(1, max_k + 1):
            # iner, _ = self.cluster(i)
            # inertia.append(iner)
            for j in range(1, n_iter+1, 1):
                self.cluster_batch(i,j)
            inertia.append(self.inertia)
        plt.plot(iterations, inertia)
        plt.xticks([i for i in range(1, max_k + 1)])
        plt.xlabel("number of clusters k")
        plt.ylabel("inertia")

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        data = np.copy(self.data)
        for i in range(self.data.shape[0]):
            index = self.data_centroid_labels[i]
            data[i] = self.centroids[index]
        self.data = data

    def sillhouette_score(self):
        '''Calculates sillhouette score for each individual data point and the sillhouette coefficient for data.
         Equations provided in the notebook '''
    
        sillhouettes = np.zeros(self.data.shape[0])
    
        for j in range(self.data.shape[0]):
            a = self.dist_pt_to_pt(self.data[j, :], self.centroids[self.data_centroid_labels[j], :])
            b = self.dist_pt_to_centroids(self.data[j, :], self.centroids)
            b_altered = np.delete(b, self.data_centroid_labels[j])
            fin_b = np.min(b_altered)
            s = (fin_b - a)/max(a, fin_b) 
            sillhouettes[j] = s

        sc = (1/self.data.shape[0])*np.sum(sillhouettes)

        return sillhouettes, sc



