'''rbf_net.py
Radial Basis Function Neural Network
Grace Moberg
CS 252: Mathematical Data Analysis Visualization
Spring 2023
'''
import numpy as np
import kmeans
import scipy.linalg
import time


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

        # number of hidden units
        self.k = num_hidden_units

        # number of classes
        self.num_classes = num_classes

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        distances = np.zeros(centroids.shape[0])

        for i in range(centroids.shape[0]):
            class_vals = data[cluster_assignments==i, :]
            dists = []
            for item in class_vals:
                dist = kmeans_obj.dist_pt_to_pt(item, centroids[i])
                dists.append(dist)
            distances[i] = np.mean(dists)

        return distances


    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        kMeansObj = kmeans.KMeans(data)
        kMeansObj.cluster_batch(k=self.k, n_iter = 10, init_method='kmeans++')
        self.prototypes = kMeansObj.get_centroids()
        cluster_assigns = kMeansObj.get_data_centroid_labels()
        sigmas = self.avg_cluster_dist(data, self.prototypes, cluster_assigns, kMeansObj)
        self.sigmas = sigmas

        
    ## code from my own proj 3 qr solver
    def linear_regression(self, A, y):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''
        A = np.hstack((A, np.ones((A.shape[0], 1))))
        Q = np.ones([A.shape[0], A.shape[1]])
        i=0 # pointer to keep track of which column of A we are in
        j=0 # pointer to keep track of which column of Q we are in
        
       
        while i < len(range(A.shape[1])):
            if i == 0:
                temp = A[:,i].copy()
                temp = temp/np.linalg.norm(temp)
                Q[:,i] = temp
                i += 1

            if A.shape[1] == 1:
                break

            temp = A[:,i].copy()
            proj = 0

            while j < i:
                u = Q[:,j].copy()
                scalar = np.dot(u,temp)
                prod = u*scalar
                proj += prod
                j += 1
            
            orthT = temp - proj
                
            norm = np.linalg.norm(orthT)

            Q[:,i] = orthT/norm
            j = 0
            i+=1

        R = Q.T@A
        rhs = Q.T @ y
        c = scipy.linalg.solve_triangular(R, rhs)

        return c

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        k_means = kmeans.KMeans(data)
        activations = np.zeros((data.shape[0], self.k))

        for i in range(data.shape[0]):
            distances = k_means.dist_pt_to_centroids(data[i,:], self.get_prototypes())**2
            num = distances/(2*self.sigmas**2 + 0.000001)
            activations[i, :] = np.exp(-num)

        return activations

    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        ones = np.ones((hidden_acts.shape[0],1))
        h = np.hstack((hidden_acts, ones))
        outputs = h@self.wts
        return outputs
    

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        # init_seconds = time.time()
        self.initialize(data)
        # print('initialize time: ', time.time()-init_seconds)
        # act_seconds = time.time()
        activations = self.hidden_act(data)
        # print('time for hidden act: ', time.time()-act_seconds)
        binary_y = np.zeros((y.shape[0], self.num_classes))
        weights = np.zeros((self.k+1, self.num_classes))

        # coding_seconds = time.time()
        for i in range(self.num_classes):
            for j in range(0, y.shape[0]):
                if y[j]==i:
                    binary_y[j][i]=1
                else:
                    binary_y[j][i]=0

        # print('recoding seconds: ', time.time()-coding_seconds)

        # lin_seconds = time.time()
        for i in range(self.num_classes):
            c = self.linear_regression(activations, binary_y[:,i])
            weights[:,i] = c

        # print('lin reg seconds: ', time.time()-lin_seconds)

        self.wts = weights
            

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        y_pred = np.zeros(data.shape[0])

        hidden = self.hidden_act(data)
        output = self.output_act(hidden)

        for i in range(output.shape[0]):
            item = np.argmax(output[i])
            y_pred[i] = item

        return y_pred

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        a = np.sum(y == y_pred) / y.shape[0]
        return a    


class RBF_Reg_Net(RBF_Net):
    '''RBF Neural Network configured to perform regression
    '''
    def __init__(self, num_hidden_units, num_classes, h_sigma_gain=5):
        '''RBF regression network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset
        h_sigma_gain: float. Multiplicative gain factor applied to the hidden unit variances

        TODO:
        - Create an instance variable for the hidden unit variance gain
        '''
        super().__init__(num_hidden_units, num_classes)

        self.gain = h_sigma_gain

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation

        TODO:
        - Copy-and-paste your classification network code here.
        - Modify your code to apply the hidden unit variance gain to each hidden unit variance.
        '''
        k_means = kmeans.KMeans(data)
        activations = np.zeros((data.shape[0], self.k))

        for i in range(data.shape[0]):
            distances = k_means.dist_pt_to_centroids(data[i,:], self.get_prototypes())**2
            denom = (2*self.gain*(self.sigmas**2) + 0.000001)
            num = distances/denom
            activations[i, :] = np.exp(-num)

        return activations

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the desired y output of each training sample.

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to perform regression on
        the actual y values instead of the y values that match a particular class. Your code should be
        simpler than before.
        - You may need to squeeze the output of your linear regression method if you get shape errors.
        '''
        self.initialize(data)
        activations = self.hidden_act(data)
        weights = np.zeros((self.k+1, self.num_classes))
        for i in range(self.num_classes):
            c = self.linear_regression(activations, y[:,i])
            weights[:,i] = c

        self.wts = weights

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_neurons). Output layer neuronPredicted "y" value of
            each sample in `data`.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to return the RAW
        output neuron activaion values. Your code should be simpler than before.
        '''

        hidden = self.hidden_act(data)
        y_pred = self.output_act(hidden)

        return y_pred
