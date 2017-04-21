import numpy as np

class Perceptron:

    def __init__(self, training_images, label):
        self.training_images = training_images[:, 0]
        self.image_labels = training_images[:, 1]
        self.label = label
        self.weights = np.zeros(len(self.training_images[0]))

    def train(self):
        learning_rate = .5
        epoch = 10
        bias = .1
        for i, image in enumerate(self.training_images):
            if self.image_labels[i] == self.label:
                this_label = 1
            else:
                this_label = 0
            num_iters = 0
            error = 100000
            while num_iters < epoch:
                y = np.dot(image, self.weights) + bias
                if y > 0:
                    y = 1
                else:
                    y = 0
                #Update weights
                for x in range(len(self.weights)):   # Cycle through each number in weight vector
                    error = this_label - y
                    self.weights[x] += error * image[x]*learning_rate
                num_iters += 1
        return self.weights

    def test(self, image):
        y = np.dot(image, self.weights)
        #error = self.label - y
        return y