import pickle
from perceptron import Perceptron
import numpy as np




def main():
    perceptrons = []
    training_images = np.array(pickle.load(open( "train.p", "rb")))
    test_images = np.array(pickle.load(open( "test.p", "rb")))

    # Fire up and train perceptrons for each number 0-9
    for i in range(10):
        perceptrons.append(Perceptron(training_images, i))
        perceptrons[i].train()

    # Test perceptrons
    correct = 0
    total = 0
    for i, test_img in enumerate(test_images):
        best_confidence = -10000
        best_guess = -1
        # put image through eachperceptron
        for y in range(10):
            confidence = perceptrons[y].test(test_img[0])
            if confidence > best_confidence:
                best_confidence = confidence
                best_guess = y
        if best_guess == test_img[1]:
            #print(best_guess)
            correct += 1
        total += 1
    #print(correct)
    print(correct/total)
main()
