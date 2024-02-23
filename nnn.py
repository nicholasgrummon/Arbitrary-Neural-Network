import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This class implements an arbitrary dimensional neural network (any number
of layers and nodes). 

Specs
The design uses a simple positive difference error function and
compresses all nodes to range 0 through 1 (a probability) with the
sigmoid function p = 1 / ( 1 + e^-x ). The network uses randomly seeded
weights and concatenation bias i.e. bias is added as an additional column of
ones in the node matrix. This increases the computational complexity of the
network without complicating the gradient calculations

About Neural Networks
A neural network is a computationally interconnected series of weights and
biases that can be tuned to provide a machine learning model. A normal neural
network design consists of several separate layers, each composed of a series 
of nodes that hold probabilities. Each node is connected to all other nodes
in the adjacent layers by a weight multiplier and sometimes by a bias offset.
These features act to preferentially highlight certain nodes and paths through
the network. As the neural network is trained, these weights and biases are
updated, and the network begins to develop reward pathways as it converges to
a solution map, almost like real neuroplasticity.

With a little matrix math and gradient calculus, a feed forward and back
propagation algorithm handle training. A deeper mathematical explanation can
be found under the feedForward(), backProp(), and train() definitions, but
these functions essentially just take the given input and pass it through an
untrained network over and over again, each time determining error from the
expected result and incrementing the weights and biases to reduce this error
in response.

As a result, a neural network must have large amounts of data to learn from
because it is effectively just reinforcing pattern recognition. However,
over-training is a common problem that arises in working with neural networks.
When a network learns the training data too well, it doesn't do a good job
predicting new data. It's important to remember that neural networks are, in
the end, approximation tools, not a form of generative intelligence. While
they are much more powerful than the linear regressions you performed in high
school, and while the types of problems they can predict are practically only
limited by the imagination, they are still just a mathematical error
reduction algorithm that cannot avoid the pitfalls of extrapolation and
sampling bias.
"""

class Network():
    ''' Arbitrary neural network class '''

    def __init__(self, dimensions):
        '''
        Object initializer.
        param: dimensions - list with the number of nodes in each layer. The number of layers is arbitrary
        '''
        self.N = len(dimensions)        # number of layers
        self.W = len(dimensions) - 1    # number of weight matrices

        self.nodes = []     # list of N node arrays
        self.probs = []     # list of N probability arrays

        self.weights = []   # list of N-1 weight arrays between each layer
        self.delta = []    # list of N-1 gradient arrays corresponding to each weight array

        for i in range(len(dimensions)):
            self.nodes.append(np.ones((1, dimensions[i])))
            self.probs.append(np.ones((1, dimensions[i])))

        self.weights.append(np.random.randn(self.nodes[0].shape[1],self.nodes[1].shape[1]))
        self.delta.append(np.ones((self.nodes[0].shape[1],self.nodes[1].shape[1])))
        for i in range(1, self.N - 1):
            self.weights.append(np.random.randn(self.nodes[i].shape[1]+1, self.nodes[i+1].shape[1]))
            self.delta.append(np.ones((self.nodes[i].shape[1]+1, self.nodes[i+1].shape[1])))

    def sig(self, inputs):
        '''
        sigmoid function
        compresses any input into the range 0 to 1 (a probability)
        '''
        return 1 / (1 + np.exp(-inputs))
        
    def dSig_d(self,inputs):
        '''
        sigmoid derivative function
        '''
        return self.sig(inputs) * (1 - self.sig(inputs))
    
    def cost(self, target):
        return np.mean(abs(self.probs[self.N - 1] - target))
    
    def feedForward(self, inputs):
        '''
        passes given inputs through network
        '''
        if inputs.shape[1] != self.nodes[0].shape[1]:
            raise Exception("Inputs do not match the specified neural network dimensions")
        else:
            self.probs[0] = inputs
            for i in range(1, self.N-1):
                self.nodes[i] = np.matmul(self.probs[i-1], self.weights[i-1])
                self.probs[i] = self.sig(self.nodes[i])

                self.bias = np.ones((self.probs[i].shape[0], 1))
                self.probs[i] = np.concatenate((self.bias, self.probs[i]), axis=1)

            self.nodes[self.N-1] = np.matmul(self.probs[self.N-2], self.weights[self.N-2])
            self.probs[self.N-1] = self.sig(self.nodes[self.N-1])

    def backProp(self, target):
        # determines gradient of NN wrt to each weight
        # target is expected result from training data
        # this model uses Error = | target - Y | where Y is solution to NN
        if target.shape != self.probs[self.N - 1].shape:
            raise Exception("Target does not match the specified neural network dimensions")
        
        else:
            base = self.probs[self.W] - target
            i = self.W-1
            while i>0:
                self.delta[i] = np.matmul(self.probs[i].T, base)
                base = base.dot(self.weights[i][1:,:].T) * self.dSig_d(self.nodes[i])
                i-=1
            
            self.delta[0] = np.matmul(self.probs[0].T, base)


    def train(self, inputs, target, epochs, lr, showErrorReport=False,):
        costs = []
        for n in range(epochs):
            self.feedForward(inputs)
            self.backProp(target)

            for w in range(self.W):
                self.weights[w] -= lr*(1/len(inputs))*self.delta[w]
            
            c = self.cost(target)
            costs.append(c)

            if showErrorReport and (n % 1000 == 0):
                print(f"iteration: {n}. Error: {c}" )

        print("Done Training")
        
        plt.plot(costs)
        plt.show()

    def solve(self, inputs):
        print("Test Case: ")
        print(inputs)
        self.feedForward(inputs)
        print("Probabilities: ")
        print(self.probs[self.N - 1])
        print("Predictions: ")
        print(np.round(self.probs[self.N - 1]))

    def loadData(self, filepath):
        data = pd.read_excel(filepath)
        inputs = pd.DataFrame()
        sol = pd.DataFrame()
        for column in data.columns:
            if column[0] == "X":
                inputs[column] = data[column]
            elif column[0] == "Y":
                sol[column] = data[column]

        inputs = np.array(inputs)
        sol = np.array(sol)

        return inputs, sol

    def saveTunedNetwork(self, filePath):
        textFile = open(filePath, "w")
        for w in range(len(self.weights)):
            textFile.write("LAYER {}n ".format(w))

            height = len(self.weights[w])
            width = len(self.weights[w][0])
            textFile.write("DIM [{} x {}]\n".format(height, width))
            for i in range(len(self.weights[w])):
                row = ""
                for j in range(len(self.weights[w][i])):
                    row += (str(self.weights[w][i][j]) + " ")
                row += "\n"
                textFile.write(row)

            textFile.write("\n\n")

        textFile.close()

    def loadPreTunedNetwork(self, filepath):
        textFile = open(filepath, "r")
        
        j = 0
        for row in textFile:
            if (row[0:5] == "LAYER"):
                layer = int(row[6:(row.find("n"))])
                height = int(row[(row.find("[")+1):(row.find("x"))])
                width = int(row[(row.find("x")+1):(row.find("]"))])
                j = 0
                continue

            if (row == "\n" or row[0] == "#"):
                continue
            
            x = 0
            i = 0
            while i < width:
                y = row.find(" ",x+1)
                self.weights[layer][j][i] = float(row[x:y])
                i += 1
                x = y

            j += 1
    
    def printNN(self):
        for w in range(len(self.weights)):
            print("LAYER {}".format(w))
            for i in range(len(self.weights[w])):
                row = ""
                for j in range(len(self.weights[w][i])):
                    row += (str(self.weights[w][i][j]) + " ")
                print(row)

            print("\n")
