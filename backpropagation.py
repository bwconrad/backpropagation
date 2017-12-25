import random
import math
import time
import copy


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1-logistic(x))

class Neuron:
    def __init__(self, attribute_weights, neuron_weights, bias_weight):
        # neuron.attribute_weights[i] = Weight of input attribute i as input to this neuron
        self.attribute_weights = attribute_weights
        # neuron.neuron_weights[i] = Weight of neuron j as input to this neuron
        self.neuron_weights = neuron_weights
        self.bias_weight = bias_weight

class ANN:
    def __init__(self, num_attributes, neurons):
        # Number of input attributes
        self.num_attributes = num_attributes
        # Number of neurons
        self.neurons = neurons
        for neuron_index, neuron in enumerate(self.neurons):
            for input_neuron, input_weight in neuron.neuron_weights.items():
                assert(input_neuron < neuron_index)

    # Calculates the output of the output neuron for given input attributes.
    def calculate(self, attributes):
        # Get outputs from hidden layer 
        num_hidden = len(self.neurons)-1
        hidden_out = [0 for x in range(num_hidden)]
        for i in range(num_hidden):
            # weights*input summation into neuron i
            neuron_sum = 0 
            for a in range(self.num_attributes):
                neuron_sum = neuron_sum + attributes[a] * self.neurons[i].attribute_weights[a] # input*weight
            neuron_sum = neuron_sum - self.neurons[i].bias_weight # Subtract bias from sum
            hidden_out[i] = logistic(neuron_sum) # Apply activation function on sum and output from neuron

        # Get output of network
        output = 0 
        # Sum all in inputs into output neuron
        for i in range(num_hidden):
            output = output + hidden_out[i]*self.neurons[-1].neuron_weights[i]
        output = output - self.neurons[-1].bias_weight # Subtract bias from sum
        output = logistic(output) # Apply activation function

        return output

    # Returns the squared error of a collection of examples:
    def squared_error(self, example_attributes, example_labels):
        error = 0 
        # Loop through all input combinations and sum errors
        for i in range(len(example_attributes)):
            error = error + (example_labels[i] - ann.calculate(example_attributes[i]))**2
        error = error * 0.5
        return error

    # Runs backpropagation on a single example in order to
    # update the network weights appropriately.
    def backpropagate_example(self, attributes, label, learning_rate=1.0):
        num_hidden = len(self.neurons)-1

        ### Propagate forward to get get inputs and outputs of neurons
        ins = [0 for x in range(len(self.neurons))] # Sums into neurons
        outs = [0 for x in range(len(self.neurons))] # values after activation

        # Hidden layer
        for i in range(num_hidden):
            for a in range(self.num_attributes):
                ins[i] = ins[i] + attributes[a] * self.neurons[i].attribute_weights[a] # input*weight
            ins[i] = ins[i] - self.neurons[i].bias_weight # Subtract bias from sum
            outs[i] = logistic(ins[i]) # Apply activation function on sum and output from neuron

        # Output layer
        for i in range(num_hidden):
            ins[-1] = ins[-1] + outs[i] * self.neurons[-1].neuron_weights[i]
        ins[-1] = ins[-1] - self.neurons[-1].bias_weight # Subtract bias from sum
        outs[-1] = logistic(ins[-1]) # Apply activation function


        ### Propagate errors backwards
        errors = [0 for x in range(len(self.neurons))]

        errors[-1] = (label-outs[-1]) * logistic_derivative(ins[-1]) # Output neuron error

        # Hidden layer errors
        for i in range(num_hidden):
            errors[i] = logistic_derivative(ins[i]) * self.neurons[-1].neuron_weights[i] * errors[-1] 


        # Update weights from hidden to output layer
        for i in range(num_hidden):
            self.neurons[-1].neuron_weights[i] = self.neurons[-1].neuron_weights[i] + (learning_rate * outs[i] * errors[-1])

        # Update weights from input to hidden layers
        for i in range(num_hidden):
            for j in range(len(self.neurons[i].attribute_weights)):
                self.neurons[i].attribute_weights[j] = self.neurons[i].attribute_weights[j] + (learning_rate * attributes[j] * errors[i])

        # Update biases
        for i in range(num_hidden+1):
            self.neurons[i].bias_weight =  self.neurons[i].bias_weight + (learning_rate * -1 * errors[i])


    # Runs backpropagation on each example, repeating this process
    # num_epochs times.
    def learn(self, example_attributes, example_labels, learning_rate=1.0, num_epochs=100):
        for i in range(num_epochs):
            for j in range(len(example_attributes)):
                ann.backpropagate_example(example_attributes[j], example_labels[j], learning_rate=learning_rate)


# XOR gate cominations and outputs
example_attributes = [ [0,0], [0,1], [1,0], [1,1] ]
example_labels = [0,1,1,0]

def random_ann(num_attributes=2, num_hidden=2):
    neurons = []
    # hidden neurons
    for i in range(num_hidden):
        attribute_weights = {attribute_index: random.uniform(-1.0,1.0) for attribute_index in range(num_attributes)}
        bias_weight = random.uniform(-1.0,1.0)
        neurons.append(Neuron(attribute_weights,{},bias_weight))
    # output neuron
    neuron_weights = {input_neuron: random.uniform(-1.0,1.0) for input_neuron in range(num_hidden)}
    bias_weight = random.uniform(-1.0,1.0)
    neurons.append(Neuron({},neuron_weights,bias_weight))
    ann = ANN(num_attributes, neurons)
    return ann


# Run network 10 times and get the best error
best_ann = None
best_error = float("inf")
for instance_index in range(10):
    ann = random_ann()
    ann.learn(example_attributes, example_labels, learning_rate=10.0, num_epochs=10000)
    error = ann.squared_error(example_attributes, example_labels)
    if error < best_error:
        best_error = error
        best_ann = ann

print('Network Error: ' + str(best_error))


