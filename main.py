import numpy as np
import matplotlib.pyplot as plt
import collections
import asyncio

'''
Activation function
'''
def activation(A): 
    return 1/(1.0+np.exp(-1*A))

'''
Derivative of activation function
'''
def activation_derivative(A):
    return A * (1 - A)

'''
NN class
'''
class NeuralNetwork:
    def __init__(self, eta, epochs, bias):
        # Define weights
        self.hidden_weights = np.array([[0.3, 0.3], [0.3, 0.3]])
        self.output_weights = np.array([[0.8], [0.8]]) 

        # Define learning rate
        self.eta = eta
        # Define number of epochs
        self.epochs = epochs
        # Define bias 
        self.output_bias = [[bias], [bias]]
        self.hidden_bias = [bias, bias]
        self.errors = collections.defaultdict(int)

    '''
    Forward propagation
    '''
    async def forward(self, input_nodes):
        print('')
        print('----------------------forward-------------------')
        print('-------------------------------------------------')

        # calculate inputs to hidden nodes
        hidden_input = np.dot(input_nodes, self.hidden_weights.T) + self.hidden_bias
        # hidden_input = np.dot(input_nodes, self.hidden_weights.T)
        hidden_output = activation(hidden_input)

        # calculate inputs to output node
        output_node_input = np.dot(hidden_output, self.output_weights) + self.output_bias
        output = activation(output_node_input)
        return output, hidden_output
    
    '''
    Backpropagation
    '''
    async def backprop(self, input_nodes, hidden_output, output, desired_output):
        print('')
        print('-----------------------')
        print('-----begin backprop----')
        error = desired_output - output

        # grad of the loss w/ respect to the output
        output_delta = error * activation_derivative(output)

        # update weights that connect hidden layer <-> output node
        hidden_output = hidden_output.reshape(-1, 1)

        # grad of loss w/ respect to output from hidden layer
        hidden_layer_delta = activation_derivative(hidden_output) * np.dot(output_delta, self.output_weights.T)

        # update bias 
        self.output_bias = self.output_bias + (self.eta * np.mean(output_delta, axis=0, keepdims=True))
        self.hidden_bias = self.hidden_bias + (self.eta * np.mean(hidden_layer_delta, axis=0, keepdims=True))

        # update weights that connect inputs between hidden layer
        self.output_weights += self.eta * hidden_output * output_delta
        self.hidden_weights += self.eta * input_nodes * hidden_layer_delta


        print('')
        print('----------------------backprop-------------------')
        print('-------------------------------------------------')
        print('hidden_weights for input: ', input_nodes, {
            input_nodes[0]: [self.hidden_weights[0][0], self.hidden_weights[1][0]],
            input_nodes[1]: [self.hidden_weights[0][1], self.hidden_weights[1][1]]
        })
        print('output_weights for input: ', input_nodes, self.output_weights)

    '''
    Training for baseline
    '''
    async def train(self, input_nodes, desired_output):
        print('self epcoh: ', self.epochs)
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(input_nodes)):
                # forward pass
                output, hidden_output = await self.forward(input_nodes[i])
        
                # backpropagation and update the weights
                await self.backprop(input_nodes[i], hidden_output, output, desired_output[i])
                # error
                total_error += 0.5 * (desired_output[i] - output) ** 2
                print('total_error', total_error)

    '''
    Method 1: For each cycle of this training procedure, present the first input/output pair,
    perform the back propagation technique to update the weights, then present the second
    input/output pair and again perform the back propagation technique to update the
    weights. This constitutes a single cycle. 
    
    Perform 15 such cycles and determine the errors
    E for each input/output pair. This method essentially updates the weights by alternately
    presenting each input/output pair ... the first pair, and then the second pair and so on.

    After the 15th training cycle, present the input values to the network and print out the
    total Error (Big E) and the final weights associated with each input/output pair.

    '''
    async def train_method_one(self, inputs, desired_output):
        total_errors = []
        for epoch in range(self.epochs):
            print('training epoch ', epoch, '/', self.epochs)
            total_error = 0
            for i in range(len(inputs)):
                # forward pass
                output, hidden_output = await self.forward(inputs[i])

                # backpropagation and update the weights
                await self.backprop(inputs[i], hidden_output, output, desired_output[i])

                # error
                curr_error = 0.5 * (desired_output[i] - output) ** 2
                total_error += float(curr_error[0])
                self.errors[str(inputs[i])] = total_error
                total_errors.append(curr_error[0])
        print('big E for method one: ', self.errors)
        return total_errors


    '''
    In this method, we update weights for one input/output pair for 15 iterations
    of the FFBP algorithm, then present the second input/output pair and run the FFBP
    algorithm to update weights for another 15 iterations. 

    Thus, for each cycle, present the first input/output pair, run the FFBP for 15 iterations
    then present the second input/output pair and run the FFBP for another 15 iterations. This second set of iterations
    therefore begins updating the weights from the values obtained after the first 15 iterations with the first input/output pair. 
    
    After the training of the second input/output pair, present the input of the first pair, print out the total Error (Big E) and the final weights associated
    with it, then present the input of the second pair and print out the total Error (Big E).

    '''
    async def train_method_two(self, inputs, desired_output):
        total_errors = []
        for i in range(len(inputs)):
            total_error = 0
            for epoch in range(self.epochs):
                # forward pass
                output, hidden_output = await self.forward(inputs[i])
                # backpropagation and update the weights
                await self.backprop(inputs[i], hidden_output, output, desired_output[i])

                # error
                curr_error = 0.5 * (desired_output[i] - output) ** 2
                total_error += float(curr_error[0])
                total_errors.append(curr_error[0])
                self.errors[str(inputs[i])] = total_error

        print('big E for method 2: ', self.errors)
        return total_errors


    '''
    Plot results of method 1 compared to method 2
    '''
    def plot(self, x, y_1, y_2, x_label, y_label, title):
        plt.plot(x, y_1)
        plt.plot(x, y_2)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    async def evaluate(self, input, desired_output):
        print('')
        print('-------------------------------------------------')
        print('-------- evaluating model on ', input, ' --------')
        output, hidden_output = await self.forward(input)
        print('desired_output: ', desired_output)
        print('final output for this input ', input, ' : ', output)
        error = 0.5 * (desired_output - output) ** 2
        self.errors[str(input)] += error[0]
        print('big E for this input ', input, self.errors[str(input)])



async def main():
    # define and train network - baseline for testing
    # inputs = np.array([[1, 2]])
    # desired_output = np.array([0.7])
    # nn = NeuralNetwork(eta=1, epochs=1, bias=0)
    # # await nn.train(inputs, desired_output)
    # await nn.train_method_one(inputs, desired_output)
    # print('')
    # print('-------------------------------------------------')
    # print('---------------baseline evaluation--------------')
    # await nn.evaluate([1, 2], 0.7)

    # method one
    inputs = np.array([[1, 1], [-1, -1]])
    desired_output = np.array([0.9, 0.05])
    epochs = 15
    x_epochs = [i for i in range(epochs*len(inputs))]

    nn = NeuralNetwork(eta=1, epochs=epochs, bias=0)
    method_one_errors = await nn.train_method_one(inputs, desired_output)
    print('')
    print('-------------------------------------------------')
    print('---------------method 1 evaluation--------------')
    await nn.evaluate([1, 1], 0.9)
    await nn.evaluate([-1, -1], 0.05)


    nn_2 = NeuralNetwork(eta=1, epochs=epochs, bias=0)
    method_two_errors = await nn_2.train_method_two(inputs, desired_output)
    print('')
    print('-------------------------------------------------')
    print('----------------method 2 evaluation--------------')
    await nn_2.evaluate([1, 1],0.9)
    await nn_2.evaluate([-1, -1],0.05)
    nn.plot(x_epochs, method_one_errors, method_two_errors, 'epoch', 'error', 'method two')

if __name__ ==  '__main__':
    loop=asyncio.get_event_loop()
    loop.run_until_complete(main())