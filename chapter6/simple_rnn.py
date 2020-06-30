'''
In this example we implement the forward pass of an RNN with numpy.
This RNN takes as input a sequence of vectors, which we will encode as a 2D tensor of size
(timesteps, input_features). It loops overs timesteps, and at each timestep, it considers
its current state at t and the input at t (of shape (input_features,)), and combines them
to obtain the output at t. We then set the state for the next step to be this previous output.
For the first timestep the previous output is not defined, hence there is no current state.
So we will initialize the state as an all-zero vector called the "initial state" of the network.

Pseudocode for the RNN:
state_t = 0
for input_t in input_sequence:
    output_t = f(input_t, state_t)
    state_t = output_t

where f() can be something like:
activation(dot(W, input_t) + dot(U, state_t) + b)
which is composed of 2 matrices: W, U and the bias vector b.
'''
import numpy as np

timesteps = 100       # number of timesteps in the input sequence
input_features = 32   # dimensionality of the input feature space
output_features = 64  # dimensionality of the output feature space

inputs = np.random.random((timesteps, input_features))  # input data: random noise for the sake of the example
state_t = np.zeros((output_features,))  # initial state: an all zero vector

# create random weight matrices
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:  # input_t is a vector of shape (input_features,)
    # combines the input with the current state (the previous output) to obtain the current output
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    # store output in a list
    successive_outputs.append(output_t)
    # update the state of the network for the next timestep
    state_t = output_t
# the final output is a 2D tensor of shape (timesteps, output_features)
final_output_sequence = np.concatenate(successive_outputs, axis=0)

final_output_sequence
'''
In summary: an RNN is a for-loop that reuses quantities computed
during the previous iteration of the loop.
RNNs are characterized by their step function which in this case is:
output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
'''

### A recurrent layer in Keras ###
'''
SimpleRNN is the layer from Keras of what we have just implemented with numpy with one
difference: SimpleRNN processes batches of sequences instead of just one sequence as in the numpy example.

SimpleRNN can be run in two different modes:
1. It can return the full sequences of successive outputs for each timestep
(a 3D tensor of shape (batch_size, timesteps, output_features)
2. It can return only the last output for each input sequence 
(a 2D tensor of shape (batch_size, output_features))
'''
from keras.models import Model
from keras.layers import Embedding, SimpleRNN, Input

input_tensor = Input((10000,))
kmodel = Embedding(10000, 32)(input_tensor)
output_tensor = SimpleRNN(32)(kmodel)
model = Model(input_tensor, output_tensor)
model.summary()

input_tensor = Input((10000,))
kmodel = Embedding(10000, 32)(input_tensor)
output_tensor = SimpleRNN(32, return_sequences=True)(kmodel)
model = Model(input_tensor, output_tensor)
model.summary()

'''
It is sometimes useful to stack several recurrent layer one after the other in order
to increase the representational power of a network. In such a setup we have to get all
of the intermediate layers to return full sequences (all except the last one).
'''
input_tensor = Input((10000,))
kmodel = Embedding(10000, 32)(input_tensor)
kmodel = SimpleRNN(32, return_sequences=True)(kmodel)
kmodel = SimpleRNN(32, return_sequences=True)(kmodel)
kmodel = SimpleRNN(32, return_sequences=True)(kmodel)
output_tensor = SimpleRNN(32)(kmodel)
model = Model(input_tensor, output_tensor)
model.summary()