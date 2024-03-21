

# %%
## Importing Libraries and Downloading Data

import sys

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import wandb
import seaborn as sns


from keras.datasets import fashion_mnist
import wandb


# %%

# Parse input parameters.

args_short = {}
args_long = {}

for i in range(1, int((len(sys.argv) - 1)/2) + 1):
    arg = sys.argv[2 * i - 1]
    val = sys.argv[2 * i]
    if(arg[0:2] == "--"):
        args_long[arg] = val
    elif(arg[0] == '-'):
        args_short[arg] = val


# Get wandb project name.
wandb_project_name = args_short.get("-wp")
if wandb_project_name == None:
    wandb_project_name = args_long.get("--wandb_project")
if wandb_project_name == None:
    wandb_project_name =  "FODL_ASSIGNMENT_01_ABHIJEET"

# Number of epoch 
epochs = int(args_short.get("-e"))
if epochs == None:
    epochs = int(args_long.get("--epochs"))
if epochs == None:
    epochs = 1

print("epochs: ", epochs)
print("Project name: ", wandb_project_name)


# %%
wandb.login()

# %%
PROJECT_NAME = "FODL_ASSIGNMENT_01_ABHIJEET"

# %%
########################################################################################
# Download data
(X, y), (X_test, y_test) = fashion_mnist.load_data()

# %%
# Reshaping the data matrices
X = X.reshape(X.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# Normalizing the pixel intensities
X = X/255.0
X_test = X_test/255.0

# Split the X_train into a training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# %%
############################################################################
# # Dataset Preprocessing
# Number of training examples
M = X_train.shape[0]

# Number of validation samples
Mval = X_val.shape[0]

# Number of test examples
Mtest = X_test.shape[0]

# Number of features in the dataset
num_features = 784

# Number of classes
num_classes = len(np.unique(y_train))

# %%
# One hot encoding for class labels
y_train_one_hot = np.zeros((10, M))
y_train_one_hot[y_train, np.array(list(range(M)))] = 1

y_val_one_hot = np.zeros((10, Mval))
y_val_one_hot[y_val, np.array(list(range(Mval)))] = 1

y_test_one_hot = np.zeros((10, Mtest))
y_test_one_hot[y_test, np.array(list(range(Mtest)))] = 1

print("Number of images in the training set =", M)
print("Number of images in the validation set =", Mval)
print("Number of images in the test set =", Mtest)
print("Number of classes =", num_classes)
print("Number of features per example =", num_features)
##########################################################################

# %%
# Modify shapes of the data matrices
X_train = X_train.T
X_val = X_val.T
X_test = X_test.T
X = X.T

# %%
#############################################################

# Number of neurons in the input and output layers
input_nodes = num_features
output_nodes = num_classes

# %%
# Number of neurons in the input and output layers
input_nodes = num_features
output_nodes = num_classes

# Class names dictionary
class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Print Sample Images
# Store the index of first occurrence of each class
example_indices = [list(y_train).index(i) for i in range(num_classes)]

fig = plt.figure(figsize=(10, 5))
count = 1
for index in example_indices:
    ax = fig.add_subplot(2, 5, count)
    ax.set_title(class_labels[y_train[index]])  # Add title with class name
    ax.imshow(X_train.T[index].reshape((28, 28)), cmap='gray')
    count += 1

plt.tight_layout()
plt.show()

# %%
from keras.datasets import fashion_mnist
import wandb

# Initialize Weights & Biases
wandb.init(project="FODL_ASSIGNMENT_01_ABHIJEET")
project_name = PROJECT_NAME

# Define class names for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the Fashion MNIST dataset
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

# Normalize pixel values to the range 0-1
trainX = trainX / 255.0
testX = testX / 255.0

def log_images():
    # Initialize lists to store images and their corresponding labels
    set_images = []
    set_labels = []
    count = 0  # Counter to keep track of how many images per class have been added
    for d in range(len(trainy)):
        if trainy[d] == count:
            # Add the image and its label to the respective lists
            set_images.append(trainX[d])
            set_labels.append(class_names[trainy[d]])
            count += 1
        else:
            pass
        if count == 10:
            break  # If images for all 10 classes have been collected, exit the loop

    # Log the images and their labels to Weights & Biases
    image_logs = [wandb.Image(img, caption=caption) for img, caption in zip(set_images, set_labels)]
    wandb.log({"Plot": image_logs})

# Call the function to log images
log_images()

wandb.finish()


# %%
# Components of the Neural Network Model
# Activation functions and their derivatives

def sigmoid(x):
    return 1. / (1.+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def Relu(x):
    return np.maximum(0,x)

def Relu_derivative(x):
    return 1*(x>0) 

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return (1 - (np.tanh(x)**2))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_derivative(x):
    return softmax(x) * (1-softmax(x))

# %%
def compute_loss_function(Y, Y_hat, batch_size, loss, lamb, parameters = None):

    if loss == 'categorical_crossentropy':
         L = (-1.0 * np.sum(np.multiply(Y, np.log(Y_hat))))/batch_size
    elif loss == 'mse':
         L = (1/2) * np.sum((Y-Y_hat)**2)/batch_size
    
    
    if parameters != None:
        
        #Add L2 regularisation
        acc = 0
        for i in range(1, len(parameters)//2 + 1):
            acc += np.sum(parameters["W"+str(i)]**2)

        L = L + (lamb/(2*batch_size))*acc

    return L

# %%
def compute_accuracy(predictions, labels):
    """Compute accuracy given the predicted labels and the true labels."""
    return np.mean(np.argmax(predictions, axis=0) == np.argmax(labels, axis=0))


# %%
# Initialize parameters
def initialize_parameters(layer_dims, init_mode="xavier"):
    '''Function to initialise weights, biases and velocities/previous updates of the NN

    Parameters
    ----------
    layer_dims: list
        list of number of neurons per layer specifying layer dimensions in the format [#input_features,#hiddenunits...#hiddenunits,#outputclasses]

    init_mode: string
        initialisation mode, default-"xavier"

    Returns
    -------
    params: dict
        contains weights and biases. eg params[W1] is weight for layer 1

    previous updates: dict
        previous updates initialisation. This is used for different perposes for different optimisers.

    '''
    np.random.seed(42)
    params = {}
    previous_updates = {}

    for i in range(1, len(layer_dims)):
        if init_mode == 'random_normal':
            
            params["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        elif init_mode == 'random_uniform':
            params["W"+str(i)] = np.random.rand(layer_dims[i], layer_dims[i-1]) * 0.01
        elif init_mode == 'xavier':
            params["W"+str(i)]= np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(2/(layer_dims[i]+layer_dims[i-1]))
        else:
                
            params["W"+str(i)]= np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(2/(layer_dims[i]+layer_dims[i-1]))
            
                
            
        params["b"+str(i)] = np.zeros((layer_dims[i], 1))
        
        previous_updates["W"+str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
        previous_updates["b"+str(i)] = np.zeros((layer_dims[i], 1))

    return params,previous_updates

# %%
# Forward Propagation
def forward_propagate(X, params, activation_function):
    '''Function to forward propagate a minibatch of data once through the NN

    Parameters
    ----------
    X: numpy array
        data in (features,batch_size) format

    params: dict
        W and b of the NN

    activation_function: string
        activation function to be used except the output layer

    Returns
    -------
    output: numpy array
        contains the output probabilities for each class and each data sample after 1 pass
    A: numpy array
        contains all post-activations
    Z: numpy array
        contsins all pre-activations

    '''
    
    L = len(params)//2 + 1
    A = [None]*L # activations
    Z = [None]*L # pre-activations
    
    A[0] = X
    
    for l in range(1, L):
        W = params["W"+str(l)]
        b = params["b"+str(l)]
        
        Z[l] = np.matmul(W,A[l-1]) + b 
        
        if l == L-1:
            A[l] = softmax(Z[l]) # activation function for output layer
        else:
            if activation_function == 'sigmoid':
                A[l] = sigmoid(Z[l])
            elif activation_function == 'relu':
                A[l] = Relu(Z[l])
            elif activation_function == 'tanh':
                A[l] = tanh(Z[l])
                
    output = A[L-1]

    return output,A,Z

# %% [markdown]
# ## BackProp

# %%
# Backpropagation

def backprop(y_hat, y,A, Z, params, activation_function, batch_size, loss, lamb):
    '''
    Function to calculate gradients for a minibatch of data once through the NN through backpropagation

    Parameters
    ----------
    y_hat: numpy array
        output from forward propagation/ class probabilities

    y: numpy array
        actual class labels
     
    A: numpy array
        post-activations

    Z: numpy array
        pre-activations   

    params: dict
        contains W and b on the NN   

    activation_function: string
        activation function to be used except the output layer

    batch_size: int
        mini-batch-size

    loss: string
        loss function (MSE/Categorical crossentropy)

    lamb: float
        L2 regularisation lambda

    Returns
    -------
    gradients: dict
        gradients wrt weights and biases

    '''

    L = len(params)//2 #no. of layers
    gradients = {}
    
    #process last layer which has softmax
    if loss == 'categorical_crossentropy':
        gradients["dZ"+str(L)] = A[L]-y
    elif loss == 'mse':
        gradients["dZ"+str(L)] = (A[L]-y) * softmax_derivative(Z[L])
    
    #process other layers
    for l in range(L,0,-1):
        gradients["dW" + str(l)] = (np.dot(gradients["dZ" + str(l)], A[l-1].T) + lamb*params["W"+str(l)]) / batch_size
        gradients["db" + str(l)] = np.sum(gradients["dZ" + str(l)], axis=1, keepdims=True) / batch_size
        
        if l>1: 
            if activation_function == 'sigmoid':
                gradients["dZ"+str(l-1)] = np.matmul(params["W" + str(l)].T, gradients["dZ" + str(l)]) * sigmoid_derivative(Z[l-1])
            elif activation_function == 'relu':
                gradients["dZ"+str(l-1)] = np.matmul(params["W" + str(l)].T, gradients["dZ" + str(l)]) * Relu_derivative(Z[l-1])
            elif activation_function == 'tanh':
                gradients["dZ"+str(l-1)] = np.matmul(params["W" + str(l)].T, gradients["dZ" + str(l)]) * tanh_derivative(Z[l-1])
        
    return gradients

    ############################################################################################################################

# %% [markdown]
#   ## Optimizers

# %%
# Optimizers
def update_params_sgd(parameters,grads,learning_rate):
    ''' Update W and b of the NN according to sgd updates

    Parameters
    ----------
    parameters: dict
        contains weights and biases of the NN

    grads: dict
        contains gradients wrt W and b returned by backpropagation

    learning_rate: float
        learning rate

    Returns
    -------
    parameters: dict
        updated NN parameters

    '''
    L = len(parameters) // 2 
    
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

def update_parameters_momentum(parameters, grads, learning_rate, beta, previous_updates):
    ''' Update W and b of the NN according to momentum updates

    Parameters
    ----------
    parameters: dict
        contains weights and biases of the NN

    grads: dict
        contains gradients wrt W and b returned by backpropagation

    learning_rate: float
        learning rate
    
    beta: float
        decay rate

    previous_updates: dict
        contains previous W and b values, accumulated in a weighted fashion along with the gradients eg.
        previous_updates[Wi] = beta*previous_updates[Wi] + (1-beta)*gradient[dWi]

    Returns
    -------
    parameters: dict
        updated NN parameters

    previous updates: dict
        updated previous updates 

    '''
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(1, L + 1):
        previous_updates["W"+str(l)] = beta*previous_updates["W"+str(l)] + (1-beta)*grads["dW" + str(l)]
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*previous_updates["W"+str(l)]
        
        previous_updates["b"+str(l)] = beta*previous_updates["b"+str(l)] + (1-beta)*grads["db" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*previous_updates["b"+str(l)]

    return parameters, previous_updates
    
def update_parameters_RMSprop(parameters, grads, learning_rate, beta, v):
    ''' Update W and b of the NN according to RMSprop updates

    Parameters
    ----------
    parameters: dict
        contains weights and biases of the NN

    grads: dict
        contains gradients wrt W and b returned by backpropagation

    learning_rate: float
        learning rate
    
    beta: float
        decay rate

    v: dict
        contains previous W and b values, accumulated in a weighted fashion along with the gradients square eg.
        v[Wi] = beta*v[Wi] + (1-beta)*(gradient[dWi]^2)

    Returns
    -------
    parameters: dict
        updated NN parameters

    v: dict
        updated "velocities"

    '''

    L = len(parameters) // 2 # number of layers in the neural network
    delta = 1e-6 # for numerical stability

    for l in range(1, L + 1):
        vdw = beta*v["W" + str(l)] + (1-beta)*np.multiply(grads["dW" + str(l)],grads["dW" + str(l)])
        vdb = beta*v["b" + str(l)] + (1-beta)*np.multiply(grads["db" + str(l)],grads["db" + str(l)])

        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)] / (np.sqrt(vdw)+delta)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)] / (np.sqrt(vdb)+delta)

        v["W" + str(l)] = vdw
        v["b" + str(l)] = vdb

    return parameters,v

def update_parameters_adam(parameters, grads, learning_rate, v, m, t):
    ''' Update W and b of the NN according to adam updates

    Parameters
    ----------
    parameters: dict
        contains weights and biases of the NN

    grads: dict
        contains gradients wrt W and b returned by backpropagation

    learning_rate: float
        learning rate

    v: dict
        contains previous W and b values, accumulated in a weighted fashion along with the gradients eg.
        v[Wi] = beta1*v[Wi] + (1-beta1)*(gradient[dWi])

    m: dict
        contains previous W and b values, accumulated in a weighted fashion along with the gradients^2 eg.
        v[Wi] = beta2*v[Wi] + (1-beta2)*(gradient[dWi]^2)

    t: int
        timestep for Adam

    Returns
    -------
    parameters: dict
        updated NN parameters

    v: dict
        updated previous updates

    m: dict
        updated "velocities"

    t: int
        updated timestep

    '''
    L = len(parameters) // 2 # number of layers in the neural network
    beta1 = 0.9 #default
    beta2 = 0.999 #default
    epsilon = 1e-8 #for numerical stability

    for l in range(1, L+1):
        mdw = beta1*m["W"+str(l)] + (1-beta1)*grads["dW"+str(l)]
        vdw = beta2*v["W"+str(l)] + (1-beta2)*np.square(grads["dW"+str(l)])
        mw_hat = mdw/(1.0 - beta1**t)
        vw_hat = vdw/(1.0 - beta2**t)

        parameters["W"+str(l)] = parameters["W"+str(l)] - (learning_rate * mw_hat)/np.sqrt(vw_hat + epsilon)

        mdb = beta1*m["b"+str(l)] + (1-beta1)*grads["db"+str(l)]
        vdb = beta2*v["b"+str(l)] + (1-beta2)*np.square(grads["db"+str(l)])
        mb_hat = mdb/(1.0 - beta1**t)
        vb_hat = vdb/(1.0 - beta2**t)

        parameters["b"+str(l)] = parameters["b"+str(l)] - (learning_rate * mb_hat)/np.sqrt(vb_hat + epsilon)

        v["dW"+str(l)] = vdw
        m["dW"+str(l)] = mdw
        v["db"+str(l)] = vdb
        m["db"+str(l)] = mdb

    t = t + 1 # timestep
    return parameters, v, m, t


# %% [markdown]
# ## Function to plot the cost curves

# %%
# Function to plot the cost curves
def plot_cost_curve(train_costs, val_costs):
    '''
    Plots the train and validation cost curves
    '''
    plt.plot(list(range(len(train_costs))), train_costs, 'r', label="Training loss")
    plt.plot(list(range(len(val_costs))), val_costs, 'lime', label="Validation loss")
    plt.title("Training and Validation Loss vs Number of Epochs", size=14)
    plt.xlabel("Number of epochs", size=14)
    plt.ylabel("Loss", size=14)
    plt.grid()
    plt.legend()
    plt.show()

# %% [markdown]
# ## Prediction and Evaluation functions

# %%
# Prediction and Evaluation functions

def NN_predict(X_test, params, activation_function):
    '''
    forward propagate once and calculate labels

    '''
    output, _, _ = forward_propagate(X_test, params, activation_function)
    predictions = np.argmax(output, axis=0)
    return predictions

def NN_evaluate(X_train, y_train, X_test, y_test, params, activation_function):
    '''
    print train,test accuracies and the classification report using sklearn

    '''
    train_predictions = NN_predict(X_train, params, activation_function)
    test_predictions = NN_predict(X_test, params, activation_function)

    print("Training accuracy = {} %".format(round(accuracy_score(y_train, train_predictions) * 100, 3)))
    print("Test accuracy = {} %".format(round(accuracy_score(y_test, test_predictions) * 100, 3)))

    print("Classification report for the test set:\n")
    print(classification_report(y_test, test_predictions))

    return train_predictions, test_predictions

# %% [markdown]
# ## NEURAL_NETWORK

# %%
########################################################################################################################
# Training on the full dataset

def NN_fit(X_train, y_train_one_hot,X_val,y_val_one_hot, learning_rate = 0.001, activation_function = 'tanh', init_mode = 'xavier', 
                optimizer = 'adam', batch_size = 512, loss = 'categorical_crossentropy', epochs = 20, L2_lamb = 0,
                layer_dims=[], num_layers = 3, hidden_size = 32):
    """This function is used to train the neural network on the dataset 

    X_train: numpy array
        train dataset

    y_train_one_hot: numpy array
        train labels with one-hot encoding

    learning_rate: float

    activation_function: string
        activation functions for all the layers except the last layer which is softmax

    init_mode: string
        initialization mode
    
    optimizer: string
        optimization routine

    bach_size: int
        minibatch size

    loss: string
        loss function

    epochs: int
        number of epochs to be used

    L2_lamb: float
        lambda for L2 regularisation of weights

    num_neurons: int
        number of neurons in every hidden layer

    num_hidden: 
        number of hidden layers

    Returns
    -------

    parameters: dict
        weights and biases of the NN model

    epoch_cost: list
        training costs with every epoch
    

    """

    layer_dims = [hidden_size for i in range(num_layers)]
    layer_dims[0] = X_train.shape[0]
    layer_dims[num_layers - 1] =  10
    params, previous_updates = initialize_parameters(layer_dims, init_mode) # initialize the parameters and past updates matrices
    
    epoch_cost = []
    validation_epoch_cost=[]
    
    count = 1
    t = 1 # initialize timestep for Adam optimizer
    v = previous_updates.copy()
    m = previous_updates.copy()
    params_look_ahead = params.copy() # initialization for nestorov
    beta = 0.9
    loss = 'categorical_crossentropy'    

    while count<=epochs:
        count = count + 1 # increment the number of epochs

        for i in range(0, X_train.shape[1], batch_size):
            batch_count = batch_size

            if i + batch_size > X_train.shape[1]: # the last mini-batch might contain fewer than "batch_size" examples
                batch_count = X_train.shape[1] - i + 1
            
            #process all nesterov accelerated optimisers

            #NAG
            if optimizer == 'nesterov':
                L = len(params)//2

                #look ahead logic
                for l in range(1, L+1):
                    params_look_ahead["W"+str(l)] = params["W"+str(l)] - beta*previous_updates["W"+str(l)]
                    params_look_ahead["b"+str(l)] = params["b"+str(l)] - beta*previous_updates["b"+str(l)]
                    
                output,A,Z = forward_propagate(X_train[:,i:i+batch_size],params_look_ahead,activation_function)
                gradients = backprop(output,y_train_one_hot[:,i:i+batch_size],A,Z,params_look_ahead,activation_function, batch_count, loss, L2_lamb)

                #call momentum
                params,previous_updates = update_parameters_momentum(params, gradients, learning_rate, beta, previous_updates)

            #nadam
            elif optimizer=='nadam':
                L = len(params)//2

                #look ahead logic
                for l in range(1, L+1):
                    params_look_ahead["W"+str(l)] = params["W"+str(l)] - beta*previous_updates["W"+str(l)]
                    params_look_ahead["b"+str(l)] = params["b"+str(l)] - beta*previous_updates["b"+str(l)]

                output,A,Z = forward_propagate(X_train[:,i:i+batch_size],params_look_ahead,activation_function)
                gradients = backprop(output,y_train_one_hot[:,i:i+batch_size],A,Z,params_look_ahead,activation_function, batch_count, loss, L2_lamb)

                #call adam
                params, v, m, t = update_parameters_adam(params, gradients, learning_rate, v, m, t)



            else:
                output,A,Z = forward_propagate(X_train[:,i:i+batch_size],params,activation_function)
                gradients = backprop(output,y_train_one_hot[:,i:i+batch_size],A,Z,params,activation_function, batch_count, loss, L2_lamb)

                if optimizer == 'sgd':
                    params = update_params_sgd(params,gradients,learning_rate)
                elif optimizer == 'momentum':
                    params,previous_updates = update_parameters_momentum(params, gradients, learning_rate, beta, previous_updates)
                elif optimizer == 'RMSprop':
                    params,previous_updates = update_parameters_RMSprop(params, gradients, learning_rate, beta, previous_updates)
                elif optimizer == 'adam':
                    params, v, m, t = update_parameters_adam(params, gradients, learning_rate, v, m, t)

                #custom
                elif optimizer == 'insert your optimiser here':
                    #insert your optimiser update routine only if it does not have nesterov 
                    pass

                    
        # Mean loss for the full training set
        full_output, _, _ = forward_propagate(X_train, params, activation_function)
        cost = compute_loss_function(y_train_one_hot, full_output, M, loss, L2_lamb, params)
        epoch_cost.append(cost)
        
        train_accuracy = compute_accuracy(full_output, y_train_one_hot)
        
        
        
        # Mean loss for the validation set
        out, _, _ = forward_propagate(X_val, params, activation_function)
        val_cost = compute_loss_function(y_val_one_hot, out, Mval, loss, L2_lamb, params)
        validation_epoch_cost.append(val_cost)
        
        val_accuracy = compute_accuracy(out, y_val_one_hot)

        
        
        

        if (count % 2 == 0):
            print("Epoch number: ", count, "\tTraining cost:", cost)
            
        
#         with wandb.init():
        wandb.log({"epoch" : count-1,
                   "train_loss": cost, 
                    "val_loss": val_cost, 
                    "train_accuracy": train_accuracy, 
                    "val_accuracy": val_accuracy})


    
    # Plot the training and validation cost curves
    plot_cost_curve(epoch_cost, validation_epoch_cost)


    return params, epoch_cost




# %%
########################################################################################################################
# Training on the full dataset

def NN_fit_w(X_train, y_train_one_hot,X_val,y_val_one_hot, learning_rate = 0.001, activation_function = 'tanh', init_mode = 'xavier', 
                optimizer = 'adam', batch_size = 512, loss = 'categorical_crossentropy', epochs = 20, L2_lamb = 0,
                layer_dims=[], num_layers = 3, hidden_size = 32):
    """This function is used to train the neural network on the dataset 

    X_train: numpy array
        train dataset

    y_train_one_hot: numpy array
        train labels with one-hot encoding

    learning_rate: float

    activation_function: string
        activation functions for all the layers except the last layer which is softmax

    init_mode: string
        initialization mode
    
    optimizer: string
        optimization routine

    bach_size: int
        minibatch size

    loss: string
        loss function

    epochs: int
        number of epochs to be used

    L2_lamb: float
        lambda for L2 regularisation of weights

    num_neurons: int
        number of neurons in every hidden layer

    num_hidden: 
        number of hidden layers

    Returns
    -------

    parameters: dict
        weights and biases of the NN model

    epoch_cost: list
        training costs with every epoch
    

    """

    layer_dims = [hidden_size for i in range(num_layers)]
    layer_dims[0] = X_train.shape[0]
    layer_dims[num_layers - 1] =  10
    params, previous_updates = initialize_parameters(layer_dims, init_mode) # initialize the parameters and past updates matrices
    
    epoch_cost = []
    validation_epoch_cost=[]
    
    count = 1
    t = 1 # initialize timestep for Adam optimizer
    v = previous_updates.copy()
    m = previous_updates.copy()
    params_look_ahead = params.copy() # initialization for nestorov
    beta = 0.9
    loss = 'categorical_crossentropy'    

    while count<=epochs:
        count = count + 1 # increment the number of epochs

        for i in range(0, X_train.shape[1], batch_size):
            batch_count = batch_size

            if i + batch_size > X_train.shape[1]: # the last mini-batch might contain fewer than "batch_size" examples
                batch_count = X_train.shape[1] - i + 1
            
            #process all nesterov accelerated optimisers

            #NAG
            if optimizer == 'nesterov':
                L = len(params)//2

                #look ahead logic
                for l in range(1, L+1):
                    params_look_ahead["W"+str(l)] = params["W"+str(l)] - beta*previous_updates["W"+str(l)]
                    params_look_ahead["b"+str(l)] = params["b"+str(l)] - beta*previous_updates["b"+str(l)]
                    
                output,A,Z = forward_propagate(X_train[:,i:i+batch_size],params_look_ahead,activation_function)
                gradients = backprop(output,y_train_one_hot[:,i:i+batch_size],A,Z,params_look_ahead,activation_function, batch_count, loss, L2_lamb)

                #call momentum
                params,previous_updates = update_parameters_momentum(params, gradients, learning_rate, beta, previous_updates)

            #nadam
            elif optimizer=='nadam':
                L = len(params)//2

                #look ahead logic
                for l in range(1, L+1):
                    params_look_ahead["W"+str(l)] = params["W"+str(l)] - beta*previous_updates["W"+str(l)]
                    params_look_ahead["b"+str(l)] = params["b"+str(l)] - beta*previous_updates["b"+str(l)]

                output,A,Z = forward_propagate(X_train[:,i:i+batch_size],params_look_ahead,activation_function)
                gradients = backprop(output,y_train_one_hot[:,i:i+batch_size],A,Z,params_look_ahead,activation_function, batch_count, loss, L2_lamb)

                #call adam
                params, v, m, t = update_parameters_adam(params, gradients, learning_rate, v, m, t)


            else:
                output,A,Z = forward_propagate(X_train[:,i:i+batch_size],params,activation_function)
                gradients = backprop(output,y_train_one_hot[:,i:i+batch_size],A,Z,params,activation_function, batch_count, loss, L2_lamb)

                if optimizer == 'sgd':
                    params = update_params_sgd(params,gradients,learning_rate)
                elif optimizer == 'momentum':
                    params,previous_updates = update_parameters_momentum(params, gradients, learning_rate, beta, previous_updates)
                elif optimizer == 'RMSprop':
                    params,previous_updates = update_parameters_RMSprop(params, gradients, learning_rate, beta, previous_updates)
                elif optimizer == 'adam':
                    params, v, m, t = update_parameters_adam(params, gradients, learning_rate, v, m, t)

                #custom
                elif optimizer == 'insert your optimiser here':
                    #insert your optimiser update routine only if it does not have nesterov 
                    pass

                    
        # Mean loss for the full training set
        full_output, _, _ = forward_propagate(X_train, params, activation_function)
        cost = compute_loss_function(y_train_one_hot, full_output, M, loss, L2_lamb, params)
        epoch_cost.append(cost)
        
        train_accuracy = compute_accuracy(full_output, y_train_one_hot)
        
        
        
        # Mean loss for the validation set
        out, _, _ = forward_propagate(X_val, params, activation_function)
        val_cost = compute_loss_function(y_val_one_hot, out, Mval, loss, L2_lamb, params)
        validation_epoch_cost.append(val_cost)
        
        val_accuracy = compute_accuracy(out, y_val_one_hot)

        
        
        

        if (count % 2 == 0):
            print("Epoch number: ", count, "\tTraining cost:", cost)
            
        
# #         with wandb.init():
#         wandb.log({"epoch" : count,
#                    "train_loss": cost, 
#                     "val_loss": val_cost, 
#                     "train_accuracy": train_accuracy, 
#                     "val_accuracy": val_accuracy})


    
    # Plot the training and validation cost curves
    plot_cost_curve(epoch_cost, validation_epoch_cost)


    return params, epoch_cost




# %% [markdown]
# ### Hyperparameter tuning using WandB.

# %%
def obj(config):
    learned_parameters, epoch_cost = NN_fit(X_train, y_train_one_hot,
                                X_val,y_val_one_hot,
                                learning_rate=config.learning_rate,
                                activation_function = config.activation_function,
                                init_mode = config.init_mode,
                                optimizer = config.optimizer,
                                batch_size = config.batch_size,
                                loss = config.loss,
                                epochs = config.epochs,
                                L2_lamb = config.L2_lamb,
                                num_layers=config.num_layers,
                                hidden_size=config.hidden_size)
#     # return epoch_cost[-1]
#     return [acc, val_acc, loss, val_loss]

# %%
import wandb

# Define your project name
project_name = PROJECT_NAME

# Define your obj function
def main():
    wandb.init(project=wandb_project_name)
    config = wandb.config  # Access configuration
    run_name = f"lr:{config.learning_rate}__activation:{config.activation_function}__init_mode:{config.init_mode}__optimizer:{config.optimizer}__batch_size:{config.batch_size}__loss:{config.loss}__epochs:{config.epochs}__L2_lamb:{config.L2_lamb}__num_layers={config.num_layers}__hidden_size:{config.hidden_size}"
#     wandb.init(name=f"lr:{config.learning_rate}__activation:{config.activation_function}__init_mode:{config.init_mode}__optimizer:{config.optimizer}__batch_size:{config.batch_size}__loss:{config.loss}__epochs:{config.epochs}__L2_lamb:{config.L2_lamb}__num_layers={config.num_layers}__hidden_size:{config.hidden_size}")
    wandb.run.name = run_name
    obj(wandb.config)

# Define your sweep configuration
sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'goal': 'maximize',
        'name': 'val_accuracy'
    },
    'parameters': {
        'epochs': {'values': [epochs]},
        'num_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128, 256, 512]},
        'weight_decay': {'values': [0, 0.0005]},
        'learning_rate': {'values': [0.1, 0.01, 0.001]},
        'optimizer': {'values': ['sgd', 'momentum', 'rmsprop', 'adam']},
        'batch_size': {'values': [16, 32, 64]},
        'init_mode': {'values': ['random_normal', 'random_uniform', 'xavier']},
        'activation_function': {'values': ['sigmoid', 'tanh', 'relu']},
        'loss': {'values': ['mse', 'categorical_crossentropy']},
        'L2_lamb': {'values': [0.05, 0.005]}
    }
}

# Define and start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
wandb.agent(sweep_id, function=main, count=30)


# %% [markdown]
# ### Result Using Tuned Hyperparameter tuning using WandB.

# %%
# Hyperparameter config1

# VAL_ACCURACY_1 = 0.885
L2_lambda_1 = 0.005
ACTIVATION_1 = "tanh"
BATCH_SIZE_1 = 64
EPOCHS_1 = 15
HIDDEN_SIZE_1 = 128
INITIALIZER_1 = "xavier"
LEARNING_RATE_1 = 0.1
LOSS_1 = 'cross-entropy'
NUM_LAYERS_1 = 3
OPTIMIZER_1 = "sgd"
WEIGHT_DECAY_1 = 0.0005

# Learned parameters
learned_parameters_best, epoch_cost_best = NN_fit_w(X_train, y_train_one_hot,
                            X_val, y_val_one_hot,
                            learning_rate=LEARNING_RATE_1,
                            activation_function=ACTIVATION_1,
                            init_mode=INITIALIZER_1,
                            optimizer=OPTIMIZER_1,
                            batch_size=BATCH_SIZE_1,
                            loss=LOSS_1,
                            epochs=EPOCHS_1,
                            L2_lamb=L2_lambda_1,
                            num_layers=NUM_LAYERS_1,
                            hidden_size=HIDDEN_SIZE_1)


# %%
# Hyperparameter config2

VAL_ACCURACY_2 = 0.8847
L2_lambda_2 = 0.005
ACTIVATION_2 = "relu"
BATCH_SIZE_2 = 64
EPOCHS_2 = 15
HIDDEN_SIZE_2 = 128
INITIALIZER_2 = "xavier"
LEARNING_RATE_2 = 0.1
LOSS_2 = 'cross-entropy'
NUM_LAYERS_2 = 4
OPTIMIZER_2 = "momentum"
WEIGHT_DECAY_2 = 0.0005

# Learned parameters
learned_parameters, epoch_cost = NN_fit_w(X_train, y_train_one_hot,
                            X_val, y_val_one_hot,
                            learning_rate=LEARNING_RATE_2,
                            activation_function=ACTIVATION_2,
                            init_mode=INITIALIZER_2,
                            optimizer=OPTIMIZER_2,
                            batch_size=BATCH_SIZE_2,
                            loss=LOSS_2,
                            epochs=EPOCHS_2,
                            L2_lamb=L2_lambda_2,
                            num_layers=NUM_LAYERS_2,
                            hidden_size=HIDDEN_SIZE_2)


# %%


# %%
# Hyperparameter config3

VAL_ACCURACY_3 = 0.8838
L2_lambda_3 = 0.005
ACTIVATION_3 = "relu"
BATCH_SIZE_3 = 64
EPOCHS_3 = 15
HIDDEN_SIZE_3 = 64
INITIALIZER_3 = "xavier"
LEARNING_RATE_3 = 0.1
LOSS_3 = 'cross-entropy'
NUM_LAYERS_3 = 3
OPTIMIZER_3 = "momentum"
WEIGHT_DECAY_3 = 0.0005

# Learned parameters
learned_parameters, epoch_cost = NN_fit_w(X_train, y_train_one_hot,
                            X_val, y_val_one_hot,
                            learning_rate=LEARNING_RATE_3,
                            activation_function=ACTIVATION_3,
                            init_mode=INITIALIZER_3,
                            optimizer=OPTIMIZER_3,
                            batch_size=BATCH_SIZE_3,
                            loss=LOSS_3,
                            epochs=EPOCHS_3,
                            L2_lamb=L2_lambda_3,
                            num_layers=NUM_LAYERS_3,
                            hidden_size=HIDDEN_SIZE_3)


# %%


# %%
# config 4 - with mse

import wandb
import matplotlib.pyplot as plt

# Initialize wandb
wandb.init(project="FODL_ASSIGNMENT_01_ABHIJEET", name='with_mse')

# Your hyperparameter configuration
VAL_ACCURACY_4 = 38.37
L2_lambda_4 = 0.005
ACTIVATION_4 = "relu"
BATCH_SIZE_4 = 64
EPOCHS_4 = 10
HIDDEN_SIZE_4 = 128
INITIALIZER_4 = "random_normal"
LEARNING_RATE_4 = 0.01
LOSS_4 = 'mse'
NUM_LAYERS_4 = 5
OPTIMIZER_4 = "momentum"
WEIGHT_DECAY_4 = 0.0005

# Learned parameters (assuming you have your data)
learned_parameters, epoch_cost = NN_fit_w(X_train, y_train_one_hot,
                            X_val, y_val_one_hot,
                            learning_rate=LEARNING_RATE_4,
                            activation_function=ACTIVATION_4,
                            init_mode=INITIALIZER_4,
                            optimizer=OPTIMIZER_4,
                            batch_size=BATCH_SIZE_4,
                            loss=LOSS_4,
                            epochs=EPOCHS_4,
                            L2_lamb=L2_lambda_4,
                            num_layers=NUM_LAYERS_4,
                            hidden_size=HIDDEN_SIZE_4)

# Generate and plot your desired graph
# For example:
plt.plot(epoch_cost)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Training Cost over Epochs')

# Log the plot to wandb
wandb.log({"Training Cost": plt})

# Close the wandb run
wandb.finish()


# %%


# %%


# %%
##########################################################################################################################

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"]

# Model Evaluation with X_test dataset
train_predictions, test_predictions = NN_evaluate(X_train, y_train, X_test, y_test, learned_parameters_best, ACTIVATION_1)
# Confusion matrix

cm = confusion_matrix(y_test, test_predictions)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# %%


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
 
wandb.init(project="FODL_ASSIGNMENT_01_ABHIJEET")

# Generate confusion matrix
cm = confusion_matrix(y_test, test_predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

# Save the plot as an image file
confusion_matrix_image_path = "confusion_matrix.png"
plt.savefig(confusion_matrix_image_path)

# Log the image file to Weights & Biases
wandb.log({"Confusion Matrix": wandb.Image(confusion_matrix_image_path)})

# Show the plot
plt.show()

wandb.finish()

# %%


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

# Define class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"]

# Initialize wandb
wandb.init(project="FODL_ASSIGNMENT_01_ABHIJEET")

# Model Evaluation with X_test dataset
train_predictions, test_predictions = NN_evaluate(X_train, y_train, X_test, y_test, learned_parameters_best, ACTIVATION_1)

# Generate confusion matrix
cm = confusion_matrix(y_test, test_predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

# Save the plot as an image file
confusion_matrix_image_path = "confusion_matrix.png"
plt.savefig(confusion_matrix_image_path)

# Log the image file to Weights & Biases
wandb.log({"Confusion Matrix": wandb.Image(confusion_matrix_image_path)})

# Show the plot
plt.show()

# Finish wandb run
wandb.finish()


# %%


# %%


# %%



