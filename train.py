import argparse
import wandb
from keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from train_modules import NN_fit

def train(wandb_entity, wandb_project, dataset, epochs, batch_size, loss, optimizer, learning_rate, momentum, beta, beta1, beta2, epsilon, weight_decay, weight_init, num_layers, hidden_size, activation):
    # Initialize wandb
    # wandb.init(entity=wandb_entity, project=wandb_project)
    
    # Your training code here
    print("Training model...")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Loss function: {loss}")
    print(f"Optimizer: {optimizer}")
    print(f"Learning rate: {learning_rate}")
    print(f"Momentum: {momentum}")
    print(f"Beta: {beta}")
    print(f"Beta1: {beta1}")
    print(f"Beta2: {beta2}")
    print(f"Epsilon: {epsilon}")
    print(f"Weight decay: {weight_decay}")
    print(f"Weight initialization: {weight_init}")
    print(f"Number of hidden layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Activation function: {activation}")
    print(f"Project name: {wandb_project}")

    # Download data
    (X, y), (X_test, y_test) = fashion_mnist.load_data()
    # Reshaping the data matrices
    X = X.reshape(X.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)

    # Normalizing the pixel intensities
    X = X/255.0
    X_test = X_test/255.0

    # Split the X_train into a training set and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Dataset Preprocessing
    # Number of training examples
    M = X_train.shape[0]

    # Number of validation samples
    Mval = X_val.shape[0]

    # Number of test examples
    Mtest = X_test.shape[0]

    # # Number of features in the dataset
    # num_features = 784

    # # Number of classes
    # num_classes = len(np.unique(y_train))
    
    # One hot encoding for class labels
    y_train_one_hot = np.zeros((10, M))
    y_train_one_hot[y_train, np.array(list(range(M)))] = 1

    y_val_one_hot = np.zeros((10, Mval))
    y_val_one_hot[y_val, np.array(list(range(Mval)))] = 1

    y_test_one_hot = np.zeros((10, Mtest))
    y_test_one_hot[y_test, np.array(list(range(Mtest)))] = 1

    X_train = X_train.T
    X_val = X_val.T
    X_test = X_test.T

    with wandb.init(entity=wandb_entity, project=wandb_project):
        # print("X_train:", X_train.shape)
        # print("y_train_one_hot:", y_train_one_hot.shape)
        # print("X_val:", X_val.shape)
        # print("y_val_one_hot:", y_val_one_hot.shape)
        NN_fit(X_train, y_train_one_hot,
            X_val,y_val_one_hot,
            learning_rate=learning_rate,
            activation_function = activation,
            init_mode = weight_init,
            optimizer = optimizer,
            batch_size = batch_size,
            loss = loss,
            epochs = epochs,
            L2_lamb = weight_decay,
            num_layers=num_layers,
            hidden_size=hidden_size
        )


def get_arg_value(args_short, args_long, short_key, long_key, default):
    """
    Utility function to get the value of an argument either from args_short or args_long dictionary.
    """
    return args_short.get(short_key) if args_short.get(short_key) else args_long.get(long_key) if args_long.get(long_key) else default

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Train neural network')
    # Add arguments
    parser.add_argument('-wp', '--wandb_project', default='myprojectname', help='Project name for Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='myname', help='Wandb Entity for Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=["mnist", "fashion_mnist"], help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=["mean_squared_error", "cross_entropy"], help='Loss function')
    parser.add_argument('-o', '--optimizer', default='sgd', choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1')
    parser.add_argument('--beta2', type=float, default=0.5, help='Beta2')
    parser.add_argument('--epsilon', type=float, default=0.000001, help='Epsilon')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('-w_i', '--weight_init', default='random', choices=["random", "Xavier"], help='Weight initialization')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, help='Number of hidden neurons')
    parser.add_argument('-a', '--activation', default='sigmoid', choices=["identity", "sigmoid", "tanh", "ReLU"], help='Activation function')

    # Parse arguments
    args = parser.parse_args()
    args.wandb_entity

    # Call train function with parsed arguments
    train(args.wandb_entity, args.wandb_project, args.dataset, args.epochs, args.batch_size, args.loss, args.optimizer, args.learning_rate, args.momentum, args.beta, args.beta1, args.beta2, args.epsilon, args.weight_decay, args.weight_init, args.num_layers, args.hidden_size, args.activation)

if __name__ == "__main__":
    print("Args:", sys.argv)
    main()