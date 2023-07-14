# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from bokeh.plotting import figure, output_file, save

from bokeh.models import Range1d, LinearAxis, FactorRange, ColumnDataSource, LabelSet, LinearColorMapper
from bokeh.io import output_notebook
from bokeh.palettes import Greys256

from py_ml_tools.setup import setup_cuda, find_available_GPUs
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.transform import transform

import matplotlib.pyplot as plt
import numpy as np

def plot_sample_digits(x_train, y_train):
    # Convert one-hot encoded y_train back to label format
    y_train_labels = np.argmax(y_train.numpy(), axis=1)

    # Convert x_train to numpy
    x_train_np = x_train.numpy()

    # Prepare an example of each digit
    examples = {}
    for i in range(10):
        examples[i] = x_train_np[np.argwhere(y_train_labels == i)[0][0]].reshape(28, 28)
        
    # Prepare the figure
    fig, axes = plt.subplots(2, 5, figsize=(10,4))

    for i in range(10):
        ax = axes[i//5, i%5]
        ax.imshow(examples[i], cmap='gray', aspect='auto')
        ax.axis('off')  # hide the axes ticks

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig("mnist_examples.png")
    
# Plot loss and accuracy over epochs
def plot_metrics(loss_per_epoch, accuracy_per_epoch):
    # Create a new plot with a range for the loss [left]
    p = figure(title="Metrics over epochs", x_axis_label='Epochs', y_axis_label='Loss')

    # Set range for loss axis
    loss_max = max(loss_per_epoch) + 0.05 * max(loss_per_epoch)  # for example, add 5% as a padding to the max value
    p.y_range = Range1d(0, loss_max)  # adjust according to your loss data
    
    # Plot loss
    p.line(np.arange(len(loss_per_epoch)), loss_per_epoch, legend_label="Loss", line_color="red")
    
    # Create a new range for the accuracy [right]
    p.extra_y_ranges = {"AccuracyRange": Range1d(start=0, end=100)} # Update the range
    p.add_layout(LinearAxis(y_range_name="AccuracyRange", axis_label="Accuracy (%)"), 'right') # Update the label

    # Scale accuracy values to percentage and plot
    accuracy_per_epoch_percentage = [acc * 100 for acc in accuracy_per_epoch] # Scale accuracy
    p.line(np.arange(len(accuracy_per_epoch_percentage)), accuracy_per_epoch_percentage, legend_label="Accuracy", line_color="blue", y_range_name="AccuracyRange") # Plot scaled accuracy

    # Save the plot
    output_file("metrics_plot.html")
    save(p)

def plot_model_output(model, output_digit, x_data, y_data):
    # Convert one-hot encoded y_train back to label format
    y_labels = np.argmax(y_data, axis=1)
    
    # Find the first instance of '0'
    zero_index = np.argwhere(y_labels == output_digit)[0][0]
    zero_input = tf.reshape(x_data[zero_index], (1, -1)) # Reshape it to match model input shape

    # Calculate model output
    output = model(zero_input).numpy()[0]

    # Prepare data for bar chart
    digits = [str(i) for i in range(10)][::-1]
    output_values = output.tolist()[::-1]
    
    # Format output values to have 2 decimal places
    formatted_output_values = [f"{val:.2E}" for val in output_values]

    # Create a new figure with a white background
    p = figure(y_range=FactorRange(factors=digits), 
               width=500, 
               title="Model Output for First Instance of Digit 0",
               y_axis_label='Digits',
               x_axis_label='Model Output (~Probability)',
               background_fill_color="white")

    # Create a ColumnDataSource with the data
    source = ColumnDataSource(data=dict(digits=digits, output_values=output_values, formatted_output_values=formatted_output_values))

    # Add a horizontal bar chart to the figure with black bars
    p.hbar(y='digits', right='output_values', height=0.5, color="black", source=source)

    # Create labels for the bars with color based on a condition
    color_mapper = LinearColorMapper(palette=['black', 'white'], low=min(output_values), high=max(output_values))
    labels = LabelSet(x='output_values', y='digits', text='formatted_output_values', level='annotation',
                      x_offset=5, y_offset=-10, text_align='left', text_color=transform('output_values', color_mapper), source=source)
    
    print(formatted_output_values)
    # Add labels to the plot
    p.add_layout(labels)

    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"
    p.xaxis.major_label_text_font_size = "15pt"
    p.yaxis.major_label_text_font_size = "15pt"

    # Remove the grid
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Save the plot
    output_file(f"output_plot_for_{output_digit}.html")
    save(p)

# Step 1: Load and prepare the MNIST dataset.
def load_data():
    # This data is already split into train and test datasets.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the images to feed into the neural network.
    x_train, x_test = tf.cast(x_train.reshape(-1, 784)/255.0, tf.float32), \
    tf.cast(x_test.reshape(-1, 784)/255.0, tf.float32)

    # Convert labels to one-hot vectors. This is necessary as our output layer 
    # will have 10 neurons, one for each digit from 0 to 9.
    y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)
    
    return x_train, y_train, x_test, y_test

# Step 2: Define the model
def define_model():
    W = tf.Variable(tf.random.normal([784, 10]), name="weights", dtype = tf.float32)
    b = tf.Variable(tf.zeros([10]), name="biases",  dtype = tf.float32)
    return W, b

# Step 3: Define the model's computations
def model(x, W, b):
    return tf.nn.softmax(tf.matmul(x, W) + b)

# Step 4: Define the loss function
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=[1]))

# Step 5: Define the training step
learning_rate = 0.01
@tf.function
def train_step(x, y, W, b):
    with tf.GradientTape() as tape:
        y_pred = model(x, W, b)
        current_loss = compute_loss(y, y_pred)
    gradients = tape.gradient(current_loss, [W, b])
    W.assign_sub(learning_rate * gradients[0])  # update weights
    b.assign_sub(learning_rate * gradients[1])  # update biases
    return current_loss

@tf.function
def compute_model(x, W, b):
    return model(x, W, b)

# Define the accuracy calculation
def compute_accuracy(y_true, y_pred):
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

def train_model(epochs, batch_size, x_train, y_train):
    # Define model
    W, b = define_model()

    # Store loss and accuracy for each epoch
    loss_per_epoch = []
    accuracy_per_epoch = []

    # Training loop
    for epoch in range(epochs):
        i = 0
        while i < len(x_train):
            start = i
            end = i + batch_size
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]
            current_loss = strategy.run(train_step, args=(x_batch, y_batch, W, b))
            i += batch_size

        # Compute loss and accuracy for each epoch
        y_pred = strategy.run(compute_model, args=(x_test, W, b))
        loss_per_epoch.append(current_loss)
        accuracy_per_epoch.append(compute_accuracy(y_test, y_pred))
        print(f'Epoch {epoch+1} completed')
        
    return loss_per_epoch, accuracy_per_epoch, W, b

def plot_results(loss_per_epoch, accuracy_per_epoch, W, b, x_train, y_train):
    # Convert the EagerTensor to numpy arrays before returning
    loss_per_epoch = [l.numpy() for l in loss_per_epoch]
    accuracy_per_epoch = [a.numpy() for a in accuracy_per_epoch]
    
    plot_sample_digits(x_train, y_train)
    plot_metrics(loss_per_epoch, accuracy_per_epoch)
    
    model_fn = lambda x: model(x, W, b)
    plot_model_output(model_fn, 0, x_train, y_train)
    plot_model_output(model_fn, 1, x_train, y_train)
    plot_model_output(model_fn, 5, x_train, y_train)

if __name__ == "__main__":
    # Run the code
    
    gpus = find_available_GPUs(10000, 1)
    strategy = setup_cuda(gpus, 8000, verbose = True)
    
    with strategy.scope():
        x_train, y_train, x_test, y_test = load_data()
        loss_per_epoch, accuracy_per_epoch, W, b = train_model(10, 32, x_train, y_train)
        
        plot_results(loss_per_epoch, accuracy_per_epoch, W, b, x_train, y_train)
