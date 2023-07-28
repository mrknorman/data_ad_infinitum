from py_ml_tools.dataset import get_ifo_data_generator, get_ifo_data, O3
from py_ml_tools.model   import ModelBuilder, DenseLayer, ConvLayer, PoolLayer, DropLayer, randomizeLayer, negative_loglikelihood
from py_ml_tools.setup   import load_label_datasets, setup_cuda, find_available_GPUs

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers, optimizers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy

import numpy as np

from bokeh.plotting import figure, output_file, save
from bokeh.models import Span, ColumnDataSource
from bokeh.models.sources import ColumnDataSource
import numpy as np
import tensorflow as tf

from scipy.stats import gamma
from scipy.special import beta as beta_function # for the Beta function B(alpha, beta)
import numpy as np

import tensorflow_probability as tfp

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()

def plot_predictions(model, tf_dataset, output_name = "amplitude", filename="output.html"):
    for i in range(10):
        batch = next(iter(tf_dataset))
        onsource = tf.convert_to_tensor(batch[0]['onsource'])
        snr_ground_truth = batch[1][output_name].numpy()

        predictions_distribution = model(onsource)

        # Use the mean() and stddev() methods to get the loc and scale of the distribution
        snr_predictions_loc = predictions_distribution.mean().numpy()
        snr_predictions_scale = predictions_distribution.stddev().numpy()

        x = np.linspace(0.0, max(snr_ground_truth), 1000)

        output_file(f"{filename}_{i}.html")

        p = figure(width=800, height=400, title='SNR Predictions vs Ground Truth')

        vline = Span(location=snr_ground_truth[i], dimension='height', line_color='green', line_width=2)
        p.renderers.extend([vline])

        vline = Span(location=snr_predictions_loc[i][0], dimension='height', line_color='blue', line_width=1)
        p.renderers.extend([vline])

        y = (1/(snr_predictions_scale[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5*((x - snr_predictions_loc[i][0]) / snr_predictions_scale[i][0])**2)
                
        source = ColumnDataSource(data=dict(x=x, y=y))
        p.line('x', 'y', source=source, line_color='blue')

        save(p)
        
def plot_predictions_f(model, tf_dataset, output_name = "amplitude", filename="output.html"):
    for i in range(10):
        batch = next(iter(tf_dataset))
        onsource = tf.convert_to_tensor(batch[0]['onsource'])
        snr_ground_truth = batch[1][output_name][0].numpy()

        predictions_distribution = model(onsource)

        # Use the mean() and stddev() methods to get the loc and scale of the distribution
        snr_predictions_loc = predictions_distribution.distribution.distribution.loc.numpy()
        snr_predictions_scale = predictions_distribution.distribution.distribution.scale.numpy()

        x = np.linspace(min(snr_ground_truth), max(snr_ground_truth), 1000)

        output_file(f"{filename}_{i}.html")

        p = figure(width=800, height=400, title=F'{output_name} Predictions vs Ground Truth')

        vline = Span(location=snr_ground_truth[i], dimension='height', line_color='green', line_width=2)
        p.renderers.extend([vline])

        vline = Span(location=snr_predictions_loc[i][0], dimension='height', line_color='blue', line_width=1)
        p.renderers.extend([vline])

        y = (1/(snr_predictions_scale[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5*((x - snr_predictions_loc[i][0]) / snr_predictions_scale[i][0])**2)
                
        source = ColumnDataSource(data=dict(x=x, y=y))
        p.line('x', 'y', source=source, line_color='blue')

        save(p)
        
def plot_predictions_b(model, tf_dataset, filename="output.html"):
    for i in range(10):
        batch = next(iter(tf_dataset))
        onsource = tf.cast(batch[0]['onsource'], tf.float32)
        snr_ground_truth = batch[1]['snr'].numpy()

        predictions_distribution = model(onsource)
        
        # Here we extract the concentration parameters (alpha and beta) directly
        alpha, beta = predictions_distribution.distribution.distribution.concentration1.numpy(), predictions_distribution.distribution.distribution.concentration0.numpy()

        # Compute the median of the distribution
        median_predictions = np.percentile(predictions_distribution.sample(1000).numpy(), 50, axis=0)

        x = np.linspace(min(snr_ground_truth), max(snr_ground_truth), 1000)

        output_file(f"{filename}_{i}.html")

        p = figure(width=800, height=400, title='SNR Predictions vs Ground Truth')

        vline = Span(location=snr_ground_truth[i], dimension='height', line_color='green', line_width=2)
        p.renderers.extend([vline])

        vline = Span(location=median_predictions[i][0], dimension='height', line_color='blue', line_width=1)
        p.renderers.extend([vline])
        
        print(alpha[i], beta[i])

        y = (x**(alpha[i][0]-1) * (1 + x)**(-alpha[i][0]-beta[i][0])) / beta_function(alpha[i][0], beta[i][0])
        
        print(y)
        
        source = ColumnDataSource(data=dict(x=x, y=y))
        p.line('x', 'y', source=source, line_color='blue')

        save(p)
        
def plot_predictions_g(model, tf_dataset, filename="output.html"):
    for i in range(10):
        batch = next(iter(tf_dataset))
        onsource = tf.cast(batch[0]['onsource'], tf.float32)
        snr_ground_truth = batch[1]['snr'].numpy()

        predictions_distribution = model(onsource)

        # Here we extract the concentration and rate parameters (alpha and beta) directly
        alpha, beta = predictions_distribution.distribution.concentration.numpy(), predictions_distribution.distribution.rate.numpy()

        # Compute the median of the distribution
        median_predictions = np.percentile(predictions_distribution.sample(1000).numpy(), 50, axis=0)

        x = np.linspace(0.0, max(snr_ground_truth), 1000)

        output_file(f"{filename}_{i}.html")

        p = figure(width=800, height=400, title='SNR Predictions vs Ground Truth')

        vline = Span(location=snr_ground_truth[i], dimension='height', line_color='green', line_width=2)
        p.renderers.extend([vline])

        vline = Span(location=median_predictions[i][0], dimension='height', line_color='blue', line_width=1)
        p.renderers.extend([vline])

        y = gamma.pdf(x, a=alpha[i][0], scale=1.0/beta[i][0])  # gamma.pdf is used to compute the PDF of the gamma distribution

        source = ColumnDataSource(data=dict(x=x, y=y))
        p.line('x', 'y', source=source, line_color='blue')

        save(p)

if __name__ == "__main__":
    
    gpus = find_available_GPUs(10000, 1)
    
    strategy = setup_cuda(gpus, 8000, verbose = True)
            
    policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_global_policy(policy)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    num_examples_per_batch = 32
    sample_rate_hertz = 8192.0
    onsource_duration_seconds = 1.0
    
    with strategy.scope():
        
        # Create TensorFlow dataset from the generator
        injection_configs = [
            {
                "type" : "cbc",
                "snr"  : \
                    {"min_value" : 0.5, "max_value": 100, "mean_value": 0.5, "std": 20, "distribution_type": "uniform"},
                "injection_chance" : 0.5,
                "padding_seconds" : {"front" : 0.3, "back" : 0.3},
                "args" : {
                    "approximant_enum" : \
                        {"value" : 1, "distribution_type": "constant", "dtype" : int}, 
                    "mass_1_msun" : \
                        {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                    "mass_2_msun" : \
                        {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                    "sample_rate_hertz" : \
                        {"value" : sample_rate_hertz, "distribution_type": "constant"},
                    "duration_seconds" : \
                        {"value" : onsource_duration_seconds, "distribution_type": "constant"},
                    "inclination_radians" : \
                        {"min_value" : 0, "max_value": np.pi, "distribution_type": "uniform"},
                    "distance_mpc" : \
                        {"min_value" : 10, "max_value": 1000, "distribution_type": "uniform"},
                    "reference_orbital_phase_in" : \
                        {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                    "ascending_node_longitude" : \
                        {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                    "eccentricity" : \
                        {"min_value" : 0, "max_value": 0.1, "distribution_type": "uniform"},
                    "mean_periastron_anomaly" : \
                        {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                    "spin_1_in" : \
                        {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3},
                    "spin_2_in" : \
                        {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3}
                }
            }
        ]

        # Setting options for data distribution
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA                
        
        layers = [
            ConvLayer(64, 8, 'relu'),
            PoolLayer(8),
            ConvLayer(32, 8, 'relu'),
            ConvLayer(32, 16, 'relu'),
            PoolLayer(8),
            ConvLayer(16, 16, 'relu'),
            ConvLayer(16, 32, 'relu'),
            ConvLayer(16, 32, 'relu'),
            DenseLayer(64, 'relu'),
            DropLayer(0.5)
        ]
        
        def transform_features_labels(features, labels):
            labels['snr'] = tf.math.sqrt(labels['snr'])
            return features, labels
        
        def transform_features_labels(features, labels):
            labels['amplitude'] = labels['amplitude'][0]
            return features, labels
        
        output_name = "amplitude"
        
        # Creating the noise dataset
        cbc_ds = get_ifo_data_generator(
            time_interval = O3,
            data_labels = ["noise", "glitches", "events"],
            ifo = 'L1',
            injection_configs = injection_configs,
            sample_rate_hertz = sample_rate_hertz,
            onsource_duration_seconds = onsource_duration_seconds,
            max_segment_size = 3600,
            num_examples_per_batch = num_examples_per_batch,
            order = "random",
            seed = 123,
            apply_whitening = True,
            input_keys = ["onsource"], 
            output_keys = [output_name]
        ).with_options(options).map(transform_features_labels)
        
        builder = ModelBuilder(
            layers, 
            optimizer = optimizers.Adam(clipvalue=1.0), 
            loss = negative_loglikelihood, 
            batch_size = num_examples_per_batch
        )
        
        input_size = int(onsource_duration_seconds*sample_rate_hertz)
        builder.build_model(
            input_shape = (input_size,), 
            output_shape = 2, 
            output_name = output_name
        )
        
        builder.summary()
        
        num_train_examples    = int(2.0E4)
        num_validate_examples = int(1.0E2)
        
        builder.train_model(cbc_ds, num_train_examples//num_examples_per_batch, 100)
        builder.model.save_weights('model_weights.h5')
    
        plot_predictions(
            builder.model, 
            cbc_ds, 
            output_name = output_name,
            filename="model_predictions.html"
        )