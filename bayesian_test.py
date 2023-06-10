from py_ml_tools.dataset import get_ifo_data_generator, get_ifo_data, O3
from py_ml_tools.model   import ModelBuilder, DenseLayer, ConvLayer, PoolLayer, DropLayer, randomizeLayer, negative_loglikelihood
from py_ml_tools.setup   import load_label_datasets, setup_cuda, find_available_GPUs

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy

import numpy as np

from bokeh.plotting import figure, save
from bokeh.models import Span, ColumnDataSource
from bokeh.io import output_file
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()

def plot_predictions(model, tf_dataset, filename="output.html"):
    for i in range(10):
        batch = next(iter(tf_dataset))
        onsource = tf.convert_to_tensor(batch[0]['onsource'])
        snr_ground_truth = batch[1]['snr'].numpy()

        predictions_distribution = model(onsource)
        snr_predictions_mean = predictions_distribution.mean().numpy()
        snr_predictions_stddev = predictions_distribution.stddev().numpy()

        x = np.linspace(min(snr_ground_truth), max(snr_ground_truth), 1000)
        
        output_file(f"{filename}_{i}.html")
        
        p = figure(width=800, height=400, title='SNR Predictions vs Ground Truth')
        
        vline = Span(location=snr_ground_truth[i], dimension='height', line_color='green', line_width=2)
        p.renderers.extend([vline])

        vline = Span(location=snr_predictions_mean[i][0], dimension='height', line_color='blue', line_width=1)
        p.renderers.extend([vline])

        y = (1/(snr_predictions_stddev[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5*((x - snr_predictions_mean[i][0]) / snr_predictions_stddev[i])**2)
        source = ColumnDataSource(data=dict(x=x, y=y))
        p.line('x', 'y', source=source, line_color='blue')

        save(p)

if __name__ == "__main__":
    
    gpus = find_available_GPUs(16000, 1)
    
    strategy = setup_cuda(gpus, 10000, verbose = True)
            
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
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
                    {"min_value" : 0.1, "max_value": 100, "mean_value": 0.0, "std": 40,  "distribution_type": "normal"},
                "injection_chance" : 1.0,
                "padding_seconds" : {"front" : 0.2, "back" : 0.1},
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
            DenseLayer(64,  'relu'),
            DropLayer(0.5)
        ]
        
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
            output_keys = ["snr"]
        ).prefetch(tf.data.experimental.AUTOTUNE).with_options(options)
        
        builder = ModelBuilder(
            layers, 
            optimizer = 'adam', 
            loss = negative_loglikelihood, 
            batch_size = num_examples_per_batch
        )
        
        input_size = int(onsource_duration_seconds*sample_rate_hertz)
        builder.build_model(input_shape = (input_size,), output_shape = 2)
        
        builder.summary()
        
        num_train_examples    = int(2.0E5)
        num_validate_examples = int(1.0E2)
        
        builder.train_model(cbc_ds, num_train_examples//num_examples_per_batch, 10)
        builder.model.save_weights('model_weights.h5')
    
        plot_predictions(builder.model, cbc_ds, filename="model_predictions.html")