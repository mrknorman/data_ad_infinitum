import numpy as np
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.palettes import Sunset11

from skopt import gp_minimize
from skopt.space import Real

def random_peak(x, y):
    a = np.random.uniform(-4.0,4.9, size = 6)
    
    return a[0] * np.exp(-((x / a[1] - a[2])**2 + (y / a[3] - a[4])**2 / (2*a[5]**2)))

def objective_function(x, y):
    
    value = 0
    for _ in range(100):
        value += random_peak(x, y)
        
    return value

def plot_objective_function(X, Y, Z, search_points = None, filename="output.html"):
    # Creating the figure object
    p = figure(width=600, height=500, toolbar_location=None, x_range=(X.min(), X.max()), y_range=(Y.min(), Y.max()),
               title="Contour plot of objective function")
    
    output_notebook()

    # Define levels
    levels = np.linspace(Z.min(), Z.max(), 11)

    # Adding a contour plot to the figure
    contour_renderer = p.contour(X, Y, Z, levels, line_color=Sunset11, line_width=2)

    # Constructing color bar
    colorbar = contour_renderer.construct_color_bar()

    # Adding color bar to the figure
    p.add_layout(colorbar, 'right')

     # Plot search points if provided
    if search_points is not None:
        p.circle(search_points[:, 0], search_points[:, 1], size=10, color="red", alpha=0.6)
    
    # Save the plot to an html file
    output_file(filename)
    show(p)

def grid_search(x_range, y_range, steps=10):
    x_points = np.linspace(x_range[0], x_range[1], steps)
    y_points = np.linspace(y_range[0], y_range[1], steps)
    X, Y = np.meshgrid(x_points, y_points)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    return points

def random_search(x_range, y_range, num_samples=100):
    x_points = np.random.uniform(x_range[0], x_range[1], num_samples)
    y_points = np.random.uniform(y_range[0], y_range[1], num_samples)
    points = np.vstack([x_points, y_points]).T
    return points

def bayesian_optimization(objective, x_range, y_range, n_calls=100):
    def wrapped_objective(xy):
        # Pass the tuple xy directly to the objective function
        return -objective(xy)  # Negate for minimization

    res = gp_minimize(wrapped_objective, [Real(x_range[0], x_range[1]), Real(y_range[0], y_range[1])],
                      n_calls=n_calls, random_state=5, n_initial_points = 100)

    return np.array(res.x_iters)

def modified_objective_function(xy):
    x, y = xy
    # Since x and y are single values, calculate the objective function value for this point
    value = objective_function(np.array(x), np.array(y))
    # Make sure to return a scalar; here we return the first element of the array
    return value.item()

np.random.seed(100)
# Define the grid for visualization purposes
x = np.linspace(-6, 6, 1000)
y = np.linspace(-6, 6, 1000)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# Call the function to plot the objective function
plot_objective_function(X, Y, Z)

grid_points = grid_search([-6, 6], [-6, 6], steps=10)
plot_objective_function(X, Y, Z, search_points=grid_points, filename="grid_search_output.html")

grid_points = random_search([-6, 6], [-6, 6], num_samples=100)
plot_objective_function(X, Y, Z, search_points=grid_points, filename="random_search_output.html")

grid_points = bayesian_optimization(modified_objective_function, [-6, 6], [-6, 6], n_calls=200)
plot_objective_function(X, Y, Z, search_points=grid_points, filename="bayesian_search_output.html")
