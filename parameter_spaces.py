import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.palettes import Sunset11

def random_peak(x, y):
    a = np.random.uniform(-4.0,4.9, size = 6)
    
    return a[0] * np.exp(-((x / a[1] - a[2])**2 + (y / a[3] - a[4])**2 / (2*a[5]**2)))

def objective_function(x, y):
    """
    Multimodal synthetic objective function.
    """
    # Three different Gaussian functions with different means and standard deviations
    value = 2.0 * np.exp(-((x - 2)**2 + (y - 2)**2) / (2 * 0.8**2))
    value += 1.5 * np.exp(-((x + 2.5)**2 + (y + 2.5)**2) / (2 * 1.5**2))
    value += 3.0 * np.exp(-((x - 1.5)**2 + (y + 3)**2) / (2 * 0.5**2))
    
    # Add a negative elliptical shape
    value -= 1.0 * np.exp(-((x / 2.0)**2 + (y / 1.5)**2))
    value += 2.2 * np.exp(-((0.1 * x / 0.1)**2 + (0.25 * y / 4.0)**2))

    value -= 1.5 * np.exp(-((x / 1.8)**2 + (4 * y / 1.5)**2))

    return value

def objective_function(x, y):
    
    value = 0
    for _ in range(100):
        value += random_peak(x, y)
        
    print(value)
    
    return value

def plot_objective_function(X, Y, Z, levels=15, filename="output.html"):
    # Reverse the Viridis palette for better visual

    """
    color_mapper = LinearColorMapper(palette=Viridis256, low=Z.min(), high=Z.max())

    # Initialize a figure
    p = figure(title="Objective Function", toolbar_location="right",
               x_range=(X.min(), X.max()), y_range=(Y.min(), Y.max()))

    # Add contour plot to the figure
    contour = Contour(x=X, y=Y, z=Z, levels=levels, line_color=None, fill_color={'field': 'z', 'transform': color_mapper})
    p.add_tools(contour)

    # Add color bar
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))
    p.add_layout(color_bar, 'right')
    """

    # Save the bokeh figure to an HTML file
    output_file(filename)
    show(p)
    
def plot_objective_function(X, Y, Z, filename="output.html"):
    # Creating the figure object
    p = figure(width=600, height=500, toolbar_location=None, x_range=(X.min(), X.max()), y_range=(Y.min(), Y.max()),
               title="Contour plot of objective function")

    # Define levels
    levels = np.linspace(Z.min(), Z.max(), 11)

    # Adding a contour plot to the figure
    contour_renderer = p.contour(X, Y, Z, levels, line_color=Sunset11, line_width=2)

    # Constructing color bar
    colorbar = contour_renderer.construct_color_bar()

    # Adding color bar to the figure
    p.add_layout(colorbar, 'right')

    # Save the plot to an html file
    output_file(filename)
    show(p)


# Define the grid for visualization purposes
x = np.linspace(-6, 6, 1000)
y = np.linspace(-6, 6, 1000)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# Call the function to plot the objective function
plot_objective_function(X, Y, Z)