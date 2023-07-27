import numpy as np

from ipywidgets import Button, VBox, HBox, HTMLMath
import bqplot.pyplot as plt
from bqplot import LinearScale

def linreg(x, y):
    '''
    computes intercept and slope for a simple
    ols regression
    '''
    b = np.cov(x, y)[0, 1] / np.var(x)
    a = np.mean(y) - b * np.mean(x)
    return a, b

x = np.linspace(-10, 10, 50)
y = 5 * x + 7 + np.random.randn(50) * 20

def update_regline(*args):
    # update the y attribute of the reg_line with 
    # the results of running the ols regression on 
    # x and y attributes of the scatter plot
    a, b = linreg(scatter.x, scatter.y)
    reg_line.y = a + b * reg_line.x
    
    # update the equation label
    equation_label.value = eqn_tmpl.format(a, b)

# Add a scatter plot and a regression line on the same figure
axes_options = {'x': {'label': 'X'},
                'y': {'label': 'Y'}}
fig = plt.figure(title='Linear Regression', animation_duration=1000)
                 
plt.scales(scales={'x': LinearScale(min=-30, max=30),
                   'y': LinearScale(min=-150, max=150)})

scatter = plt.scatter(x, y, colors=['orangered'], default_size=100, 
                      enable_move=True, stroke='black')
reg_line = plt.plot(np.arange(-30, 31), [], 'g', stroke_width=8,
                    opacities=[.5], axes_options=axes_options)

fig.layout.width = '800px'
fig.layout.height = '550px'

reset_button = Button(description='Reset', button_style='success')
reset_button.layout.margin = '0px 30px 0px 60px'

eqn_tmpl = 'Regression Line: ${:.2f} + {:.2f}x$'
equation_label = HTMLMath()

def reset_points(*args):
    '''
    resets the scatter's x and y points 
    to the original values
    '''
    with scatter.hold_sync():
        # hold_sync will send trait updates 
        # (x and y here) to front end in one trip
        scatter.x = x
        scatter.y = y

# on button click reset the scatter points
reset_button.on_click(lambda btn: reset_points())
# recompute reg line when new points are added
scatter.observe(update_regline, ['x', 'y'])

# compute the reg line
update_regline(None)

VBox([fig, HBox([reset_button, equation_label])])

