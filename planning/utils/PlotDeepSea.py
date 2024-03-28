from filecmp import cmp
import matplotlib.pyplot as plt
import numpy as np


class PlotDeepSea():

    def __init__(self, params, action_name):

        self.grid_size = params.environment_params.grid_size
        self.num_actions = 2
        self.data_size = int(((self.grid_size * (self.grid_size+1))/2))
        self.row_indexes = [np.sum(np.arange(i)) for i in range(1,self.grid_size+1)]
        self.grid_data = np.zeros((self.grid_size, self.grid_size))
        self.min_value = -0.01
        # self.max_value = 1 + self.min_value
        #keeping maximum sensitive to the size of the grid
        self.max_value = 0.01/self.grid_size

        # displaying
        self.fig = plt.figure(figsize=(8, 8), tight_layout=True)
        self.fig.suptitle(action_name)
        self.viewer = self.fig.add_subplot(111)
        plt.ion() # Turns interactive mode on (probably unnecessary)
        self.fig.show() # Initially shows the figure
        init_data = np.random.uniform(self.min_value, self.max_value, (self.grid_size, self.grid_size))
        te = self.viewer.imshow(init_data, cmap='ocean', vmin=self.min_value, vmax=self.max_value) # Loads the new image
        self.color_bar = self.fig.colorbar(te, label="Value", orientation="vertical", cmap='ocean')
        # plt.pause(0.001)

    
    def convert_to_grid(self, data):
        for i in range(self.grid_size):
            for j in range(i+1):
                self.grid_data[i][j] = data[self.row_indexes[i] + j]

    def update_plot(self, data):
        self.convert_to_grid(data)
        self.viewer.clear() # Clears the previous image
        # te = self.viewer.imshow(self.grid_data, cmap='ocean', vmin=self.min_value, vmax=self.max_value) # Loads the new image
        self.color_bar.remove()
        te = self.viewer.imshow(self.grid_data, cmap='ocean') # Loads the new image
        self.color_bar = self.fig.colorbar(te, label="Value", orientation="vertical", cmap='ocean')
        plt.pause(.001) # Delay in seconds
        self.fig.canvas.draw() # Draws the image to the screen







