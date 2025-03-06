# sciNeurotech Lab 
# Theodore


# in this file, we will define utility functions used across the project.


#import libs
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import ipywidgets as widgets
from ipywidgets import interact


def standardize_vector(vec: np.ndarray) -> np.ndarray:
    """standardize a vector

    Args:
        vec (np.ndarray): the array you want to standardize

    Returns:
        np.ndarray: the standardized vector
    """
    mean = np.mean(vec)
    std = np.std(vec)
    return (vec - mean) / std if std != 0 else vec/vec # avoid divide with zero

def map_plot(my_map, title: str = 'map', figsize=(6, 6)):
    """
    Plots a 2D map with color scale and values displayed at each cell.

    This function visualizes a 2D matrix (`my_map`) using a color map, with the values of the 
    matrix displayed at each corresponding location. The color range is automatically scaled 
    to the minimum and maximum values of the matrix, and a color bar is included to indicate 
    the value scale. The function allows customization of the plot title.

    Args:
        my_map (numpy.ndarray): A 2D numpy array to be plotted.
        title (str, optional): The title of the plot. Default is 'map'.

    Returns:
        None: The function displays the plot but does not return anything.
    
    Example:
        map_plot(np.array([[1, 2], [3, 4]]), 'Sample Map')
    """
    # Set up the figure with a fixed size
    plt.figure(figsize=figsize)
    
    # Display the map using a color map and scale it according to the minimum and maximum values
    plt.imshow(my_map, cmap='coolwarm', vmin=np.nanmin(my_map), vmax=np.nanmax(my_map))
    
    # Add a color bar to indicate the value scale
    plt.colorbar(label="values")
    
    # Set the title of the plot
    plt.title(title)

    # Loop over each cell of the matrix to display the value in the center of the cell
    for i in range(my_map.shape[0]):
        for j in range(my_map.shape[1]):
            plt.text(j, i, f"{my_map[i, j]:.3g}", ha='center', va='center', color="white", fontsize=8)

    # Show the plot
    plt.show()

def labeled_inputs_plot(train_input, train_label, comb_idx: int = 0, values: bool = False, figsize=(10, 8)):
    """
    Plots labeled inputs and their corresponding labels with an option to display values.
    
    Args:
        train_input (torch.Tensor): Tensor of input maps with shape (n_combinations, n_features, height, width).
        train_label (torch.Tensor): Tensor of label maps with shape (n_combinations, n_outputs, height, width).
        comb_idx (int): the index of the data you want to plot
        values (bool): If True, display the value of each cell in the plots.
    """
    
    # Define the figure and axes for the subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize) 
    
    # Define the maps for the plots
    maps = [train_input[comb_idx, 0], train_input[comb_idx, 3], train_label[comb_idx, 0], 
            train_input[comb_idx, 1], train_input[comb_idx, 2], train_label[comb_idx, 1]]
    
    # Titles for the plots
    titles = ["mean map (before)", "elem_mean(query_x) * query_y", "mean map (after)",
              "std map (before)", "elem_std(query_x)", "std map (after)"]
    
    # Normalize the color scale for the std maps and other maps
    std_maps = [train_input[comb_idx, 1], train_input[comb_idx, 2], train_label[comb_idx, 1]]
    other_maps = [train_input[comb_idx, 0], train_input[comb_idx, 3], train_label[comb_idx, 0]]
    
    std_min = torch.min(torch.stack(std_maps))
    std_max = torch.max(torch.stack(std_maps))
    other_min = torch.min(torch.stack(other_maps))
    other_max = torch.max(torch.stack(other_maps))
    
    # Plot the maps
    for idx, ax in enumerate(axes.flat):
        # Select colormap and normalization based on the type of map
        if idx in [3, 4, 5]:  # std maps
            cax = ax.imshow(maps[idx], cmap='cividis', vmin=0, vmax=std_max)
        else:  # other maps
            cax = ax.imshow(maps[idx], cmap='coolwarm', vmin=other_min, vmax=other_max)
        
        ax.set_title(titles[idx])
        plt.colorbar(cax, ax=ax)
        
        # Add cell values as text if values=True
        if values:
            data = maps[idx].cpu().numpy()  # Convert tensor to numpy array
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{data[i, j]:.3g}", ha='center', va='center', 
                            color="white", fontsize=6)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def compare_output_label(train_input, train_label, output, comb_idx: int = 0, values: bool = False, figsize = (15, 8)):
    """
    Plots labeled inputs and their corresponding labels with an option to display values.
    
    Args:
        train_input (torch.Tensor): Tensor of input maps with shape (n_combinations, n_features, height, width).
        train_label (torch.Tensor): Tensor of label maps with shape (n_combinations, n_outputs, height, width).
        comb_idx (int): the index of the data you want to plot
        values (bool): If True, display the value of each cell in the plots.
    """
    
    # Define the figure and axes for the subplots
    fig, axes = plt.subplots(2, 5, figsize=figsize) 
    
    # Define the maps for the plots
    maps = [train_input[comb_idx, 0], train_input[comb_idx, 3], train_label[comb_idx, 0], output[comb_idx, 0], output[comb_idx, 0]-train_label[comb_idx, 0],
            train_input[comb_idx, 1], train_input[comb_idx, 2], train_label[comb_idx, 1], output[comb_idx, 1], output[comb_idx, 1]-train_label[comb_idx, 1]]
    
    # Titles for the plots
    titles = ["mean map (nb_query=N)", "elem_mean(query_x) * query_y", "mean map (nb_query=N+1)", "mean map output", "error mean", 
              "std map (nb_query=N)", "elem_std(query_x)", "std map (nb_query=N+1)", "std map output", "error std"]
    
    # Normalize the color scale for the std maps and other maps
    std_maps = [train_input[comb_idx, 1], train_input[comb_idx, 2], train_label[comb_idx, 1], output[comb_idx, 1]]
    mean_maps = [train_input[comb_idx, 0], train_input[comb_idx, 3], train_label[comb_idx, 0], output[comb_idx, 0]]
    
    std_min = torch.min(torch.stack(std_maps))
    std_max = torch.max(torch.stack(std_maps))
    other_min = torch.min(torch.stack(mean_maps))
    other_max = torch.max(torch.stack(mean_maps))
    
    # Plot the maps
    for idx, ax in enumerate(axes.flat):
        # Select colormap and normalization based on the type of map
        if idx in [5, 6, 7, 8]:  # std maps
            cax = ax.imshow(maps[idx], cmap='cividis', vmin=0, vmax=std_max)
        elif idx in [0, 1, 2, 3]:  # mean maps
            cax = ax.imshow(maps[idx], cmap='coolwarm', vmin=other_min, vmax=other_max)
        else: # error map
            cax = ax.imshow(maps[idx], cmap='PiYG', vmin=-torch.max(torch.abs(maps[idx])), vmax=torch.max(torch.abs(maps[idx])))
        
        ax.set_title(titles[idx])
        plt.colorbar(cax, ax=ax)
        
        # Add cell values as text if values=True
        if values:
            data = maps[idx].cpu().numpy()  # Convert tensor to numpy array
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{data[i, j]:.3g}", ha='center', va='center', 
                            color="white", fontsize=6)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_combinations(train_input, train_label, output=None, comb_idx_range=(0, 9), values=False):
    """
    Function to plot multiple combinations with a button to switch between cases.

    Args:
        train_input (torch.Tensor): Tensor of input maps with shape (n_combinations, n_features, height, width).
        train_label (torch.Tensor): Tensor of label maps with shape (n_combinations, n_outputs, height, width).
        output (torch.Tensor): Tensor of predicted output maps with shape (n_combinations, n_outputs, height, width).
        comb_idx_range (tuple): Range of the combination indices to visualize (default 0-9).
        values (bool): If True, display the value of each cell in the plots.
    """

    if output is None:
        def update_plot(comb_idx):
            # Plot the comparison for the selected index
            labeled_inputs_plot(train_input, train_label, comb_idx, values)
    else:
        def update_plot(comb_idx):
            # Plot the comparison for the selected index
            compare_output_label(train_input, train_label, output, comb_idx, values)
    
    # Create the interactive widget
    interact(update_plot, comb_idx=widgets.IntSlider(value=comb_idx_range[0], min=comb_idx_range[0], max=comb_idx_range[1], step=1, description="Comb Index"))
    
def fill_nans_with_mean(map_emg):
    """
    Remplace toutes les valeurs NaN dans l'array par la moyenne des valeurs voisines valides.
    
    Args:
        map_emg (np.ndarray): Array 2D contenant des valeurs et éventuellement des NaN.
    
    Returns:
        np.ndarray: Array où les NaN ont été remplacés par la moyenne de leurs voisins valides.
    """
    # Obtenir les dimensions de la carte
    rows, cols = map_emg.shape
    
    # Copier l'array pour éviter de modifier l'original
    filled_map = map_emg.copy()
    
    # Parcourir toutes les positions de l'array
    for x in range(rows):
        for y in range(cols):
            if np.isnan(filled_map[x, y]):  # Vérifier si la cellule est NaN
                # Collecter les valeurs voisines
                neighbors = []
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        # S'assurer que les indices sont valides (dans les limites de l'array)
                        if 0 <= x + i < rows and 0 <= y + j < cols and not (i == 0 and j == 0):
                            neighbor_value = filled_map[x + i, y + j]
                            if not np.isnan(neighbor_value):
                                neighbors.append(neighbor_value)
                
                # Si des voisins valides existent, calculer la moyenne et remplacer le NaN
                if neighbors:
                    filled_map[x, y] = np.mean(neighbors)
    
    return filled_map
