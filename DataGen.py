# sciNeurotech Lab 
# Theodore

# This file contains the DataGen class, which is used to generate, process, and manipulate data for Gaussian Process optimization and modeling tasks.

# import lib
import torch
import numpy as np
from utils import *
import itertools
import random
import pickle
import gpytorch
from typing import Union, List

# import file
# from FixedGP import FixedGP
from GPcustom.models import FixedGP, GPytorchFixed

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

class DataGen():
    """
    A class to generate, process, and manipulate data for Gaussian Process optimization and modeling tasks.

    Attributes:
        map (np.ndarray): The input 2D array representing the spatial map.
        space_dim (int): Dimensionality of the input space derived from the shape of the map.
        space_size (int): Total number of elements in the input space (flattened size of the map).
        ch2xy (np.ndarray): Coordinates associated with the map elements.
        map_idx (np.ndarray): Array of indices corresponding to the flattened map.
        X_test_normed (np.ndarray): Normalized input space coordinates within [0, 1].
        kernel (GPy.kern): Kernel object for the Gaussian Process model.
        maps_kernel (np.ndarray): Kernel values mapped to the spatial structure of the map.

    Methods:
        __init__(map, ch2xy=None):
            Initializes the class with a given map and optional coordinates.

        define_ch2xy(ch2xy=None):
            Defines the coordinates of the input space if not provided.

        define_map_idx():
            Maps indices of flattened input space to the spatial map.

        vec_to_map(the_vec):
            Converts a vector to the spatial map format.

        map_to_img(the_map):
            Converts a 2D map to a 4D PyTorch tensor suitable for neural networks.

        normalized_input_space():
            Normalizes the input space coordinates to the range [0, 1].

        set_kernel(kernel_type, noise_std, output_std, lengthscale):
            Configures the kernel for the Gaussian Process.

        get_elem_maps(kernel_type, noise_std, output_std, lengthscale):
            Generates element maps for all elements in the map.

        pkl_save(L, name):
            Saves a list to a pickle file.

        pkl_load(name):
            Loads a list from a pickle file.

        generate_combinations(nb_queries, nb_comb=None):
            Generates random combinations of indices for queries.

        new_generate_combinations(self, nb_queries: int, nb_comb: int = None):
            Generate lots of differents (not necessarily unique) combinations of indices of a given size.
            Because "itertools.combinations(range(self.space_size)" in generate_combinations method could 
            require too much memory, we implement this new method.

        generate_selections(self, nb_queries: int, nb_comb: int = None):
            Generate lots of differents (not necessarily unique) selections of indices of a given size.
            In contrast to generate_combinations and new_generate_combinations, which perform draws without discount 
            (they don't allow repetitions), generate_selections allows repetitions. 

        generate_idx_inputs(self, nb_queries: int, nb_comb: int = None, combination: bool = False, combination_without_rep: bool = False):
            Creates input indices for Gaussian Process training and querying.

        get_inputs_for_GPs(train_X_idx, query_x_idx):
            Prepares inputs and labels for training and querying Gaussian Processes.

        generate_pre_labeled_inputs(name, nb_queries, nb_comb=None, kernel_type='rbf', noise_std=0.1, output_std=1, lengthscale=0.05):
            Generates and saves pre-labeled inputs for Gaussian Processes.

        format_labeled_inputs(name):
            Loads, formats, and saves labeled inputs for PyTorch-based models.
    """
    def __init__(self, map: np.ndarray, ch2xy: np.ndarray = None) -> None:
        """
        Initializes the DataGen object with a given map and optional coordinates.

        Args:
            map (np.ndarray): A 2D array representing the data map.
            ch2xy (np.ndarray, optional): Coordinates of channels in the map. Defaults to None.
        """
        self.map = map
        self.space_dim = len(map.shape)
        self.space_size = map.size

        self.define_ch2xy(ch2xy)
        self.define_map_idx()
        self.normalized_input_space()

    def define_ch2xy(self, ch2xy: np.ndarray = None) -> None:
        """
        Defines the mapping of channels to coordinates.

        If no mapping is provided, creates one based on the map's shape.

        Args:
            ch2xy (np.ndarray, optional): Channel-to-coordinate mapping. Defaults to None.
        """
        if ch2xy is None:
            nb_ch_x = self.map.shape[0]
            nb_ch_y = self.map.shape[1]

            # Define the ranges of values
            x = np.arange(1, nb_ch_x+1)
            y = np.arange(1, nb_ch_y+1)

            # Create a grid of all possible combinations between x and y values
            X, Y = np.meshgrid(x, y)

            # Stack the grids and reshape to get an array with the correct shape
            # Each row in 'combinations' represents a unique (x, y) combination
            ch2xy = np.column_stack([X.ravel(), Y.ravel()])

        self.ch2xy = ch2xy

    def define_map_idx(self) -> None:
        """
        Maps each index in the flattened map to its corresponding value.

        Uses `ch2xy` to assign values from `map` to `map_idx`.
        """
        self.map_idx = np.full(self.space_size, np.nan)
        for i in range(self.space_size):
            self.map_idx[i] = self.map[self.ch2xy[i][0]-1,self.ch2xy[i][1]-1]

    def vec_to_map(self, the_vec) -> None:
        """
        Converts a vector to the corresponding map.

        Args:
            the_vec (np.ndarray): A vector of values.

        Returns:
            np.ndarray: A 2D array matching the shape of the original map.
        """
        the_map = np.full((self.map.shape[0], self.map.shape[1]), np.nan)
        for i in range(self.space_size):
            the_map[int(self.ch2xy[i,0]-1), int(self.ch2xy[i,1]-1)] = the_vec[i]
        return(the_map)
    
    def map_to_img(self, the_map) -> None:
        """
        Converts a map to an image tensor suitable for PyTorch models.

        Args:
            the_map (np.ndarray): A 2D array to convert.

        Returns:
            torch.Tensor: A PyTorch tensor with shape (1, 1, height, width).
        """
        the_img = torch.full((1, 1, the_map.shape[0], the_map.shape[1]), torch.nan)
        the_img[0,0] = torch.from_numpy(the_map)
        return the_img

    def normalized_input_space(self) -> None:
        """
        Normalizes the input space coordinates to a [0, 1] range.

        Normalization ensures consistency for Gaussian Process inputs.
        
        Array:
            - `X_test_normed`: An array of normalized coordinates representing the input space for the Gaussian Process. 
                               The shape of the tensor is the same as 'ch2xy'. The normalization ensures that all  
                               coordinates are in the range [0, 1]. shape is (space_size, space_dim)
        """
        # Normalize the coordinates to the range [0, 1]
        self.X_test_normed = ((self.ch2xy - np.min(self.ch2xy, axis=0)) /
                                        (np.max(self.ch2xy, axis=0) - np.min(self.ch2xy, axis=0)))
        self.X_test_normed_tensor = torch.from_numpy(self.X_test_normed).double()
    
    # def set_kernel(self, kernel_type, noise_std, output_std, lengthscale) -> None:
    #     """
    #     Sets the kernel for Gaussian Process modeling.

    #     Args:
    #         kernel_type (str): The type of kernel ('rbf', 'Mat32', 'Mat52').
    #         noise_std (float): Noise standard deviation.
    #         output_std (float): Output standard deviation.
    #         lengthscale (float or list): Lengthscale(s) for the kernel.

    #     Raises:
    #         ValueError: If the kernel_type or lengthscale is not well defined.
    #     """
    #     if isinstance(lengthscale, float) or (isinstance(lengthscale, list) and len(lengthscale) == 1):     
    #         if kernel_type == 'rbf':
    #             self.kernel = GPy.kern.RBF(input_dim=self.space_dim, variance=output_std**2, lengthscale=lengthscale)
    #         elif kernel_type == 'Mat32':
    #             self.kernel = GPy.kern.Matern32(input_dim=self.space_dim, variance=output_std**2, lengthscale=lengthscale)
    #         elif kernel_type == 'Mat52':
    #             self.kernel = GPy.kern.Matern52(input_dim=self.space_dim, variance=output_std**2, lengthscale=lengthscale)
    #         else:
    #             raise ValueError("The attribute kernel_type is not well defined")
    #     elif len(lengthscale) == self.space_dim:
    #         if kernel_type == 'rbf':
    #             self.kernel = GPy.kern.RBF(input_dim=self.space_dim, variance=output_std**2, lengthscale=lengthscale, ARD=True)
    #         elif kernel_type == 'Mat32':
    #             self.kernel = GPy.kern.Matern32(input_dim=self.space_dim, variance=output_std**2, lengthscale=lengthscale, ARD=True)
    #         elif kernel_type == 'Mat52':
    #             self.kernel = GPy.kern.Matern52(input_dim=self.space_dim, variance=output_std**2, lengthscale=lengthscale, ARD=True)
    #         else:
    #             raise ValueError("The attribute kernel_type is not well defined")
    #     else:
    #         raise ValueError("The attribute lengthscale is not well defined")
        
    def get_elem_maps(self, kernel_type, noise, outputscale, lengthscale, using_GPytorch=False) -> None:
        """
        Generates element maps for all elements in the map.
        For each 'pixel', we will generate mean and std map for a unique query : 
                    x="this pixel"      and     y=1.

        Args:
            kernel_type (str): The type of kernel ('RBF', 'Matern32', 'Matern52').
            noise (float): Noise variance.
            outputscale (float): Output variance.
            lengthscale (float or list): Lengthscale(s) for the kernel.
            using_GPytorch (bool): If True, we use GPytorchFixed. If False, we use FixedGP.
        """
        self.maps_elem_mean = np.full((self.space_size, self.map.shape[0], self.map.shape[1]), np.nan)
        self.maps_elem_std = np.full((self.space_size, self.map.shape[0], self.map.shape[1]), np.nan)

        if using_GPytorch:
            for i in range(self.space_size):
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                elem_gp = GPytorchFixed(
                                train_x=self.X_test_normed_tensor[i:i+1], train_y=torch.tensor([1.]), 
                                likelihood=likelihood, kernel_type=kernel_type, noise=noise, 
                                outputscale=outputscale, lengthscale=lengthscale)
                elem_gp.double()
                elem_mean, elem_std = elem_gp.predict(self.X_test_normed_tensor)
                for j in range(self.space_size):
                    self.maps_elem_mean[i, int(self.ch2xy[j,0]-1), int(self.ch2xy[j,1]-1)] = elem_mean[j].detach().item()
                    self.maps_elem_std[i, int(self.ch2xy[j,0]-1), int(self.ch2xy[j,1]-1)] = elem_std[j].detach().item()
        else:
            #self.set_kernel(kernel_type, np.sqrt(noise), np.sqrt(outputscale), lengthscale)
            for i in range(self.space_size):
                elem_gp = FixedGP(input_space=self.X_test_normed, 
                                train_X=self.X_test_normed[i:i+1], train_Y=np.array([[1.]]), 
                                kernel_type=kernel_type, noise_std=np.sqrt(noise), 
                                output_std=np.sqrt(outputscale), lengthscale=lengthscale)
                elem_mean, elem_std = elem_gp.predict()
                for j in range(self.space_size):
                    self.maps_elem_mean[i, int(self.ch2xy[j,0]-1), int(self.ch2xy[j,1]-1)] = elem_mean[j]
                    self.maps_elem_std[i, int(self.ch2xy[j,0]-1), int(self.ch2xy[j,1]-1)] = elem_std[j]

    def pkl_save(self, L: list, name: str):
        """Save a list to a pickle file.

        Args:
            L (list): The list to save.
            name (str): The name of the pickle file (without extension).
        """
        with open('dataset/pkl_files/'+name+'.pkl', 'wb') as f:
            pickle.dump(L, f)

    def pkl_load(self, name: str):
        """Load a list from a pickle file.

        Args:
            name (str): The name of the pickle file (without extension).

        Returns:
            list: The loaded list from the pickle file.
        """
        with open('dataset/pkl_files/'+name+'.pkl', 'rb') as f:
            L = pickle.load(f)
        return L

    def generate_combinations(self, nb_queries: int, nb_comb: int = None) -> list:
        """Generate unique combinations of indices of a given size.

        Args:
            nb_queries (int): The number of elements in each combination.
            nb_comb (int, optional): The number of unique combinations to generate. 
                Defaults to the total number of possible combinations.

        Returns:
            list: A list of randomly selected unique combinations of indices.

        Raises:
            ValueError: If the requested number of combinations exceeds the total number of possible combinations.
        """        
        # Generate all possible combinations of size `nb_queries` from integers 0 to space_size-1
        all_combinations = list(itertools.combinations(range(self.space_size), nb_queries))

        if nb_comb is None:
            nb_comb = len(all_combinations)

        # Check if there are enough unique combinations to select `nb_comb` sets
        if nb_comb > len(all_combinations):
            raise ValueError("Not enough unique combinations to select `nb_comb` sets")
        
        # Randomly select `nb_comb` distinct combinations from the list
        selected_combinations = [list(comb) for comb in random.sample(all_combinations, nb_comb)]
        
        return selected_combinations

    def new_generate_combinations(self, nb_queries: int, nb_comb: int) -> list:
        """Generate lots of differents (not necessarily unique) combinations of indices of a given size.
        Because "itertools.combinations(range(self.space_size)" in generate_combinations method could 
        require too much memory, we implement this new method.

        Args:
            nb_queries (int): The number of elements in each combination.
            nb_comb (int): The number of combinations to generate. 

        Returns:
            list: A list of randomly selected combinations of indices.
        """        

        random_combinations = [list(random.sample(range(self.space_size), nb_queries)) for _ in range(nb_comb)]
        
        return random_combinations

    def generate_selections(self, nb_queries: Union[int, List[int]], nb_comb: Union[int, List[int]]) -> list:
        """Generate lots of differents (not necessarily unique) selections of indices of a given size.
        In contrast to generate_combinations and new_generate_combinations, which perform draws without discount 
        (they don't allow repetitions), generate_selections allows repetitions. 

        Args:
            nb_queries (int or list(int)): The number of elements in each selections.
            nb_comb (int or list(int)): The number of unique selections to generate. 

        Returns:
            list: A list of randomly selected selections of indices.
        """     
        if (type(nb_queries) != type(nb_comb)):
            raise ValueError("nb_queries & nb_comb should have the same type.")
        

        
        if type(nb_queries) == int:
            random_selections = [list(random.choices(range(self.space_size), k=nb_queries)) for _ in range(nb_comb)]

        else:
            if len(nb_queries)!=len(nb_comb):
                raise ValueError("The lists nb_queries & nb_comb should have the same length.")
            
            random_selections = []
            for i in range(len(nb_comb)):
                random_selections += [list(random.choices(range(self.space_size), k=nb_queries[i])) for _ in range(nb_comb[i])]
        
        return random_selections
    
    def generate_idx_inputs(self, nb_queries: Union[int, List[int]], nb_comb: Union[int, List[int]] = None, combination: bool = False, combination_without_rep: bool = False) -> list:
        """Generate indexed inputs with random selection for queries.

        Args:
            nb_queries (int or list(int)): The number of queries per combination.
            nb_comb (int or list(int), optional): The number of combinations to generate. Defaults to None.
            combination (bool, optional): If False, we use sampling with replacement. If True, we use sampling without replacement. 
                                          Defaults to False.  
            combination_without_rep (bool, optional): If False, combinations could be repetitive in the dataset. 
                                                      If True, each combination is unique. Defaults to False.

        Returns:
            list: A list of indexed inputs, each containing a selection and a randomly selected query index.
        """
        if combination:
            if combination_without_rep:
                # Generate nb_comb combinations of size `nb_queries` from integers 0 to space_size-1
                selected_selections = self.generate_combinations(nb_queries=nb_queries, nb_comb=nb_comb)
            else:
                # Generate nb_comb combinations of size `nb_queries` from integers 0 to space_size-1
                selected_selections = self.new_generate_combinations(nb_queries=nb_queries, nb_comb=nb_comb)

            # Generate a random vector of size `nb_comb` with integers from 0 to `nb_queries-1`
            random_idx = np.random.randint(0, nb_queries, size=nb_comb)

            idx_inputs = []
            for i in range(nb_comb):
                q = selected_selections[i].pop(random_idx[i])
                idx_inputs.append([selected_selections[i], q])

        else:
            # Generate nb_comb selections of size `nb_queries` from integers 0 to space_size-1
            selected_selections = self.generate_selections(nb_queries=nb_queries, nb_comb=nb_comb)

            idx_inputs = []
            for i in range(len(selected_selections)):
                q = selected_selections[i].pop()
                idx_inputs.append([selected_selections[i], q])
        
        return(idx_inputs)
    
    def get_inputs_for_GPs(self, train_X_idx: list, query_x_idx: int, method: str = 'same_std') -> tuple:
        """
        Prepare inputs and labels for Gaussian Process training.

        This function standardizes inputs and labels based on a selected approach 
        to ensure consistency in data representation for Gaussian Processes.

        We have several options: 
            1- Standardize Y_n and Y_{n+1} independently and transform y_{n+1} using the mean and standard 
               deviation of Y_{n+1}.
            2- Standardize Y_n, and transform both Y_n and y_{n+1} using the mean and standard deviation 
               of Y_n.
            3- Standardize Y_{n+1}, and transform both Y_n and y_{n+1} using the mean and standard deviation 
               of Y_{n+1}.

        Option 1 is quite convincing because the model is intended to work online. With this method, 
        the label at iteration n is the input for iteration n+1, which is desirable.

        However, option 1 also has the disadvantage of modifying the first n queries when transitioning 
        from Y_n to Y_{n+1}, since Y_n and Y_{n+1} are standardized differently. 
        In other words, train_label != train_input.append(query_y).

        Therefore, we choose option 3 for the method 'same_std', which certainly has the disadvantage that the label at iteration n 
        will be different from the input at iteration n+1 (as they are standardized differently), 
        but it has the advantage that train_label = train_input.append(query_y), which we consider very 
        beneficial for training our model.

        Args:
            train_X_idx (list): List of indices for training input.
            query_x_idx (int): Index of the query point.
            method (str, optional): choose one of two methods presented before (1 & 3). Default is 'same_std' (method 3)

        Returns:
            tuple: Contains training inputs, training labels, query data, and full labeled data.
        """
        if method == 'same_std':

            train_X_label = self.X_test_normed[train_X_idx + [query_x_idx]]
            train_Y_label = self.map_idx[train_X_idx + [query_x_idx]].reshape(-1,1)
            train_Y_label = standardize_vector(train_Y_label)

            train_X_input = self.X_test_normed[train_X_idx]
            train_Y_input = train_Y_label[:-1]

            query_x = self.X_test_normed[query_x_idx]
            query_y = train_Y_label[-1,0]

            return train_X_input, train_Y_input, query_x_idx, query_x, query_y, train_X_label, train_Y_label
        
        elif method == 'independent_std':

            train_X_label = self.X_test_normed[train_X_idx + [query_x_idx]]
            train_Y_label = self.map_idx[train_X_idx + [query_x_idx]].reshape(-1,1)
            train_Y_label = standardize_vector(train_Y_label)

            train_X_input = self.X_test_normed[train_X_idx]
            train_Y_input = self.map_idx[train_X_idx].reshape(-1,1)
            train_Y_input = standardize_vector(train_Y_input)

            query_x = self.X_test_normed[query_x_idx]
            query_y = train_Y_label[-1,0]

            return train_X_input, train_Y_input, query_x_idx, query_x, query_y, train_X_label, train_Y_label


    def generate_pre_labeled_inputs(self, name: str, nb_queries: Union[int, List[int]], nb_comb: Union[int, List[int]] = None, 
                                    kernel_type: str = 'RBF', noise=0.1, 
                                    outputscale=1, lengthscale=0.05, 
                                    combination: bool = False, combination_without_rep: bool = False,
                                    using_GPytorch: bool = False, method_std: str = 'same_std') -> None:
        """Generate pre-labeled data for Gaussian Processes and save it.

        Args:
            name (str): The name used to save the dataset.
            nb_queries (int or list(int)): The number of queries per combination.
            nb_comb (int or list(int), optional): The number of combinations to generate. Defaults to None.
            kernel_type (str): The kernel type for the Gaussian Process. Defaults to 'RBF'.
            noise (float): The noise variance. Defaults to 0.1.
            outputscale (float): The output variance. Defaults to 1.
            lengthscale (float): The kernel lengthscale. Defaults to 0.05.
            combination (bool, optional): If False, we use sampling with replacement. If True, we use sampling without replacement. 
                                          Defaults to False.  
            combination_without_rep (bool, optional): If False, combinations could be repetitive in the dataset. 
                                                      If True, each combination is unique. Defaults to False.
            using_GPytorch (bool, optional): If True, we use GPytorchFixed. If False, we use FixedGP. Defaults to False.
            method_std (str, optional): choose one of two methods presented before in get_inputs_for_GPs (1 & 3). Default is 'same_std' (method 3)
        """

        if (not combination) and (combination_without_rep):
            raise ValueError("combination_without_rep cannot be True if combination is False")
        # Generate indexed inputs for GP training.
        idx_inputs = self.generate_idx_inputs(nb_queries=nb_queries, nb_comb=nb_comb, 
                                              combination=combination, combination_without_rep=combination_without_rep)
        pre_labeled_inputs = []

        if using_GPytorch:
            for i in range(len(idx_inputs)):
                # Prepare inputs and labels for GP models.
                train_X_input, train_Y_input, query_x_idx, query_x, query_y, train_X_label, train_Y_label = self.get_inputs_for_GPs(
                                                            train_X_idx=idx_inputs[i][0],query_x_idx=idx_inputs[i][1], method=method_std)
                
                # Initialize GP models for inputs and labels.
                likelihood_input = gpytorch.likelihoods.GaussianLikelihood()
                gp_input = GPytorchFixed( 
                                train_x=torch.from_numpy(train_X_input), train_y=torch.from_numpy(train_Y_input[:,0]), 
                                likelihood=likelihood_input, kernel_type=kernel_type, noise=noise, 
                                outputscale=outputscale, lengthscale=lengthscale)
                likelihood_label = gpytorch.likelihoods.GaussianLikelihood()
                gp_label = GPytorchFixed( 
                                train_x=torch.from_numpy(train_X_label), train_y=torch.from_numpy(train_Y_label[:,0]), 
                                likelihood=likelihood_label, kernel_type=kernel_type, noise=noise, 
                                outputscale=outputscale, lengthscale=lengthscale)
                
                # Generate predictions for inputs and labels.
                mean_input, std_input = gp_input.predict(self.X_test_normed_tensor)
                mean_label, std_label = gp_label.predict(self.X_test_normed_tensor)

                # Transform vector predictions to maps.
                map_mean_input = self.vec_to_map(mean_input.detach().numpy())
                map_std_input = self.vec_to_map(std_input.detach().numpy())
                map_mean_label = self.vec_to_map(mean_label.detach().numpy())
                map_std_label = self.vec_to_map(std_label.detach().numpy())

                # Save processed data.
                pre_labeled_inputs.append([map_mean_input, map_std_input, 
                                    query_x_idx, query_x, query_y, 
                                    map_mean_label, map_std_label])
        else:
            for i in range(len(idx_inputs)):
                # Prepare inputs and labels for GP models.
                train_X_input, train_Y_input, query_x_idx, query_x, query_y, train_X_label, train_Y_label = self.get_inputs_for_GPs(
                                                            train_X_idx=idx_inputs[i][0],query_x_idx=idx_inputs[i][1], method=method_std)
                
                # Initialize GP models for inputs and labels.
                gp_input = FixedGP(input_space=self.X_test_normed, 
                                train_X=train_X_input, train_Y=train_Y_input,
                                kernel_type=kernel_type, noise_std=np.sqrt(noise), 
                                output_std=np.sqrt(outputscale), lengthscale=lengthscale)
                gp_label = FixedGP(input_space=self.X_test_normed, 
                                train_X=train_X_label, train_Y=train_Y_label, 
                                kernel_type=kernel_type, noise_std=np.sqrt(noise), 
                                output_std=np.sqrt(outputscale), lengthscale=lengthscale)
                
                # Generate predictions for inputs and labels.
                mean_input, std_input = gp_input.predict()
                mean_label, std_label = gp_label.predict()

                # Transform vector predictions to maps.
                map_mean_input = self.vec_to_map(mean_input)
                map_std_input = self.vec_to_map(std_input)
                map_mean_label = self.vec_to_map(mean_label)
                map_std_label = self.vec_to_map(std_label)

                # Save processed data.
                pre_labeled_inputs.append([map_mean_input, map_std_input, 
                                    query_x_idx, query_x, query_y, 
                                    map_mean_label, map_std_label])
       

        if combination:
            if combination_without_rep:
                generation_method = "we have used sampling without replacement to choose the queries. And all the samples are uniques"
                self.LETTER = 'A_'
            else:
                generation_method = "we have used sampling without replacement to choose the queries. And all the samples are NOT uniques"
                self.LETTER = 'B_'
        else:
            generation_method = "we have used sampling with replacement to choose the queries. And all the samples are NOT uniques"
            if using_GPytorch:
                self.LETTER = 'C4_'
            else:
                self.LETTER = 'C3_'

        if method_std == 'independent_std':
            self.LETTER = 'i'+self.LETTER

        # Add metadata and save the dataset.
        pre_labeled_inputs.append((nb_queries, nb_comb, kernel_type, noise, outputscale, lengthscale, using_GPytorch, generation_method))  
        pre_labeled_inputs.append("These pre-labeled inputs were made with the following method : " + generation_method + f" \nand these parameters:\n name: {self.LETTER+name}\n nb_queries: {nb_queries}\n nb_comb: {nb_comb}\n kernel_type: {kernel_type}\n noise: {noise}\n outputscale: {outputscale}\n lengthscale: {lengthscale}")       
        
        if combination:
            if combination_without_rep:
                self.pkl_save(L=pre_labeled_inputs, name=self.LETTER+name)
            else:
                self.pkl_save(L=pre_labeled_inputs, name=self.LETTER+name)
        else:
            self.pkl_save(L=pre_labeled_inputs, name=self.LETTER+name)
        

    def format_labeled_inputs(self, name: str):
        """Format pre-labeled inputs for training and save them as tensors.

        Args:
            name (str): The name of the pre-labeled dataset.
        """
        # Load pre-labeled inputs.
        pre_labeled_inputs = self.pkl_load(self.LETTER+name)
        print(pre_labeled_inputs[-1])

        # Extract metadata and initialize tensors for formatted data.
        nb_queries, nb_comb, kernel_type, noise_std, output_std, lengthscale, using_GPytorch, generation_method = pre_labeled_inputs[-2]
        self.get_elem_maps(kernel_type, noise_std, output_std, lengthscale, using_GPytorch)

        if type(nb_comb) == int:
            nb_comb_tot = nb_comb
        else:
            nb_comb_tot = sum(nb_comb)

        train_input = torch.full((nb_comb_tot, 4, self.map.shape[0], self.map.shape[1]), torch.nan)
        train_label = torch.full((nb_comb_tot, 2, self.map.shape[0], self.map.shape[1]), torch.nan)

        for i in range(nb_comb_tot):
            # Convert maps to images for training inputs and labels.
            img_mean_input = self.map_to_img(pre_labeled_inputs[i][0])
            img_std_input = self.map_to_img(pre_labeled_inputs[i][1])
            img_query_x = self.map_to_img(self.maps_elem_std[pre_labeled_inputs[i][2]])
            img_query_y = self.map_to_img(self.maps_elem_mean[pre_labeled_inputs[i][2]]) * pre_labeled_inputs[i][4]

            train_input[i] = torch.cat((img_mean_input, img_std_input, img_query_x, img_query_y), dim=1)

            img_mean_label = self.map_to_img(pre_labeled_inputs[i][5])
            img_std_label = self.map_to_img(pre_labeled_inputs[i][6])

            train_label[i] = torch.cat((img_mean_label, img_std_label), dim=1)
        
        # Save formatted data with metadata.
        data_to_save = {
            "train_input": train_input,
            "train_label": train_label,
            "description": f"These labeled inputs have the original form.\nThese pre-labeled inputs were made with the following method : " + generation_method + f" \nand these parameters:\n name: {name}\n nb_queries: {nb_queries}\n nb_comb: {nb_comb}\n kernel_type: {kernel_type}\n noise_std: {noise_std}\n output_std: {output_std}\n lengthscale: {lengthscale}"
        }

        torch.save(data_to_save, 'dataset/single_map/og_'+self.LETTER+name+'.pth')


        