import numpy as np

class QueriesInfo:
    def __init__(self, space_shape):
        self.space_shape = space_shape
        self.query_map = np.full(space_shape, 0)
        self.mean_map = np.full(space_shape, np.nan)
        self.var_map = np.full(space_shape, np.nan)
        
    def update_map(self, query_x, query_y):
        self.query_map[query_x] += 1
        if self.query_map[query_x] == 1:
            self.mean_map[query_x] = query_y
        elif self.query_map[query_x] == 2:
            new_mean_map = (query_y + self.mean_map[query_x]) / 2
            self.var_map[query_x] = ((query_y-new_mean_map)**2 + (self.mean_map[query_x]-new_mean_map)**2) / 2
            self.mean_map[query_x] = new_mean_map
        else:
            new_mean_map = ((self.query_map[query_x]-1)*self.mean_map[query_x] + query_y) / self.query_map[query_x]
            self.var_map[query_x] = ((self.query_map[query_x]-1) * (self.var_map[query_x] + (new_mean_map-self.mean_map[query_x])**2) + (query_y-self.mean_map[query_x])**2) / self.query_map[query_x]
            self.mean_map[query_x] = new_mean_map

    def idx2coord(self, idx):
        coord = []
        for i in range(len(idx)):
            coord.append(idx[i]/self.space_shape[i])
        return tuple(coord)
    
    def coord2idx(self, coord):
        idx = []
        for i in range(len(coord)):
            idx.append(round(coord[i]*self.space_shape[i]))
        return tuple(idx)
    
    def get_mean_queries(self):
        mean_queries_list = []
        for idx in np.ndindex(self.space_shape):
            if not np.isnan(self.mean_map[idx]):
                mean_queries_list.append([self.idx2coord(idx), self.mean_map[idx]])
        return mean_queries_list
