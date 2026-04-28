# --- File: python/ion_flux/compiler/coloring.py ---
from typing import Set, Tuple, List, Dict
from collections import defaultdict

class HybridGraphColorer:
    """
    Applies Curtis-Powell-Reid (CPR) column intersection graph coloring to 
    minimize Forward-Mode AD JVP sweeps. Safely isolates dense global states 
    (Arrowhead matrices) into a distinct Reverse-Mode VJP pathway to prevent 
    O(N) scaling collapse.
    """
    def __init__(self, n_states: int, triplets: Set[Tuple[int, int]], dense_threshold: int = 20):
        self.n_states = n_states
        self.triplets = triplets
        self.dense_threshold = dense_threshold
        
        self.dense_rows: List[int] = []
        self.sparse_triplets: Set[Tuple[int, int]] = set()
        
        self.n_colors: int = 0
        self.color_map: Dict[int, int] = {}
        self.color_seeds: List[List[float]] = []

        self._segregate()
        self._color_graph()
        self._generate_seeds()

    def _segregate(self):
        """Identifies globally coupled equations (e.g. 0D Algebraic constraints)."""
        row_counts = defaultdict(int)
        for r, c in self.triplets:
            row_counts[r] += 1
            
        for r, count in row_counts.items():
            if count > self.dense_threshold:
                self.dense_rows.append(r)
                
        dense_rows_set = set(self.dense_rows)
        for r, c in self.triplets:
            if r not in dense_rows_set:
                self.sparse_triplets.add((r, c))

    def _color_graph(self):
        """
        Builds the Column Intersection Graph for the sparse bulk and applies 
        Welsh-Powell greedy coloring. If two columns have non-zeros in the same 
        row, they collide and cannot share a color.
        """
        col_in_sparse_row = defaultdict(list)
        for r, c in self.sparse_triplets:
            col_in_sparse_row[r].append(c)
            
        adj = defaultdict(set)
        for r, cols in col_in_sparse_row.items():
            # Create cliques for all columns sharing this row
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    u, v = cols[i], cols[j]
                    adj[u].add(v)
                    adj[v].add(u)
                    
        # Welsh-Powell heuristic: Color nodes with the highest degree first
        sorted_nodes = sorted(range(self.n_states), key=lambda x: len(adj[x]), reverse=True)
        
        for node in sorted_nodes:
            # Isolated columns or columns only existing in dense rows safely take color 0
            if len(adj[node]) == 0:
                self.color_map[node] = 0
                continue
                
            neighbor_colors = {self.color_map[neighbor] for neighbor in adj[node] if neighbor in self.color_map}
            
            color = 0
            while color in neighbor_colors:
                color += 1
                
            self.color_map[node] = color
            self.n_colors = max(self.n_colors, color + 1)
            
        if self.n_colors == 0 and self.n_states > 0:
            self.n_colors = 1
            for node in range(self.n_states):
                self.color_map[node] = 0

    def _generate_seeds(self):
        """
        Emits the binary perturbation vectors to be executed by the compiled 
        Forward-Mode AD (JVP) function.
        """
        self.color_seeds = [[0.0] * self.n_states for _ in range(self.n_colors)]
        for col, color in self.color_map.items():
            self.color_seeds[color][col] = 1.0