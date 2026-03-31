import numpy as np
from typing import Dict, Any, List, Optional

class Variable:
    """Wrapper mapping flat FFI arrays back into intuitive multidimensional structures."""
    __slots__ = ["data", "result", "name"]
    def __init__(self, data: np.ndarray, result: Optional[Any] = None, name: str = ""): 
        self.data = data
        self.result = result
        self.name = name
    def __repr__(self) -> str: return f"<Variable: {self.name} shape={self.data.shape}>"



class SimulationResult:
    __slots__ = ["_data", "parameters", "status", "engine", "trajectory"]
    def __init__(self, data: Dict[str, np.ndarray], parameters: Dict[str, float], status: str = "completed", engine: Optional[Any] = None, trajectory: Optional[Dict] = None):
        self._data = data
        self.parameters = parameters
        self.status = status
        self.engine = engine
        self.trajectory = trajectory

    def __getitem__(self, key: str) -> Variable:
        if key not in self._data: raise KeyError(f"Variable '{key}' not found.")
        return Variable(self._data[key], result=self, name=key)
        
    def to_dict(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        keys = variables or self._data.keys()
        return {k: self._data[k].tolist() for k in keys if k in self._data}

    def plot_dashboard(self):
        """Generates an interactive 2x4 Matplotlib dashboard to visualize full-cell internal states over time."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        import numpy as np

        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        
        time = self["Time [s]"].data / 3600.0
        
        # Helper to dynamically slice internal arrays
        def get_surf(state_name, r_res):
            data = self[state_name].data
            return data[:, r_res-1::r_res]

        # Explicitly stitch the isolated spatial domains together
        c_e = np.concatenate([self["c_e_n"].data, self["c_e_s"].data, self["c_e_p"].data], axis=1)
        phi_e = np.concatenate([self["phi_e_n"].data, self["phi_e_s"].data, self["phi_e_p"].data], axis=1)
        
        # Assume 10 radial nodes from the DFN model defaults
        c_s_n_surf = get_surf("c_s_n", 10)
        c_s_p_surf = get_surf("c_s_p", 10)
        
        # Coordinates
        x_n_len, x_s_len, x_p_len = 20, 10, 20
        x_ce = np.linspace(0, 100, x_n_len + x_s_len + x_p_len)
        x_n = np.linspace(0, 40, x_n_len)
        x_p = np.linspace(60, 100, x_p_len)
        
        lines = []
        
        # --- Top Row ---
        axs[0,0].set_title("Negative particle surface concentration")
        l1, = axs[0,0].plot(x_n, c_s_n_surf[0], 'r-')
        lines.append((l1, c_s_n_surf))
        
        axs[0,1].set_title("Electrolyte concentration")
        l2, = axs[0,1].plot(x_ce, c_e[0], 'r-')
        axs[0,1].axvline(40, color='gray', linestyle='-')
        axs[0,1].axvline(60, color='gray', linestyle='-')
        lines.append((l2, c_e))
        
        axs[0,2].set_title("Positive particle surface concentration")
        l3, = axs[0,2].plot(x_p, c_s_p_surf[0], 'r-')
        lines.append((l3, c_s_p_surf))
        
        axs[0,3].set_title("Current [A]")
        l4, = axs[0,3].plot(time, self["i_app"].data, 'r-')
        t_line1 = axs[0,3].axvline(time[0], color='k', linestyle='--')
        
        # --- Bottom Row ---
        axs[1,0].set_title("Negative electrode potential [V]")
        l5, = axs[1,0].plot(x_n, self["phi_s_n"].data[0], 'r-')
        lines.append((l5, self["phi_s_n"].data))
        
        axs[1,1].set_title("Electrolyte potential [V]")
        l6, = axs[1,1].plot(x_ce, phi_e[0], 'r-')
        axs[1,1].axvline(40, color='gray', linestyle='-')
        axs[1,1].axvline(60, color='gray', linestyle='-')
        lines.append((l6, phi_e))
        
        axs[1,2].set_title("Positive electrode potential [V]")
        l7, = axs[1,2].plot(x_p, self["phi_s_p"].data[0], 'r-')
        lines.append((l7, self["phi_s_p"].data))
        
        axs[1,3].set_title("Voltage [V]")
        l8, = axs[1,3].plot(time, self["V_cell"].data, 'r-')
        t_line2 = axs[1,3].axvline(time[0], color='k', linestyle='--')
        
        for ax in axs.flat:
            ax.relim()
            ax.autoscale_view()
            
        plt.subplots_adjust(bottom=0.2, wspace=0.3, hspace=0.3)
        ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
        slider = Slider(ax_slider, 'Time [h]', 0, len(time)-1, valinit=0, valfmt='%0.0f')
        
        def update(val):
            idx = int(slider.val)
            for line, data in lines:
                line.set_ydata(data[idx])
            t_line1.set_xdata([time[idx]])
            t_line2.set_xdata([time[idx]])
            
            for ax in axs.flat:
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw_idle()
            
        slider.on_changed(update)
        plt.show()

    def plot_dashboard(self, variables: Optional[List[Any]] = None):
        """
        Generates an interactive Matplotlib dashboard to visualize simulated states.
        Dynamically adapts to the provided variables and handles both 0D and 1D spatial data.

        Args:
            variables: List of variable names or lists of variable names (for grouping on the same axes).
                       If None, attempts to use the model's `default_quick_plot_variables` or falls back 
                       to automatically selecting available outputs.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        import numpy as np

        # 1. Resolve which variables to plot
        if variables is None:
            if hasattr(self.engine.model, "default_quick_plot_variables") and self.engine.model.default_quick_plot_variables:
                variables = self.engine.model.default_quick_plot_variables
            else:
                # Fallback: Auto-select up to 6 interesting states (ignoring time)
                variables = [k for k in self._data.keys() if "Time" not in k][:6]

        if not variables:
            print("No variables available to plot.")
            return

        time = self["Time [s]"].data
        
        # 2. Layout logic
        n_plots = len(variables)
        n_cols = min(3, n_plots)
        n_rows = (n_plots - 1) // n_cols + 1
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)
        axs = axs.flatten()
        
        # Tracking references for the interactive slider updates
        lines_spatial = []
        lines_time = []
        
        # Styling cycles for groups
        styles = ['-', ':', '--', '-.']

        # 3. Build subplots
        for i, var_group in enumerate(variables):
            ax = axs[i]
            
            # Standardize format to handle both single strings and grouped lists
            if isinstance(var_group, str):
                var_group = [var_group]
                
            has_0d = False
            
            for j, var_name in enumerate(var_group):
                if var_name not in self._data:
                    continue
                
                data = self[var_name].data
                style = styles[j % len(styles)]
                
                if data.ndim == 1:
                    # 0D time-series data
                    ax.plot(time, data, label=var_name, color='r', linestyle=style)
                    has_0d = True
                elif data.ndim == 2:
                    # 1D spatial data
                    # Assuming a normalized domain 0..1 if exact physical bounds are abstracted away
                    x_axis = np.linspace(0, 1, data.shape[1])
                    line, = ax.plot(x_axis, data[0], label=var_name, color='r', linestyle=style)
                    lines_spatial.append((line, data))
                    ax.set_xlabel("Normalized Space")
            
            # Format aesthetics
            title = var_group[0] if len(var_group) == 1 else ", ".join(var_group)
            ax.set_title(title[:40] + "..." if len(title) > 40 else title)
            
            if has_0d:
                ax.set_xlim(time[0], time[-1])
                ax.set_xlabel("Time [s]")
                t_line = ax.axvline(time[0], color='k', linestyle='--')
                lines_time.append((ax, t_line))
                
            if len(var_group) > 1:
                ax.legend(loc='best', fontsize=9)
                
        # Hide any unused grid cells
        for i in range(len(variables), len(axs)):
            axs[i].set_visible(False)
            
        plt.subplots_adjust(bottom=0.2, wspace=0.3, hspace=0.4)
        
        # 4. Interactive Slider Hook
        ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
        slider = Slider(ax_slider, 'Time [s]', 0, len(time)-1, valinit=0, valfmt='%0.0f')
        
        def update(val):
            idx = int(slider.val)
            for line, data in lines_spatial:
                line.set_ydata(data[idx])
            for ax, t_line in lines_time:
                t_line.set_xdata([time[idx], time[idx]])
            for ax in axs:
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw_idle()
            
        slider.on_changed(update)
        plt.show()


