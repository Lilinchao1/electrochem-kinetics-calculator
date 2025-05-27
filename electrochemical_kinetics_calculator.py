import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from itertools import product
import time
from matplotlib.ticker import FuncFormatter

# Constants (all in eV units)
kB_eV = 8.617333e-5  # Boltzmann constant (eV/K)
h_eV = 4.135667e-15  # Planck constant (eV·s)
F = 96485           # Faraday constant (C/mol)
R = 8.314           # Gas constant (J/(mol·K))
eV_to_J = 1.602176634e-19  # 1 eV = 1.602e-19 J

class ElectrochemicalKineticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Electrochemical Kinetics Calculator")
        self.root.geometry("1400x900")
        
        # Variables
        self.mechanism_var = tk.StringVar(value="VH")
        self.model_var = tk.StringVar(value="BV")
        self.cmap_var = tk.StringVar(value="viridis")
        self.plot_type_var = tk.StringVar(value="3D")
        self.cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                     'coolwarm', 'rainbow', 'jet', 'turbo', 'seismic']
        
        # Parameters
        self.variable_params = {
            "η": {"min": tk.DoubleVar(value=-1.0), "max": tk.DoubleVar(value=1.0),
                  "step": tk.DoubleVar(value=0.01), "is_var": tk.BooleanVar(value=False),
                  "fixed_value": tk.DoubleVar(value=0.0)},
            "pH": {"min": tk.DoubleVar(value=-2), "max": tk.DoubleVar(value=2),
                   "step": tk.DoubleVar(value=0.1), "is_var": tk.BooleanVar(value=False),
                   "fixed_value": tk.DoubleVar(value=0.0)}
        }
        
        self.fixed_params = {}
        self.results = None
        self.current_plot_page = 0
        self.plots_per_page = 5
        self.plot_windows = []
        
        self.create_widgets()

    def create_widgets(self):
        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel (Parameters and Results)
        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        # Mechanism and Model Selection Frame
        selection_frame = ttk.LabelFrame(left_panel, text="Mechanism and Model Selection")
        selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Mechanism Selection
        ttk.Label(selection_frame, text="Mechanism:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(selection_frame, text="Volmer-Heyrovsky (VH)", variable=self.mechanism_var,
                       value="VH", command=self.update_parameter_ui).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(selection_frame, text="Volmer-Tafel (VT)", variable=self.mechanism_var,
                       value="VT", command=self.update_parameter_ui).grid(row=0, column=2, sticky="w")
        
        # Model Selection
        ttk.Label(selection_frame, text="Model:").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(selection_frame, text="Butler-Volmer (BV)", variable=self.model_var,
                       value="BV", command=self.update_parameter_ui).grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(selection_frame, text="Marcus (M)", variable=self.model_var,
                       value="M", command=self.update_parameter_ui).grid(row=1, column=2, sticky="w")
        ttk.Radiobutton(selection_frame, text="Marcus-Gerischer (MG)", variable=self.model_var,
                       value="MG", command=self.update_parameter_ui).grid(row=1, column=3, sticky="w")
        
        # Temperature
        ttk.Label(selection_frame, text="Temperature (K):").grid(row=2, column=0, sticky="w")
        self.T = tk.DoubleVar(value=298.15)
        ttk.Entry(selection_frame, textvariable=self.T, width=10).grid(row=2, column=1, sticky="w")
        
        # Colormap Selection
        ttk.Label(selection_frame, text="Colormap:").grid(row=2, column=2, sticky="w")
        cmap_menu = ttk.OptionMenu(selection_frame, self.cmap_var, self.cmap_var.get(), *self.cmaps)
        cmap_menu.grid(row=2, column=3, sticky="w")
        
        # Plot Type Selection
        ttk.Label(selection_frame, text="Plot Type:").grid(row=3, column=0, sticky="w")
        ttk.Radiobutton(selection_frame, text="2D", variable=self.plot_type_var,
                       value="2D").grid(row=3, column=1, sticky="w")
        ttk.Radiobutton(selection_frame, text="3D", variable=self.plot_type_var,
                       value="3D").grid(row=3, column=2, sticky="w")
        
        # Parameter Frames Container
        self.param_container = ttk.Frame(left_panel)
        self.param_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create empty parameter frames
        self.volmer_frame = ttk.LabelFrame(self.param_container, text="Volmer Parameters")
        self.other_frame = ttk.LabelFrame(self.param_container, text="")
        self.variable_frame = ttk.LabelFrame(self.param_container, text="Variable Parameters")
        
        self.update_parameter_ui()
        
        # Button Frame
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(button_frame, text="Calculate", command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # Progress Bar
        self.progress = ttk.Progressbar(left_panel, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Results Frame with Scrollbar
        results_container = ttk.Frame(left_panel)
        results_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_canvas = tk.Canvas(results_container)
        self.results_scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=self.results_canvas.yview)
        self.results_scrollbar_x = ttk.Scrollbar(results_container, orient="horizontal", command=self.results_canvas.xview)
        
        self.results_frame = ttk.Frame(self.results_canvas)
        self.results_frame.bind("<Configure>", lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all")))
        
        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set, xscrollcommand=self.results_scrollbar_x.set)
        
        self.results_canvas.grid(row=0, column=0, sticky="nsew")
        self.results_scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        results_container.grid_rowconfigure(0, weight=1)
        results_container.grid_columnconfigure(0, weight=1)
        
        # Right Panel (Graphs)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Graph Frame with Scrollbar
        graph_container = ttk.Frame(right_panel)
        graph_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.graph_canvas = tk.Canvas(graph_container)
        self.graph_scrollbar = ttk.Scrollbar(graph_container, orient="vertical", command=self.graph_canvas.yview)
        self.graph_scrollbar_x = ttk.Scrollbar(graph_container, orient="horizontal", command=self.graph_canvas.xview)
        
        self.graph_frame = ttk.Frame(self.graph_canvas)
        self.graph_frame.bind("<Configure>", lambda e: self.graph_canvas.configure(scrollregion=self.graph_canvas.bbox("all")))
        
        self.graph_canvas.create_window((0, 0), window=self.graph_frame, anchor="nw")
        self.graph_canvas.configure(yscrollcommand=self.graph_scrollbar.set, xscrollcommand=self.graph_scrollbar_x.set)
        
        self.graph_canvas.grid(row=0, column=0, sticky="nsew")
        self.graph_scrollbar.grid(row=0, column=1, sticky="ns")
        self.graph_scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        graph_container.grid_rowconfigure(0, weight=1)
        graph_container.grid_columnconfigure(0, weight=1)
        
        # Configure grid weights
        main_frame.pack_propagate(False)

    def update_parameter_ui(self):
        """Update the parameter input UI based on mechanism and model selection"""
        for widget in self.param_container.winfo_children():
            widget.destroy()
        
        mechanism = self.mechanism_var.get()
        model = self.model_var.get()
        
        # Recreate the parameter frames
        self.volmer_frame = ttk.LabelFrame(self.param_container, text="Volmer Parameters")
        self.volmer_frame.pack(fill=tk.X, padx=5, pady=5)
        
        if mechanism == "VH":
            self.other_frame = ttk.LabelFrame(self.param_container, text="Heyrovsky Parameters")
        else:
            self.other_frame = ttk.LabelFrame(self.param_container, text="Tafel Parameters")
        self.other_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.variable_frame = ttk.LabelFrame(self.param_container, text="Variable Parameters")
        self.variable_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add parameters based on model and mechanism
        if model == "BV":
            self.create_bv_parameters(mechanism)
        elif model == "M":
            self.create_marcus_parameters(mechanism)
        elif model == "MG":
            self.create_mg_parameters(mechanism)
        
        # Add variable parameters (η and pH)
        self.create_variable_parameters()

    def create_bv_parameters(self, mechanism):
        """Create Butler-Volmer parameters"""
        # Volmer parameters (common to both mechanisms)
        ttk.Label(self.volmer_frame, text="ΔGV (eV):").grid(row=0, column=0, sticky="w")
        self.fixed_params["ΔGV"] = tk.DoubleVar(value=0.5)
        ttk.Entry(self.volmer_frame, textvariable=self.fixed_params["ΔGV"], width=10).grid(row=0, column=1, sticky="w")
        
        ttk.Label(self.volmer_frame, text="γ_V:").grid(row=1, column=0, sticky="w")
        self.fixed_params["γ_V"] = tk.DoubleVar(value=0.5)
        ttk.Entry(self.volmer_frame, textvariable=self.fixed_params["γ_V"], width=10).grid(row=1, column=1, sticky="w")
        
        ttk.Label(self.volmer_frame, text="β_V:").grid(row=2, column=0, sticky="w")
        self.fixed_params["β_V"] = tk.DoubleVar(value=0.5)
        ttk.Entry(self.volmer_frame, textvariable=self.fixed_params["β_V"], width=10).grid(row=2, column=1, sticky="w")
        
        ttk.Label(self.volmer_frame, text="Ea0_V (eV):").grid(row=3, column=0, sticky="w")
        self.fixed_params["Ea0_V"] = tk.DoubleVar(value=0.5)
        ttk.Entry(self.volmer_frame, textvariable=self.fixed_params["Ea0_V"], width=10).grid(row=3, column=1, sticky="w")
        
        # Other step parameters
        if mechanism == "VH":
            # Heyrovsky parameters
            ttk.Label(self.other_frame, text="ΔGH (eV):").grid(row=0, column=0, sticky="w")
            self.fixed_params["ΔGH"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["ΔGH"], width=10).grid(row=0, column=1, sticky="w")
            
            ttk.Label(self.other_frame, text="γ_H:").grid(row=1, column=0, sticky="w")
            self.fixed_params["γ_H"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["γ_H"], width=10).grid(row=1, column=1, sticky="w")
            
            ttk.Label(self.other_frame, text="β_H:").grid(row=2, column=0, sticky="w")
            self.fixed_params["β_H"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["β_H"], width=10).grid(row=2, column=1, sticky="w")
            
            ttk.Label(self.other_frame, text="Ea0_H (eV):").grid(row=3, column=0, sticky="w")
            self.fixed_params["Ea0_H"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["Ea0_H"], width=10).grid(row=3, column=1, sticky="w")
        else:
            # Tafel parameters
            ttk.Label(self.other_frame, text="ΔGT (eV):").grid(row=0, column=0, sticky="w")
            self.fixed_params["ΔGT"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["ΔGT"], width=10).grid(row=0, column=1, sticky="w")
            
            ttk.Label(self.other_frame, text="Ea0_T (eV):").grid(row=1, column=0, sticky="w")
            self.fixed_params["Ea0_T"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["Ea0_T"], width=10).grid(row=1, column=1, sticky="w")
            
            ttk.Label(self.other_frame, text="γ_T:").grid(row=2, column=0, sticky="w")
            self.fixed_params["γ_T"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["γ_T"], width=10).grid(row=2, column=1, sticky="w")

    def create_marcus_parameters(self, mechanism):
        """Create Marcus model parameters"""
        # Volmer parameters
        ttk.Label(self.volmer_frame, text="ΔGV (eV):").grid(row=0, column=0, sticky="w")
        self.fixed_params["ΔGV"] = tk.DoubleVar(value=0.5)
        ttk.Entry(self.volmer_frame, textvariable=self.fixed_params["ΔGV"], width=10).grid(row=0, column=1, sticky="w")
        
        ttk.Label(self.volmer_frame, text="λV:").grid(row=1, column=0, sticky="w")
        self.fixed_params["λV"] = tk.DoubleVar(value=2.0)
        ttk.Entry(self.volmer_frame, textvariable=self.fixed_params["λV"], width=10).grid(row=1, column=1, sticky="w")
        
        # Other step parameters
        if mechanism == "VH":
            # Heyrovsky parameters
            ttk.Label(self.other_frame, text="ΔGH (eV):").grid(row=0, column=0, sticky="w")
            self.fixed_params["ΔGH"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["ΔGH"], width=10).grid(row=0, column=1, sticky="w")
            
            ttk.Label(self.other_frame, text="λH:").grid(row=1, column=0, sticky="w")
            self.fixed_params["λH"] = tk.DoubleVar(value=2.0)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["λH"], width=10).grid(row=1, column=1, sticky="w")
        else:
            # Tafel parameters
            ttk.Label(self.other_frame, text="ΔGT (eV):").grid(row=0, column=0, sticky="w")
            self.fixed_params["ΔGT"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["ΔGT"], width=10).grid(row=0, column=1, sticky="w")
            
            ttk.Label(self.other_frame, text="Ea0_T (eV):").grid(row=1, column=0, sticky="w")
            self.fixed_params["Ea0_T"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["Ea0_T"], width=10).grid(row=1, column=1, sticky="w")
            
            ttk.Label(self.other_frame, text="γ_T:").grid(row=2, column=0, sticky="w")
            self.fixed_params["γ_T"] = tk.DoubleVar(value=0.5)
            ttk.Entry(self.other_frame, textvariable=self.fixed_params["γ_T"], width=10).grid(row=2, column=1, sticky="w")

    def create_mg_parameters(self, mechanism):
        """Create Marcus-Gerischer model parameters"""
        self.create_marcus_parameters(mechanism)

    def create_variable_parameters(self):
        """Create UI for variable parameters (η and pH)"""
        # Headers
        headers = ["Parameter", "Is Variable", "Min", "Max", "Step", "Fixed Value"]
        for col, header in enumerate(headers):
            ttk.Label(self.variable_frame, text=header).grid(row=0, column=col, padx=2, pady=2)
        
        # Add rows for η and pH
        for row, param in enumerate(self.variable_params.keys()):
            ttk.Label(self.variable_frame, text=param).grid(row=row+1, column=0, padx=2, pady=2)
            ttk.Checkbutton(self.variable_frame, variable=self.variable_params[param]["is_var"]).grid(row=row+1, column=1, padx=2, pady=2)
            ttk.Entry(self.variable_frame, textvariable=self.variable_params[param]["min"], width=8).grid(row=row+1, column=2, padx=2, pady=2)
            ttk.Entry(self.variable_frame, textvariable=self.variable_params[param]["max"], width=8).grid(row=row+1, column=3, padx=2, pady=2)
            ttk.Entry(self.variable_frame, textvariable=self.variable_params[param]["step"], width=8).grid(row=row+1, column=4, padx=2, pady=2)
            ttk.Entry(self.variable_frame, textvariable=self.variable_params[param]["fixed_value"], width=8).grid(row=row+1, column=5, padx=2, pady=2)

    def calculate(self):
        try:
            # Get parameters
            mechanism = self.mechanism_var.get()
            model = self.model_var.get()
            T = self.T.get()
            f = F / (R * T)  # F/RT (1/V)
            
            # Determine which parameters are variables
            var_params = []
            fixed_params = {}
            
            # Add fixed parameters
            for param, var in self.fixed_params.items():
                fixed_params[param] = var.get()
            
            # Check variable parameters
            for param, data in self.variable_params.items():
                if data["is_var"].get():
                    var_params.append(param)
                else:
                    fixed_params[param] = data["fixed_value"].get()
            
            if not var_params:
                messagebox.showerror("Error", "Please select at least one variable parameter (η or pH)")
                return
            
            # Prepare parameter ranges
            param_ranges = {}
            for param in var_params:
                min_val = self.variable_params[param]["min"].get()
                max_val = self.variable_params[param]["max"].get()
                step = self.variable_params[param]["step"].get()
                param_ranges[param] = np.arange(min_val, max_val + step/2, step)
            
            # Calculate number of iterations
            total_iterations = 1
            for param in var_params:
                total_iterations *= len(param_ranges[param])
            
            # Initialize results dictionary
            results = {param: [] for param in var_params}
            results.update({
                "kVa": [], "k-Va": [], "kVb": [], "k-Vb": [],
                "θ*": [], "θH*": [], "rV": [], "r-V": [], "RV": [], "LOGV": []
            })
            
            if mechanism == "VH":
                results.update({
                    "kHa": [], "k-Ha": [], "kHb": [], "k-Hb": [],
                    "rH": [], "r-H": [], "RH": [], "LOGH": []
                })
            else:
                results.update({
                    "kT": [], "k-T": [], "rT": [], "r-T": [], "RT": [], "LOGT": []
                })
            
            # Initialize progress
            self.progress["maximum"] = total_iterations
            self.progress["value"] = 0
            
            # Optimized calculation based on number of variables
            if len(var_params) == 1:
                # Single variable case - optimized
                param = var_params[0]
                param_values = param_ranges[param]
                n = len(param_values)
                
                # Create arrays for all parameters
                current_params = {k: np.full(n, fixed_params.get(k)) for k in fixed_params.keys()}
                current_params[param] = param_values
                current_params["T"] = np.full(n, T)
                
                # Calculate rate constants
                if model == "BV":
                    k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.butler_volmer_vectorized(current_params, "Volmer")
                    if mechanism == "VH":
                        k_Ha, k_minus_Ha, k_Hb, k_minus_Hb = self.butler_volmer_vectorized(current_params, "Heyrovsky")
                elif model == "M":
                    k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.marcus_vectorized(current_params, "Volmer")
                    if mechanism == "VH":
                        k_Ha, k_minus_Ha, k_Hb, k_minus_Hb = self.marcus_vectorized(current_params, "Heyrovsky")
                elif model == "MG":
                    for i in range(n):
                        single_params = {k: v[i] for k, v in current_params.items()}
                        k_Va_i, k_minus_Va_i, k_Vb_i, k_minus_Vb_i = self.marcus_gerischer(single_params, "Volmer")
                        if i == 0:
                            k_Va = np.zeros(n)
                            k_minus_Va = np.zeros(n)
                            k_Vb = np.zeros(n)
                            k_minus_Vb = np.zeros(n)
                            if mechanism == "VH":
                                k_Ha = np.zeros(n)
                                k_minus_Ha = np.zeros(n)
                                k_Hb = np.zeros(n)
                                k_minus_Hb = np.zeros(n)
                        k_Va[i] = k_Va_i
                        k_minus_Va[i] = k_minus_Va_i
                        k_Vb[i] = k_Vb_i
                        k_minus_Vb[i] = k_minus_Vb_i
                        if mechanism == "VH":
                            k_Ha_i, k_minus_Ha_i, k_Hb_i, k_minus_Hb_i = self.marcus_gerischer(single_params, "Heyrovsky")
                            k_Ha[i] = k_Ha_i
                            k_minus_Ha[i] = k_minus_Ha_i
                            k_Hb[i] = k_Hb_i
                            k_minus_Hb[i] = k_minus_Hb_i
                
                # pH terms
                pH = current_params.get("pH", np.zeros(n))
                k_V = k_Va * (10 ** -pH) + k_Vb
                k_minus_V = k_minus_Va + k_minus_Vb * (10 ** (pH - 14))
                
                if mechanism == "VH":
                    k_H = k_Ha * (10 ** -pH) + k_Hb
                    k_minus_H = k_minus_Ha + k_minus_Hb * (10 ** (pH - 14))
                    
                    # Calculate coverages with improved denominator handling
                    denominator = k_V + k_minus_H + k_minus_V + k_H
                    denominator = np.maximum(denominator, 1e-30)  # Prevent division by zero
                    theta_star = (k_minus_V + k_H) / denominator
                    theta_H = (k_V + k_minus_H) / denominator
                    
                    # Calculate rates
                    r_V = k_V * theta_star
                    r_minus_V = k_minus_V * theta_H
                    R_V = r_minus_V - r_V
                    LOGV = np.log10(np.abs(R_V) + 1e-20)
                    
                    r_H = k_H * theta_H
                    r_minus_H = k_minus_H * theta_star
                    R_H = r_minus_H - r_H
                    LOGH = np.log10(np.abs(R_H) + 1e-20)
                    
                    # Store results
                    results["kHa"] = k_Ha
                    results["k-Ha"] = k_minus_Ha
                    results["kHb"] = k_Hb
                    results["k-Hb"] = k_minus_Hb
                    results["rH"] = r_H
                    results["r-H"] = r_minus_H
                    results["RH"] = R_H
                    results["LOGH"] = LOGH
                else:
                    # VT mechanism calculations
                    Ea0 = current_params.get("Ea0_T", 0.5)
                    γ = current_params.get("γ_T", 0.5)
                    ΔGT = current_params.get("ΔGT", 0.5)
                    k_T = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGT) / (kB_eV * T))
                    k_minus_T = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGT) / (kB_eV * T))
                    
                    a = 2 * (k_minus_T - k_T)
                    b_V = k_V + k_minus_V + 4 * k_minus_T
                    c_V = k_V + 2 * k_minus_T
                    
                    sqrt_term = b_V**2 - 4 * a * c_V
                    sqrt_term[sqrt_term < 0] = 0
                    theta_H = (b_V - np.sqrt(sqrt_term)) / (2 * a)
                    
                    b_H = k_V + k_minus_V + 4 * k_T
                    c_H = k_minus_V + 2 * k_T
                    sqrt_term = b_H**2 + 4 * a * c_H
                    theta_star = (-b_H + np.sqrt(sqrt_term)) / (2 * a)
                    
                    r_V = k_V * theta_star
                    r_minus_V = k_minus_V * theta_H
                    R_V = r_minus_V - r_V
                    LOGV = np.log10(np.abs(R_V) + 1e-20)
                    
                    r_T = 2 * k_T * theta_H**2
                    r_minus_T = 2 * k_minus_T * theta_star**2
                    R_T = r_minus_T - r_T
                    LOGT = np.log10(np.abs(R_T) + 1e-20)
                    
                    results["kT"] = k_T
                    results["k-T"] = k_minus_T
                    results["rT"] = r_T
                    results["r-T"] = r_minus_T
                    results["RT"] = R_T
                    results["LOGT"] = LOGT
                
                # Store common results
                for param_name in var_params:
                    results[param_name] = current_params[param_name]
                results["kVa"] = k_Va
                results["k-Va"] = k_minus_Va
                results["kVb"] = k_Vb
                results["k-Vb"] = k_minus_Vb
                results["θ*"] = theta_star
                results["θH*"] = theta_H
                results["rV"] = r_V
                results["r-V"] = r_minus_V
                results["RV"] = R_V
                results["LOGV"] = LOGV
                
                self.progress["value"] = total_iterations
            
            elif len(var_params) == 2:
                # Two variables case - optimized
                param1, param2 = var_params
                param1_values = param_ranges[param1]
                param2_values = param_ranges[param2]
                n1 = len(param1_values)
                n2 = len(param2_values)
                
                # Create meshgrid for parameters
                param1_mesh, param2_mesh = np.meshgrid(param1_values, param2_values)
                
                # Create arrays for all parameters
                current_params = {k: np.full((n2, n1), fixed_params.get(k)) for k in fixed_params.keys()}
                current_params[param1] = param1_mesh
                current_params[param2] = param2_mesh
                current_params["T"] = np.full((n2, n1), T)
                
                # Calculate rate constants
                if model == "BV":
                    k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.butler_volmer_vectorized(current_params, "Volmer")
                    if mechanism == "VH":
                        k_Ha, k_minus_Ha, k_Hb, k_minus_Hb = self.butler_volmer_vectorized(current_params, "Heyrovsky")
                elif model == "M":
                    k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.marcus_vectorized(current_params, "Volmer")
                    if mechanism == "VH":
                        k_Ha, k_minus_Ha, k_Hb, k_minus_Hb = self.marcus_vectorized(current_params, "Heyrovsky")
                elif model == "MG":
                    for i in range(n2):
                        for j in range(n1):
                            single_params = {k: v[i,j] for k, v in current_params.items()}
                            k_Va_ij, k_minus_Va_ij, k_Vb_ij, k_minus_Vb_ij = self.marcus_gerischer(single_params, "Volmer")
                            if i == 0 and j == 0:
                                k_Va = np.zeros((n2, n1))
                                k_minus_Va = np.zeros((n2, n1))
                                k_Vb = np.zeros((n2, n1))
                                k_minus_Vb = np.zeros((n2, n1))
                                if mechanism == "VH":
                                    k_Ha = np.zeros((n2, n1))
                                    k_minus_Ha = np.zeros((n2, n1))
                                    k_Hb = np.zeros((n2, n1))
                                    k_minus_Hb = np.zeros((n2, n1))
                            k_Va[i,j] = k_Va_ij
                            k_minus_Va[i,j] = k_minus_Va_ij
                            k_Vb[i,j] = k_Vb_ij
                            k_minus_Vb[i,j] = k_minus_Vb_ij
                            if mechanism == "VH":
                                k_Ha_ij, k_minus_Ha_ij, k_Hb_ij, k_minus_Hb_ij = self.marcus_gerischer(single_params, "Heyrovsky")
                                k_Ha[i,j] = k_Ha_ij
                                k_minus_Ha[i,j] = k_minus_Ha_ij
                                k_Hb[i,j] = k_Hb_ij
                                k_minus_Hb[i,j] = k_minus_Hb_ij
                
                # pH terms
                pH = current_params.get("pH", np.zeros((n2, n1)))
                k_V = k_Va * (10 ** -pH) + k_Vb
                k_minus_V = k_minus_Va + k_minus_Vb * (10 ** (pH - 14))
                
                if mechanism == "VH":
                    k_H = k_Ha * (10 ** -pH) + k_Hb
                    k_minus_H = k_minus_Ha + k_minus_Hb * (10 ** (pH - 14))
                    
                    # Calculate coverages with improved denominator handling
                    denominator = k_V + k_minus_H + k_minus_V + k_H
                    denominator = np.maximum(denominator, 1e-30)  # Prevent division by zero
                    theta_star = (k_minus_V + k_H) / denominator
                    theta_H = (k_V + k_minus_H) / denominator
                    
                    # Calculate rates
                    r_V = k_V * theta_star
                    r_minus_V = k_minus_V * theta_H
                    R_V = r_minus_V - r_V
                    LOGV = np.log10(np.abs(R_V) + 1e-20)
                    
                    r_H = k_H * theta_H
                    r_minus_H = k_minus_H * theta_star
                    R_H = r_minus_H - r_H
                    LOGH = np.log10(np.abs(R_H) + 1e-20)
                    
                    # Store results
                    results["kHa"] = k_Ha.flatten()
                    results["k-Ha"] = k_minus_Ha.flatten()
                    results["kHb"] = k_Hb.flatten()
                    results["k-Hb"] = k_minus_Hb.flatten()
                    results["rH"] = r_H.flatten()
                    results["r-H"] = r_minus_H.flatten()
                    results["RH"] = R_H.flatten()
                    results["LOGH"] = LOGH.flatten()
                else:
                    # VT mechanism calculations
                    Ea0 = current_params.get("Ea0_T", 0.5)
                    γ = current_params.get("γ_T", 0.5)
                    ΔGT = current_params.get("ΔGT", 0.5)
                    k_T = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGT) / (kB_eV * T))
                    k_minus_T = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGT) / (kB_eV * T))
                    
                    a = 2 * (k_minus_T - k_T)
                    b_V = k_V + k_minus_V + 4 * k_minus_T
                    c_V = k_V + 2 * k_minus_T
                    
                    sqrt_term = b_V**2 - 4 * a * c_V
                    sqrt_term[sqrt_term < 0] = 0
                    theta_H = (b_V - np.sqrt(sqrt_term)) / (2 * a)
                    
                    b_H = k_V + k_minus_V + 4 * k_T
                    c_H = k_minus_V + 2 * k_T
                    sqrt_term = b_H**2 + 4 * a * c_H
                    theta_star = (-b_H + np.sqrt(sqrt_term)) / (2 * a)
                    
                    r_V = k_V * theta_star
                    r_minus_V = k_minus_V * theta_H
                    R_V = r_minus_V - r_V
                    LOGV = np.log10(np.abs(R_V) + 1e-20)
                    
                    r_T = 2 * k_T * theta_H**2
                    r_minus_T = 2 * k_minus_T * theta_star**2
                    R_T = r_minus_T - r_T
                    LOGT = np.log10(np.abs(R_T) + 1e-20)
                    
                    results["kT"] = k_T.flatten()
                    results["k-T"] = k_minus_T.flatten()
                    results["rT"] = r_T.flatten()
                    results["r-T"] = r_minus_T.flatten()
                    results["RT"] = R_T.flatten()
                    results["LOGT"] = LOGT.flatten()
                
                # Store common results
                for param_name in var_params:
                    results[param_name] = current_params[param_name].flatten()
                results["kVa"] = k_Va.flatten()
                results["k-Va"] = k_minus_Va.flatten()
                results["kVb"] = k_Vb.flatten()
                results["k-Vb"] = k_minus_Vb.flatten()
                results["θ*"] = theta_star.flatten()
                results["θH*"] = theta_H.flatten()
                results["rV"] = r_V.flatten()
                results["r-V"] = r_minus_V.flatten()
                results["RV"] = R_V.flatten()
                results["LOGV"] = LOGV.flatten()
                
                self.progress["value"] = total_iterations
            
            else:
                # More than 2 variables - use original method
                param_combinations = product(*[param_ranges[param] for param in var_params])
                for combination in param_combinations:
                    self.progress["value"] += 1
                    self.root.update_idletasks()
                    
                    current_params = fixed_params.copy()
                    for i, param in enumerate(var_params):
                        current_params[param] = combination[i]
                    current_params["T"] = T
                    
                    for param in var_params:
                        results[param].append(current_params[param])
                    
                    if model == "BV":
                        k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.butler_volmer(current_params, "Volmer")
                        if mechanism == "VH":
                            k_Ha, k_minus_Ha, k_Hb, k_minus_Hb = self.butler_volmer(current_params, "Heyrovsky")
                    elif model == "M":
                        k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.marcus(current_params, "Volmer")
                        if mechanism == "VH":
                            k_Ha, k_minus_Ha, k_Hb, k_minus_Hb = self.marcus(current_params, "Heyrovsky")
                    elif model == "MG":
                        k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.marcus_gerischer(current_params, "Volmer")
                        if mechanism == "VH":
                            k_Ha, k_minus_Ha, k_Hb, k_minus_Hb = self.marcus_gerischer(current_params, "Heyrovsky")
                    
                    pH = current_params.get("pH", 0)
                    k_V = k_Va * (10 ** -pH) + k_Vb
                    k_minus_V = k_minus_Va + k_minus_Vb * (10 ** (pH - 14))
                    
                    if mechanism == "VH":
                        k_H = k_Ha * (10 ** -pH) + k_Hb
                        k_minus_H = k_minus_Ha + k_minus_Hb * (10 ** (pH - 14))
                        
                        denominator = k_V + k_minus_H + k_minus_V + k_H
                        denominator = max(denominator, 1e-30)  # Prevent division by zero
                        theta_star = (k_minus_V + k_H) / denominator
                        theta_H = (k_V + k_minus_H) / denominator
                        
                        r_V = k_V * theta_star
                        r_minus_V = k_minus_V * theta_H
                        R_V = r_minus_V - r_V
                        LOGV = np.log10(np.abs(R_V) + 1e-20)
                        
                        r_H = k_H * theta_H
                        r_minus_H = k_minus_H * theta_star
                        R_H = r_minus_H - r_H
                        LOGH = np.log10(np.abs(R_H) + 1e-20)
                        
                        results["kHa"].append(k_Ha)
                        results["k-Ha"].append(k_minus_Ha)
                        results["kHb"].append(k_Hb)
                        results["k-Hb"].append(k_minus_Hb)
                        results["rH"].append(r_H)
                        results["r-H"].append(r_minus_H)
                        results["RH"].append(R_H)
                        results["LOGH"].append(LOGH)
                    else:
                        Ea0 = current_params.get("Ea0_T", 0.5)
                        γ = current_params.get("γ_T", 0.5)
                        ΔGT = current_params.get("ΔGT", 0.5)
                        k_T = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGT) / (kB_eV * T))
                        k_minus_T = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGT) / (kB_eV * T))
                        
                        a = 2 * (k_minus_T - k_T)
                        b_V = k_V + k_minus_V + 4 * k_minus_T
                        c_V = k_V + 2 * k_minus_T
                        
                        sqrt_term = b_V**2 - 4 * a * c_V
                        sqrt_term = max(sqrt_term, 0)
                        theta_H = (b_V - np.sqrt(sqrt_term)) / (2 * a)
                        
                        b_H = k_V + k_minus_V + 4 * k_T
                        c_H = k_minus_V + 2 * k_T
                        sqrt_term = b_H**2 + 4 * a * c_H
                        theta_star = (-b_H + np.sqrt(sqrt_term)) / (2 * a)
                        
                        r_V = k_V * theta_star
                        r_minus_V = k_minus_V * theta_H
                        R_V = r_minus_V - r_V
                        LOGV = np.log10(np.abs(R_V) + 1e-20)
                        
                        r_T = 2 * k_T * theta_H**2
                        r_minus_T = 2 * k_minus_T * theta_star**2
                        R_T = r_minus_T - r_T
                        LOGT = np.log10(np.abs(R_T) + 1e-20)
                        
                        results["kT"].append(k_T)
                        results["k-T"].append(k_minus_T)
                        results["rT"].append(r_T)
                        results["r-T"].append(r_minus_T)
                        results["RT"].append(R_T)
                        results["LOGT"].append(LOGT)
                    
                    results["kVa"].append(k_Va)
                    results["k-Va"].append(k_minus_Va)
                    results["kVb"].append(k_Vb)
                    results["k-Vb"].append(k_minus_Vb)
                    results["θ*"].append(theta_star)
                    results["θH*"].append(theta_H)
                    results["rV"].append(r_V)
                    results["r-V"].append(r_minus_V)
                    results["RV"].append(R_V)
                    results["LOGV"].append(LOGV)
            
            self.results = results
            self.current_plot_page = 0
            
            self.display_results()
            self.plot_results()
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    def butler_volmer(self, params, step):
        """Butler-Volmer model calculation"""
        pH = params.get("pH", 0)
        η = params.get("η", 0.0)
        T = params.get("T", self.T.get())
        
        pH_term = -2.303 * pH / (F / (R * T))
        
        if step == "Volmer":
            ΔGV = params.get("ΔGV", 0.5)
            γ = params.get("γ_V", 0.5)
            β = params.get("β_V", 0.5)
            Ea0 = params.get("Ea0_V", 0.5)
            
            k_Va = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGV) / (kB_eV * T)) * np.exp(-β * (η + pH_term) / (kB_eV * T))
            k_minus_Va = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGV) / (kB_eV * T)) * np.exp((1 - β) * (η + pH_term) / (kB_eV * T))
            
            ΔGV_b = ΔGV - (2.303 * (- 14)) / (F / (R * T))
            k_Vb = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGV_b) / (kB_eV * T)) * np.exp(-β * (η + pH_term) / (kB_eV * T))
            k_minus_Vb = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGV_b) / (kB_eV * T)) * np.exp((1 - β) * (η + pH_term) / (kB_eV * T))
            
            return k_Va, k_minus_Va, k_Vb, k_minus_Vb
        
        elif step == "Heyrovsky":
            ΔGH = params.get("ΔGH", 0.5)
            γ = params.get("γ_H", 0.5)
            β = params.get("β_H", 0.5)
            Ea0 = params.get("Ea0_H", 0.5)
            
            k_Ha = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGH) / (kB_eV * T)) * np.exp(-β * (η + pH_term) / (kB_eV * T))
            k_minus_Ha = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGH) / (kB_eV * T)) * np.exp((1 - β) * (η + pH_term) / (kB_eV * T))
            
            ΔGH_b = ΔGH - (2.303 * (- 14)) / (F / (R * T))
            k_Hb = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGH_b) / (kB_eV * T)) * np.exp(-β * (η + pH_term) / (kB_eV * T))
            k_minus_Hb = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGH_b) / (kB_eV * T)) * np.exp((1 - β) * (η + pH_term) / (kB_eV * T))
            
            return k_Ha, k_minus_Ha, k_Hb, k_minus_Hb

    def butler_volmer_vectorized(self, params, step):
        """Vectorized Butler-Volmer model calculation"""
        pH = params.get("pH", 0)
        η = params.get("η", 0.0)
        T = params.get("T", self.T.get())
        
        pH_term = -2.303 * pH / (F / (R * T))
        
        if step == "Volmer":
            ΔGV = params.get("ΔGV", 0.5)
            γ = params.get("γ_V", 0.5)
            β = params.get("β_V", 0.5)
            Ea0 = params.get("Ea0_V", 0.5)
            
            k_Va = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGV) / (kB_eV * T)) * np.exp(-β * (η + pH_term) / (kB_eV * T))
            k_minus_Va = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGV) / (kB_eV * T)) * np.exp((1 - β) * (η + pH_term) / (kB_eV * T))
            
            ΔGV_b = ΔGV - (2.303 * (- 14)) / (F / (R * T))
            k_Vb = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGV_b) / (kB_eV * T)) * np.exp(-β * (η + pH_term) / (kB_eV * T))
            k_minus_Vb = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGV_b) / (kB_eV * T)) * np.exp((1 - β) * (η + pH_term) / (kB_eV * T))
            
            return k_Va, k_minus_Va, k_Vb, k_minus_Vb
        
        elif step == "Heyrovsky":
            ΔGH = params.get("ΔGH", 0.5)
            γ = params.get("γ_H", 0.5)
            β = params.get("β_H", 0.5)
            Ea0 = params.get("Ea0_H", 0.5)
            
            k_Ha = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGH) / (kB_eV * T)) * np.exp(-β * (η + pH_term) / (kB_eV * T))
            k_minus_Ha = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGH) / (kB_eV * T)) * np.exp((1 - β) * (η + pH_term) / (kB_eV * T))
            
            ΔGH_b = ΔGH - (2.303 * (- 14)) / (F / (R * T))
            k_Hb = (kB_eV * T / h_eV) * np.exp(-(Ea0 + γ * ΔGH_b) / (kB_eV * T)) * np.exp(-β * (η + pH_term) / (kB_eV * T))
            k_minus_Hb = (kB_eV * T / h_eV) * np.exp(-(Ea0 - γ * ΔGH_b) / (kB_eV * T)) * np.exp((1 - β) * (η + pH_term) / (kB_eV * T))
            
            return k_Ha, k_minus_Ha, k_Hb, k_minus_Hb

    def marcus(self, params, step):
        """Marcus model calculation"""
        pH = params.get("pH", 0)
        η = params.get("η", 0.0)
        T = params.get("T", self.T.get())
        
        pH_term = -2.303 * pH / (F / (R * T))
        
        if step == "Volmer":
            ΔGV = params.get("ΔGV", 0.5)
            λ = params.get("λV", 2.0)
            
            k_Va = (kB_eV * T / h_eV) * np.exp(-( (ΔGV + (η + pH_term) + λ)**2 / (4 * λ * kB_eV * T) ))
            k_minus_Va = (kB_eV * T / h_eV) * np.exp(-( (-ΔGV - (η + pH_term) + λ)**2 / (4 * λ * kB_eV * T) ))
            
            pH_term_b = -2.303 * (pH - 14) / (F / (R * T))
            k_Vb = (kB_eV * T / h_eV) * np.exp(-( (ΔGV + (η + pH_term_b) + λ)**2 / (4 * λ * kB_eV * T) ))
            k_minus_Vb = (kB_eV * T / h_eV) * np.exp(-( (-ΔGV - (η + pH_term_b) + λ)**2 / (4 * λ * kB_eV * T) ))
            
            return k_Va, k_minus_Va, k_Vb, k_minus_Vb
        
        elif step == "Heyrovsky":
            ΔGH = params.get("ΔGH", 0.5)
            λ = params.get("λH", 2.0)
            
            k_Ha = (kB_eV * T / h_eV) * np.exp(-( (ΔGH + (η + pH_term) + λ)**2 / (4 * λ * kB_eV * T) ))
            k_minus_Ha = (kB_eV * T / h_eV) * np.exp(-( (-ΔGH - (η + pH_term) + λ)**2 / (4 * λ * kB_eV * T) ))
            
            pH_term_b = -2.303 * (pH - 14) / (F / (R * T))
            k_Hb = (kB_eV * T / h_eV) * np.exp(-( (ΔGH + (η + pH_term_b) + λ)**2 / (4 * λ * kB_eV * T) ))
            k_minus_Hb = (kB_eV * T / h_eV) * np.exp(-( (-ΔGH - (η + pH_term_b) + λ)**2 / (4 * λ * kB_eV * T) ))
            
            return k_Ha, k_minus_Ha, k_Hb, k_minus_Hb

    def marcus_vectorized(self, params, step):
        """Vectorized Marcus model calculation"""
        pH = params.get("pH", 0)
        η = params.get("η", 0.0)
        T = params.get("T", self.T.get())
        
        pH_term = -2.303 * pH / (F / (R * T))
        
        if step == "Volmer":
            ΔGV = params.get("ΔGV", 0.5)
            λ = params.get("λV", 2.0)
            
            k_Va = (kB_eV * T / h_eV) * np.exp(-( (ΔGV + (η + pH_term) + λ)**2 / (4 * λ * kB_eV * T) ))
            k_minus_Va = (kB_eV * T / h_eV) * np.exp(-( (-ΔGV - (η + pH_term) + λ)**2 / (4 * λ * kB_eV * T) ))
            
            pH_term_b = -2.303 * (pH - 14) / (F / (R * T))
            k_Vb = (kB_eV * T / h_eV) * np.exp(-( (ΔGV + (η + pH_term_b) + λ)**2 / (4 * λ * kB_eV * T) ))
            k_minus_Vb = (kB_eV * T / h_eV) * np.exp(-( (-ΔGV - (η + pH_term_b) + λ)**2 / (4 * λ * kB_eV * T) ))
            
            return k_Va, k_minus_Va, k_Vb, k_minus_Vb
        
        elif step == "Heyrovsky":
            ΔGH = params.get("ΔGH", 0.5)
            λ = params.get("λH", 2.0)
            
            k_Ha = (kB_eV * T / h_eV) * np.exp(-( (ΔGH + (η + pH_term) + λ)**2 / (4 * λ * kB_eV * T) ))
            k_minus_Ha = (kB_eV * T / h_eV) * np.exp(-( (-ΔGH - (η + pH_term) + λ)**2 / (4 * λ * kB_eV * T) ))
            
            pH_term_b = -2.303 * (pH - 14) / (F / (R * T))
            k_Hb = (kB_eV * T / h_eV) * np.exp(-( (ΔGH + (η + pH_term_b) + λ)**2 / (4 * λ * kB_eV * T) ))
            k_minus_Hb = (kB_eV * T / h_eV) * np.exp(-( (-ΔGH - (η + pH_term_b) + λ)**2 / (4 * λ * kB_eV * T) ))
            
            return k_Ha, k_minus_Ha, k_Hb, k_minus_Hb

    def marcus_gerischer(self, params, step):
        """Marcus-Gerischer model calculation"""
        pH = params.get("pH", 0)
        η = params.get("η", 0.0)
        T = params.get("T", self.T.get())
        ε_range = (-20, 20)  # Wider integration range in eV for better accuracy
        
        pH_term = -2.303 * pH / (F / (R * T))
        
        if step == "Volmer":
            ΔGV = params.get("ΔGV", 0.5)
            λ = params.get("λV", 2.0)
            
            def integrand_Va(ε):
                term = (ΔGV + (η + pH_term) - ε + λ)**2 / (4 * λ * kB_eV * T)
                fermi = 1 / (1 + np.exp(ε / (kB_eV * T)))
                return np.exp(-term) * fermi
            
            k_Va, _ = quad(integrand_Va, ε_range[0], ε_range[1])
            k_Va *= (kB_eV * T / h_eV)
            
            def integrand_minus_Va(ε):
                term = (-ΔGV - (η + pH_term) + ε + λ)**2 / (4 * λ * kB_eV * T)
                fermi = 1 / (1 + np.exp(-ε / (kB_eV * T)))
                return np.exp(-term) * fermi
            
            k_minus_Va, _ = quad(integrand_minus_Va, ε_range[0], ε_range[1])
            k_minus_Va *= (kB_eV * T / h_eV)
            
            pH_term_b = -2.303 * (pH - 14) / (F / (R * T))
            
            def integrand_Vb(ε):
                term = (ΔGV + (η + pH_term_b) - ε + λ)**2 / (4 * λ * kB_eV * T)
                fermi = 1 / (1 + np.exp(ε / (kB_eV * T)))
                return np.exp(-term) * fermi
            
            k_Vb, _ = quad(integrand_Vb, ε_range[0], ε_range[1])
            k_Vb *= (kB_eV * T / h_eV)
            
            def integrand_minus_Vb(ε):
                term = (-ΔGV - (η + pH_term_b) + ε + λ)**2 / (4 * λ * kB_eV * T)
                fermi = 1 / (1 + np.exp(-ε / (kB_eV * T)))
                return np.exp(-term) * fermi
            
            k_minus_Vb, _ = quad(integrand_minus_Vb, ε_range[0], ε_range[1])
            k_minus_Vb *= (kB_eV * T / h_eV)
            
            return k_Va, k_minus_Va, k_Vb, k_minus_Vb
        
        elif step == "Heyrovsky":
            ΔGH = params.get("ΔGH", 0.5)
            λ = params.get("λH", 2.0)
            
            def integrand_Ha(ε):
                term = (ΔGH + (η + pH_term) - ε + λ)**2 / (4 * λ * kB_eV * T)
                fermi = 1 / (1 + np.exp(ε / (kB_eV * T)))
                return np.exp(-term) * fermi
            
            k_Ha, _ = quad(integrand_Ha, ε_range[0], ε_range[1])
            k_Ha *= (kB_eV * T / h_eV)
            
            def integrand_minus_Ha(ε):
                term = (-ΔGH - (η + pH_term) + ε + λ)**2 / (4 * λ * kB_eV * T)
                fermi = 1 / (1 + np.exp(-ε / (kB_eV * T)))
                return np.exp(-term) * fermi
            
            k_minus_Ha, _ = quad(integrand_minus_Ha, ε_range[0], ε_range[1])
            k_minus_Ha *= (kB_eV * T / h_eV)
            
            pH_term_b = -2.303 * (pH - 14) / (F / (R * T))
            
            def integrand_Hb(ε):
                term = (ΔGH + (η + pH_term_b) - ε + λ)**2 / (4 * λ * kB_eV * T)
                fermi = 1 / (1 + np.exp(ε / (kB_eV * T)))
                return np.exp(-term) * fermi
            
            k_Hb, _ = quad(integrand_Hb, ε_range[0], ε_range[1])
            k_Hb *= (kB_eV * T / h_eV)
            
            def integrand_minus_Hb(ε):
                term = (-ΔGH - (η + pH_term_b) + ε + λ)**2 / (4 * λ * kB_eV * T)
                fermi = 1 / (1 + np.exp(-ε / (kB_eV * T)))
                return np.exp(-term) * fermi
            
            k_minus_Hb, _ = quad(integrand_minus_Hb, ε_range[0], ε_range[1])
            k_minus_Hb *= (kB_eV * T / h_eV)
            
            return k_Ha, k_minus_Ha, k_Hb, k_minus_Hb

    def display_results(self):
        """Display results in the scrollable frame"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # Create DataFrame from results
        df = pd.DataFrame(self.results)
        
        # Create treeview
        tree = ttk.Treeview(self.results_frame)
        
        # Add columns
        tree["columns"] = list(df.columns)
        tree.column("#0", width=0, stretch=tk.NO)
        for col in df.columns:
            tree.column(col, anchor=tk.CENTER, width=100)
            tree.heading(col, text=col)
        
        # Add data (limit to 1000 rows for performance)
        for i in range(min(1000, len(df))):
            tree.insert("", tk.END, values=list(df.iloc[i]))
        
        # Add scrollbars
        vsb = ttk.Scrollbar(self.results_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.results_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Update canvas scroll region
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def plot_results(self):
        """Plot results in groups as requested"""
        # Clear previous plots
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # Determine which parameters are variables
        var_params = [p for p in self.variable_params.keys() if self.variable_params[p]["is_var"].get()]
        
        if not var_params:
            return
        
        mechanism = self.mechanism_var.get()
        cmap_name = self.cmap_var.get()
        plot_type = self.plot_type_var.get()
        
        # Create 5 fixed plots
        plots = [
            ("Coverages", ["θ*", "θH*"], "Coverage"),
            ("Forward Rates", ["rV", "rH"] if mechanism == "VH" else ["rV", "rT"], "Rate"),
            ("Backward Rates", ["r-V", "r-H"] if mechanism == "VH" else ["r-V", "r-T"], "Rate"),
            ("Net Rates", ["RV", "RH"] if mechanism == "VH" else ["RV", "RT"], "Net Rate"),
            ("LOG Values", ["LOGV", "LOGH"] if mechanism == "VH" else ["LOGV", "LOGT"], "log10(Rate)")
        ]
        
        for i, (title, columns, ylabel) in enumerate(plots):
            frame = ttk.Frame(self.graph_frame)
            frame.grid(row=i, column=0, padx=5, pady=5, sticky="nsew")
            
            if len(var_params) == 1:
                # Single variable - line plot
                fig = plt.Figure(figsize=(8, 4), dpi=100)
                ax = fig.add_subplot(111)
                
                x_param = var_params[0]
                x_values = self.results[x_param]
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
                
                for idx, col in enumerate(columns):
                    if col in self.results and len(self.results[col]) > 0:
                        ax.plot(x_values, self.results[col],
                               label=col,
                               color=colors[idx],
                               linewidth=2.5,
                               marker='o' if len(x_values) < 20 else None,
                               markersize=6,
                               markeredgecolor='white',
                               markeredgewidth=0.5)
                
                ax.set_xlabel(x_param, fontsize=11, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
                ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
                
                legend = ax.legend(fontsize=10, framealpha=0.7, loc='best')
                legend.get_frame().set_edgecolor('0.8')
                
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                fig.tight_layout()
                
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            elif len(var_params) == 2:
                # Two variables - surface or contour plots
                x_param, y_param = var_params
                x_vals = np.unique(self.results[x_param])
                y_vals = np.unique(self.results[y_param])
                
                if plot_type == "3D":
                    fig = plt.Figure(figsize=(8*len(columns), 4), dpi=100)
                    
                    for col_idx, col in enumerate(columns):
                        if col in self.results and len(self.results[col]) > 0:
                            ax = fig.add_subplot(1, len(columns), col_idx+1, projection='3d')
                            
                            Z = np.array(self.results[col]).reshape(len(y_vals), len(x_vals))
                            X, Y = np.meshgrid(x_vals, y_vals)
                            
                            # For coverage plots, enforce 0-1 range
                            if "θ" in col:  # Coverage plots
                                Z = np.clip(Z, 0, 1)  # Ensure values are between 0 and 1
                                ax.set_zlim(0, 1)
                                ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                            
                            cmap = plt.colormaps.get_cmap(cmap_name)
                            surf = ax.plot_surface(
                                X, Y, Z,
                                cmap=cmap,
                                edgecolor='none',
                                alpha=0.9,
                                rstride=1,
                                cstride=1,
                                linewidth=0,
                                antialiased=True,
                                shade=True
                            )
                            
                            # For coverage plots, set colorbar range to 0-1
                            if "θ" in col:  # Coverage plots
                                surf.set_clim(0, 1)
                            
                            offset = np.min(Z) - 0.1 * (np.max(Z) - np.min(Z))
                            ax.contourf(X, Y, Z, zdir='z', offset=offset, cmap=cmap, alpha=0.3)
                            
                            cbar = fig.colorbar(
                                surf,
                                ax=ax,
                                shrink=0.6,
                                aspect=10,
                                pad=0.1
                            )
                            cbar.set_label(col, fontsize=10)
                            cbar.ax.tick_params(labelsize=8)
                            
                            ax.set_xlabel(x_param, fontsize=10, labelpad=10)
                            ax.set_ylabel(y_param, fontsize=10, labelpad=10)
                            ax.set_zlabel(ylabel, fontsize=10, labelpad=10)
                            ax.set_title(f"{col}", fontsize=11, pad=15)
                            
                            ax.view_init(elev=30, azim=45)
                            ax.grid(True, linestyle=':', alpha=0.5)
                            ax.tick_params(axis='both', which='major', labelsize=8, pad=5)
                            ax.xaxis.pane.fill = False
                            ax.yaxis.pane.fill = False
                            ax.zaxis.pane.fill = False
                            ax.xaxis.pane.set_edgecolor('w')
                            ax.yaxis.pane.set_edgecolor('w')
                            ax.zaxis.pane.set_edgecolor('w')
                            
                            surf.set_facecolor((0,0,0,0))
                            surf.set_edgecolors(cmap(surf._A))
                            surf.set_alpha(0.9)
                    
                    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.0)
                    fig.tight_layout()
                    
                    canvas = FigureCanvasTkAgg(fig, master=frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                else:
                    # 2D plots
                    fig = plt.Figure(figsize=(10, 8), dpi=100)
                    
                    for col_idx, col in enumerate(columns):
                        if col in self.results and len(self.results[col]) > 0:
                            Z = np.array(self.results[col]).reshape(len(y_vals), len(x_vals))
                            X, Y = np.meshgrid(x_vals, y_vals)
                            
                            # For coverage plots, enforce 0-1 range
                            if "θ" in col:  # Coverage plots
                                Z = np.clip(Z, 0, 1)  # Ensure values are between 0 and 1
                            
                            # Contour plot
                            ax1 = fig.add_subplot(2, len(columns), col_idx+1)
                            contour = ax1.contourf(X, Y, Z, levels=20, cmap=cmap_name)
                            
                            # For coverage plots, set color range to 0-1
                            if "θ" in col:  # Coverage plots
                                contour.set_clim(0, 1)
                            
                            def fmt(x):
                                if "θ" in col:  # Coverage plots
                                    return f"{x:.2f}"  # 覆盖度显示2位小数
                                else:
                                    return f"{x:.2e}"  # 其他值显示科学计数法
                            
                            CS = ax1.contour(X, Y, Z, levels=10, colors='k', linewidths=0.5)
                            ax1.clabel(CS, inline=True, fontsize=8, fmt=fmt)
                            
                            cbar = fig.colorbar(contour, ax=ax1)
                            cbar.ax.tick_params(labelsize=8)
                            ax1.set_xlabel(x_param, fontsize=9)
                            ax1.set_ylabel(y_param, fontsize=9)
                            ax1.set_title(f"{col} - Contour", fontsize=10)
                            ax1.tick_params(axis='both', which='major', labelsize=8)
                            
                            # Heatmap plot
                            ax2 = fig.add_subplot(2, len(columns), col_idx+1+len(columns))
                            heatmap = ax2.imshow(Z, extent=[min(x_vals), max(x_vals), min(y_vals), max(y_vals)], 
                                                 origin='lower', aspect='auto', cmap=cmap_name)
                            
                            # For coverage plots, set color range to 0-1
                            if "θ" in col:  # Coverage plots
                                heatmap.set_clim(0, 1)
                            
                            cbar = fig.colorbar(heatmap, ax=ax2)
                            cbar.ax.tick_params(labelsize=8)
                            ax2.set_xlabel(x_param, fontsize=9)
                            ax2.set_ylabel(y_param, fontsize=9)
                            ax2.set_title(f"{col} - Heatmap", fontsize=10)
                            ax2.tick_params(axis='both', which='major', labelsize=8)
                    
                    fig.suptitle(title, fontsize=12, fontweight='bold')
                    fig.tight_layout()
                    
                    canvas = FigureCanvasTkAgg(fig, master=frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            else:
                # For more than 2 variables
                fig = plt.Figure(figsize=(8, 4), dpi=100)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5,
                       "Visualization for >2 variables not implemented.\nCheck results table.",
                       ha='center', va='center', fontsize=10)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')
                
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update canvas scroll region
        self.graph_canvas.configure(scrollregion=self.graph_canvas.bbox("all"))

    def export_results(self):
        """Export results to Excel file"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.results)
                df.to_excel(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def clear(self):
        """Clear results"""
        self.results = None
        self.current_plot_page = 0
        self.progress["value"] = 0
        
        # Clear results display
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Clear plot
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Close all plot windows
        if hasattr(self, 'plot_windows'):
            for window in self.plot_windows:
                try:
                    window.destroy()
                except:
                    pass
            self.plot_windows = []

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ElectrochemicalKineticsApp(root)
    root.mainloop()
