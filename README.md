## 1.项目概述

这是一个基于 Python 的图形用户界面(GUI)应用程序，用于计算和可视化电化学动力学过程。该工具支持多种反应机制(Volmer-Heyrovsky 和 Volmer-Tafel)和动力学模型(Butler-Volmer、Marcus 和 Marcus-Gerischer)，并提供了丰富的参数设置和结果可视化功能。

以下是核心代码，此代码实现了电化学动力学计算的核心数学模型，并提供了基本的用户交互界面，可以扩展添加更多复杂功能和可视化选项。

```python
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 物理常数（eV单位）
kB_eV = 8.617333e-5   # 玻尔兹曼常数
h_eV = 4.135667e-15   # 普朗克常数
F = 96485             # 法拉第常数
R = 8.314             # 气体常数

class ElectrochemicalKineticsApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.setup_variables()
        
    def setup_ui(self):
        """初始化用户界面"""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建参数输入区域
        self.create_parameter_frame()
        
        # 创建结果展示区域
        self.create_results_area()

    def create_parameter_frame(self):
        """参数输入框架"""
        param_frame = ttk.LabelFrame(self.main_frame, text="Parameters")
        param_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # 模型选择
        ttk.Label(param_frame, text="Model:").grid(row=0, column=0)
        self.model_var = tk.StringVar(value="BV")
        ttk.Combobox(param_frame, textvariable=self.model_var, 
                    values=["BV", "Marcus", "Marcus-Gerischer"]).grid(row=0, column=1)

        # 动力学参数输入
        self.entries = {}
        params = [("ΔG", 0.5), ("λ", 2.0), ("β", 0.5), ("η", 0.0), ("pH", 7.0)]
        for i, (name, val) in enumerate(params):
            ttk.Label(param_frame, text=f"{name}:").grid(row=i+1, column=0)
            self.entries[name] = tk.DoubleVar(value=val)
            ttk.Entry(param_frame, textvariable=self.entries[name]).grid(row=i+1, column=1)

    def calculate_rates(self):
        """核心计算逻辑"""
        # 获取输入参数
        params = {name: var.get() for name, var in self.entries.items()}
        model = self.model_var.get()
        
        # 根据不同模型计算速率
        if model == "BV":
            return self.butler_volmer(params)
        elif model == "Marcus":
            return self.marcus_model(params)
        else:
            return self.marcus_gerischer(params)

    def butler_volmer(self, params):
        """Butler-Volmer 模型计算"""
        ΔG = params['ΔG']
        β = params['β']
        η = params['η']
        T = 298  # 温度
        
        k_forward = (kB_eV*T/h_eV) * np.exp(-(ΔG + η*β)/(kB_eV*T))
        k_backward = (kB_eV*T/h_eV) * np.exp((ΔG - η*(1-β))/(kB_eV*T))
        return k_forward, k_backward

    def marcus_model(self, params):
        """Marcus 模型计算"""
        ΔG = params['ΔG']
        λ = params['λ']
        η = params['η']
        
        k_forward = (kB_eV*T/h_eV) * np.exp(-(ΔG + η + λ)**2/(4*λ*kB_eV*T))
        k_backward = (kB_eV*T/h_eV) * np.exp(-(ΔG - η + λ)**2/(4*λ*kB_eV*T))
        return k_forward, k_backward

    def update_plot(self):
        """更新结果图表"""
        kf, kb = self.calculate_rates()
        
        # 清除旧图表
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # 绘制新图表
        fig = plt.Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot([kf], [kb], 'ro')
        ax.set_xlabel('Forward Rate')
        ax.set_ylabel('Backward Rate')
        
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = ElectrochemicalKineticsApp(root)
    root.mainloop()
```



## 2.主要功能

## 2.1多种反应机制支持

程序运行之后创建的GUI界面上有机理选择区，可以选择Volmer-Heyrovsky机理 或 Volmer-Tafel机理。

```python
# 在 __init__ 方法中定义机制变量
self.mechanism_var = tk.StringVar(value="VH")

# 在 create_widgets 方法中创建单选按钮
ttk.Radiobutton(selection_frame, text="Volmer-Heyrovsky (VH)", variable=self.mechanism_var,
               value="VH", command=self.update_parameter_ui).grid(row=0, column=1, sticky="w")
ttk.Radiobutton(selection_frame, text="Volmer-Tafel (VT)", variable=self.mechanism_var,
               value="VT", command=self.update_parameter_ui).grid(row=0, column=2, sticky="w")

# 在 update_parameter_ui 方法中动态更新参数界面
if mechanism == "VH":
    self.other_frame = ttk.LabelFrame(self.param_container, text="Heyrovsky Parameters")
else:
    self.other_frame = ttk.LabelFrame(self.param_container, text="Tafel Parameters")
```

## 2.2多种动力学模型

Butler-Volmer (BV) 模型Marcus (M) 模型Marcus-Gerischer (MG) 模型

### 2.2.1模型选择与UI控制

程序运行之后有模型选择区，可以选择Butler-Volmer (BV) 模型、Marcus (M) 模型或者Marcus-Gerischer (MG) 模型。

```python
# 定义模型变量
self.model_var = tk.StringVar(value="BV")

# 创建模型选择按钮
ttk.Radiobutton(selection_frame, text="Butler-Volmer (BV)", variable=self.model_var,
               value="BV", command=self.update_parameter_ui).grid(row=1, column=1, sticky="w")
ttk.Radiobutton(selection_frame, text="Marcus (M)", variable=self.model_var,
               value="M", command=self.update_parameter_ui).grid(row=1, column=2, sticky="w")
ttk.Radiobutton(selection_frame, text="Marcus-Gerischer (MG)", variable=self.model_var,
               value="MG", command=self.update_parameter_ui).grid(row=1, column=3, sticky="w")

# 动态参数界面
def create_bv_parameters(self, mechanism):
    # BV模型参数（β系数）
    ttk.Label(self.volmer_frame, text="β_V:").grid(row=2, column=0, sticky="w")
    self.fixed_params["β_V"] = tk.DoubleVar(value=0.5)

def create_marcus_parameters(self, mechanism):
    # Marcus模型参数（重组能λ）
    ttk.Label(self.volmer_frame, text="λV:").grid(row=1, column=0, sticky="w")
    self.fixed_params["λV"] = tk.DoubleVar(value=2.0)
```

## 2.3参数设置

### 2.3.1可调节温度参数

#### (1)温度参数输入界面部分

界面上会生成一个单独的温度输入框，在此可以输入温度。

```python
# 在create_widgets方法中
ttk.Label(selection_frame, text="Temperature (K):").grid(row=2, column=0, sticky="w")
self.T = tk.DoubleVar(value=298.15)  # 默认温度298.15K
ttk.Entry(selection_frame, textvariable=self.T, width=10).grid(row=2, column=1, sticky="w")
```

#### (2)温度参数获取与传递

```python
# 在calculate方法中获取温度值
T = self.T.get()

# 在参数传递中保持温度值
# 单变量情况
current_params["T"] = np.full(n, T)
# 双变量情况 
current_params["T"] = np.full((n2, n1), T)
```

#### (3)温度在计算公式中的应用

```python
# 在butler_volmer方法中
kB_eV * T / h_eV  # 计算指前因子
np.exp(-(Ea0 + γ * ΔGV) / (kB_eV * T))  # 计算指数项

# 在marcus方法中
np.exp(-( (ΔGV + (η + pH_term) + λ)**2 / (4 * λ * kB_eV * T) ))  # 马库斯项

# 在marcus_gerischer方法中
fermi = 1 / (1 + np.exp(ε / (kB_eV * T)))  # 费米-狄拉克分布计算
```

### 2.3.2可变的过电位(η)和pH值

#### (1)参数定义和初始化部分

此部分是参数的定义和初始化设置，包括范围、步长和默认值。

```python
# 在 __init__ 方法中初始化参数范围
self.variable_params = {
    "η": {"min": tk.DoubleVar(value=-1.0), "max": tk.DoubleVar(value=1.0),
          "step": tk.DoubleVar(value=0.01), "is_var": tk.BooleanVar(value=False),
          "fixed_value": tk.DoubleVar(value=0.0)},
    "pH": {"min": tk.DoubleVar(value=-2), "max": tk.DoubleVar(value=2),
           "step": tk.DoubleVar(value=0.1), "is_var": tk.BooleanVar(value=False),
           "fixed_value": tk.DoubleVar(value=0.0)}
}
```

#### (2)参数界面创建部分

GUI界面生成时有专门的参数输入区，在此可以自行设置想要的参数范围。

```python
# 在 create_variable_parameters 方法中创建输入界面
def create_variable_parameters(self):
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
```

#### (3)参数范围生成逻辑

在 calculate 方法中处理可变参数

```python
# 在 calculate 方法中处理可变参数
# Check variable parameters
var_params = []
for param, data in self.variable_params.items():
    if data["is_var"].get():
        var_params.append(param)

# Prepare parameter ranges
param_ranges = {}
for param in var_params:
    min_val = self.variable_params[param]["min"].get()
    max_val = self.variable_params[param]["max"].get()
    step = self.variable_params[param]["step"].get()
    param_ranges[param] = np.arange(min_val, max_val + step/2, step)
```

### 2.3.3各步骤的动力学参数设置

#### (1)参数界面设置代码

根据选择的机理和模型，创建不同的参数界面。

```python
# 在 update_parameter_ui 方法中根据机理和模型动态创建参数界面
def update_parameter_ui(self):
    mechanism = self.mechanism_var.get()  # 获取机理选择（VH/VT）
    model = self.model_var.get()          # 获取模型选择（BV/M/MG）
    
    # 根据模型创建不同参数输入界面
    if model == "BV":
        self.create_bv_parameters(mechanism)  # 创建 Butler-Volmer 参数
    elif model == "M":
        self.create_marcus_parameters(mechanism)  # 创建 Marcus 参数
    elif model == "MG":
        self.create_mg_parameters(mechanism)      # 创建 Marcus-Gerischer 参数
```

#### (2)Butler-Volmer 模型参数设置

##### ①Volmer 步骤参数

```python
def create_bv_parameters(self, mechanism):
    # Volmer 参数（所有机理共有）
    ttk.Label(self.volmer_frame, text="ΔGV (eV):").grid(row=0, column=0)
    self.fixed_params["ΔGV"] = tk.DoubleVar(value=0.5)  # 吉布斯自由能变化
    ttk.Entry(...)  # 输入框
    
    ttk.Label(...text="γ_V:")  # 对称因子
    self.fixed_params["γ_V"] = tk.DoubleVar(value=0.5)
    
    ttk.Label(...text="β_V:")  # 电荷转移系数
    self.fixed_params["β_V"] = tk.DoubleVar(value=0.5)
    
    ttk.Label(...text="Ea0_V (eV):")  # 基准活化能
    self.fixed_params["Ea0_V"] = tk.DoubleVar(value=0.5)
```

##### ②Heyrovsky 步骤参数（VH 机理）

```python
if mechanism == "VH":
    ttk.Label(...text="ΔGH (eV):")  # 吉布斯自由能变化
    self.fixed_params["ΔGH"] = tk.DoubleVar(value=0.5)
    
    ttk.Label(...text="γ_H:")       # 对称因子
    self.fixed_params["γ_H"] = tk.DoubleVar(value=0.5)
    
    ttk.Label(...text="β_H:")       # 电荷转移系数
    self.fixed_params["β_H"] = tk.DoubleVar(value=0.5)
    
    ttk.Label(...text="Ea0_H (eV):") # 基准活化能
    self.fixed_params["Ea0_H"] = tk.DoubleVar(value=0.5)

```

##### ③Tafel 步骤参数（VT 机理）

```python
else:
    ttk.Label(...text="ΔGT (eV):")  # 吉布斯自由能变化
    self.fixed_params["ΔGT"] = tk.DoubleVar(value=0.5)
    
    ttk.Label(...text="Ea0_T (eV):") # 基准活化能
    self.fixed_params["Ea0_T"] = tk.DoubleVar(value=0.5)
    
    ttk.Label(...text="γ_T:")       # 对称因子
    self.fixed_params["γ_T"] = tk.DoubleVar(value=0.5)
```

#### （3）Marcus 模型参数设置

```python
def create_marcus_parameters(self, mechanism):
    # Volmer 参数
    ttk.Label(...text="ΔGV (eV):")    # 吉布斯自由能变化
    self.fixed_params["ΔGV"] = tk.DoubleVar(value=0.5)
    
    ttk.Label(...text="λV:")          # 重组能
    self.fixed_params["λV"] = tk.DoubleVar(value=2.0)

    # Heyrovsky（VH 机理）参数
    if mechanism == "VH":
        ttk.Label(...text="ΔGH (eV):")
        self.fixed_params["ΔGH"] = tk.DoubleVar(value=0.5)
        
        ttk.Label(...text="λH:") 
        self.fixed_params["λH"] = tk.DoubleVar(value=2.0)
    
    # Tafel（VT 机理）参数
    else:
        ttk.Label(...text="ΔGT (eV):")
        self.fixed_params["ΔGT"] = tk.DoubleVar(value=0.5)
        
        ttk.Label(...text="Ea0_T (eV):")
        self.fixed_params["Ea0_T"] = tk.DoubleVar(value=0.5)
```

#### （4）Marcus-Gerischer模型

参数设置同Marcus 模型

## 2.4计算模块

### 2.4.1核心计算逻辑入口（`calculate`方法片段）

```python
def calculate(self):
    # 获取温度参数
    T = self.T.get()
    
    # 计算速率常数（根据模型选择不同方法）
    if model == "BV":
        k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.butler_volmer(current_params, "Volmer")
    elif model == "M":
        k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.marcus(current_params, "Volmer")
    elif model == "MG":
        k_Va, k_minus_Va, k_Vb, k_minus_Vb = self.marcus_gerischer(current_params, "Volmer")

    # 计算总反应速率常数（考虑pH依赖）
    k_V = k_Va * (10 ** -pH) + k_Vb
    k_minus_V = k_minus_Va + k_minus_Vb * (10 ** (pH - 14))

    # 表面覆盖度计算（分VH和VT机制）
    if mechanism == "VH":
        denominator = k_V + k_minus_H + k_minus_V + k_H
        theta_star = (k_minus_V + k_H) / denominator  # θ*
        theta_H = (k_V + k_minus_H) / denominator     # θH*
    else:  # VT机制
        # 使用二次方程求解覆盖度
        a = 2 * (k_minus_T - k_T)
        b_V = k_V + k_minus_V + 4 * k_minus_T
        sqrt_term = b_V**2 - 4 * a * c_V
        theta_H = (b_V - np.sqrt(sqrt_term)) / (2 * a)
```

### 2.4.2Butler-Volmer模型计算模块

```python
def butler_volmer(self, params, step):
    # 获取关键参数
    η = params.get("η", 0.0)
    T = params.get("T", self.T.get())
    β = params.get("β_V", 0.5) if step=="Volmer" else params.get("β_H", 0.5)

    # 计算正向/逆向速率常数
    k_forward = (kB_eV*T/h_eV) * np.exp(
        -(Ea0 + γ*ΔG) / (kB_eV*T)) * np.exp(-β*(η+pH_term)/(kB_eV*T)
    )
    k_backward = (kB_eV*T/h_eV) * np.exp(
        -(Ea0 - γ*ΔG) / (kB_eV*T)) * np.exp((1-β)*(η+pH_term)/(kB_eV*T)
    )
    return k_forward, k_backward
```

### 2.4.3Marcus模型计算模块

```python
def marcus_vectorized(self, params, step):
    # 重组能计算
    λ = params.get("λV", 2.0) if step=="Volmer" else params.get("λH", 2.0)
    
    # 马库斯理论公式
    k_forward = (kB_eV*T/h_eV) * np.exp(
        -((ΔG + overpotential + λ)**2) / (4*λ*kB_eV*T)
    )
    k_backward = (kB_eV*T/h_eV) * np.exp(
        -((-ΔG - overpotential + λ)**2) / (4*λ*kB_eV*T)
    )
    return k_forward, k_backward
```

### 2.4.4Marcus-Gerischer模型（含积分计算）

```python
def marcus_gerischer(self, params, step):
    # 定义积分函数
    def integrand(ε):
        term = (ΔG + overpotential - ε + λ)**2 / (4*λ*kB_eV*T)
        fermi = 1 / (1 + np.exp(ε/(kB_eV*T)))  # 费米-狄拉克分布
        return np.exp(-term) * fermi

    # 数值积分计算速率常数
    k_forward, _ = quad(integrand, -20, 20)
    k_forward *= (kB_eV*T/h_eV)
    return k_forward, ...  # 逆向速率计算类似
```

### 2.4.5反应速率计算模块

```python
# Volmer步骤速率
r_V = k_V * theta_star
r_minus_V = k_minus_V * theta_H

# Heyrovsky步骤速率
r_H = k_H * theta_H
r_minus_H = k_minus_H * theta_star

# Tafel步骤速率（二次依赖）
r_T = 2 * k_T * theta_H**2
r_minus_T = 2 * k_minus_T * theta_star**2

# 净反应速率
R_V = r_minus_V - r_V
R_H = r_minus_H - r_H
R_T = r_minus_T - r_T

# 对速率取对数（避免零值）
LOGV = np.log10(np.abs(R_V) + 1e-20)
LOGH = np.log10(np.abs(R_H) + 1e-20)
LOGT = np.log10(np.abs(R_T) + 1e-20)
```

## 2.5结果可视化

### 2.5.12D 和 3D 图形展示

以下是创建图形区和生成2D 和 3D 图形的关键代码。

```python
# 导入相关库（需要保留的部分）
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_results(data, var_params, plot_type="3D", cmap_name="viridis"):
    """通用绘图函数"""
    # 创建5组固定绘图（示例保留3组关键绘图）
    plots = [
        ("Coverages", ["θ*", "θH*"], "Coverage"),
        ("Net Rates", ["RV", "RH"], "Net Rate"),
        ("LOG Values", ["LOGV", "LOGH"], "log10(Rate)")
    ]
    
    figures = []
    
    for title, columns, ylabel in plots:
        fig = plt.figure(figsize=(10, 6))
        
        if len(var_params) == 1:
            # 2D线图
            ax = fig.add_subplot(111)
            x_param = var_params[0]
            x_values = data[x_param]
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
            for idx, col in enumerate(columns):
                ax.plot(x_values, data[col], 
                       color=colors[idx],
                       linewidth=2,
                       marker='o',
                       markersize=5,
                       label=col)
            
            ax.set(xlabel=x_param, ylabel=ylabel, title=title)
            ax.grid(True, linestyle='--')
            ax.legend()
            
        elif len(var_params) == 2 and plot_type == "3D":
            # 3D曲面图
            x_param, y_param = var_params
            x_vals = np.unique(data[x_param])
            y_vals = np.unique(data[y_param])
            
            for col_idx, col in enumerate(columns):
                ax = fig.add_subplot(1, len(columns), col_idx+1, projection='3d')
                Z = np.array(data[col]).reshape(len(y_vals), len(x_vals))
                X, Y = np.meshgrid(x_vals, y_vals)
                
                surf = ax.plot_surface(X, Y, Z, cmap=cmap_name,
                                      linewidth=0, antialiased=True)
                
                fig.colorbar(surf, ax=ax, shrink=0.6)
                ax.set(xlabel=x_param, ylabel=y_param, zlabel=ylabel, title=col)
                ax.view_init(30, 45)
                
        elif len(var_params) == 2 and plot_type == "2D":
            # 2D等高线+热力图
            x_param, y_param = var_params
            x_vals = np.unique(data[x_param])
            y_vals = np.unique(data[y_param])
            
            for col_idx, col in enumerate(columns):
                # 等高线图
                ax1 = fig.add_subplot(2, len(columns), col_idx+1)
                Z = np.array(data[col]).reshape(len(y_vals), len(x_vals))
                X, Y = np.meshgrid(x_vals, y_vals)
                
                contour = ax1.contourf(X, Y, Z, 20, cmap=cmap_name)
                fig.colorbar(contour, ax=ax1)
                ax1.set(xlabel=x_param, ylabel=y_param, title=f"{col} Contour")
                
                # 热力图
                ax2 = fig.add_subplot(2, len(columns), col_idx+1+len(columns))
                heatmap = ax2.imshow(Z, extent=[x_vals.min(), x_vals.max(), 
                                               y_vals.min(), y_vals.max()],
                                   origin='lower', aspect='auto', cmap=cmap_name)
                fig.colorbar(heatmap, ax=ax2)
                ax2.set(xlabel=x_param, ylabel=y_param, title=f"{col} Heatmap")
        
        figures.append(fig)
    
    return figures

# 使用示例（需要准备数据）
if __name__ == "__main__":
    # 生成示例数据（替换为实际数据）
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    example_data = {
        "η": X.flatten(),
        "pH": Y.flatten(),
        "θ*": np.sin(X**2 + Y**2).flatten(),
        "RV": (X**3 - Y**2).flatten(),
        "LOGV": np.log(np.abs(X + Y) + 1e-10).flatten()
    }
    
    # 生成3D图
    figures_3d = plot_results(example_data, ["η", "pH"], plot_type="3D")
    
    # 生成2D图
    figures_2d = plot_results(example_data, ["η", "pH"], plot_type="2D")
    
    # 显示图形
    plt.show()
```

### 2.5.2多种颜色映射选择

可以自行选择需要的颜色。

```python
import matplotlib.pyplot as plt
from matplotlib import cm
import tkinter as tk
from tkinter import ttk

class ColorMapSelector:
    def __init__(self):
        self.root = tk.Tk()
        
        # 颜色映射参数
        self.cmap_var = tk.StringVar(value="viridis")
        self.cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                     'coolwarm', 'rainbow', 'jet', 'turbo', 'seismic']
        
        # 创建选择组件
        ttk.Label(self.root, text="Colormap:").pack()
        cmap_menu = ttk.OptionMenu(self.root, self.cmap_var, 
                                  self.cmap_var.get(), *self.cmaps)
        cmap_menu.pack()
        
        # 示例按钮
        ttk.Button(self.root, text="Show Example", 
                 command=self.show_example).pack()
        
        self.root.mainloop()
    
    def show_example(self):
        # 获取当前选择的颜色映射
        cmap_name = self.cmap_var.get()
        
        # 生成示例数据
        import numpy as np
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        
        # 创建3D图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap_name,
                              linewidth=0, antialiased=True)
        fig.colorbar(surf)
        ax.set_title(f"Using {cmap_name} colormap")
        plt.show()

if __name__ == "__main__":
    ColorMapSelector()
```

### 2.5.1覆盖度、反应速率、净速率等关键参数的图形显示

```python
def plot_key_parameters(self):
    """专门绘制电化学关键参数图形的核心代码"""
    if not self.results:
        return
    
    # 获取当前设置
    mechanism = self.mechanism_var.get()
    cmap_name = self.cmap_var.get()
    plot_type = self.plot_type_var.get()
    var_params = [p for p in self.variable_params.keys() 
                 if self.variable_params[p]["is_var"].get()]
    
    if not var_params:
        return
    
    # 关键参数定义
    coverage_params = ["θ*", "θH*"]  # 覆盖度参数
    forward_rate_params = ["rV", "rH"] if mechanism == "VH" else ["rV", "rT"]  # 正向反应速率
    backward_rate_params = ["r-V", "r-H"] if mechanism == "VH" else ["r-V", "r-T"]  # 逆向反应速率
    net_rate_params = ["RV", "RH"] if mechanism == "VH" else ["RV", "RT"]  # 净反应速率
    log_params = ["LOGV", "LOGH"] if mechanism == "VH" else ["LOGV", "LOGT"]  # 对数速率
    
    # 1. 覆盖度图形 (θ*和θH*)
    self._plot_single_category(
        title="Surface Coverages",
        params=coverage_params,
        ylabel="Coverage Fraction",
        var_params=var_params,
        plot_type=plot_type,
        cmap_name=cmap_name,
        zlim=(0, 1)  # 强制覆盖度范围0-1
    )
    
    # 2. 反应速率图形 (正向和逆向)
    self._plot_single_category(
        title="Reaction Rates",
        params=forward_rate_params + backward_rate_params,
        ylabel="Rate (s⁻¹)",
        var_params=var_params,
        plot_type=plot_type,
        cmap_name=cmap_name
    )
    
    # 3. 净速率图形
    self._plot_single_category(
        title="Net Reaction Rates",
        params=net_rate_params,
        ylabel="Net Rate (s⁻¹)",
        var_params=var_params,
        plot_type=plot_type,
        cmap_name=cmap_name
    )
    
    # 4. 对数速率图形
    self._plot_single_category(
        title="Logarithmic Rates",
        params=log_params,
        ylabel="log10(Rate)",
        var_params=var_params,
        plot_type=plot_type,
        cmap_name=cmap_name
    )

def _plot_single_category(self, title, params, ylabel, var_params, plot_type, cmap_name, zlim=None):
    """绘制单个参数类别的通用函数"""
    frame = ttk.Frame(self.graph_frame)
    frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    if len(var_params) == 1:
        # 单变量2D线图
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        x_param = var_params[0]
        x_values = self.results[x_param]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(params)))
        
        for idx, param in enumerate(params):
            if param in self.results:
                ax.plot(x_values, self.results[param],
                       label=param,
                       color=colors[idx],
                       linewidth=2)
        
        ax.set_xlabel(x_param)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
    elif len(var_params) == 2 and plot_type == "3D":
        # 双变量3D曲面图
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        
        x_param, y_param = var_params
        x_vals = np.unique(self.results[x_param])
        y_vals = np.unique(self.results[y_param])
        
        for i, param in enumerate(params):
            if param in self.results:
                ax = fig.add_subplot(1, len(params), i+1, projection='3d')
                
                Z = np.array(self.results[param]).reshape(len(y_vals), len(x_vals))
                X, Y = np.meshgrid(x_vals, y_vals)
                
                surf = ax.plot_surface(X, Y, Z, cmap=cmap_name,
                                     rstride=1, cstride=1,
                                     linewidth=0, antialiased=True)
                
                if zlim:  # 特别处理覆盖度的范围
                    ax.set_zlim(*zlim)
                    surf.set_clim(*zlim)
                
                fig.colorbar(surf, ax=ax, shrink=0.6)
                ax.set(xlabel=x_param, ylabel=y_param, zlabel=ylabel, title=param)
                ax.view_init(30, 45)
    
    elif len(var_params) == 2 and plot_type == "2D":
        # 双变量2D组合图
        fig = plt.Figure(figsize=(12, 8), dpi=100)
        
        x_param, y_param = var_params
        x_vals = np.unique(self.results[x_param])
        y_vals = np.unique(self.results[y_param])
        
        for i, param in enumerate(params):
            if param in self.results:
                Z = np.array(self.results[param]).reshape(len(y_vals), len(x_vals))
                X, Y = np.meshgrid(x_vals, y_vals)
                
                # 等高线图
                ax1 = fig.add_subplot(2, len(params), i+1)
                contour = ax1.contourf(X, Y, Z, 20, cmap=cmap_name)
                if zlim:  # 特别处理覆盖度的范围
                    contour.set_clim(*zlim)
                fig.colorbar(contour, ax=ax1)
                ax1.set(xlabel=x_param, ylabel=y_param, title=f"{param} Contour")
                
                # 热力图
                ax2 = fig.add_subplot(2, len(params), i+1+len(params))
                im = ax2.imshow(Z, extent=[x_vals.min(), x_vals.max(), 
                                          y_vals.min(), y_vals.max()],
                               origin='lower', aspect='auto', cmap=cmap_name)
                if zlim:
                    im.set_clim(*zlim)
                fig.colorbar(im, ax=ax2)
                ax2.set(xlabel=x_param, ylabel=y_param, title=f"{param} Heatmap")
    
    # 嵌入图形到Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
```



## 2.6数据导出

结果可导出为Excel文件。

```python
def export_results(self):
    """导出结果到Excel文件"""
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
```



# 3.安装与运行

### 3.1系统要求

Python 3.6 或更高版本

## 3.2主程序入口

```python
# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ElectrochemicalKineticsApp(root)
    root.mainloop()
```

## 3.3依赖库导入

```python
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
```

# 4.使用说明

## 4.1计算功能

```python
def calculate(self):
    """核心计算函数"""
    try:
        # 获取参数
        mechanism = self.mechanism_var.get()
        model = self.model_var.get()
        T = self.T.get()
        
        # 确定变量参数
        var_params = [p for p, data in self.variable_params.items() if data["is_var"].get()]
        
        # 根据变量参数数量选择计算方法
        if len(var_params) == 1:
            # 单变量优化计算
            pass
        elif len(var_params) == 2:
            # 双变量优化计算
            pass
        else:
            # 多变量常规计算
            pass
```

4.2参数界面更新

```python
def update_parameter_ui(self):
    """根据选择的机制和模型更新参数界面"""
    for widget in self.param_container.winfo_children():
        widget.destroy()
    
    mechanism = self.mechanism_var.get()
    model = self.model_var.get()
    
    # 根据模型创建参数输入界面
    if model == "BV":
        self.create_bv_parameters(mechanism)
    elif model == "M":
        self.create_marcus_parameters(mechanism)
    elif model == "MG":
        self.create_mg_parameters(mechanism)
```

# 5. 示例

### 5.1 Butler-Volmer 模型实现

```python
def butler_volmer(self, params, step):
    """Butler-Volmer模型计算"""
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
```

## 示例

1. 选择"Volmer-Heyrovsky"机制和"Butler-Volmer"模型
2. 设置Volmer和Heyrovsky步骤的参数
3. 选择η作为变量参数，范围从-1.0到1.0，步长0.1
4. 点击"Calculate"按钮
5. 查看生成的覆盖度、反应速率等图形

# 6.开发说明

该项目使用Python的tkinter库构建GUI界面，matplotlib用于数据可视化，numpy和scipy用于数值计算。

## 6.1常量定义

```python
# Constants (all in eV units)
kB_eV = 8.617333e-5  # Boltzmann constant (eV/K)
h_eV = 4.135667e-15  # Planck constant (eV·s)
F = 96485           # Faraday constant (C/mol)
R = 8.314           # Gas constant (J/(mol·K))
eV_to_J = 1.602176634e-19  # 1 eV = 1.602e-19 J
```

## 6.2界面布局

```python
def create_widgets(self):
    """创建GUI界面"""
    # Main Frame
    main_frame = ttk.Frame(self.root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Left Panel (Parameters and Results)
    left_panel = ttk.Frame(main_frame, width=400)
    left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
    
    # Right Panel (Graphs)
    right_panel = ttk.Frame(main_frame)
    right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
```



## 许可证

此项目采用 [MIT许可证](https://license/)。

## 贡献

欢迎提交问题和拉取请求。

## 联系方式

如有任何问题或建议，请联系项目维护者。
