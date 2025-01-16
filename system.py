import numpy as np
import skfuzzy as fuzz
import sys
import matplotlib.pyplot as plt
from functools import reduce

class FLSInventoryManagement:
    x_demand = np.linspace(0, 4, 51)  # Input 3
    x_inventory = np.linspace(0, 50, 51)  # Input 1
    x_days_before_expiry = np.linspace(0, 50, 51)  # Input 2
    x_quantity = np.linspace(0, 50, 51)  # Output 1
    x_urgency = np.linspace(0, 100, 51)  # Output 2

    rule_list = [ # must covers 27 outcomes
        # StateDemand, StateInv, StateDays - Outcomes [None if not available or ignored]
        [['High','Medium','Low'],['Medium','High']], # 1o - HML
        [['High','Low','Low'],['High','High']], # 1o - HLL
        [[None,"High","High"],["Zero","Zero"]], # 3o - LHH, MHH, HHH
        [[None,'High','Low'],['Medium','High']], # 3o - LHL, MHL, HHL
        [['Low', 'Medium', None],['Low','Low']], # 3o - LML, LMM, LMH
        [['Medium', None, 'High'],['Zero','Low']], # 3o - MLH, MMH, MHH
        [['High', "Medium", None],['Medium', 'Medium']], # 3o - HML, HMM, HMH
        [[None, 'High', 'Medium'],['Low','Low']], # 3o - LHM, MHM, HHM
        [[None, None,'Low'],['Medium','High']], # 9o - LLL, LML, LHL, MLL, MML, MHL, HLL, HML, HHL
        [[None, 'Low', None],['High', 'High']] # 9o - LLL, LLM, LLH, MLL, MLM, MLH, HLL, HLM, HLH
    ]

    def __init__(self):
        self.construct_fuzzy_system()
    
    def construct_fuzzy_system(self):
        x = self.x_demand
        demand_low = fuzz.trapmf(x, [0, 0, 0.5, 2])
        demand_med = fuzz.trimf(x, [0.5, 2, 3.5])
        demand_hig = fuzz.trapmf(x, [2, 3.5, 4, 4])
        self.demand = [demand_low, demand_med, demand_hig]

        x = self.x_inventory
        inventory_low = fuzz.trapmf(x, [0, 0, 5, 25])
        inventory_med = fuzz.trimf(x, [5, 25, 45])
        inventory_hig = fuzz.trapmf(x, [25, 45, 50, 50])
        self.inventory = [inventory_low, inventory_med, inventory_hig]

        x = self.x_days_before_expiry
        days_low = fuzz.trapmf(x, [0, 0, 5, 15])
        days_med = fuzz.trapmf(x, [5, 20, 30, 45])
        days_hig = fuzz.trapmf(x, [30, 45, 50, 50])
        self.days = [days_low, days_med, days_hig]

        x = self.x_quantity
        quantity_zero = fuzz.trapmf(x, [0, 0, 5, 5])
        quantity_low = fuzz.trimf(x, [5, 5, 25])
        quantity_med = fuzz.trimf(x, [5, 25, 45])
        quantity_hig = fuzz.trapmf(x, [25, 45, 50, 50])
        self.quantity = [quantity_zero, quantity_low, quantity_med, quantity_hig]

        x = self.x_urgency
        urgent_zero = fuzz.trapmf(x, [0, 0, 5, 5])
        urgent_low = fuzz.trimf(x, [5, 5, 30])
        urgent_med = fuzz.trapmf(x, [5, 45, 55, 95])
        urgent_hig = fuzz.trapmf(x, [65, 95, 100, 100])

        self.urgency = [urgent_zero, urgent_low, urgent_med, urgent_hig]
    
    def get_result(self, inv_input:int, days_input:int, demand_input:float, method_quantity:str = 'centroid', method_urgency:str = 'centroid'):
        aggregated_quantity, aggregated_urgency = self.get_aggregated(inv_input, days_input, demand_input)
        quantity_result, urgency_result = self.defuzzification(aggregated_quantity, aggregated_urgency, method_quantity, method_urgency)
        return quantity_result, urgency_result
    
    def get_aggregated(self, inv_input:int, days_input:int, demand_input:float):
        invFuzz, daysFuzz, demandFuzz = self.fuzzification(inv_input, days_input, demand_input)
        quantity_agg, urgency_agg = self.inference_loop(invFuzz, daysFuzz, demandFuzz)

        return quantity_agg, urgency_agg

    def get_variables(self, kind:str):
        if kind == 'Inventory':
            x = self.x_inventory
            func = self.inventory
        elif kind == 'Days':
            x = self.x_days_before_expiry
            func = self.days
        elif kind == 'Quantity':
            x = self.x_quantity
            func = self.quantity
        elif kind == 'Urgency':
            x = self.x_urgency
            func = self.urgency
        elif kind == 'Demand':
            x = self.x_demand
            func = self.demand
        else:
            raise(ValueError("Invalid variables!"))
        return x, func
    
    def interpert_membership_inputs(self, val, kind):
            result = {
                "Low": 0,
                "Medium": 0,
                "High": 0
            }
            linguistics = ['Low', 'Medium', 'High']
            for ind in range(3):
                if kind == 'Inventory':
                    result[linguistics[ind]] = fuzz.interp_membership(self.x_inventory, self.inventory[ind], val)
                elif kind == 'Days':
                    result[linguistics[ind]] = fuzz.interp_membership(self.x_days_before_expiry, self.days[ind], val)
                elif kind == 'Demand':
                    result[linguistics[ind]] = fuzz.interp_membership(self.x_demand, self.demand[ind], val)
                else:
                    raise(ValueError("Wrong Variables!"))
            return result
    
    def fuzzification(self, inv_input:int, days_input:int, demand_input:float):
        
        inv_input = np.clip(inv_input, min(self.x_inventory), max(self.x_inventory))
        days_input = np.clip(days_input, min(self.x_days_before_expiry), max(self.x_days_before_expiry))
        demand_input = np.clip(demand_input, min(self.x_demand), max(self.x_demand))
        invFuzz = self.interpert_membership_inputs(inv_input, "Inventory")
        dayFuzz = self.interpert_membership_inputs(days_input, "Days")
        demandFuzz = self.interpert_membership_inputs(demand_input, "Demand")

        return invFuzz, dayFuzz, demandFuzz
    
    def get_set(self, output:str, kind:str):
        if kind == "Quantity":
            setlist = self.quantity
        elif kind == 'Urgency':
            setlist = self.urgency
        else:
            raise(ValueError("Invalid Variables!"))
        
        if output == 'Low':
            set_outcome = setlist[1]
        elif output == 'Medium':
            set_outcome = setlist[2]
        elif output == 'High':
            set_outcome = setlist[3]
        elif output == 'Zero':
            set_outcome = setlist[0]
        else:
            raise(ValueError("Invalid States!"))
        
        return set_outcome
    
    inputsVar = None
    outputsVar = None
    def inference_loop(self, invFuzz, daysFuzz, demandFuzz):
        # Initialize aggregated outputs
        quantity_agg = np.zeros_like(self.x_quantity)
        urgency_agg = np.zeros_like(self.x_urgency)

        for ind, rule in enumerate(self.rule_list):
            self.inputsVar = rule[0]
            self.outputsVar = rule[1]
            quan, urg = self.inference(ind, invFuzz, daysFuzz, demandFuzz)

            # Aggregate outputs using max operation
            quantity_agg = np.fmax(quantity_agg, quan)
            urgency_agg = np.fmax(urgency_agg, urg)

        return quantity_agg, urgency_agg

    def inference(self, ind:int, invFuzz=None, daysFuzz=None, demandFuzz=None):
        # Input states
        stateDemand = self.inputsVar[0]
        stateInv = self.inputsVar[1]
        stateDay = self.inputsVar[2]

        # Output fuzzy sets
        outputQuantity = self.get_set(self.outputsVar[0], "Quantity")
        outputUrgent = self.get_set(self.outputsVar[1], "Urgency")

        # Handle fuzzy input states
        getDemandFuzz = np.ones_like(self.x_demand) if stateDemand is None else demandFuzz[stateDemand]
        getInvFuzz = np.ones_like(self.x_inventory) if stateInv is None else invFuzz[stateInv]
        getDaysFuzz = np.ones_like(self.x_days_before_expiry) if stateDay is None else daysFuzz[stateDay]

        # Apply fuzzy AND to compute rule strength
        rule_strength = np.fmin(getDemandFuzz, np.fmin(getInvFuzz, getDaysFuzz))
        print((ind, rule_strength))

        # Compute resulting fuzzy sets for outputs
        quantity_res = np.fmin(rule_strength, outputQuantity)
        urgency_res = np.fmin(rule_strength, outputUrgent)

        return quantity_res, urgency_res

    
    def defuzzification(self, aggregated_quantity, aggregated_urgency, method1, method2):
        # Continous output
        urgency_result = fuzz.defuzz(self.x_urgency, aggregated_urgency, method2)
        
        # For quantity (discrete output)
        if method1 == 'centroid':
            sum_agg = sum(aggregated_quantity)
            if sum_agg == 0:
                quantity_result = 0
            else:
                sum_numerator = sum([self.x_quantity[i] * aggregated_quantity[i] for i in range(len(self.x_quantity))])
                quantity_result = sum_numerator / sum_agg
        elif method1 == 'bisector':
            cumulative_sum = 0
            total_area = sum(aggregated_quantity)
            for i, val in enumerate(aggregated_quantity):
                cumulative_sum += val
                if cumulative_sum >= total_area / 2:
                    quantity_result = self.x_quantity[i]
                    break
        elif method1 == 'som':
            quantity_result = self.x_quantity[next(i for i, val in enumerate(aggregated_quantity) if val == max(aggregated_quantity))]
        elif method1 == 'lom':
            quantity_result = self.x_quantity[next(i for i, val in reversed(list(enumerate(aggregated_quantity))) if val == max(aggregated_quantity))]
        elif method1 == 'mom':
            max_val = max(aggregated_quantity)
            indices = [i for i, val in enumerate(aggregated_quantity) if val == max_val]
            quantity_result = sum([self.x_quantity[i] for i in indices]) / len(indices)
        else:
            raise ValueError("Invalid defuzzification method for quantity.")
        
        return quantity_result, urgency_result
    

def plotting_subplot(axes, x, functionList, clr, labels, title, kind, symbol, index, fuzzfied_vals=None, aggregated=None, specific_x_val=None):
    """
    Plots membership functions, fuzzified values, and aggregated curves on a subplot.

    Parameters:
        axes (ndarray): Array of subplot axes.
        x (array-like): x-axis values.
        functionList (list): List of membership functions to plot.
        clr (list): List of colors for each function.
        labels (list): List of labels for each function.
        title (str): Title for the subplot.
        kind (str): Type of variable (e.g., 'Inventory', 'Demand').
        symbol (str): Symbol to represent the variable.
        index (int): Index of the subplot in the grid.
        fuzzfied_vals (dict, optional): Dictionary of fuzzy degrees (e.g., {'Low': 0.1, 'Medium': 0.9}).
        aggregated (array-like, optional): Aggregated fuzzy curve to plot.
        specific_x_val (float, optional): Specific x-value to highlight on the plot.
    """
    # Determine the subplot
    if axes.ndim == 1:
        ax = axes[index]
    else:
        rows, cols = axes.shape
        row = index // cols
        col = index % cols
        ax = axes[row, col]

    # Plot membership functions
    for ind, func in enumerate(functionList):
        ax.plot(x, func, color=clr[ind], label=labels[ind])

    # Plot fuzzified values
    if fuzzfied_vals:
        for key, val in fuzzfied_vals.items():
            ax.plot([min(x),max(x)], [val, val], color='purple', label=f"{key}: {val:.2f}")

    # Plot and shade aggregated curve
    if aggregated is not None:
        ax.plot(x, aggregated, color='purple', label="Aggregated")
        ax.fill_between(x, aggregated, color='purple', alpha=0.3)

    # Plot specific x-value
    if specific_x_val is not None:
        y_min, y_max = ax.get_ylim()
        ax.plot([specific_x_val, specific_x_val], [y_min, y_max], color='brown', linestyle='--', label=f"Value = {specific_x_val}")

    # Set title, grid, and labels
    ax.set_title(f"{title} ({kind} {symbol})", fontsize=10)
    ax.grid(True)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("Membership Degree")

    # Adjust legend
    # ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.1, 1))


def get_valid_input(prompt, datatype):
    while True:
        try:
            user_input = input(prompt).strip()

            # Check if the user wants to exit
            if user_input.lower() == "exit":
                print("Ending the program.")
                sys.exit()

            # Handle different data types
            if datatype == 'float':
                value = float(user_input)
            elif datatype == 'int':
                value = int(user_input)
            else:
                raise ValueError("Unsupported datatype provided.")

            return value
        except ValueError:
            print("Invalid input. Please enter a valid numeric value.")


system = FLSInventoryManagement()
x = [system.get_variables("Demand")[0],system.get_variables("Inventory")[0], system.get_variables("Days")[0],
                system.get_variables("Quantity")[0], system.get_variables("Urgency")[0]]
func = [system.get_variables("Demand")[1],system.get_variables("Inventory")[1], system.get_variables("Days")[1],
                    system.get_variables("Quantity")[1], system.get_variables("Urgency")[1]]
clrs = [['r','g','b'],['r','g','b'],['r','g','b'],['k','r','g','b'], ['k','r','g','b']]
labels = [['Low','Medium','High'],['Low','Medium','High'],['Low','Medium','High'],['Zero','Low','Medium','High'],['Zero','Low','Medium','High']]
title = ["Demand","Inventory","Days Before Expired","Reorder Quantity", "Reorder Urgency"]
kind = ['Input','Input','Input','Output','Output']
symbol = [" RM in k",' Units', ' Days', ' Units', ' %']


method_quantity = 'centroid'
method_urgency = 'som'
while(True):
    input_demand = get_valid_input("Insert the demand (RM in k): ", 'float')
    input_inventory = get_valid_input("Insert the inventory of quantity for a specified product (units): ", 'int')
    input_days = get_valid_input("Insert the days of that product before it expires (days): ", 'int')

    fuzz_demand = system.interpert_membership_inputs(np.clip(input_demand, min(system.x_demand), max(system.x_demand)), "Demand")
    fuzz_inv = system.interpert_membership_inputs(np.clip(input_inventory, min(system.x_inventory), max(system.x_inventory)), "Inventory")
    fuzz_days = system.interpert_membership_inputs(np.clip(input_days, min(system.x_days_before_expiry), max(system.x_days_before_expiry)), "Days")

    quantity_result, urgency_result = system.get_result(input_inventory, input_days, input_demand, method_quantity, method_urgency)
    quantity_agg, urgency_agg = system.get_aggregated(input_inventory, input_days, input_demand)


    print("\nInputs")
    print("Demand (RM in k): ", input_demand)
    print("Inventory (Units): ", input_inventory)
    print("Days Before Expired (Days):", input_days)
    print()
    print("Outputs")
    print("Reorder Quantity (Units): ", quantity_result)
    print("Reorder Urgency (%): ", urgency_result)
    print()
    print()

    # Example inputs
    fuzzfied_vals = [fuzz_demand, fuzz_inv, fuzz_days, None, None]
    aggregated = [None,None,None,quantity_agg,urgency_agg]
    specific_x_val = [input_demand,input_inventory, input_days, quantity_result, urgency_result]

            # Create subplots
    rows = 2  # Number of rows in the grid
    cols = 3  # Number of columns in the grid
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    axes = axes.flatten()  # Flatten axes for easier access   

                # Loop through and plot on subplots
    for i in range(len(x)):
                plotting_subplot(axes, x[i], func[i], clrs[i], labels[i], title[i], kind[i], symbol[i], i, fuzzfied_vals[i], aggregated[i], specific_x_val[i])

            # Remove any extra subplots (if len(x) < rows * cols)
    for j in range(len(x), rows * cols):
                fig.delaxes(axes[j])

            # Adjust layout
    plt.tight_layout()
    plt.show() 
    
