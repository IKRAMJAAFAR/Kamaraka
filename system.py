import numpy as np
import skfuzzy as fuzz
from functools import reduce

class FLSInventoryManagement:
    x_inventory = np.arange(0, 51, 1) # Input 1
    x_days_before_expiry = np.arange(0, 51, 1) # Input 2
    x_quantity = np.arange(0, 51, 1) # Output 1
    x_urgency = np.arange(0, 100.01, 0.01) # Output 2

    basic_rule_list = [
        # Inventory, Days - Reorder_q, Reorder_urgency
        #[['Low','Low'],['High','High']],
        #[['Medium', 'Low'],['Medium', 'High']],
        # [['High', 'Low'],['Medium','High']],
        [['Low', 'Medium'],['High','Medium']],
        [['Medium','Medium'],['Medium','Medium']],
        [['High','Medium'],['Low','Low']],
        # [['Low','High'],['Medium','Low']],
        # [['Medium','High'],['Low','Zero']],
        # [['High','High'],['Zero','Zero']]
    ]

    def __init__(self):
        self.construct_fuzzy_system()
    
    def construct_fuzzy_system(self):
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
        urgent_low = fuzz.trapmf(x, [5, 5, 5, 30])
        urgent_med = fuzz.trapmf(x, [5, 45, 55, 95])
        urgent_hig = fuzz.trapmf(x, [65, 95, 100, 100])

        self.urgency = [urgent_zero, urgent_low, urgent_med, urgent_hig]
    
    def get_result(self, inv_input:int, days_input:int, method_quantity:str, method_urgency:str):
        aggregated_quantity, aggregated_urgency = self.get_aggregated(inv_input, days_input)
        quantity_result, urgency_result = self.defuzzification(aggregated_quantity, aggregated_urgency, method_quantity, method_urgency)
        return quantity_result, urgency_result
    
    def get_aggregated(self, inv_input:int, days_input:int):
        invFuzz, daysFuzz = self.fuzzification(inv_input, days_input)
        quantity_agg, urgency_agg = self.inference_loop(invFuzz, daysFuzz)
        # Aggregate all quantity rules
        aggregated_quantity = reduce(np.fmax, quantity_agg)
        # Aggregate all urgency rules
        aggregated_urgency = reduce(np.fmax, urgency_agg)
        return aggregated_quantity, aggregated_urgency

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
        else:
            raise(ValueError("Invalid variables!"))
        return x, func
    
    def fuzzification(self, inv_input:int, days_input:int):
        def interpert_membership_inputs(val, kind):
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
            return result
        
        inv_input = np.clip(inv_input, min(self.x_inventory), max(self.x_inventory))
        days_input = np.clip(days_input, min(self.x_days_before_expiry), max(self.x_days_before_expiry))
        invFuzz = interpert_membership_inputs(inv_input, "Inventory")
        dayFuzz = interpert_membership_inputs(inv_input, "Days")
        return invFuzz, dayFuzz
    
    inputsVar = None
    outputsVar = None
    def inference_loop(self, invFuzz, daysFuzz):
        quantity_agg = []
        urgency_agg = []
        for ind in range(len(self.basic_rule_list)):
            self.inputsVar = self.basic_rule_list[ind][0]
            self.outputsVar = self.basic_rule_list[ind][1]
            quan, urg = self.inference(invFuzz, daysFuzz)
            quantity_agg.append(quan)
            urgency_agg.append(urg)
        
        rule = invFuzz['Low']
        quantity_res = np.fmin(rule, self.quantity[3])
        urgency_res = np.fmin(rule, self.urgency[3])
        quantity_agg.append(quantity_res)
        urgency_agg.append(urgency_res)

        rule = daysFuzz['Low']
        quantity_res = np.fmin(rule, self.quantity[3])
        urgency_res = np.fmin(rule, self.urgency[3])
        quantity_agg.append(quantity_res)
        urgency_agg.append(urgency_res)

        rule = invFuzz['High']
        quantity_res = np.fmin(rule, self.quantity[0])
        urgency_res = np.fmin(rule, self.urgency[0])
        quantity_agg.append(quantity_res)
        urgency_agg.append(urgency_res)

        rule = daysFuzz['High']
        quantity_res = np.fmin(rule, self.quantity[0])
        urgency_res = np.fmin(rule, self.urgency[0])
        quantity_agg.append(quantity_res)
        urgency_agg.append(urgency_res)

        return quantity_agg, urgency_agg
    
    def inference(self, invFuzz, daysFuzz):
            def intialize_quantity(outputQuantity):
                if outputQuantity == 'Low':
                    quantity_set = self.quantity[1]
                elif outputQuantity == 'Medium':
                    quantity_set = self.quantity[2]
                elif outputQuantity == 'High':
                    quantity_set = self.quantity[3]
                else:
                    quantity_set = self.quantity[0]
                return quantity_set

            def intialize_urgency(outputUrgent):
                if outputUrgent == 'Low':
                    urgency_set = self.urgency[1]
                elif outputUrgent == 'Medium':
                    urgency_set = self.urgency[2]
                elif outputUrgent == 'High':
                    urgency_set = self.urgency[3]
                else:
                    urgency_set = self.urgency[0]
                return urgency_set
            
            inputsInv = self.inputsVar[0]
            inputsDay = self.inputsVar[1]
            outputQuantity = intialize_quantity(self.outputsVar[0])
            outputUrgent = intialize_urgency(self.outputsVar[1])

            rule = np.fmin(invFuzz[inputsInv], daysFuzz[inputsDay])
            quantity_res = np.fmin(rule, outputQuantity)
            urgency_res = np.fmin(rule, outputUrgent)

            return quantity_res, urgency_res
    
    def defuzzification(self, aggregated_quantity, aggregated_urgency, method1, method2):
        quantity_result = fuzz.defuzz(self.x_quantity, aggregated_quantity, method1)
        urgency_result = fuzz.defuzz(self.x_urgency, aggregated_urgency, method2)

        return round(quantity_result), round(urgency_result,2)