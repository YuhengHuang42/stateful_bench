from typing import Callable, Any, Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import copy
import random
from faker import Faker
from loguru import logger
from collections import defaultdict
from collections import OrderedDict
import requests
import json
import traceback
import torch

from Sgenerator.utils import get_nested_path_string
from Sgenerator.state import State, Transition, Schema, RandomInitializer, USER_FUNCTION_PARAM_FLAG, RESPONSE_VARIABLE_TEMP
from Sgenerator.state import INDENT, RESULT_NAME, ProgramEvaluator, LocalVariable

MAX_LINEAR_FEATURES = 32
MAX_KERNEL_SIZE = 7
MAX_CHANNELS = 64
FLOAT_THRESHOLD = 1e-2

TENSOR_SUMMARY_PROMPT = '''The below code is about a set of PyTorch tensor manipulation APIs that provide essential operations for deep learning and scientific computing. 
These APIs enable efficient reshaping, transformation, and combination of multi-dimensional arrays (tensors), 
which are fundamental for building and training neural networks.

API Functions:
conv2d - Applies a 2D convolution over input tensors, commonly used in image processing and convolutional neural networks (CNNs) to extract spatial features.
permute - Returns a view of the input tensor with its dimensions reordered, useful for changing the arrangement of data without copying.
split - Splits a tensor into multiple chunks along a specified dimension, either into equal parts or custom sizes, facilitating batch processing or data partitioning.
cat - Concatenates a sequence of tensors along a given dimension, allowing for the merging of data from different sources or model outputs.
linear - Applies a linear (fully connected) transformation to the input tensor, a core operation in neural network layers for feature transformation.
transpose - Swaps two specified dimensions of a tensor, often used in matrix operations and for aligning data shapes.
'''

TENSOR_GENERATION_PROMPT = '''All APIs, Pytorch library, user-related variables, and constants have been preloaded into memory and are available for direct use.
Please begin your Python code generation with a code block (with ```), for example:
```
response_1 = torch.cat((user_variable, user_variable), 0)
```
Your code:
'''

def normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key in parameters:
        if isinstance(parameters[key], LocalVariable):
            result[key] = parameters[key].name
        elif isinstance(parameters[key], TensorState):
            result[key] = parameters[key].identifier
        else:
            result[key] = parameters[key]
    return result

class TensorRandomInitializer(RandomInitializer):
    def __init__(self):
        super().__init__()
    
    def random_generate_state(self, dim_range=None):
        # Generate random dimensions
        if dim_range is not None:
            lower = dim_range[0]
            upper = dim_range[1]
        else:
            lower = 1
            upper = 64
        dim1 = random.randint(lower, upper)
        dim2 = random.randint(lower, upper)
        
        dim3 = random.randint(max(lower // 2, 1), max(upper // 2, 2)) * 2  # Generate even number between 2 and 64
        dim4 = dim3  # Third and fourth dimensions are equal
        
        # Create tensor with random values using torch.randn
        tensor_data = torch.randn(dim1, dim2, dim3, dim4)
        return tensor_data

class TensorState(State):
    def __init__(self,
                 name:str,
                 data: Optional[Any] = None):
        '''
        Other tensor information is hidden from the program instead of its name and data (print(data)).
        While the shape can be obtained by print(data.shape), the LLM cannot directly access it.
        '''
        super().__init__(identifier=name)
        self.initial_value = {
            "data": data,
            "shape": data.shape,
            "dtype": data.dtype,
        }
        self.current_value = copy.deepcopy(self.initial_value)

    def __str__(self):
        return f"Tensor(name={self.name}, data={self.current_value})"
    
    def get_current_value(self):
        return_value = copy.deepcopy(self.current_value)
        return return_value

class TensorVariableSchema(Schema):
    def __init__(self):
        super().__init__()
        self.local_states = {
             "variables": [],
        }
        self.implicit_states = {
            "tensor_info": {},
            "latest_call": {},
        }
        self.transitions = [
            PermuteTransition,
            SplitTransition,
            CatTransition,
            TransposeTransition,
            Conv2dTransition,
            LinearTransition
        ]
        self.local_call_map = {}
        self.implicit_call_map = {}
        self.init_implict_dict = {}
        self.init_tensor_counter = 0
        self.init_weight_counter = 0
    
    def add_implicit_variable(self, implicit_variable: Any, latest_call: int):
        assert implicit_variable.identifier not in self.implicit_states["tensor_info"]
        self.implicit_states["tensor_info"][implicit_variable.identifier] = implicit_variable
        self.implicit_states["latest_call"][implicit_variable.identifier] = latest_call
        self.init_implict_dict[implicit_variable.identifier] = implicit_variable
    
    def add_local_variable(self, local_variable: Any, update_implicit=False):
        self.local_states["variables"].append(local_variable)
        if update_implicit:
            state = TensorState(
                name=local_variable.name,
                data=local_variable.value
            )
            self.add_implicit_variable(state, local_variable.latest_call)
            
    def prepare_initial_state(self, random_generator: TensorRandomInitializer, config: Dict[str, Any], random_generate_config: Dict[str, Any]):
        self.clear_state()
        
        local_state_num = random.randint(config["init_local_state_num_range"][0], config["init_local_state_num_range"][1])
        for i in range(local_state_num):
            if "dim_range" in random_generate_config:
                state = random_generator.random_generate_state(dim_range=random_generate_config["dim_range"])
            else:
                state = random_generator.random_generate_state()
            tensor_name = f"user_tensor_{self.init_tensor_counter}"
            self.init_tensor_counter += 1
            local_variable = LocalVariable(
                name=tensor_name,
                value=state,
                latest_call=0,
                created_by=USER_FUNCTION_PARAM_FLAG,
                updated=True
            )
            self.add_local_variable(local_variable, update_implicit=True)
            #self.add_implicit_variable(implicit_variable, 0)
            self.init_load_info[tensor_name] = state
            
        self.align_initial_state()
    
    def clear_state(self):
        self.local_states["variables"] = []
        self.implicit_states["tensor_info"] = {}
        self.implicit_states["latest_call"] = {}
        self.init_implict_dict = {}
        self.implicit_call_map = {}
        self.local_call_map = {}
        self.init_load_info = {}
        self.init_local_info = []
        self.init_tensor_counter = 0
        self.init_weight_counter = 0
    
    def get_implicit_states(self, current_value: bool = True):
        result = {}
        if current_value is True:
            tensor_info = self.implicit_states["tensor_info"]
            for key, value in tensor_info.items():
                result[key] = value.current_value
        else:
            tensor_info = self.init_implict_dict
            for key, value in tensor_info.items():
                result[key] = value
        return result
    
    def align_initial_state(self):
        pass
    
    def get_latest_call_map(self):
        """Create mappings of latest_call timestamps to variable indices/IDs"""
        local_call_map = defaultdict(list)
        for idx, var in enumerate(self.local_states["variables"]):
            local_call_map[var.latest_call].append(idx)
        implicit_call_map = defaultdict(list)
        for tensor_name in self.implicit_states["tensor_info"]:
            latest_call = self.implicit_states["latest_call"][tensor_name]
            implicit_call_map[latest_call].append(tensor_name)
        
        self.local_call_map = local_call_map
        self.implicit_call_map = implicit_call_map
        
        return local_call_map, implicit_call_map
        
    
    def transform_parameters_to_str(self, parameters: Dict[str, Any]):
        '''
        Transform the parameters to the format that the transition expects.
        '''
        result = ""
        for key in sorted(parameters.keys()):
            value = parameters[key]
            if isinstance(value, torch.Tensor):
                result += f"{key}={value.shape}, "
            elif isinstance(value, LocalVariable):
                result += f"{key}={value.value.shape}, "
            elif isinstance(value, TensorState):
                result += f"{key}={value.current_value['data'].shape}, "
            else:
                result += f"{key}={value}, "
        return result[:-2]
    
    def find_local_variable_by_name(self, name: str):
        for local_idx in range(len(self.local_states["variables"])-1, -1, -1):
            if self.local_states["variables"][local_idx].name == name:
                return local_idx
        return None
    
    def determine_whether_to_keep_pair(self, 
                                       previous_transition_info: Tuple,
                                       current_transition_info: Tuple):
        '''
        previous_transition_info: Tuple: (previous_transition_name, previous_transition_parameters)
        current_transition_info: Tuple: (current_transition_name, current_transition_parameters)
        '''
        return True
    
    def get_available_transitions(self, 
                                  random_generator: Any, 
                                  current_call: int, 
                                  max_call: int, 
                                  duplicate_local_variable_map: Dict[str, Set[str]], 
                                  previous_transition_info: Tuple):
        """
        Return:
            {
                "required_parameters": target_parameters, # Dict
                "latest_call": local_variable.latest_call,
                "whether_updated": local_variable.updated,
                "producer_variable_idx": idx,
                "transition_pairs": transition_pairs,
                "local_variable_idx": idx,
            }
        """
        available_transitions = {}
        self.get_latest_call_map()
        # Outer loop: transition
        # Inner loop: variables
        # Check:
        # 1. Whether the transition is valid.
        # 2. whether it passes determine_whether_to_keep_pair
        # 3. Whether the transition passes duplicate_local_variable_map
        # 4. For the same transition, if there is updated variable as its parameter, we should choose it.
        for transition in self.transitions:
            if transition.__name__ not in duplicate_local_variable_map:
                duplicate_local_variable_map[transition.__name__] = set([])
            if transition.__name__ not in available_transitions:
                available_transitions[transition.__name__] = []
            for tensor_name in self.implicit_states["tensor_info"]:
                tensor_variable = self.implicit_states["tensor_info"][tensor_name]
                local_variable_idx = self.find_local_variable_by_name(tensor_variable.identifier)
                local_variable = self.local_states["variables"][local_variable_idx]
                input_shape = tensor_variable.current_value["data"].shape
                updated = local_variable.updated
                if transition.__name__ == "Conv2dTransition":
                    # [["input", "weight", "stride", "padding", "dilation"]]
                    # First, check if the current variables can serve as the parameters.
                    create_new = random.random() < 0.5
                    whether_valid = False
                    if create_new:
                        for lvar in self.local_states["variables"]:
                            lvar_shape = lvar.value.shape
                            whether_valid, parameters = Conv2dTransition.is_possible(input_shape, lvar_shape)
                            if whether_valid:
                                updated = updated or lvar.updated
                                parameters["weight"] = lvar
                                break
                    if not whether_valid:
                        whether_valid, parameters = Conv2dTransition.generate_valid_parameters(input_shape)
                    if not whether_valid:
                        continue
                    
                    parameters["input"] = local_variable
                    parameters_str = self.transform_parameters_to_str(parameters)
                    if parameters_str in duplicate_local_variable_map[transition.__name__]:
                        continue
                    if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, parameters)):
                        continue
                    transition_pairs = [self.form_pair_transition(tensor_variable, transition.__name__)]
                    target_parameters = {
                        "required_parameters": parameters,
                        "latest_call": local_variable.latest_call,
                        "whether_updated": updated,
                        "producer_variable_idx": local_variable_idx, # From where
                        "transition_pairs": transition_pairs,
                        "local_variable_idx": local_variable_idx,
                    }
                    available_transitions[transition.__name__].append(target_parameters)
                elif transition.__name__ == "LinearTransition":
                    # [["input", "weight"]]
                    # First, check if the current variables can serve as the parameters.
                    create_new = random.random() < 0.5
                    whether_valid = False
                    if create_new:
                        for lvar in self.local_states["variables"]:
                            lvar_shape = lvar.value.shape
                            whether_valid, parameters = LinearTransition.is_possible(input_shape, lvar_shape)
                            if whether_valid:
                                updated = updated or lvar.updated
                                parameters["weight"] = lvar
                                break
                    if not whether_valid:
                        whether_valid, parameters = LinearTransition.generate_valid_parameters(input_shape)
                    if not whether_valid:
                        continue
                    parameters["input"] = local_variable
                    parameters_str = self.transform_parameters_to_str(parameters)
                    if parameters_str in duplicate_local_variable_map[transition.__name__]:
                        continue
                    if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, parameters)):
                        continue
                    transition_pairs = [self.form_pair_transition(tensor_variable, transition.__name__)]
                    target_parameters = {
                        "required_parameters": parameters,
                        "latest_call": local_variable.latest_call,
                        "whether_updated": updated,
                        "producer_variable_idx": local_variable_idx, # From where
                        "transition_pairs": transition_pairs,
                        "local_variable_idx": local_variable_idx,
                    }
                    available_transitions[transition.__name__].append(target_parameters)
                elif transition.__name__ == "TransposeTransition":
                    whether_valid, parameters = TransposeTransition.generate_valid_parameters(input_shape)
                    if not whether_valid:
                        continue
                    parameters["input"] = local_variable
                    parameters_str = self.transform_parameters_to_str(parameters)
                    if parameters_str in duplicate_local_variable_map[transition.__name__]:
                        continue
                    if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, parameters)):
                        continue
                    transition_pairs = [self.form_pair_transition(tensor_variable, transition.__name__)]
                    target_parameters = {
                        "required_parameters": parameters,
                        "latest_call": local_variable.latest_call,
                        "whether_updated": updated,
                        "producer_variable_idx": local_variable_idx, # From where
                        "transition_pairs": transition_pairs,
                        "local_variable_idx": local_variable_idx,
                    }
                    available_transitions[transition.__name__].append(target_parameters)
                elif transition.__name__ == "PermuteTransition":
                    whether_valid, parameters = PermuteTransition.generate_valid_parameters(input_shape)
                    if not whether_valid:
                        continue
                    parameters["input"] = local_variable
                    parameters_str = self.transform_parameters_to_str(parameters)
                    if parameters_str in duplicate_local_variable_map[transition.__name__]:
                        continue
                    if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, parameters)):
                        continue
                    transition_pairs = [self.form_pair_transition(tensor_variable, transition.__name__)]
                    target_parameters = {
                        "required_parameters": parameters,
                        "latest_call": local_variable.latest_call,
                        "whether_updated": updated,
                        "producer_variable_idx": local_variable_idx, # From where
                        "transition_pairs": transition_pairs,
                        "local_variable_idx": local_variable_idx,
                    }
                    available_transitions[transition.__name__].append(target_parameters)
                elif transition.__name__ == "SplitTransition":
                    whether_valid, parameters = SplitTransition.generate_valid_parameters(input_shape)
                    if not whether_valid:
                        continue
                    parameters["input"] = local_variable
                    parameters_str = self.transform_parameters_to_str(parameters)
                    if parameters_str in duplicate_local_variable_map[transition.__name__]:
                        continue
                    if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, parameters)):
                        continue
                    transition_pairs = [self.form_pair_transition(tensor_variable, transition.__name__)]
                    target_parameters = {
                        "required_parameters": parameters,
                        "latest_call": local_variable.latest_call,
                        "whether_updated": updated,
                        "producer_variable_idx": local_variable_idx, # From where
                        "transition_pairs": transition_pairs,
                        "local_variable_idx": local_variable_idx,
                    }
                    available_transitions[transition.__name__].append(target_parameters)
                    
                elif transition.__name__ == "CatTransition":
                    whether_valid = False
                    transition_pairs = []
                    for lvar in self.local_states["variables"]:
                        lvar_shape = lvar.value.shape
                        whether_valid, parameters = CatTransition.is_possible(input_shape, lvar_shape)
                        if whether_valid:
                            updated = updated or lvar.updated
                            parameters["tensors"] = [local_variable, lvar]
                            transition_pairs.append(self.form_pair_transition(self.implicit_states["tensor_info"][lvar.name], transition.__name__))
                            break
                    if not whether_valid:
                        # Directly create a similar tensor.
                        parameters = {}
                        new_var = torch.randn_like(local_variable.value)
                        parameters["tensors"] = [local_variable, new_var]
                        parameters["dim"] = random.randint(0, len(input_shape)-1)
                    parameter_str = self.transform_parameters_to_str(parameters)
                    if parameter_str in duplicate_local_variable_map[transition.__name__]:
                        continue
                    if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, parameters)):
                        continue
                    transition_pairs.append(self.form_pair_transition(tensor_variable, transition.__name__))
                    target_parameters = {
                        "required_parameters": parameters,
                        "latest_call": local_variable.latest_call,
                        "whether_updated": updated,
                        "producer_variable_idx": local_variable_idx, # From where
                        "transition_pairs": transition_pairs,
                        "local_variable_idx": local_variable_idx,
                    }
                    available_transitions[transition.__name__].append(target_parameters)
        
        # For the same transition, if there is updated variable as its parameter, we should choose it.
        for transition_name in available_transitions:
            this_transition_updated = []
            for target_parameters in available_transitions[transition_name]:
                if target_parameters["whether_updated"]:
                    this_transition_updated.append(target_parameters)
            if len(this_transition_updated) > 0:
                available_transitions[transition_name] = this_transition_updated
        return available_transitions

    def create_new_init_tensor(self, param_name, param_value):
        if "weight" in param_name:
            name = f"user_weight_{self.init_weight_counter}"
            self.init_weight_counter += 1
        else:
            name = f"user_tensor_{self.init_tensor_counter}"
            self.init_tensor_counter += 1
        # maintain the load info data structure.
        implicit_variable = TensorState(name, param_value)
        assert name not in self.implicit_states["tensor_info"]
        self.add_implicit_variable(implicit_variable, 0)
        local_variable = LocalVariable(
            name=name,
            value=param_value,
            latest_call=0,
            created_by=f"user",
            updated=False # It will be referenced by the transition, so we set it to False.
        )
        self.add_local_variable(local_variable)
        # Add init_load_info
        self.init_load_info[name] = param_value
        return local_variable
        
        
    def craft_transition(self, transition_info, calling_timestamp, transition, producer="None"):
        # Creating new local variables if necessary
        # Creating transition classes using transition_class = globals()[transition]
        # Update producer
        if producer == "None":
            producer = self.local_states["variables"][transition_info["producer_variable_idx"]]
            
        parameters = transition_info["required_parameters"]
        parameters_name = list(parameters.keys())
        for param_name in parameters_name:
            param_value = parameters[param_name]
            if isinstance(param_value, torch.Tensor):
                # Create a new local variable.
                new_local_variable = self.create_new_init_tensor(param_name, param_value)
                parameters[param_name] = new_local_variable
            if isinstance(param_value, List) or isinstance(param_value, Tuple):
                for idx, item in enumerate(param_value):
                    if isinstance(item, torch.Tensor):
                        new_local_variable = self.create_new_init_tensor(f"{param_name}_{idx}", item)
                        param_value[idx] = new_local_variable
                parameters[param_name] = param_value
                
        transition_class = globals()[transition]
        new_transition = transition_class(
            parameters=parameters, 
            calling_timestamp=calling_timestamp
        )
        new_transition.producer = producer
        return new_transition

    def postprocess_transitions(self, remaining_call: int) -> Tuple[bool, List[str]]:
        '''
        No need to postprocess. We do not have remote database to be updated.
        '''
        return False, []
    
    def postprocess_choose_result(self):
        result_var_list = []
        for idx in range(len(self.local_states["variables"])-1, -1, -1):
            local_variable = self.local_states["variables"][idx]
            #transitions = local_variable.value.transitions
            if local_variable.updated == True:
                # Either being updated or being queried.
                #result_str = f"{RESULT_NAME} = {local_variable.name}"
                result_var_list.append(local_variable)
        if len(result_var_list) == 0:
            return None
        elif len(result_var_list) == 1:
            return f"{RESULT_NAME} = {result_var_list[0].name}"
        else:
            result_str = f"{RESULT_NAME} = "
            for idx, result_var in enumerate(result_var_list):
                result_str += f"{result_var.name}.sum() + "
            result_str = result_str[:-3]
            return result_str
    
    def get_load_info(self, init_load_info=None):
        # Return: program_string, list of (variable_name, variable_value)
        if init_load_info is None:
            init_load_info = self.init_load_info
        if init_load_info is None or len(init_load_info) == 0:
            return None, None
        init_program = "# == The variables below will be pre-loaded into the memory. Here we only show the shape of the variables. ==\n"
        all_init_pairs = []
        for name, value in init_load_info.items():
            init_program += f"{name} # shape: {value.shape}\n"
            all_init_pairs.append((name, value))
        init_program += "# == The variables above will be pre-loaded into the memory at evaluation runtime. ==\n"
        return init_program, all_init_pairs
    
    def obtain_if_condition(self):
        """
        Obtain the condition for the if-else transition.
         (left_variable_name, index, right_value)
         Return:
            if_condition: The condition for the if-else transition.
            whether_replace_by_variable: Whether the if-else transition is replaced by a variable.
            additional_content: Additional statement for the if-else transition.
        """
        condition_types = [
            'shape', 'value', 'dtype'
        ]
        shape_sub_types = ["single", "all", "every"]
        if_condition = None
        whether_replace_by_variable = True
        for idx in range(len(self.local_states["variables"])-1, -1, -1):
            local_var = self.local_states["variables"][idx]
            
            tensor = local_var.value
            
            cond_type = random.choice(condition_types)
            
            if cond_type == 'shape':
                shape_sub_type = random.choice(shape_sub_types)
                if shape_sub_type == "single":
                    single_dim = random.randint(0, len(tensor.shape)-1)
                    if_condition = (f"{local_var.name}.shape", [single_dim], tensor.shape[single_dim])
                elif shape_sub_type == "all":
                    if_condition = (f"len({local_var.name}.shape)", None, len(tensor.shape))
                elif shape_sub_type == "every":
                    if_condition = (f"list({local_var.name}.shape)", None, list(tensor.shape))
            elif cond_type == "dtype":
                if_condition = (f"str({local_var.name}.dtype)", None, str(tensor.dtype))
            elif cond_type == "value":
                right_value = torch.sum(tensor > torch.mean(tensor)).item()
                if_condition = (f"torch.sum({local_var.name} > torch.mean({local_var.name}))", None, right_value)
            
        return if_condition, whether_replace_by_variable, None
    
class Conv2dTransition(Transition):
    """Handles tensor concatenation"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "input" in parameters
        assert isinstance(parameters["input"], LocalVariable)
        assert "weight" in parameters
        # Weight should be lazily initialized and a user-defined variable.
        # We will not track its transitions here.
        assert isinstance(parameters["weight"], LocalVariable)
        assert "stride" in parameters
        assert isinstance(parameters["stride"], int)
        assert "padding" in parameters
        assert isinstance(parameters["padding"], int) or isinstance(parameters["padding"], str)
        assert "dilation" in parameters
        assert isinstance(parameters["dilation"], int)
        
        
        super().__init__("Conv2dTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {
            "shape": None
        }

    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["input", "weight", "stride", "padding", "dilation"]]
    
    def get_effected_states(self, variable_schema: TensorVariableSchema) -> List[str]:
        local_states = []
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == self.parameters["input"].name:
                local_states.append(local_idx)
                break
        implicit_states = [variable_schema.implicit_states["tensor_info"][self.parameters["input"].name]]
        return implicit_states, local_states
    
    def apply(self, implicit_states: List[TensorState], local_states: List[str], variable_schema: TensorVariableSchema):
        assert len(implicit_states) == 1
        assert len(local_states) == 1
        old_tensor = variable_schema.implicit_states["tensor_info"][implicit_states[0].identifier]
        variable_schema.implicit_states["latest_call"][old_tensor.identifier] = self.calling_timestamp
        new_tensor_value = torch.nn.functional.conv2d(old_tensor.current_value["data"], 
                                                      self.parameters["weight"].value, 
                                                      stride=self.parameters["stride"], 
                                                      padding=self.parameters["padding"], 
                                                      dilation=self.parameters["dilation"]
                                                      )
        self.string_parameters["shape"] = new_tensor_value.shape
        new_tensor = TensorState(self.new_variable_name, new_tensor_value)
        new_tensor.transitions = copy.deepcopy(old_tensor.transitions)
        new_tensor.transitions.append({
            "name": self.name,
            "parameters": normalize_parameters(self.parameters),
        })
        variable_schema.add_implicit_variable(new_tensor, self.calling_timestamp)
        
        new_local_variable = LocalVariable(
            name=self.new_variable_name,
            value=new_tensor_value,
            latest_call=self.calling_timestamp,
            created_by=f"{self.name}@{self.calling_timestamp}",
            updated=True
        )
        # Latest call update
        variable_schema.local_states["variables"][local_states[0]].updated = False # Already been used.
        variable_schema.local_states["variables"][local_states[0]].latest_call = self.calling_timestamp
        variable_schema.add_local_variable(new_local_variable)
        
        return variable_schema
    
    def __str__(self):
        return f"torch.nn.functional.conv2d({self.parameters['input'].name}, {self.parameters['weight'].name}, stride={self.parameters['stride']}, padding={self.parameters['padding']}, dilation={self.parameters['dilation']})"

    def get_program_str(self) -> Tuple[List[str], str]:
        padding = self.parameters['padding'] if isinstance(self.parameters['padding'], int) else f"'{self.parameters['padding']}'"
        result = [
            f"{self.new_variable_name} = torch.nn.functional.conv2d({self.parameters['input'].name}, {self.parameters['weight'].name}, stride={self.parameters['stride']}, padding={padding}, dilation={self.parameters['dilation']}) # Output shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""

    @staticmethod
    def calculate_valid_parameters(input_shape: Tuple[int]):
        if len(input_shape) != 4:
            return None
        if input_shape[0] == 0:
            return None
        _, in_channels, in_height, in_width = input_shape
        max_kernel = min(MAX_KERNEL_SIZE, in_height, in_width) # input is 224x224 --> 7*7
        valid_ranges = {
            'weight': {
                'min_channels': 1,
                'max_channels': min(in_channels * 16, MAX_CHANNELS),  # Arbitrary reasonable limit
                'kernel_sizes': [(k, k) for k in range(1, max_kernel+1)] # input is 224x224 --> 7*7
            },
            'stride': {
                'min': 1,
                'max': min(in_height, in_width)
            },
            #'padding': ["valid", "same"],
            'dilation': {
                'min': 1,
                'max': (min(in_height, in_width) - 1) // 2
            }
        }
        valid_ranges['padding'] = {
                'valid': 0,
                'same': (max_kernel - 1) // 2  # Only works when stride=1
            }
        return valid_ranges
    
    @staticmethod
    def generate_valid_parameters(input_shape: Tuple[int]):
        '''
        Given input shape, generate valid parameters for conv2d.
        '''
        valid_range = Conv2dTransition.calculate_valid_parameters(input_shape)
        if valid_range is None:
            return False, None
        _, _, in_height, in_width = input_shape
        max_attempts = 10  # Prevent infinite loops
        valid_found = False
        
        for _ in range(max_attempts):
            # Generate parameters with constraints
            if valid_range['weight']['min_channels'] == valid_range['weight']['max_channels']:
                out_channels = valid_range['weight']['min_channels']
            elif valid_range['weight']['min_channels'] > valid_range['weight']['max_channels']:
                return False, None
            else:
                out_channels = random.randint(
                    valid_range['weight']['min_channels'],
                    valid_range['weight']['max_channels']
                )
            if valid_range['dilation']['min'] == valid_range['dilation']['max']:
                dilation = valid_range['dilation']['min']
            elif valid_range['dilation']['min'] < valid_range['dilation']['max']:
                dilation = random.randint(
                    valid_range['dilation']['min'],
                    valid_range['dilation']['max']
                )
            else:
                return False, None
            
            # Calculate max kernel size considering dilation
            max_kernel_h = (in_height - 1) // dilation + 1
            max_kernel_w = (in_width - 1) // dilation + 1
            valid_kernels = [
                k for k in valid_range['weight']['kernel_sizes']
                if k[0] <= max_kernel_h and k[1] <= max_kernel_w
            ]
            
            if len(valid_kernels) == 0:
                continue  # No valid kernels for this dilation
                
            kernel_size = random.choice(valid_kernels)
            padding = random.choice(list(valid_range['padding'].keys()))
            
            # Handle 'same' padding constraint
            if padding == 'same':
                stride = 1
                pad_h = (kernel_size[0] - 1) * dilation // 2
                pad_w = (kernel_size[1] - 1) * dilation // 2
            else:  # 'valid'
                max_stride = min(valid_range['stride']['max'], in_height, in_width)
                if valid_range['stride']['min'] == max_stride:
                    stride = valid_range['stride']['min']
                else:
                    stride = random.randint(
                        valid_range['stride']['min'],
                        max_stride
                    )
                pad_h = 0
                pad_w = 0
                
            # Calculate output dimensions
            out_h = (in_height + 2*pad_h - dilation*(kernel_size[0]-1)-1) // stride + 1
            out_w = (in_width + 2*pad_w - dilation*(kernel_size[1]-1)-1) // stride + 1
            
            if out_h > 0 and out_w > 0:
                valid_found = True
                break
                
        #if not valid_found:
        #    raise ValueError("Could not find valid parameters for input shape")
        
        if not valid_found:
            return False, None
            
        weight = torch.randn(out_channels, input_shape[1], kernel_size[0], kernel_size[1])
        return True, {
            "weight": weight,
            "stride": stride,
            "padding": padding,
            "dilation": dilation
        }

    @staticmethod
    def is_possible(input_shape: Tuple[int], weight_shape: Tuple[int]):
        """Determine if valid conv parameters exist for given input/weight"""
        if len(weight_shape) != 4 or len(input_shape) != 4:
            return False, None
            
        in_channels = input_shape[1]
        _, weight_in, kH, kW = weight_shape
        
        # Basic channel and kernel size checks
        if weight_in != in_channels:
            return False, None
        if kH < 1 or kW < 1:
            return False, None
            
        # With adjustable padding, convolution is always possible
        # Extract kernel dimensions
        _, _, kH, kW = weight_shape
        _, _, inH, inW = input_shape
        
        # Generate stride (1 to input size)
        max_stride = min(inH, inW)
        if max_stride == 1:
            stride = 1
        elif max_stride < 1:
            return False, None
        else:
            stride = random.randint(1, max_stride)
        
        # Generate dilation (1 to max that fits in input)
        max_dilationH = (inH - 1) // (kH - 1) if kH > 1 else 1
        max_dilationW = (inW - 1) // (kW - 1) if kW > 1 else 1
        min_dilation = min(max_dilationH, max_dilationW)
        if min_dilation < 1:
            return False, None
        elif min_dilation == 1:
            dilation = 1
        else:
            dilation = random.randint(1, min_dilation)
        
        # Calculate minimal required padding
        def calc_pad(in_dim, k_size):
            required = dilation * (k_size - 1) + 1 - in_dim
            return max(0, (required + 1) // 2) if required > 0 else 0
        
        padH = calc_pad(inH, kH)
        padW = calc_pad(inW, kW)
        
        parameters =  {
            "stride": stride,
            "padding": max(padH, padW),
            "dilation": dilation
        }
        return True, parameters
        


class PermuteTransition(Transition):
    """Handles dimension permutation"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "input" in parameters
        assert isinstance(parameters["input"], LocalVariable)
        assert "dims" in parameters
        super().__init__("PermuteTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {
            "shape": None
        }
    
    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["input", "dims"]]
    
    def get_effected_states(self, variable_schema: TensorVariableSchema) -> List[str]:
        local_states = []
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == self.parameters["input"].name:
                local_states.append(local_idx)
                break
        implicit_states = [variable_schema.implicit_states["tensor_info"][self.parameters["input"].name]]
        return implicit_states, local_states
    
    def apply(self, implicit_states: List[TensorState], local_states: List[str], variable_schema: TensorVariableSchema):
        assert len(implicit_states) == 1
        assert len(local_states) == 1
        old_tensor = variable_schema.implicit_states["tensor_info"][implicit_states[0].identifier]
        variable_schema.implicit_states["latest_call"][old_tensor.identifier] = self.calling_timestamp
        new_tensor_value = torch.permute(old_tensor.current_value["data"], self.parameters["dims"])
        self.string_parameters["shape"] = new_tensor_value.shape
        new_tensor = TensorState(self.new_variable_name, new_tensor_value)
        new_tensor.transitions = copy.deepcopy(old_tensor.transitions)
        new_tensor.transitions.append({
            "name": self.name,
            "parameters": normalize_parameters(self.parameters),
        })
        variable_schema.add_implicit_variable(new_tensor, self.calling_timestamp)
        new_local_variable = LocalVariable(
            name=self.new_variable_name,
            value=new_tensor_value,
            latest_call=self.calling_timestamp,
            created_by=f"{self.name}@{self.calling_timestamp}",
            updated=True
        )
        variable_schema.local_states["variables"][local_states[0]].updated = False # Already been used.
        variable_schema.local_states["variables"][local_states[0]].latest_call = self.calling_timestamp
        variable_schema.add_local_variable(new_local_variable)
        
        return variable_schema
    
    def __str__(self):
        return f"torch.permute({self.parameters['input'].name}, {self.parameters['dims']})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        result = [
            f"{self.new_variable_name} = torch.permute({self.parameters['input'].name}, {self.parameters['dims']}) # Output shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""
    

    @staticmethod
    def is_possible(input_shape: Tuple[int]):
        """Generate valid dimension permutation"""
        ndim = len(input_shape)
        
        #if ndim < 2:
        #    raise ValueError("Input tensor must have at least 2 dimensions for permutation")
        
        if ndim < 2:
            return False, None
        # Generate a random permutation of dimensions
        dims = list(range(ndim))
        random.shuffle(dims)
        
        # Ensure permutation is different from original
        if dims == list(range(ndim)):
            # Swap two random dimensions if we got the original order
            i, j = random.sample(range(ndim), 2)
            dims[i], dims[j] = dims[j], dims[i]
        
        return True, {'dims': tuple(dims)}
    
    @staticmethod
    def generate_valid_parameters(input_shape: Tuple[int]):
        """Generate valid dimension permutation parameters"""
        ndim = len(input_shape)
        
        if ndim < 2:
            return False, None
            
        # Generate a random permutation of dimensions
        dims = list(range(ndim))
        random.shuffle(dims)
        
        # Ensure permutation is different from original
        if dims == list(range(ndim)):
            # Swap two random dimensions if we got the original order
            i, j = random.sample(range(ndim), 2)
            dims[i], dims[j] = dims[j], dims[i]
        
        return True, {'dims': tuple(dims)}

class SplitTransition(Transition):
    """Handles tensor splitting"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "input" in parameters
        assert isinstance(parameters["input"], LocalVariable)
        assert "split_size_or_sections" in parameters
        #assert isinstance(parameters["split_size_or_sections"], Union[int, List[int], Tuple[int]])
        assert "dim" in parameters
        super().__init__("SplitTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {
            "shape": None
        }
    
    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["input", "split_size_or_sections", "dim"]]
    
    def get_effected_states(self, variable_schema: TensorVariableSchema) -> List[str]:
        local_states = []
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == self.parameters["input"].name:
                local_states.append(local_idx)
                break
        implicit_states = [variable_schema.implicit_states["tensor_info"][self.parameters["input"].name]]
        return implicit_states, local_states
    
    def apply(self, implicit_states: List[TensorState], local_states: List[str], variable_schema: TensorVariableSchema):
        assert len(implicit_states) == 1
        assert len(local_states) == 1
        old_tensor = variable_schema.implicit_states["tensor_info"][implicit_states[0].identifier]
        variable_schema.implicit_states["latest_call"][old_tensor.identifier] = self.calling_timestamp
        new_tensor_value = torch.split(old_tensor.current_value["data"], self.parameters["split_size_or_sections"], self.parameters["dim"]) # The returned value is a tuple of tensors.
        self.string_parameters["shape"] = []
        for idx, item in enumerate(new_tensor_value):
            self.string_parameters["shape"].append(item.shape)
            new_tensor_index_name = f"{self.new_variable_name}_{idx}"
            new_tensor = TensorState(new_tensor_index_name, item)
            new_tensor.transitions = copy.deepcopy(old_tensor.transitions)
            new_tensor.transitions.append({
                "name": self.name,
                "parameters": normalize_parameters(self.parameters),
            })
            variable_schema.add_implicit_variable(new_tensor, self.calling_timestamp)
            new_local_variable = LocalVariable(
                name=new_tensor_index_name,
                value=item,
                latest_call=self.calling_timestamp,
                created_by=f"{self.name}@{self.calling_timestamp}",
                updated=True
            )
            variable_schema.local_states["variables"][local_states[0]].updated = False # Already been used.
            variable_schema.local_states["variables"][local_states[0]].latest_call = self.calling_timestamp
            variable_schema.add_local_variable(new_local_variable)
        
        return variable_schema
    
    def __str__(self):
        return f"torch.split({self.parameters['input'].name}, {self.parameters['split_size_or_sections']}, {self.parameters['dim']})"

    def get_program_str(self) -> Tuple[List[str], str]:
        returned_left = "("
        for idx, shape in enumerate(self.string_parameters["shape"]):
            returned_left += f"{self.new_variable_name}_{idx}, "
        if len(self.string_parameters["shape"]) == 1:
            returned_left = returned_left + ")" # (a, ) = xxxxx
        else:
            returned_left = returned_left[:-2] + ")"
        returned_right = "shape: "
        for idx, shape in enumerate(self.string_parameters["shape"]):
            returned_right += f"{shape}, "
        returned_right = returned_right[:-2]
        result = [
            f"{returned_left} = torch.split({self.parameters['input'].name}, {self.parameters['split_size_or_sections']}, {self.parameters['dim']}) # {returned_right}\n",
        ]
        return result, ""

    @staticmethod
    def generate_valid_parameters(input_shape: Tuple[int]):
        """Generate valid split parameters based on input tensor shape"""
        ndim = len(input_shape)
        if ndim == 0:
            return False, None
            
        # Randomly select a dimension to split along
        dim = random.randint(0, ndim-1)
        dim_size = input_shape[dim]
        
        # Decide between split_size or sections
        if random.random() < 0.5 or dim_size < 2:
            # Split by size (integer)
            max_split_size = max(1, dim_size - 1)
            split_size = random.randint(1, max_split_size)
            return True, {
                'split_size_or_sections': split_size,
                'dim': dim
            }
        else:
            # Split into sections (list)
            num_splits = random.randint(2, min(4, dim_size))
            base_size = dim_size // num_splits
            remainder = dim_size % num_splits
            sections = [base_size + 1 if i < remainder else base_size 
                       for i in range(num_splits)]
            return True, {
                'split_size_or_sections': sections,
                'dim': dim
            }
    
class CatTransition(Transition):
    """Handles tensor concatenation"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "tensors" in parameters
        assert "dim" in parameters
        assert isinstance(parameters["dim"], int)
        tensor_shape = [parameters["tensors"][0].value.shape]
        for tensor in parameters["tensors"][1:]:
            assert isinstance(tensor, LocalVariable)
            for idx, item_shape in enumerate(tensor.value.shape):
                if idx == parameters["dim"]:
                    continue
                assert item_shape == tensor_shape[0][idx], f"The shape of the tensor {tensor.name} is not the same as the first tensor {parameters['tensors'][0].name} at dimension {idx}."
        super().__init__("CatTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {
            "shape": None
        }
    
    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["tensors", "dim"]]
    
    def get_effected_states(self, variable_schema: TensorVariableSchema) -> List[str]:
        local_states = []
        implicit_states = []
        target_tensor_names = [item.name for item in self.parameters["tensors"]]
        for tensor_name in target_tensor_names:
            for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
                c_name = variable_schema.local_states["variables"][local_idx].name
                if c_name == tensor_name:
                    local_states.append(local_idx)
                    break
            implicit_states.append(variable_schema.implicit_states["tensor_info"][tensor_name])
        return implicit_states, local_states

    def apply(self, implicit_states: List[TensorState], local_states: List[str], variable_schema: TensorVariableSchema):
        input_tensors = []
        longest_transitions = []
        for state in implicit_states:
            input_tensors.append(state.current_value["data"])
            variable_schema.implicit_states["latest_call"][state.identifier] = self.calling_timestamp
            if len(state.transitions) > len(longest_transitions):
                longest_transitions = copy.deepcopy(state.transitions)
        try:
            new_tensor_value = torch.cat(input_tensors, self.parameters["dim"])
        except Exception as e:
            for idx, item in enumerate(local_states):
                logger.error(f"Local state name: {variable_schema.local_states['variables'][item].name}, value: {variable_schema.local_states['variables'][item].value.shape}")
            raise Exception(f"Error in cat transition: {e}")
        self.string_parameters["shape"] = new_tensor_value.shape
        new_tensor = TensorState(self.new_variable_name, new_tensor_value)
        
        new_tensor.transitions = longest_transitions
        new_tensor.transitions.append({
            "name": self.name,
            "parameters": normalize_parameters(self.parameters),
        })
        variable_schema.add_implicit_variable(new_tensor, self.calling_timestamp)
        new_local_variable = LocalVariable(
            name=self.new_variable_name,
            value=new_tensor_value,
            latest_call=self.calling_timestamp,
            created_by=f"{self.name}@{self.calling_timestamp}",
            updated=True
        )
        for local_idx in local_states:
            variable_schema.local_states["variables"][local_idx].updated = False # Already been used.
            variable_schema.local_states["variables"][local_idx].latest_call = self.calling_timestamp
        variable_schema.add_local_variable(new_local_variable)
        return variable_schema
    
    def __str__(self):
        return f"torch.cat({', '.join([tensor.name for tensor in self.parameters['tensors']])}, {self.parameters['dim']})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        input_parameters = "("
        for tensor in self.parameters["tensors"]:
            input_parameters += f"{tensor.name}, "
        input_parameters = input_parameters[:-2] + ")"
        result = [
            f"{self.new_variable_name} = torch.cat({input_parameters}, {self.parameters['dim']}) # Output shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""

    @staticmethod
    def generate_valid_parameters(shape1: Tuple[int], shape2: Tuple[int]):
        """
        Generate valid concatenation parameters for two input shapes.
        Returns: (success, parameters) where parameters contains 'dim' if successful
        """
        # Check dimensional compatibility
        if len(shape1) != len(shape2):
            return False, None
            
        valid_dims = []
        # Find all dimensions where other dimensions match
        for dim in range(len(shape1)):
            valid = True
            for i in range(len(shape1)):
                if i != dim and shape1[i] != shape2[i]:
                    valid = False
                    break
            if valid:
                valid_dims.append(dim)
                
        if not valid_dims:
            return False, None
            
        # Randomly select a valid dimension
        selected_dim = random.choice(valid_dims)
        return True, {'dim': selected_dim}

    @staticmethod
    def is_possible(shape1: Tuple[int], shape2: Tuple[int]):
        """
        Check if two tensors can be concatenated along any dimension.
        Returns: (success, parameters) where parameters contains 'dim' if successful
        """
        if len(shape1) != len(shape2):
            return False, None

        valid_dims = []
        for dim in range(len(shape1)):
            valid = True
            for i in range(len(shape1)):
                if i != dim and shape1[i] != shape2[i]:
                    valid = False
                    break
            if valid:
                valid_dims.append(dim)

        if not valid_dims:
            return False, None

        return True, {'dim': random.choice(valid_dims)}


class LinearTransition(Transition):
    """Handles linear transformation"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "input" in parameters
        assert isinstance(parameters["input"], LocalVariable)
        assert "weight" in parameters
        assert isinstance(parameters["weight"], LocalVariable)
        super().__init__("LinearTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {
            "shape": None
        }
    
    @staticmethod
    def get_required_parameters() -> List[str]:
        # Weight should be lazily initialized.
        return [["input", "weight"]]
    
    def get_effected_states(self, variable_schema: TensorVariableSchema) -> List[str]:
        local_states = []
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == self.parameters["input"].name:
                local_states.append(local_idx)
                break
        implicit_states = [variable_schema.implicit_states["tensor_info"][self.parameters["input"].name]]
        return implicit_states, local_states
    
    def apply(self, implicit_states: List[TensorState], local_states: List[str], variable_schema: TensorVariableSchema):
        assert len(implicit_states) == 1
        assert len(local_states) == 1
        old_tensor = variable_schema.implicit_states["tensor_info"][implicit_states[0].identifier]
        variable_schema.implicit_states["latest_call"][old_tensor.identifier] = self.calling_timestamp
        new_tensor_value = torch.nn.functional.linear(old_tensor.current_value["data"], self.parameters["weight"].value)
        self.string_parameters["shape"] = new_tensor_value.shape
        new_tensor = TensorState(self.new_variable_name, new_tensor_value)
        new_tensor.transitions = copy.deepcopy(old_tensor.transitions)
        new_tensor.transitions.append({
            "name": self.name,
            "parameters": normalize_parameters(self.parameters),
        })
        variable_schema.add_implicit_variable(new_tensor, self.calling_timestamp)
        new_local_variable = LocalVariable(
            name=self.new_variable_name,
            value=new_tensor_value,
            latest_call=self.calling_timestamp,
            created_by=f"{self.name}@{self.calling_timestamp}",
            updated=True
        )
        variable_schema.local_states["variables"][local_states[0]].updated = False # Already been used.
        variable_schema.local_states["variables"][local_states[0]].latest_call = self.calling_timestamp
        variable_schema.add_local_variable(new_local_variable)
        return variable_schema
    
    def __str__(self):
        return f"torch.nn.functional.linear({self.parameters['input'].name}, {self.parameters['weight'].name})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        result = [
            f"{self.new_variable_name} = torch.nn.functional.linear({self.parameters['input'].name}, {self.parameters['weight'].name}) # Output shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""

    @staticmethod
    def calculate_valid_parameters(input_shape: Tuple[int]):
        """Valid parameters for linear transformation"""
        valid_ranges = {
            'weight': {
                'min_features': 2,
                'max_features': min(input_shape[-1] * 4, MAX_LINEAR_FEATURES)  # Allow up to 4x input features
            }
        }
        return valid_ranges

    @staticmethod
    def generate_valid_parameters(input_shape: Tuple[int]):
        """Generate valid weight matrix for linear transformation"""
        valid_range = LinearTransition.calculate_valid_parameters(input_shape)
        if valid_range['weight']['min_features'] > valid_range['weight']['max_features']:
            if valid_range["weight"]["min_features"] == valid_range["weight"]["max_features"]:
                out_features = valid_range["weight"]["min_features"]
            elif valid_range["weight"]["max_features"] == 1:
                out_features = 1
            else:
                return False, None
        out_features = random.randint(
            valid_range['weight']['min_features'],
            valid_range['weight']['max_features']
        )
        weight = torch.randn(out_features, input_shape[-1])
        return True, {'weight': weight}

    @staticmethod
    def is_possible(input_shape: Tuple[int], weight_shape: Tuple[int]):
        """Check if linear transformation is possible with given weight"""
        # Input must have at least 1 dimension (features dimension)
        # Input: (*, in_features)
        if len(input_shape) < 1:
            return False, None
            
        # Weight must be 2D: (out_features, in_features)
        if len(weight_shape) != 2:
            return False, None
            
        # Check if input's last dimension matches weight's in_features
        in_features = input_shape[-1]
        if weight_shape[1] != in_features:
            return False, None
            
        # No additional parameters needed for linear transformation
        return True, {}
    
class TransposeTransition(Transition):
    """Handles tensor transposition"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        # For Transpose, the dim is purely number instead of a variable.
        assert "input" in parameters
        assert isinstance(parameters["input"], LocalVariable)
        assert "dim0" in parameters
        assert isinstance(parameters["dim0"], int)
        assert "dim1" in parameters
        super().__init__("TransposeTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {
            "shape": None
        }
    
    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["input", "dim0", "dim1"]]
    
    def get_effected_states(self, variable_schema: TensorVariableSchema) -> List[str]:
        local_states = []
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == self.parameters["input"].name:
                local_states.append(local_idx)
                break
        implicit_states = [variable_schema.implicit_states["tensor_info"][self.parameters["input"].name]]
        return implicit_states, local_states

    def apply(self, implicit_states: List[TensorState], local_states: List[str], variable_schema: TensorVariableSchema):
        assert len(implicit_states) == 1
        assert len(local_states) == 1
        old_tensor = variable_schema.implicit_states["tensor_info"][implicit_states[0].identifier]
        variable_schema.implicit_states["latest_call"][old_tensor.identifier] = self.calling_timestamp
        new_tensor_value = torch.transpose(old_tensor.current_value["data"], self.parameters["dim0"], self.parameters["dim1"])
        self.string_parameters["shape"] = new_tensor_value.shape
        new_tensor = TensorState(self.new_variable_name, new_tensor_value)
        
        new_tensor.transitions = copy.deepcopy(old_tensor.transitions)
        new_tensor.transitions.append({
            "name": self.name,
            "parameters": normalize_parameters(self.parameters),
        })
        variable_schema.add_implicit_variable(new_tensor, self.calling_timestamp)
        variable_schema.local_states["variables"][local_states[0]].updated = False # Already been used.
        variable_schema.local_states["variables"][local_states[0]].latest_call = self.calling_timestamp
        new_local_variable = LocalVariable(
            name=self.new_variable_name,
            value=new_tensor_value,
            latest_call=self.calling_timestamp,
            created_by=f"{self.name}@{self.calling_timestamp}",
            updated=True
        )
        variable_schema.add_local_variable(new_local_variable)
        
        return variable_schema

    @staticmethod
    def calculate_valid_parameters(input_shape: Tuple[int]):
        """Valid dimension indices for transposition based on input shape"""
        ndim = len(input_shape)
        return {
            'dims': {
                'min': 0,
                'max': ndim - 1
            }
        }

    @staticmethod
    def generate_valid_parameters(input_shape: Tuple[int]):
        """Generate valid dimension indices for transposition"""
        valid_ranges = TransposeTransition.calculate_valid_parameters(input_shape)
        ndim = len(input_shape)
        
        if ndim < 2:
            return False, None
            
        # Randomly select two distinct dimensions
        #dims = random.sample(range(ndim), 2)
        dim0 = random.randint(valid_ranges['dims']['min'], valid_ranges['dims']['max'])
        remain_dim = [i for i in range(ndim) if i != dim0]
        dim1 = random.choice(remain_dim)
        return True, {
            'dim0': dim0,
            'dim1': dim1
        }

    def __str__(self):
        return f"torch.transpose({self.parameters['input'].name}, {self.parameters['dim0']}, {self.parameters['dim1']})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        result = [
            f"{self.new_variable_name} = torch.transpose({self.parameters['input'].name}, {self.parameters['dim0']}, {self.parameters['dim1']}) # Output shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""

class TensorEvaluator(ProgramEvaluator):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_cases = []
        
    def prepare_environment(self, init_implicit_dict, init_local_info, init_load_info=None):
        # 
        pass


    def store(self, file_path: str):
        """Save test cases to disk. Handles torch tensors by converting them to lists."""
        
        def tensor_to_list(obj, level=0):
            if level > 2:
                return obj
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                result = []
                for item in obj:
                    if isinstance(item, torch.Tensor):
                        result.append(item.tolist())
                    else:
                        # Only one level of list is supported.
                        result.append(tensor_to_list(item, level+1))
                return result
                #return [tensor_to_list(item) for item in obj]
            return obj

        # Convert test cases to serializable format
        serializable_test_cases = []
        for test_case in self.test_cases:
            serializable_case = {
                "result": tensor_to_list(test_case["result"]),
                "state_oracle": test_case["state_oracle"],
                "program_info": {
                    "init_local_str": test_case["program_info"]["init_local_str"],
                    "init_local_info": test_case["program_info"]["init_local_info"],
                    "init_implicit_dict": None, # not necessary for this evaluator
                    "end_implict_list": None, # not necessary for this evaluator
                    "init_load_str": test_case["program_info"]["init_load_str"],
                    "init_load_info": tensor_to_list(test_case["program_info"]["init_load_info"])
                },
                "program": test_case["program"]
            }
            serializable_test_cases.append(serializable_case)

        # Save to file
        saved_info = {  
            "test_cases": serializable_test_cases,
            "config": self.config
        }
        with open(file_path, 'w') as f:
            json.dump(saved_info, f, indent=2)
            
    @classmethod
    def load(cls, file_path: str, config: Dict[str, Any] = None):
        """Load test cases from disk. Converts lists back to torch tensors."""
        
        # Load from file
        with open(file_path, 'r') as f:
            saved_info = json.load(f)
        if config is None:
            config = saved_info["config"]
        created_cls = cls(config)
        test_cases = saved_info["test_cases"]
        
        # Convert test cases back to tensor format
        for test_case in test_cases:
            if isinstance(test_case["result"], int) or isinstance(test_case["result"], float):
                result = test_case["result"]
            else:
                result = torch.tensor(test_case["result"])
            init_load_info = []
            for (name, value) in test_case["program_info"]["init_load_info"]:
                if isinstance(value, int) or isinstance(value, float):
                    init_load_info.append((name, value))
                else:
                    init_load_info.append((name, torch.tensor(value)))
            converted_case = {
                "result": result,
                "state_oracle": test_case["state_oracle"],
                "program_info": {
                    "init_local_str": test_case["program_info"]["init_local_str"],
                    "init_local_info": test_case["program_info"]["init_local_info"],
                    "init_implicit_dict": None, # not necessary for this evaluator
                    "end_implict_list": None, # not necessary for this evaluator
                    "init_load_str": test_case["program_info"]["init_load_str"],
                    "init_load_info": init_load_info
                },
                "program": test_case["program"]
            }
            created_cls.test_cases.append(converted_case)

        return created_cls
    
    def collect_test_case(self, program_info, program):
        self.prepare_environment(program_info['init_implicit_dict'], program_info['init_local_info'], program_info["init_load_info"])
        namespace = {
            "torch": torch,
        }
        init_load_info = program_info["init_load_info"]
        complete_program = program_info["init_local_str"] + program
        if init_load_info is not None:
            for (name, value) in init_load_info:
                namespace[name] = value.clone()
        try:
            exec(complete_program, namespace)
        except Exception as e:
            error_info = traceback.format_exc()
            logger.warning(f"Error in executing the program: {error_info}")
            return None
        
        if f"{RESULT_NAME}" in namespace:
            result = namespace[RESULT_NAME]
            if isinstance(result, torch.Tensor) and result.numel() == 1:
                result = result.item()
        else:
            result = None
        
        test_case = {
            "result": result,
            "state_oracle": None,
            "program_info": program_info,
            "program": program
        }
        self.test_cases.append(test_case)
        return test_case
    
    def evaluate(self, program: str):
        pass_list = []
        test_case_pass_detail = []
        for test_case in self.test_cases:
            namespace = {
                "torch": torch
            }
            if test_case["program_info"]["init_load_info"] is not None:
                for (name, value) in test_case["program_info"]["init_load_info"]:
                    namespace[name] = value.clone()
            complete_program = test_case["program_info"]["init_local_str"] + program
            try:
                exec(complete_program, namespace)
            except Exception as e:
                error_info = traceback.format_exc()
                pass_list.append(False)
                test_case_pass_detail.append({
                    "result_pass": False,
                    "error_info": error_info,
                    "state_pass": None,
                    "state_pass_detail": None,
                })
                continue
            try:
                result_pass = True
                if f"{RESULT_NAME}" in namespace:
                    result = namespace[RESULT_NAME]
                    if isinstance(result, torch.Tensor):
                        if isinstance(test_case["result"], float) or isinstance(test_case["result"], int):
                            if result.numel() != 1:
                                result_pass = False
                            result = result.item()
                            if abs(result - test_case["result"]) > FLOAT_THRESHOLD:
                                result_pass = False
                        elif result.shape != test_case["result"].shape:
                            result_pass = False
                        elif not torch.allclose(result, test_case["result"], rtol=FLOAT_THRESHOLD, atol=FLOAT_THRESHOLD):
                            result_pass = False
                    elif isinstance(result, (float, int)):
                        if abs(result - test_case["result"]) > FLOAT_THRESHOLD:
                            result_pass = False
                    else:
                        if result != test_case["result"]:
                            result_pass = False
                else:
                    if test_case["result"] is not None:
                        result_pass = False
            except Exception as e:
                error_info = traceback.format_exc()
                pass_list.append(False)
                test_case_pass_detail.append({
                    "result_pass": False,
                    "error_info": error_info,
                    "state_pass": None,
                    "state_pass_detail": None
                })
                continue
            pass_list.append(result_pass)
            test_case_pass_detail.append({
                "result_pass": result_pass,
                "error_info": None,
                "state_pass": True,
                "state_pass_detail": None
            })
        return pass_list, test_case_pass_detail
                
    
    
def test_transpose():
    stateschema = TensorVariableSchema()
    new_tensor_value = torch.randn(3, 4)
    new_local_variable = LocalVariable(
        name="user_variable_1",
        value=new_tensor_value,
        latest_call=0,
        created_by=f"user"
    )
    new_implicit_variable = TensorState(
        name="user_variable_1",
        data=new_tensor_value
    )
    stateschema.add_implicit_variable(new_implicit_variable, 0)
    stateschema.add_local_variable(new_local_variable)
    transitions = TransposeTransition(
        {
            "input": new_local_variable,
            "dim0": 0,
            "dim1": 1
        },
        1
    )
    implicit_states, local_states = transitions.get_effected_states(stateschema)
    assert len(implicit_states) == 1
    assert len(local_states) == 1
    assert 'response_1 = torch.transpose(user_variable_1, 0, 1)' in transitions.get_program_str()[0][0]
    transitions.apply(implicit_states, local_states, stateschema)
    assert stateschema.local_states["variables"][local_states[0]].name == "user_variable_1"
    assert stateschema.implicit_states["tensor_info"][implicit_states[0].identifier].identifier == "user_variable_1"
    assert stateschema.implicit_states["latest_call"]['response_1'] == 1
    

def test_permute():
    stateschema = TensorVariableSchema()
    new_tensor_value = torch.randn(3, 4)
    new_local_variable = LocalVariable(
        name="user_variable_1",
        value=new_tensor_value,
        latest_call=0,
        created_by=f"user"
    )
    new_implicit_variable = TensorState(
        name="user_variable_1",
        data=new_tensor_value
    )
    stateschema.add_implicit_variable(new_implicit_variable, 0)
    stateschema.add_local_variable(new_local_variable)
    transitions = PermuteTransition(
        {
            "input": new_local_variable,
            "dims": (1, 0)
        },
        1
    )
    implicit_states, local_states = transitions.get_effected_states(stateschema)
    assert len(implicit_states) == 1
    assert len(local_states) == 1
    assert 'response_1 = torch.permute(user_variable_1, (1, 0))' in transitions.get_program_str()[0][0]
    transitions.apply(implicit_states, local_states, stateschema)
    assert len(stateschema.local_states["variables"]) == 2
    assert "response_1" in stateschema.implicit_states["tensor_info"]

def test_split():
    stateschema = TensorVariableSchema()

    new_tensor_value = torch.randn(3, 4)

    new_local_variable = LocalVariable(
        name="user_variable_1",
        value=new_tensor_value,
        latest_call=0,
        created_by=f"user"
    )

    new_implicit_variable = TensorState(
        name="user_variable_1",
        data=new_tensor_value
    )

    stateschema.add_implicit_variable(new_implicit_variable, 0)
    stateschema.add_local_variable(new_local_variable)

    transitions = SplitTransition(
        {
            "input": new_local_variable,
            "split_size_or_sections": 2,
            "dim": 1
        },
        1
    )

    implicit_states, local_states = transitions.get_effected_states(stateschema)
    transitions.apply(implicit_states, local_states, stateschema)
    assert "(response_1_0, response_1_1) = torch.split(user_variable_1, 2, 1)" in transitions.get_program_str()[0][0]
    assert len(stateschema.local_states["variables"]) == 3
    assert len(stateschema.local_states['variables']) == 3

def test_cat():
    stateschema = TensorVariableSchema()

    new_tensor_value_1 = torch.randn(3, 4)
    user_variable_1 = new_tensor_value_1

    new_local_variable = LocalVariable(
        name="user_variable_1",
        value=new_tensor_value_1,
        latest_call=0,
        created_by=f"user"
    )

    new_tensor_value_2 = torch.randn(3, 4)
    user_variable_2 = new_tensor_value_2

    new_local_variable_2 = LocalVariable(
        name="user_variable_2",
        value=new_tensor_value_2,
        latest_call=0,
        created_by=f"user"
    )

    new_implicit_variable = TensorState(
        name="user_variable_1",
        data=new_tensor_value_1
    )

    new_implicit_variable_2 = TensorState(
        name="user_variable_2",
        data=new_tensor_value_2
    )

    stateschema.add_implicit_variable(new_implicit_variable, 0)
    stateschema.add_local_variable(new_local_variable)

    stateschema.add_implicit_variable(new_implicit_variable_2, 0)
    stateschema.add_local_variable(new_local_variable_2)



    transitions = CatTransition(
        {
            "tensors": [new_local_variable, new_local_variable_2],
            "dim": 0
        },
        1
    )

    implicit_states, local_states = transitions.get_effected_states(stateschema)
    transitions.apply(implicit_states, local_states, stateschema)
    
    assert "response_1 = torch.cat((user_variable_1, user_variable_2), 0)" in transitions.get_program_str()[0][0]
    assert len(stateschema.local_states['variables']) == 3
    assert len(stateschema.implicit_states['tensor_info']) == 3

def test_linear():
    stateschema = TensorVariableSchema()

    new_tensor_value_1 = torch.randn(3, 4)
    user_variable_1 = new_tensor_value_1

    new_local_variable = LocalVariable(
        name="user_variable_1",
        value=new_tensor_value_1,
        latest_call=0,
        created_by=f"user"
    )

    new_tensor_value_2 = torch.randn(4, 4)
    user_variable_2 = new_tensor_value_2

    new_local_variable_2 = LocalVariable(
        name="user_variable_2",
        value=new_tensor_value_2,
        latest_call=0,
        created_by=f"user"
    )

    new_implicit_variable = TensorState(
        name="user_variable_1",
        data=new_tensor_value_1
    )

    new_implicit_variable_2 = TensorState(
        name="user_variable_2",
        data=new_tensor_value_2
    )

    stateschema.add_implicit_variable(new_implicit_variable, 0)
    stateschema.add_local_variable(new_local_variable)

    stateschema.add_implicit_variable(new_implicit_variable_2, 0)
    stateschema.add_local_variable(new_local_variable_2)



    transitions = LinearTransition(
        {
            "input": new_local_variable,
            "weight": new_local_variable_2
        },
        1
    )

    implicit_states, local_states = transitions.get_effected_states(stateschema)
    transitions.apply(implicit_states, local_states, stateschema)
    assert "response_1 = torch.nn.functional.linear(user_variable_1, user_variable_2)" in transitions.get_program_str()[0][0]
    assert len(stateschema.local_states['variables']) == 3
    assert len(stateschema.implicit_states['tensor_info']) == 3

    inputs = torch.randn(1, 4, 5, 5)
    temp = LinearTransition.generate_valid_parameters(inputs.shape)
    result = torch.nn.functional.linear(inputs, 
                            temp['weight'])

def test_conv2d():
    stateschema = TensorVariableSchema()

    new_tensor_value_1 = torch.randn(24, 4, 3, 3)
    user_variable_1 = new_tensor_value_1

    new_local_variable = LocalVariable(
        name="user_variable_1",
        value=new_tensor_value_1,
        latest_call=0,
        created_by=f"user"
    )

    new_tensor_value_2 = torch.randn(1, 4, 5, 5)
    user_variable_2 = new_tensor_value_2

    new_local_variable_2 = LocalVariable(
        name="user_variable_2",
        value=new_tensor_value_2,
        latest_call=0,
        created_by=f"user"
    )

    new_implicit_variable = TensorState(
        name="user_variable_1",
        data=new_tensor_value_1
    )

    new_implicit_variable_2 = TensorState(
        name="user_variable_2",
        data=new_tensor_value_2
    )

    stateschema.add_implicit_variable(new_implicit_variable, 0)
    stateschema.add_local_variable(new_local_variable)

    stateschema.add_implicit_variable(new_implicit_variable_2, 0)
    stateschema.add_local_variable(new_local_variable_2)



    transitions = Conv2dTransition(
        {
            "input": new_local_variable_2,
            "weight": new_local_variable,
            "stride": 1, 
            "padding": 0, 
            "dilation": 1
        },
        1
    )

    implicit_states, local_states = transitions.get_effected_states(stateschema)
    transitions.apply(implicit_states, local_states, stateschema)
    assert "response_1 = torch.nn.functional.conv2d(user_variable_2, user_variable_1, stride=1, padding=0, dilation=1)" in transitions.get_program_str()[0][0]
    assert len(stateschema.local_states['variables']) == 3
    assert len(stateschema.implicit_states['tensor_info']) == 3
    
    
    inputs = torch.randn(1, 4, 5, 5)
    temp = Conv2dTransition.generate_valid_parameters(inputs.shape)
    result = torch.nn.functional.conv2d(inputs, 
                               temp['weight'], 
                               padding=temp["padding"], 
                               stride=temp['stride'], 
                               dilation=temp["dilation"])


