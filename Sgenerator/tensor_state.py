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

TENSOR_SUMMARY_PROMPT = '''
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
            Conv2dTransition,
            LinearTransition
        ]
        self.local_call_map = {}
        self.implicit_call_map = {}
        self.init_implict_dict = {}
    
    def add_implicit_variable(self, implicit_variable: Any, latest_call: int):
        assert implicit_variable.identifier not in self.implicit_states["tensor_info"]
        self.implicit_states["tensor_info"][implicit_variable.identifier] = implicit_variable
        self.implicit_states["latest_call"][implicit_variable.identifier] = latest_call
        self.init_implict_dict[implicit_variable.identifier] = implicit_variable
    
    def add_local_variable(self, local_variable: Any):
        self.local_states["variables"].append(local_variable)
        

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
        assert isinstance(parameters["padding"], int)
        assert "dilation" in parameters
        assert isinstance(parameters["dilation"], int)
        
        
        super().__init__("conv2d", parameters=parameters, func=None)
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
        variable_schema.local_states["variables"][local_states[0]].updated = False # Already been used.
        variable_schema.add_local_variable(new_local_variable)
        
        return variable_schema
    
    def __str__(self):
        return f"torch.nn.functional.conv2d({self.parameters['input'].name}, {self.parameters['weight'].name}, stride={self.parameters['stride']}, padding={self.parameters['padding']}, dilation={self.parameters['dilation']})"

    def get_program_str(self) -> Tuple[List[str], str]:
        result = [
            f"{self.new_variable_name} = torch.nn.functional.conv2d({self.parameters['input'].name}, {self.parameters['weight'].name}, stride={self.parameters['stride']}, padding={self.parameters['padding']}, dilation={self.parameters['dilation']}) # shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""

class PermuteTransition(Transition):
    """Handles dimension permutation"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "input" in parameters
        assert isinstance(parameters["input"], LocalVariable)
        assert "dims" in parameters
        super().__init__("permute", parameters=parameters, func=None)
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
        variable_schema.add_local_variable(new_local_variable)
        
        return variable_schema
    
    def __str__(self):
        return f"torch.permute({self.parameters['input'].name}, {self.parameters['dims']})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        result = [
            f"{self.new_variable_name} = torch.permute({self.parameters['input'].name}, {self.parameters['dims']}) # shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""
    
    

class SplitTransition(Transition):
    """Handles tensor splitting"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "input" in parameters
        assert isinstance(parameters["input"], LocalVariable)
        assert "split_size_or_sections" in parameters
        assert isinstance(parameters["split_size_or_sections"], Union[int, List[int], Tuple[int]])
        assert "dim" in parameters
        super().__init__("split", parameters=parameters, func=None)
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
            variable_schema.add_local_variable(new_local_variable)
        
        return variable_schema
    
    def __str__(self):
        return f"torch.split({self.parameters['input'].name}, {self.parameters['split_size_or_sections']}, {self.parameters['dim']})"

    def get_program_str(self) -> Tuple[List[str], str]:
        returned_left = "("
        for idx, shape in enumerate(self.string_parameters["shape"]):
            returned_left += f"{self.new_variable_name}_{idx}, "
        returned_left = returned_left[:-2] + ")"
        returned_right = "shape: "
        for idx, shape in enumerate(self.string_parameters["shape"]):
            returned_right += f"{shape}, "
        returned_right = returned_right[:-2]
        result = [
            f"{returned_left} = torch.split({self.parameters['input'].name}, {self.parameters['split_size_or_sections']}, {self.parameters['dim']}) # {returned_right}\n",
        ]
        return result, ""
    
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
        super().__init__("cat", parameters=parameters, func=None)
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
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            tensor_name = variable_schema.local_states["variables"][local_idx].name
            if tensor_name in self.parameters["tensors"]:
                local_states.append(local_idx)
            implicit_states.append(variable_schema.implicit_states["tensor_info"][tensor_name])
        return implicit_states, local_states

    def apply(self, implicit_states: List[TensorState], local_states: List[str], variable_schema: TensorVariableSchema):
        input_tensors = []
        longest_transitions = []
        for state in implicit_states:
            input_tensors.append(state.current_value["data"])
            if len(state.transitions) > len(longest_transitions):
                longest_transitions = copy.deepcopy(state.transitions)
        new_tensor_value = torch.cat(input_tensors, self.parameters["dim"])
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
            f"{self.new_variable_name} = torch.cat({input_parameters}, {self.parameters['dim']}) # shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""
    

class LinearTransition(Transition):
    """Handles linear transformation"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "input" in parameters
        assert isinstance(parameters["input"], LocalVariable)
        assert "weight" in parameters
        assert isinstance(parameters["weight"], LocalVariable)
        super().__init__("linear", parameters=parameters, func=None)
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
        variable_schema.add_local_variable(new_local_variable)
        return variable_schema
    
    def __str__(self):
        return f"torch.nn.functional.linear({self.parameters['input'].name}, {self.parameters['weight'].name})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        result = [
            f"{self.new_variable_name} = torch.nn.functional.linear({self.parameters['input'].name}, {self.parameters['weight'].name}) # shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""
    
class TransposeTransition(Transition):
    """Handles tensor transposition"""
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        # For Transpose, the dim is purely number instead of a variable.
        assert "input" in parameters
        assert isinstance(parameters["input"], LocalVariable)
        assert "dim0" in parameters
        assert isinstance(parameters["dim0"], int)
        assert "dim1" in parameters
        super().__init__("transpose", parameters=parameters, func=None)
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
        new_local_variable = LocalVariable(
            name=self.new_variable_name,
            value=new_tensor_value,
            latest_call=self.calling_timestamp,
            created_by=f"{self.name}@{self.calling_timestamp}",
            updated=True
        )
        variable_schema.add_local_variable(new_local_variable)
        
        return variable_schema
    
    def __str__(self):
        return f"torch.transpose({self.parameters['input'].name}, {self.parameters['dim0']}, {self.parameters['dim1']})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        result = [
            f"{self.new_variable_name} = torch.transpose({self.parameters['input'].name}, {self.parameters['dim0']}, {self.parameters['dim1']}) # shape: {self.string_parameters['shape']}\n",
        ]
        return result, ""
    

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


