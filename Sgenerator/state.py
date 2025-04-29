from typing import Callable, Any, Dict, List, Optional, Iterable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random

@dataclass
class Transition:
    """
    Represents a state transformation operation on a single value.
    - name: a human-readable identifier for the operation
    - parameters: arguments to pass into func
    - func: a callable that takes (current_value, parameters) and returns a new value
    """
    name: str
    parameters: Dict[str, Any]
    func: Callable[[Any, Dict[str, Any]], Any]
    
    
    def apply(self, effected_states: List[str], variable_schema) -> Any:
        """
        Apply this transition and record the effect on the variable schema.
        """
        pass
        #return self.func(value, self.parameters)

    def check(self, name: str, parameters: Dict[str, Any]) -> bool:
        """
        Check if this transition matches the given name and parameters.
        """
        if self.name != name:
            return False
        for key, value in parameters.items():
            if self.parameters[key] != value:
                return False
        return True
    
    @abstractmethod
    def get_effected_states(self, variable_schema) -> List[str]:
        """
        Get the states that are affected by this transition.
        Return a list of state identifiers.
        """
        pass
    
    @abstractmethod
    def apply(self, states, variable_schema):
        """Apply the transition to the list of state. Return the list of updated states."""
        pass

@dataclass
class State:
    """
    Tracks a single state variable.
    - identifier: unique name of the state variable
    - initial_value: starting value
    - transitions: sequence of state updates
    """
    identifier: str
    initial_value: Any = None
    transitions: List[Transition] = field(default_factory=list)
    exist: bool = True
    created_by: str = None

class Schema:
    def __init__(self):
        pass
    
    @abstractmethod
    def add_local_variable(self, local_variable: Any):
        """
        Add a local variable to the global schema.
        """
        pass
    
    @abstractmethod
    def add_implicit_variable(self, implicit_variable: Any, latest_call: int):
        """
        Add an implicit variable to the global schema.
        """
        pass
    
    @abstractmethod
    def add_local_variable_using_state(self, state: Any, latest_call: int, updated: bool, created_by: str):
        """
        Add a local variable to the global schema using a state.
        Wrapper of add_local_variable.
        """
        pass
    
    @abstractmethod
    def get_available_transitions(self, random_generator: Any):
        """
        Get the available transitions for the global schema.
        Return a dictionary of transition name to possible parameters.
        """
        pass
    
    @abstractmethod
    def craft_transition(self, parameters: Any, calling_timestamp: int, transition: Any):
        """
        Craft a transition for the global schema.
        Return a transition object. Some side effects may be applied to the global schema in calling this function..
        """
        pass
    
    @abstractmethod
    def craft_ifelse(self):
        """
        Craft an ifelse condition for the given trace.
        This function will called by TraceGenerator and after this, two separate traces will be generated.
        """
        pass
    
    @abstractmethod
    def align_initial_state(self):
        """
        Align the initial state with the parameter space.
        Might not be needed for all schemas. Align the states where both 
        implicit and local variables are generated randomly but they are not aligned.
        """
        pass
    
    def form_pair_transition(self, state, new_transition: str):
        """
        Form a pair transition for the given state and new transition.
        """
        if len(state.transitions) == 0:
            last_transition = "NONE"
        else:
            last_transition = state.transitions[-1]["name"]
        transition_pair = (last_transition, new_transition, )
        return transition_pair
    
class RandomInitializer:
    """
    Initialize a random generator.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def random_generate_state(self):
        """
        Random generate a state.
        """
        pass
    

class TraceGenerator:
    """
    Generate a trace of the state changes.
    Initialization --> transition selection --> trace generation
    """
    def __init__(self, 
                 state_schema: Schema, 
                 random_generator: Any, 
                 config: Dict[str, Any],
                 coverage_book: Dict[str, Any]):
        """
        config:
            - init_local_state_num_range: the range of the number of local states to be initialized
            - init_implicit_state_num_range: the range of the number of implicit states to be initialized
            - random_generate_config: the config for the random generator (func random_generate_state) Dict [str, Any]
        """
        self.state_schema = state_schema
        self.random_generator = random_generator
        self.trace = []
        self.config = config
        self.random_generate_config = config["random_generate_config"] if "random_generate_config" in config else {}
        self.call_num = config["call_num"]
        self.coverage_book = coverage_book
        
    def prepare_initial_state(self):
        """
        Prepare the initial state for the trace generation.
        """
        local_state_num = random.randint(self.config["init_local_state_num_range"][0], self.config["init_local_state_num_range"][1])
        implicit_state_num = random.randint(self.config["init_implicit_state_num_range"][0], self.config["init_implicit_state_num_range"][1])
        for i in range(implicit_state_num):
            self.state_schema.add_implicit_variable(self.random_generator.random_generate_state(**self.random_generate_config), 0)
        for i in range(local_state_num):
            state = self.random_generator.random_generate_state(**self.random_generate_config)
            self.state_schema.add_local_variable_using_state(state, latest_call=0)
        self.state_schema.align_initial_state()
        
        
    def generate_trace(self):
        # 1. Two consecutive function calls with exact same parameters should be avoided.
        # 2. The trace should allow control flow.
        # 3. Increase pair coverage as much as possible.
        for i in range(self.call_num):
            available_transitions = self.state_schema.get_available_transitions(self.random_generator)
            selection_to_coverage_map = dict()
            for transition in available_transitions:
                for idx, transition_info in enumerate(available_transitions[transition]):
                    if transition_info["transition_pairs"] not in self.coverage_book:
                        pass

