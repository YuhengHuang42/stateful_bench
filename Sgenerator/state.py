from typing import Callable, Any, Dict, List, Optional, Iterable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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

class TraceGenerator:
    """
    Generate a trace of the state changes.
    """
    def __init__(self, state_schema):
        self.state_schema = state_schema

    def generate_trace(self) -> List[str]:
        pass