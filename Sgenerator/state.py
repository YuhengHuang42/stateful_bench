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
    
    
    def apply(self, value: Any) -> Any:
        """
        Apply this transition to `value` and return the result.
        """
        return self.func(value, self.parameters)

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
    def get_effected_states(self, state_list) -> List[str]:
        """
        Get the states that are affected by this transition.
        """
        pass
    
    @abstractmethod
    def apply(self, states):
        """Apply the transition to the list of state. Return the list of updated states."""
        pass

@dataclass
class State:
    """
    Tracks a single program state variable.
    - identifier: unique name of the state variable
    - initial_value: starting value
    - transitions: sequence of state updates
    """
    identifier: str
    initial_value: Any = None
    transitions: List[Transition] = field(default_factory=list)
    exist: bool = True
    
    def __post_init__(self):
        # set current_value to initial when created
        self.current_value = self.initial_value

    def add_transition(self, transition: Transition) -> None:
        """Record a transition."""
        self.transitions.append(transition)
    
    def apply_transition(self, transition: Transition) -> None:
        """Apply a transition to the current value."""
        self.current_value = transition.apply(self.current_value)

@dataclass
class StateSchema:
    """
    Manages a collection of state variables and their transitions.
    - states: dictionary mapping state identifiers to State instances
    - side_effect_table: dictionary mapping transition names to lists of state identifiers that are affected by the transition
    """
    states: Dict[str, State] = field(default_factory=dict)
    side_effect_table: Dict[str, List[str]] = field(default_factory=dict)

    def add_state(self, state: State) -> None:
        """Register a new state variable."""
        self.states[state.identifier] = state

    def get_state(self, identifier: str) -> Optional[State]:
        """Retrieve a state variable by its identifier."""
        return self.states.get(identifier)

    def check_equal(self, other: 'StateSchema') -> bool:
        """Check if two state schemas are equal."""
        invalid_dict = {}
        for state_id in self.states:
            if state_id not in other.states: # other schema doesn't have this state
                invalid_dict[state_id] = [state_id, None]
                continue
            this_state = self.states[state_id]
            other_state = other.states[state_id]
            if len(this_state.transitions) != len(other_state.transitions): # other schema has a different number of transitions
                invalid_dict[state_id] = [state_id, None]
                continue
            for i, transition in enumerate(this_state.transitions):
                if not other_state.transitions[i].check(transition.name, transition.parameters): # other schema has a different transition
                    invalid_dict[state_id] = [state_id, (transition.name, transition.parameters)]
                    break
        if len(invalid_dict) == 0:
            return True, invalid_dict
        return False, invalid_dict

    def get_latest_transition_for_state(self, state_id: str) -> Optional[Transition]:
        """Get the latest transition for a state."""
        state = self.get_state(state_id)
        if state is None:
            return None
        return state.transitions[-1].name
    
    def get_latest_transition(self) -> Optional[Transition]:
        """Get the latest transition for all states."""
        latest_transitions = {}
        for state_id in self.states:
            transition_name = self.get_latest_transition_for_state(state_id)
            if transition_name is not None:
                latest_transitions[state_id] = transition_name
        return latest_transitions

