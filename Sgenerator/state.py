from typing import Callable, Any, Dict, List, Optional, Iterable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
import numpy as np
from loguru import logger
import copy
USER_FUNCTION_PARAM_FLAG = "User-provided local variable"

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
    def get_effected_states(self, variable_schema) -> Tuple[List[str], List[str]]:
        """
        Get the states that are affected by this transition.
        Return a list of state identifiers.
        """
        pass
    
    @abstractmethod
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema):
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
    def get_available_transitions(self, random_generator: Any, current_call: int, max_call: int):
        """
        Get the available transitions for the global schema.
        Return a dictionary of transition name to possible parameters.
        """
        pass
    
    @abstractmethod
    def craft_transition(self, parameters: Any, calling_timestamp: int, transition: str):
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
    
    def clear_state(self):
        """
        Clear the state of the schema.
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
                 occurence_book: Dict[str, Any]):
        """
        config:
            - init_local_state_num_range: the range of the number of local states to be initialized. [min, max]
            - init_implicit_state_num_range: the range of the number of implicit states to be initialized. [min, max]
            - random_generate_config: the config for the random generator (func random_generate_state) Dict [str, Any]
        """
        self.state_schema = state_schema
        self.random_generator = random_generator
        self.trace = []
        self.config = config
        self.random_generate_config = config["random_generate_config"] if "random_generate_config" in config else {}
        self.call_num = config["call_num"]
        self.occurence_book = occurence_book # pair -> occurence
        self.this_trace_recorder = dict()
        
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
        # 1. Two function calls with exact same parameters should be avoided.
        # 3. Increase pair coverage as much as possible.
        # Data structure that being effected: self.occurence_book, self.trace, self.state_schema, self.random_generator
        self.this_trace_recorder = dict()
        self.this_trace_duplicate_local_variable_map = dict()
        for i in range(self.call_num):
            available_transitions = self.state_schema.get_available_transitions(self.random_generator, 
                                                                                i+1, 
                                                                                self.call_num, 
                                                                                self.this_trace_duplicate_local_variable_map)
            selection_to_coverage_map = dict()
            self.energy_map = dict()
            # Compute coverage information
            has_new_coverage = False
            normalize_term = 0
            for transition in available_transitions:
                if transition not in self.this_trace_recorder:
                    self.this_trace_recorder[transition] = []
                for idx, transition_info in enumerate(available_transitions[transition]):
                    # If the transition with exactly the same parameters has been called, skip it.
                    for parameters in self.this_trace_recorder[transition]:
                        if parameters == transition_info["required_parameters"]:
                            continue
                    # Compute the coverage information
                    transition_pairs = transition_info["transition_pairs"]
                    uncovered_pairs = [
                                    pair for pair in transition_pairs
                                    if pair not in self.occurence_book
                    ]
                    if len(uncovered_pairs) > 0:
                        has_new_coverage = True
                        normalize_term += len(uncovered_pairs)
                    selection_to_coverage_map[(transition, idx)] = [len(uncovered_pairs), uncovered_pairs, transition_pairs]
            if has_new_coverage:
                # If there is new coverage, the next selection should be made from the transitions with new coverage.
                assert normalize_term > 0
                for transition, idx in selection_to_coverage_map:
                    if selection_to_coverage_map[(transition, idx)][0] > 0:
                        self.energy_map[(transition, idx)] = selection_to_coverage_map[(transition, idx)][0] / normalize_term
            else:
                # If there is no new coverage, the next selection should be made from the transitions with lower occurence.
                for transition, idx in selection_to_coverage_map:
                    local_occurence = 0
                    for pair in selection_to_coverage_map[(transition, idx)][2]:
                        local_occurence += self.occurence_book[pair]
                    ave_occurence = local_occurence / len(selection_to_coverage_map[(transition, idx)][2])
                    self.energy_map[(transition, idx)] = ave_occurence
                    normalize_term += ave_occurence
                for transition, idx in selection_to_coverage_map:
                    self.energy_map[(transition, idx)] = np.log(normalize_term / self.energy_map[(transition, idx)]) # IDF term in TF-IDF
                    
            candidates = [(key, self.energy_map[key]) for key in self.energy_map]
            if len(candidates) == 1:
                selected = candidates[0][0]
            elif len(candidates) == 0:
                logger.warning(f"No available transitions for the {i}-th call. Terminate the trace generation.")
                return self.trace
            else:
                selected = random.choices(candidates, weights=[c[1] for c in candidates], k=1)[0][0] # [0] for random.choices list return, [0] for the selected transition
            for pair in selection_to_coverage_map[selected][2]:
                self.occurence_book[pair] = self.occurence_book.get(pair, 0) + 1
            target_transition_info = available_transitions[selected[0]][selected[1]]
            new_transition = self.state_schema.craft_transition(target_transition_info["required_parameters"], i, selected[0])
            if selected[0] not in self.this_trace_duplicate_local_variable_map:
                self.this_trace_duplicate_local_variable_map[selected[0]] = set([])
            self.this_trace_duplicate_local_variable_map[selected[0]].add(str(sorted(target_transition_info["required_parameters"].items())))
            implicit, local = new_transition.get_effected_states(self.state_schema)
            new_transition.apply(implicit, local, self.state_schema)
            self.trace.append([selected[0], copy.deepcopy(target_transition_info["required_parameters"])])
        
        return self.trace
            
                    
                    
