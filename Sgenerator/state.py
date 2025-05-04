from typing import Callable, Any, Dict, List, Optional, Set, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
import numpy as np
from loguru import logger
import copy
USER_FUNCTION_PARAM_FLAG = "User-variable"
RESPONSE_VARIABLE_TEMP = "response_{}"

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
    producer = "None" # The input parameter source of this transition.
    func: Callable[[Any, Dict[str, Any]], Any] = None

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
    def get_available_transitions(self, 
                                  random_generator: Any, 
                                  current_call: int, 
                                  max_call: int, 
                                  duplicate_local_variable_map: Dict[str, Set[str]], 
                                  previous_transition_info: Tuple):
        """
        Get the available transitions for the global schema.
        Return a dictionary of transition name to possible parameters.
        ---
        Args:
            random_generator: a random generator
            current_call: the current call number (1-indexed)
            max_call: the maximum call number
            duplicate_local_variable_map: a dictionary of duplicate local variable map. Transition name -> set of parameters.
            previous_transition_info: the previous transition info. (transition name, parameters)
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

    def determine_whether_to_keep_pair(self, previous_transition_info: Tuple, current_transition_info: Tuple) -> bool:
        """
        Determine whether to keep the pair of transitions based on the already choosen one and the current candidate.
        This function is used to filter out the transition pairs that seem to be stupid. (e.g., query the same variable twice in a row)
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

    def transform_parameters_to_str(self, parameters: Dict[str, Any]) -> str:
        """
        Transform the parameters to a string.
        """
        def sorted_deep(obj):
            if isinstance(obj, dict):
                return {k: sorted_deep(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return [sorted_deep(elem) for elem in obj]
            return obj
        
        normalized = sorted_deep(parameters)
        return str(sorted(normalized.items()))
    
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
        self.config = config
        self.random_generate_config = config["random_generate_config"] if "random_generate_config" in config else {}
        self.occurence_book = occurence_book # pair -> occurence
        
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
        
        
    def generate_trace(self, call_num, this_trace_recorder=dict(), this_trace_duplicate_local_variable_map=dict()):
        # 1. Two function calls with exact same parameters should be avoided.
        # 2. Increase pair coverage as much as possible.
        # Data structure that being effected: self.occurence_book, self.trace, self.state_schema, self.random_generator
        previous_transition_info = None
        trace = []
        trace_str = []
        for i in range(call_num):
            available_transitions = self.state_schema.get_available_transitions(self.random_generator, 
                                                                                i+1, 
                                                                                call_num, 
                                                                                copy.deepcopy(this_trace_duplicate_local_variable_map),
                                                                                previous_transition_info)
            selection_to_coverage_map = dict()
            energy_map = dict()
            # Compute coverage information
            has_new_coverage = False
            normalize_term = 0
            for transition in available_transitions:
                if transition not in this_trace_recorder:
                    this_trace_recorder[transition] = []
                for idx, transition_info in enumerate(available_transitions[transition]):
                    # If the transition with exactly the same parameters has been called, skip it.
                    for parameters in this_trace_recorder[transition]:
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
                        energy_map[(transition, idx)] = selection_to_coverage_map[(transition, idx)][0] / normalize_term
            else:
                # If there is no new coverage, the next selection should be made from the transitions with lower occurence.
                for transition, idx in selection_to_coverage_map:
                    local_occurence = 0
                    for pair in selection_to_coverage_map[(transition, idx)][2]:
                        local_occurence += self.occurence_book[pair]
                    ave_occurence = local_occurence / len(selection_to_coverage_map[(transition, idx)][2])
                    energy_map[(transition, idx)] = ave_occurence
                    normalize_term += ave_occurence
                for transition, idx in selection_to_coverage_map:
                    energy_map[(transition, idx)] = np.log(normalize_term / energy_map[(transition, idx)]) # IDF term in TF-IDF
                    
            candidates = [(key, energy_map[key]) for key in energy_map]
            if len(candidates) == 1:
                selected = candidates[0][0]
            elif len(candidates) == 0:
                logger.warning(f"No available transitions for the {i+1}-th call. Terminate the trace generation.")
                return trace, this_trace_recorder, this_trace_duplicate_local_variable_map
            else:
                selected = random.choices(candidates, weights=[c[1] for c in candidates], k=1)[0][0] # [0] for random.choices list return, [0] for the selected transition
            for pair in selection_to_coverage_map[selected][2]:
                self.occurence_book[pair] = self.occurence_book.get(pair, 0) + 1
            target_transition_info = available_transitions[selected[0]][selected[1]]
            producer = target_transition_info["producer_variable_idx"]
            if producer is not None:
                producer = copy.deepcopy(self.state_schema.local_states["variables"][producer])
            else:
                producer = None
            #producer_info = f"FROM local_variable_idx: {producer}, created_by: {self.state_schema.local_states['variables'][producer].created_by}"
            new_transition = self.state_schema.craft_transition(target_transition_info["required_parameters"], i+1, selected[0], producer)
            if selected[0] not in this_trace_duplicate_local_variable_map:
                this_trace_duplicate_local_variable_map[selected[0]] = set([])
            this_trace_duplicate_local_variable_map[selected[0]].add(self.state_schema.transform_parameters_to_str(target_transition_info["required_parameters"]))
            implicit, local = new_transition.get_effected_states(self.state_schema)
            new_transition.apply(implicit, local, self.state_schema)
            trace.append([selected[0], copy.deepcopy(target_transition_info["required_parameters"])])
            try:
                trace_str.append(str(new_transition))
            except:
                trace_str.append(f"Error: {new_transition}")
            previous_transition_info = (selected[0], target_transition_info["required_parameters"])
        
        return (trace, trace_str), this_trace_recorder, this_trace_duplicate_local_variable_map
            
                    
                    
