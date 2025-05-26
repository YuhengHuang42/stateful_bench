from typing import Callable, Any, Dict, List, Optional, Set, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
import numpy as np
from loguru import logger
import copy
from . import utils
USER_FUNCTION_PARAM_FLAG = "user_variable"
USER_CONSTANT_FLAG = "user_constant"
RESPONSE_VARIABLE_TEMP = "response_{}"
RESULT_NAME = "RESULT"
INDENT = "    "

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
        self.init_local_info: List[Tuple[int, str, str]] = field(default_factory=list) # [(idx, name, value)] Used for the initialization of the user-defined variables.
        self.local_states = None
        self.implicit_states = None
        self.init_load_info = None
        
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
        Return:
            A list of transitions.
            Each transition is a dictionary with the following keys:
            - required_parameters: the required parameters for the transition
            - latest_call: the latest call number
            - whether_updated: whether the transition updates the state
            - producer_variable_idx: the index of the producer variable
            - transition_pairs: the transition pairs
            - transition_name: the name of the transition
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
            try:
                last_transition = state.transitions[-1]["name"]
            except:
                print(state.transitions)
                raise ValueError("Invalid transition list")
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
    
    @abstractmethod
    def postprocess_transitions(self, remaining_call: int) -> Tuple[bool, List[str]]:
        """
        Postprocess the transitions.
        Some local states need to be submitted to the implicit states (e.g., server/return value).
        Given the Number of API call as the budget, this function check whether we need to enter the postprocessing stage
        and submit the local states to the implicit states.
        Return a tuple of (whether to enter the postprocessing stage, list of transitions for the postprocessing stage).
        """
        pass
    
    @abstractmethod
    def postprocess_choose_result(self):
        """
        Postprocess the program state to set the leaf of the data flow graph as the result.
        """
        pass
    
    @abstractmethod
    def get_program_str(self) -> Tuple[List[str], str]:
        """
        Return the program string in line and the indent.
        """
        pass
    
    @abstractmethod
    def prepare_initial_state(self):
        """
        Prepare the initial state for the trace generation.
        """
        pass

    @abstractmethod
    def get_load_info(self, init_load_info=None):
        """
        Get the load info for the trace generation.
        Return: program_string, list of (variable_name, variable_value)
        """
        pass

    @staticmethod
    def reverse_if_condition(value):
        """
        Reverse the if condition for a quoted string.
        If value is a quoted string (e.g., '"foo"'), change the content inside the quotes.
        Otherwise, append '_diff'.
        """
        if isinstance(value, str):
            if (len(value) >= 2) and (value[0] == value[-1]) and value[0] in ('"', "'"):
                # Quoted string
                inner = value[1:-1]
                # You can make the modification more robust (e.g., add a suffix, or flip a char)
                new_inner = inner + "_diff"
                return value[0] + new_inner + value[-1]
            else:
                return value + "_diff"
        elif isinstance(value, tuple) or isinstance(value, list):
            new_value = [i for i in value]
            new_value.append(value[0])
            return new_value
        elif isinstance(value, int) or isinstance(value, float):
            return value + 1
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
    
    def add_local_constant(self, value, name=None):
        already_exist = False
        if name is None:
            for item in self.init_local_info:
                if item[1] == value:
                    already_exist = True
                    return (item[0], already_exist)
            name = self.get_new_local_constant_name()
        if isinstance(value, str):    
            value = f"\"{value}\""
        self.init_local_info.append([name, value])
        return (name, already_exist)
    
    def get_new_local_constant_name(self):
        name = f"{USER_CONSTANT_FLAG}_{len(self.init_local_info)}"
        return name
    
    @staticmethod
    def return_init_local_info(init_local_info):
        init_program = ""
        returned_init_local_info = []
        for init_str in init_local_info:
            init_program += f"{init_str[0]} = {init_str[1]}\n"
            returned_init_local_info.append((init_str[0], init_str[1]))
        return init_program, returned_init_local_info

@dataclass
class LocalVariable:
    value: Any
    name: str
    # whether the local variable is updated before the implicit state is actually updated. E.g., those local 
    # varibales that are updated by the local edit transition but not submitted to the backend database yet.
    updated: bool = False  
    latest_call: int = 0
    exist: bool = True
    created_by: str = None
    is_indexed: bool = False
    variable_type: str = None
    transitions: List[Transition] = field(default_factory=list) # We only use this in voice_state.py
    
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
        self.occurence_book = copy.deepcopy(occurence_book) # pair -> occurence
        
    def prepare_initial_state(self):
        """
        Prepare the initial state for the trace generation.
        """
        self.state_schema.prepare_initial_state(self.random_generator, self.config, self.random_generate_config)
        '''
        self.state_schema.clear_state()
        local_state_num = random.randint(self.config["init_local_state_num_range"][0], self.config["init_local_state_num_range"][1])
        implicit_state_num = random.randint(self.config["init_implicit_state_num_range"][0], self.config["init_implicit_state_num_range"][1])
        for i in range(implicit_state_num):
            self.state_schema.add_implicit_variable(self.random_generator.random_generate_state(**self.random_generate_config), 0)
        for i in range(local_state_num):
            state = self.random_generator.random_generate_state(**self.random_generate_config)
            self.state_schema.add_local_variable_using_state(state, latest_call=0)
        self.state_schema.align_initial_state()
        '''
    
    def copy_if_state(self, if_trace_generator):
        self.occurence_book = copy.deepcopy(if_trace_generator.occurence_book)
        self.state_schema.init_local_info = copy.deepcopy(if_trace_generator.state_schema.init_local_info)
        if hasattr(if_trace_generator.state_schema, "init_load_info"):
            self.state_schema.init_load_info = copy.deepcopy(if_trace_generator.state_schema.init_load_info)
        if hasattr(if_trace_generator.state_schema, "init_tensor_counter"):
            self.state_schema.init_tensor_counter = copy.deepcopy(if_trace_generator.state_schema.init_tensor_counter)
            self.state_schema.init_weight_counter = copy.deepcopy(if_trace_generator.state_schema.init_weight_counter)
        
    
    def generate_trace(self, 
                       call_num, 
                       this_trace_duplicate_local_variable_map=None, 
                       disable_postprocess=False, 
                       base_call_num=0,
                       enable_coverage=True):
        # 1. Two function calls with exact same parameters should be avoided.
        # 2. Increase pair coverage as much as possible.
        # Data structure that being effected: self.occurence_book, self.trace, self.state_schema, self.random_generator
        if this_trace_duplicate_local_variable_map is None:
            this_trace_duplicate_local_variable_map = dict()
        previous_transition_info = None
        trace = []
        trace_str = []
        for i in range(call_num):
            enter_postprocess_stage, postprocess_transitions = self.state_schema.postprocess_transitions(call_num - i)
            if enter_postprocess_stage and not disable_postprocess:
                for idx, transition in enumerate(postprocess_transitions):
                    transition_pairs = transition["transition_pairs"]
                    for pair in transition_pairs:
                        self.occurence_book[pair] = self.occurence_book.get(pair, 0) + 1
                    transition_name = transition["transition_name"]
                    if transition_name not in this_trace_duplicate_local_variable_map:
                        this_trace_duplicate_local_variable_map[transition_name] = set([])
                    parameter_str = self.state_schema.transform_parameters_to_str(transition["required_parameters"])
                    this_trace_duplicate_local_variable_map[transition_name].add(parameter_str)
                    producer = transition["producer_variable_idx"]
                    if producer is not None:
                        producer = copy.deepcopy(self.state_schema.local_states["variables"][producer])
                    else:
                        producer = None
                    new_transition = self.state_schema.craft_transition(transition, i + idx + 1 + base_call_num, transition_name, producer)
                    implicit, local = new_transition.get_effected_states(self.state_schema)
                    new_transition.apply(implicit, local, self.state_schema)
                    trace.append([transition_name, copy.deepcopy(transition["required_parameters"])])
                    trace_str.append(new_transition.get_program_str())
                break
            else:
                available_transitions = self.state_schema.get_available_transitions(self.random_generator, 
                                                                                i+1, 
                                                                                call_num, 
                                                                                copy.deepcopy(this_trace_duplicate_local_variable_map),
                                                                                previous_transition_info)
                selection_to_coverage_map = dict()
                # Selection includes which transition with which index.
                # energy_map: (transition_pair) -> energy
                # Because energy_map is indexed by pairs,
                # any transition results with multiple pairs (but same energy), will have higher chance to be selected.
                # selection_to_coverage_map: (transition_pair) -> List of idx.
                energy_map = dict()
                # ====== Compute coverage information ======
                new_coverage_list = []
                #normalize_term = 0
                #for pair in self.occurence_book:
                #    normalize_term += self.occurence_book[pair]
                if len(available_transitions) == 0:
                    logger.warning(f"No available transitions for the {i+1}-th call. Terminate the trace generation.")
                    return (trace, trace_str), this_trace_duplicate_local_variable_map, False
                
                has_transition = False
                for transition in available_transitions:
                    if transition not in this_trace_duplicate_local_variable_map:
                        this_trace_duplicate_local_variable_map[transition] = set()
                    for idx, transition_info in enumerate(available_transitions[transition]):
                        # If the transition with exactly the same parameters has been called, skip it.
                        string_parameters = self.state_schema.transform_parameters_to_str(transition_info["required_parameters"])
                        if string_parameters in this_trace_duplicate_local_variable_map[transition]:
                            continue
                        # Compute the coverage information
                        transition_pairs = transition_info["transition_pairs"]
                        has_transition = True
                        #uncovered_pairs = [
                        #                pair for pair in transition_pairs
                        #                if pair not in self.occurence_book
                        #]
                        ## for pair in transition_pairs:
                        ##    if pair not in selection_to_coverage_map:
                        ##        selection_to_coverage_map[pair] = []
                        ##    selection_to_coverage_map[pair].append([transition, idx])
                        ##    if pair not in energy_map:
                        ##        energy_map[pair] = self.occurence_book.get(pair, 0)
                        ##        if energy_map[pair] == 0:
                        ##            has_new_coverage = True
                        #if len(uncovered_pairs) > 0:
                        #    has_new_coverage = True
                            #normalize_term += len(uncovered_pairs)
                        #selection_to_coverage_map[(transition, idx)] = [len(uncovered_pairs), uncovered_pairs, transition_pairs]
                        selection_to_coverage_map[(transition, idx)] = {}
                        for pair in transition_pairs:
                            occ = self.occurence_book.get(pair, 0)
                            selection_to_coverage_map[(transition, idx)][pair] = occ
                            if occ == 0:
                                new_coverage_list.append((transition, idx))
                if not has_transition:
                    logger.warning(f"No available transitions for the {i+1}-th call. Terminate the trace generation.")
                    return (trace, trace_str), this_trace_duplicate_local_variable_map, False
                if len(new_coverage_list) > 0 and enable_coverage:
                    # If there is new coverage, the next selection should be made from the transitions with new coverage.
                    # The more transitions with new coverage, the higher chance to be selected.
                    candidates = [(item, sum(1 for x in selection_to_coverage_map[item].values() if x == 0)) for item in new_coverage_list]
                    selected = random.choices(candidates, weights=[c[1] for c in candidates], k=1)[0][0]
                elif enable_coverage:
                    for transition, idx in selection_to_coverage_map:
                        energy_map[(transition, idx)] = min(selection_to_coverage_map[(transition, idx)].values())
                        energy_map[(transition, idx)] = 1 / (energy_map[(transition, idx)] + 1e-7)
                    # If there is no new coverage, the next selection should be made from the transitions with lower occurence.
                    #for transition, idx in selection_to_coverage_map:
                        #local_occurence = 0
                        #local_occurence = self.occurence_book.get(selection_to_coverage_map[(transition, idx)][2][0], 0)
                        #for pair in selection_to_coverage_map[(transition, idx)][2]:
                            #local_occurence += self.occurence_book.get(pair, 0)
                            #local_occurence = min(local_occurence, self.occurence_book.get(pair, 0))
                        #energy_map[(transition, idx)] = local_occurence
                        #ave_occurence = local_occurence / len(selection_to_coverage_map[(transition, idx)][2])
                        #energy_map[(transition, idx)] = ave_occurence
                        #normalize_term += ave_occurence
                        #print(transition, ave_occurence)
                    #if normalize_term == 0:
                    #    normalize_term = 1
                    #for transition, idx in selection_to_coverage_map:
                        #energy_map[(transition, idx)] = np.log(normalize_term / (energy_map[(transition, idx)] + 1e-7)) # IDF term in TF-IDF
                    #    energy_map[(transition, idx)] = 1 / (energy_map[(transition, idx)] + 1e-7)
                    
                    
                        
                    candidates = [(key, energy_map[key]) for key in energy_map]
                    selected = random.choices(candidates, weights=[c[1] for c in candidates], k=1)[0][0]
                else:
                    candidates = [(transition, idx) for transition in available_transitions.keys() for idx in range(len(available_transitions[transition]))]
                    selected = random.choices(candidates, k=1)[0]
                '''
                sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                for item in sorted_candidates:
                    print(item)
                    print(selection_to_coverage_map[item[0]])
                    print_pairs = selection_to_coverage_map[item[0]][2]
                    for pair in print_pairs:
                        if pair in self.occurence_book:
                            print("occurence: ", self.occurence_book[pair])
                    print('-----')
                '''
                '''
                else:
                    if enable_coverage:
                        if has_new_coverage:
                            selected = random.choices(candidates, k=1)[0][0] # All remaning should be new.
                        else:
                            selected = random.choices(candidates, weights=[c[1] for c in candidates], k=1)[0][0] # [0] for random.choices list return, [0] for the selected transition
                        
                    else:
                        candidates = [(transition, idx) for transition in available_transitions.keys() for idx in range(len(available_transitions[transition]))]
                        selected_idx = random.choices(candidates, k=1)[0]
                        #selected = random.choices(candidates, k=1)[0][0] # [0] for random.choices list return, [0] for the selected transition
                    if enable_coverage:
                        selected_idx = random.choices(selection_to_coverage_map[selected], k=1)[0]
                '''
                for pair in available_transitions[selected[0]][selected[1]]["transition_pairs"]:
                    self.occurence_book[pair] = self.occurence_book.get(pair, 0) + 1
                target_transition_info = available_transitions[selected[0]][selected[1]]
                producer = target_transition_info["producer_variable_idx"]
                if producer is not None:
                    producer = copy.deepcopy(self.state_schema.local_states["variables"][producer])
                else:
                    producer = None
                #producer_info = f"FROM local_variable_idx: {producer}, created_by: {self.state_schema.local_states['variables'][producer].created_by}"
                new_transition = self.state_schema.craft_transition(target_transition_info, i+1+base_call_num, selected[0], producer)
                if selected[0] not in this_trace_duplicate_local_variable_map:
                    this_trace_duplicate_local_variable_map[selected[0]] = set([])
                this_trace_duplicate_local_variable_map[selected[0]].add(self.state_schema.transform_parameters_to_str(target_transition_info["required_parameters"]))
                implicit, local = new_transition.get_effected_states(self.state_schema)
                new_transition.apply(implicit, local, self.state_schema)
                trace.append([selected[0], copy.deepcopy(target_transition_info["required_parameters"])])
                trace_str.append(new_transition.get_program_str())
                previous_transition_info = (selected[0], target_transition_info["required_parameters"]) # (Transition, parameters)
        return (trace, trace_str), this_trace_duplicate_local_variable_map, True


class ProgramEvaluator(ABC):
    """Base class for program evaluators defining common interface"""
    
    @classmethod
    @abstractmethod
    def load(cls, file_path: str, config: Dict[str, Any] = None):
        """Load evaluator state from file"""
        pass

    @abstractmethod
    def store(self, file_path: str):
        """Store evaluator state to file"""
        pass

    @abstractmethod
    def prepare_environment(self, init_implicit_dict, init_local_info, init_load_info=None):
        """Prepare testing environment with initial states"""
        pass

    @abstractmethod 
    def collect_test_case(self, program_info, program):
        """Collect a new test case for evaluation"""
        pass

    @abstractmethod
    def evaluate(self, program: str, threshold: float = 1e-4):
        """Evaluate program against collected test cases"""
        pass
    
def generate_program(trace_generator: TraceGenerator, 
                     trace_length: int, 
                     control_position_candidate: List[int]=None, 
                     enable_if_else: bool = True,
                     enable_coverage: bool = True):
    """
    Generate a program from the trace.
    """
    
    def synthesize_trace_str(program, trace, condition_string="", global_indent="", block_ending="\n\n"):
        if program != "":
            program += "\n" + condition_string
        else:
            program += condition_string
        for block_info in trace[1]:
            block = block_info[0]
            local_indent = block_info[1]
            for line in block:
                program += global_indent + local_indent + line
            program += block_ending
        if len(block_ending) > 0:
            program = program[:-len(block_ending)] + "\n"
        return program
    
    trace_generator.prepare_initial_state()
    result = {
        "init_load_info": None,
        "init_block": None,
        "program": None,
        "main_trace": None,
        "if_trace": None,
        "else_trace": None,
        "occurence_book": None,
        "condition_info": None,
        "init_implict_dict": None,
        "state_schema": {
            "main": None,
            "if": None,
            "else": None,
        },
        "debug_info": None
    }
    is_success = True
    occurence_book_backup = copy.deepcopy(trace_generator.occurence_book)
    result_variable_flag = False
    if not enable_if_else:
        trace, this_trace_duplicate_local_variable_map, is_success = trace_generator.generate_trace(trace_length, 
                                                                                                    disable_postprocess=False,
                                                                                                    this_trace_duplicate_local_variable_map=dict(),
                                                                                                    enable_coverage=enable_coverage)
        if not is_success:
            result["occurence_book"] = occurence_book_backup
            result["debug_info"] = [trace, trace_generator]
            return result, is_success
        result["main_trace"] = [trace, trace_generator.state_schema.get_implicit_states()]
        program = synthesize_trace_str("", trace)
        init_program, returned_init_local_info = Schema.return_init_local_info(trace_generator.state_schema.init_local_info)
        result["program"] = program
        result["init_block"] = [init_program, returned_init_local_info]
        result["init_load_info"] = trace_generator.state_schema.get_load_info()
        result_str = trace_generator.state_schema.postprocess_choose_result()
        if result_str is not None:
            result["program"] += "\n" + result_str
        result["init_implict_dict"] = trace_generator.state_schema.get_implicit_states(current_value=False)
        result["occurence_book"] = trace_generator.occurence_book
        result["state_schema"]["main"] = trace_generator.state_schema
    else:
        #control_position = random.randint(0, trace_length - 1)
        assert control_position_candidate is not None
        assert max(control_position_candidate) <= trace_length - 1
        control_position = control_position_candidate[random.randint(0, len(control_position_candidate) - 1)]
        if control_position == trace_length - 1:
            # The same as non-ifelse case.
            # Left here for diversity.
            trace, this_trace_duplicate_local_variable_map, is_success = trace_generator.generate_trace(trace_length, 
                                                                                                    disable_postprocess=False, 
                                                                                                    this_trace_duplicate_local_variable_map=dict(),
                                                                                                    enable_coverage=enable_coverage)
            if not is_success:
                result["occurence_book"] = occurence_book_backup
                result["debug_info"] = [trace, trace_generator]
                return result, is_success
            result["main_trace"] = [trace, trace_generator.state_schema.get_implicit_states()]
            program = synthesize_trace_str("", trace)
            init_program, returned_init_local_info = Schema.return_init_local_info(trace_generator.state_schema.init_local_info)
            result["program"] = program
            result["init_block"] = [init_program, returned_init_local_info]
            result["init_load_info"] = trace_generator.state_schema.get_load_info()
            result_str = trace_generator.state_schema.postprocess_choose_result()
            if result_str is not None:
                result["program"] += "\n" + result_str
            result["init_implict_dict"] = trace_generator.state_schema.get_implicit_states(current_value=False)
            result["occurence_book"] = trace_generator.occurence_book
            result["state_schema"]["main"] = trace_generator.state_schema
        else:
            program = ""
            if control_position != 0:
                trace, this_trace_duplicate_local_variable_map, is_success = trace_generator.generate_trace(control_position, 
                                                                                                            disable_postprocess=True,
                                                                                                            this_trace_duplicate_local_variable_map=dict(),
                                                                                                            enable_coverage=enable_coverage)
                if not is_success:
                    result["occurence_book"] = occurence_book_backup
                    result["debug_info"] = [trace, trace_generator]
                    return result, is_success
                program = synthesize_trace_str(program, trace)
                result["main_trace"] = [trace, trace_generator.state_schema.get_implicit_states()]
                result["state_schema"]["main"] = trace_generator.state_schema
            else:
                this_trace_duplicate_local_variable_map = dict()
            disable_postprocess = random.random() < 0.5
            if_condition, whether_replace_by_variable, additional_content = trace_generator.state_schema.obtain_if_condition()
            if whether_replace_by_variable:
                if_condition_name, already_exist = trace_generator.state_schema.add_local_constant(if_condition[2])
                if already_exist:
                    # Trivial if-else, change name
                    # Fix this if more if-else is introduced for program generation.
                    if_condition_name, already_exist = trace_generator.state_schema.add_local_constant(if_condition[2], name="condition_variable")
            else:
                if_condition_name = if_condition[2]
            if isinstance(if_condition[1], tuple) or isinstance(if_condition[1], list):
                if_string = f"if {utils.get_nested_path_string(if_condition[0], if_condition[1])} == {if_condition_name}:\n"
            else:
                if_string = f"if {if_condition[0]} == {if_condition_name}:\n"
            
            if additional_content is not None:
                additional_text = "".join(additional_content[0])
                if_string = additional_text + if_string
            if_trace_generator = copy.deepcopy(trace_generator)
            else_trace_generator = copy.deepcopy(trace_generator)
            if_trace, if_this_trace_duplicate_local_variable_map, is_success = if_trace_generator.generate_trace(
                trace_length - control_position,
                this_trace_duplicate_local_variable_map, 
                disable_postprocess=disable_postprocess,
                base_call_num=control_position,
                enable_coverage=enable_coverage
            )
            result["state_schema"]["if"] = if_trace_generator.state_schema
            if not is_success:
                result["occurence_book"] = occurence_book_backup
                result["debug_info"] = [if_trace, if_trace_generator]
                return result, is_success
            program = synthesize_trace_str(program, if_trace, if_string, INDENT, block_ending="\n\n")
            result["if_trace"] = [if_trace, if_trace_generator.state_schema.get_implicit_states()]
            result["condition_info"] = {"if_statement": if_string}
            result["condition_info"]["if_condition_name"] = if_condition_name
            result["condition_info"]["if_condition_value"] = if_condition[2]
            if_result_str = if_trace_generator.state_schema.postprocess_choose_result()
            if if_result_str is not None:
                result_variable_flag = True
                program = synthesize_trace_str(program,
                                               [None, [([if_result_str], "")], ], 
                                               '', 
                                               global_indent=INDENT,
                                               block_ending="")
            
            #else_trace_generator.occurence_book = copy.deepcopy(if_trace_generator.occurence_book)
            #else_trace_generator.state_schema.init_local_info = copy.deepcopy(if_trace_generator.state_schema.init_local_info)
            else_trace_generator.copy_if_state(if_trace_generator)
            
            else_trace, else_this_trace_duplicate_local_variable_map, is_success = else_trace_generator.generate_trace(trace_length - control_position,
                                                                                                           if_this_trace_duplicate_local_variable_map, 
                                                                                                           disable_postprocess=not disable_postprocess,
                                                                                                           base_call_num=control_position,
                                                                                                           enable_coverage=enable_coverage)
            result["state_schema"]["else"] = else_trace_generator.state_schema
            if not is_success:
                result["occurence_book"] = occurence_book_backup
                result["debug_info"] = [else_trace, else_trace_generator]
                return result, is_success
            result["else_trace"] = [else_trace, else_trace_generator.state_schema.get_implicit_states()]
            result["occurence_book"] = copy.deepcopy(else_trace_generator.occurence_book)
            
            else_string = f"else:\n"
            program = synthesize_trace_str(program, else_trace, else_string, INDENT, block_ending="\n\n")
            else_result_str = else_trace_generator.state_schema.postprocess_choose_result()
            if else_result_str is not None:
                result_variable_flag = True
                program = synthesize_trace_str(program,
                                               [None, [([else_result_str], "")], ], 
                                               '', 
                                               global_indent=INDENT,
                                               block_ending="")
            all_init_str = []
            for init_str in if_trace_generator.state_schema.init_local_info:
                # We do not use set because some elements might be non-hashable.
                if (init_str[0], init_str[1]) not in all_init_str:
                    all_init_str.append((init_str[0], init_str[1]))
            for init_str in else_trace_generator.state_schema.init_local_info:
                if (init_str[0], init_str[1]) not in all_init_str:
                    all_init_str.append((init_str[0], init_str[1]))
            init_program = ""
            for init_str in all_init_str:
                init_program += f"{init_str[0]} = {init_str[1]}\n"
            if result_variable_flag:
                init_program += f"{RESULT_NAME} = None\n"
            
            # For loaded variables.
            init_load_info_dict = utils.merge_dicts(if_trace_generator.state_schema.init_load_info, 
                                               else_trace_generator.state_schema.init_load_info)
            if len(init_load_info_dict) > 0:
                result["init_load_info"] = else_trace_generator.state_schema.get_load_info(init_load_info_dict)
                        
            result["program"] = program
            result["init_block"] = [init_program, all_init_str]
            result["init_implict_dict"] = else_trace_generator.state_schema.get_implicit_states(current_value=False)
    return result, is_success


def generate_and_collect_test_case(
    schema_class,
    random_init_class,
    evaluator_class,
    trace_config,
    num_of_apis=5,
    control_position_candidate=[3, 4],
    occurence_book={},
    evaluation_config={},
    enable_if_else=True,
    enable_coverage=True
    ):
    state_schema = schema_class()
    random_init = random_init_class()
    trace_generator = TraceGenerator(
        state_schema,
        random_init,
        trace_config,
        occurence_book
    )
    trace_generator.prepare_initial_state()
    result, is_success = generate_program(trace_generator, num_of_apis, control_position_candidate, enable_if_else, enable_coverage)
    added_changes = utils.get_added_changes(occurence_book, result["occurence_book"])
    occurence_book = result["occurence_book"]
    if not is_success:
        return None, is_success, None, None

    evaluator = evaluator_class(evaluation_config)

    if result["main_trace"] is None:
        implict_list = []
    else:
        implict_list = list(result['main_trace'][1].values())
    if result["if_trace"] is not None:
        for key in result["if_trace"][1]:
            if result["main_trace"] is None or key not in result['main_trace'][1]:
                implict_list.append(result["if_trace"][1][key])

    if result["else_trace"] is not None:
        for key in result["else_trace"][1]:
            if result["main_trace"] is None or key not in result['main_trace'][1]:
                implict_list.append(result["else_trace"][1][key])
    
    program_info = {
            "init_local_str": result["init_block"][0],
            "init_local_info": result["init_block"][1],
            "init_implicit_dict": result['init_implict_dict'],
            "end_implict_list": implict_list,
            "init_load_str": result["init_load_info"][0] if result["init_load_info"] is not None else None,
            "init_load_info": result["init_load_info"][1] if result["init_load_info"] is not None else None
        }
    
    try:
        test_cases = evaluator.collect_test_case(
            program_info = program_info,
            program = result['program']
        )
        if test_cases is None:
            return None, False, None, None

        pass_list, test_case_pass_detail = evaluator.evaluate(result['program'])
        assert pass_list[0] == True
    except Exception as e:
        logger.warning(f"Error in collecting test case: {e}, skip.")
        return evaluator, False, None, None
    
    if result["condition_info"] is not None:
        init_local_info_new = copy.deepcopy(result["init_block"][1])
        idx = None
        for idx, item in enumerate(init_local_info_new):
            if item[0] == result["condition_info"]["if_condition_name"]:
                break
        init_local_info_new[idx] = (init_local_info_new[idx][0], Schema.reverse_if_condition(init_local_info_new[idx][1]))
        init_local_str_new = Schema.return_init_local_info(init_local_info_new)[0]
        program_info = {
            "init_local_str": init_local_str_new,
            "init_local_info": init_local_info_new,
            "init_implicit_dict": result['init_implict_dict'],
            "end_implict_list": implict_list,
            "init_load_str": result["init_load_info"][0] if result["init_load_info"] is not None else None,
            "init_load_info": result["init_load_info"][1] if result["init_load_info"] is not None else None
        }
        try:
            evaluator.collect_test_case(
                program_info = program_info,
                program = result['program']
            )
            pass_list, test_case_pass_detail = evaluator.evaluate(result['program'])
            assert pass_list[0] == True
        except Exception as e:
            logger.warning(f"Error in generating and collecting test case: {e} when reversing the if condition, skip this test case.")
        
    return evaluator, is_success, occurence_book, added_changes