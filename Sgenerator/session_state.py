from typing import Callable, Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import copy
import random
from faker import Faker
from loguru import logger
from collections import defaultdict
from collections import OrderedDict

from Sgenerator.utils import get_nested_path_string
from Sgenerator.state import State, Transition, Schema, RandomInitializer, USER_FUNCTION_PARAM_FLAG, RESPONSE_VARIABLE_TEMP

class SessionType(str, Enum):
    MAIN_SESSION = "main_session"
    VIRTUAL_STUDY = "virtual_study"
    GROUP = "group"
    COMPARISON_SESSION = "comparison_session"
    SETTINGS = "settings"
    CUSTOM_DATA = "custom_data"
    GENOMIC_CHART = "genomic_chart"
    CUSTOM_GENE_LIST = "custom_gene_list"

    def __str__(self):
        return self.value

QUERY_SET = set(["GetSession", "GetSessions", "GetSessionByQuery"])

class SessionRandomInitializer(RandomInitializer):
    """
    Initialize random states for a transition trace.
    """
    def __init__(self):
        super().__init__()
        fake = Faker()
        self.parameter_space = {
            "source": ["collaboration", 
                       "user_portal", 
                       "clinical_portal", 
                       "test_source", 
                       "lab_portal"],
            "type": ["main_session", 
                     "virtual_study", 
                     "group", 
                     "comparison_session", 
                     "settings", 
                     "custom_data", 
                     "genomic_chart", 
                     "custom_gene_list"
                     ],
            "data": {"title": ["Primary Liver Cancer Analysis", 
                               "Breast Cancer Analysis", 
                               "Colorectal Cancer Analysis",
                               "Lung Cancer Analysis",
                               "Ovarian Cancer Analysis",
                               "Pancreatic Cancer Analysis",
                               "Prostate Cancer Analysis",
                               "Renal Cancer Analysis",
                               "Skin Cancer Analysis",
                               "Thyroid Cancer Analysis",
                               "Pan-Cancer TP53 Analysis"], 
                     "description": [ "Main workspace for HCC cohort analysis",
                                     "Cross-cancer study of TP53 mutations",
                                     "Genomic analysis of colorectal cancer",
                                     "Lung cancer research",
                                     "Ovarian cancer study",
                                     "Pancreatic cancer analysis",
                                     "Prostate cancer research",
                                     "Renal cancer study",
                                     "Skin cancer research",
                                     "Thyroid cancer study",
                                     ],
                     "members": [
                         fake.unique.email() for i in range(50)
                     ],
                     "similarities": [i for i in range(100)],
                     "significantDifferences": [i/100 for i in range(100)],
                     }
        }
        self.already_generated = set([])
        self.field_order = list(self.parameter_space["data"].keys())
    
    def transform_data_field_to_str(self, data) -> str:
        target_str = ""
        for field in self.field_order:
            if field in data:
                if isinstance(data[field], (int, float)):
                    target_str += f"{field}: {data[field]:.2f}"
                else:
                    target_str += f"{field}: {data[field]}"
        return target_str
    
    def random_generate_state(self, field_num: int=3, max_random_attempt: int=10):
        assert field_num <= len(self.parameter_space["data"].keys())
        assert field_num >= 1
        source = random.choice(self.parameter_space["source"])
        type = random.choice(self.parameter_space["type"])
        data_field = ["title"]
        if field_num > 1:
            candidate_fields = list(self.parameter_space["data"].keys())
            candidate_fields.remove("title")
            data_field.extend(random.sample(candidate_fields, field_num - 1))
        data = {field: random.choice(self.parameter_space["data"][field]) for field in data_field}
        data_str = self.transform_data_field_to_str(data)
        random_attempt = 1
        while data_str in self.already_generated:
            data_str = self.transform_data_field_to_str(data)   
            if random_attempt > max_random_attempt:
                logger.warning(f"Failed to generate a unique session after {max_random_attempt} attempts")
                break
            random_attempt += 1
        
        return Session(
            id=None,
            source=source,
            type=type,
            checksum="",
            data=data
        )
        
    def random_generate_session_data(self, meta_field: str, field: str=None):
        assert meta_field in ["source", "type", "data"]
        if meta_field != "data":
            return random.choice(self.parameter_space[meta_field])
        else:
            assert field in self.parameter_space["data"]
            return random.choice(self.parameter_space["data"][field])
    

class Session(State):
    """
    Represents a session entity from the session-service API
    """
    def __init__(self, 
                 id: str, 
                 source: str, 
                 type: str, 
                 checksum: str = "", 
                 data: Dict[str, Any] = None):
        if id is None:
            identifier = "local_variable"
        else:
            identifier = id
        super().__init__(identifier=identifier)
        self.id = id
        session_type = SessionType(type) if type is not None else None
        self.initial_value = OrderedDict([
            ("id", id),
            ("source", source),
            ("type", session_type),
            ("checksum", checksum),
            ("data", data)
        ])
        self.current_value = OrderedDict([
            ("id", id),
            ("source", source),
            ("type", session_type),
            ("checksum", checksum),
            ("data", data)
        ])
    
    def get_id(self):
        return self.id

    def __str__(self):
        return_str = "{"
        for key, value in self.current_value.items():
            if isinstance(value, float) or isinstance(value, int):
                return_str += f"{key}: {value:.2f}, "
            else:
                return_str += f"{key}: '{value}', "
        return_str = return_str[:-2] + "}"
        return return_str
    
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
    
class SessionVariableSchema(Schema):
    def __init__(self):
        super().__init__()
        self.local_states = {
            "variables": [],
        }
        self.implicit_states = {
            "sessions": {},
            "latest_call": {},
        }
        self.transitions = [
            GetSessions,
            AddSession,
            GetSessionByQuery,
            GetSession,
            UpdateSession,
            DeleteSession,
            LocalEdit,
        ]
        self.local_call_map = {}
        self.implicit_call_map = {}
        self.init_local_str = []
    
    def add_local_variable(self, local_variable: LocalVariable):
        self.local_states["variables"].append(local_variable)
    
    def clear_state(self):
        self.local_states["variables"] = []
        self.implicit_states["sessions"] = {}
        self.implicit_states["latest_call"] = {}
    
    def add_local_variable_using_state(self, state: Session, latest_call=0, updated=True, created_by=USER_FUNCTION_PARAM_FLAG):
        local_variable = LocalVariable(
            name=created_by+f"_{len(self.local_states['variables'])}",
            value=state,
            updated=updated,
            latest_call=latest_call,
            exist=True,
            created_by=created_by
        )
        self.local_states["variables"].append(local_variable)
    
    def add_implicit_variable(self, session: Session, latest_call: int):
        if session.current_value["id"] is None:
            if len(self.implicit_states["sessions"]) == 0:
                current_max_id = 1
            else:
                current_max_id = max([int(i) for i in self.implicit_states["sessions"].keys()])
            session.current_value["id"] = str(current_max_id + 1)
        self.implicit_states["sessions"][session.current_value["id"]] = session
        self.implicit_states["latest_call"][session.current_value["id"]] = latest_call
    
    def get_latest_call_map(self):
        """Create mappings of latest_call timestamps to variable indices/IDs"""
        local_call_map = defaultdict(list)
        for idx, var in enumerate(self.local_states["variables"]):
            local_call_map[var.latest_call].append(idx)
        
        implicit_call_map = defaultdict(list)
        for session_id in self.implicit_states["sessions"]:
            latest_call = self.implicit_states["latest_call"][session_id]
            implicit_call_map[latest_call].append(session_id)
        
        self.local_call_map = local_call_map
        self.implicit_call_map = implicit_call_map
        
        return local_call_map, implicit_call_map
        
    def align_initial_state(self):
        """
        Align the initial state with the parameter space.
        """
        replace_local_variable = False
        # Maximum number of local variables to be replaced is 2.
        replace_local_variable_num = random.randint(0, min(2, len(self.local_states["variables"])))
        if replace_local_variable_num > 0:
            replace_local_variable = True
            drop_idx = random.sample(range(len(self.local_states["variables"])), replace_local_variable_num)
            for idx in sorted(drop_idx, reverse=True):
                self.local_states["variables"].pop(idx)
        if len(self.local_states["variables"]) != 0:
            implicit_set = set([])
            for key in self.implicit_states["sessions"]:
                s_source = self.implicit_states["sessions"][key].current_value["source"]
                s_type = self.implicit_states["sessions"][key].current_value["type"]
                implicit_set.add((s_source, s_type))
            has_one_corresponding = False
            for local_variable in self.local_states["variables"]:
                if (local_variable.value.current_value["source"], local_variable.value.current_value["type"]) in implicit_set:
                    has_one_corresponding = True
                    break
            # Align the initial state with the parameter space
            # so at least we can call GetSessions.
            if not has_one_corresponding:
                # Choose a random source and type from the parameter space.
                source, type = random.choice(list(implicit_set))
                chosen_local_variable = random.choice(self.local_states["variables"])
                chosen_local_variable.value.initial_value["source"] = source
                chosen_local_variable.value.initial_value["type"] = type
                chosen_local_variable.value.current_value["source"] = source
                chosen_local_variable.value.current_value["type"] = type
        
        if replace_local_variable:
            # Add a new local variable with id available.
            chosen_implicit_variable = random.sample(list(self.implicit_states["sessions"].keys()), replace_local_variable_num)
            for i in chosen_implicit_variable:
                chosen_implicit_variable = self.implicit_states["sessions"][i]
                new_session = Session(id=chosen_implicit_variable.current_value["id"],
                                    source=chosen_implicit_variable.current_value["source"],
                                    type=chosen_implicit_variable.current_value["type"],
                                    data=None)
                self.add_local_variable_using_state(new_session, latest_call=0, updated=True, created_by=USER_FUNCTION_PARAM_FLAG)

        for idx, local_variable in enumerate(self.local_states["variables"]):
            self.init_local_str.append([idx, local_variable.name, str(local_variable.value)])
            
    def obtain_if_condition(self):
        """
        Obtain the condition for the if-else transition.
        """
        for idx in range(len(self.local_states["variables"])-1, -1, -1):
            local_variable = self.local_states["variables"][idx]
            if local_variable.created_by == USER_FUNCTION_PARAM_FLAG:
                meta_field = random.choice([key for key in local_variable.value.current_value.keys() \
                    if key in ["source", "type", "data"] and local_variable.value.current_value[key] is not None])
            else:
                meta_field = "data"
            if meta_field == "source" or meta_field == "type":
                if_condition = (idx, (meta_field, ), local_variable.value.current_value[meta_field])
            elif meta_field == "data":
                field = random.choice(list(local_variable.value.current_value["data"].keys()))  
                if_condition = (idx, ("data", field), local_variable.value.current_value["data"][field])
            return if_condition
            
                
    
    def determine_whether_to_keep_pair(self, previous_transition_info: Tuple, current_transition_info: Tuple) -> bool:
        """
        Determine whether to keep the pair of transitions based on the already choosen one and the current candidate.
        """
        if previous_transition_info is None:
            return True
        if previous_transition_info[0] == current_transition_info[0]:
            if previous_transition_info[0] == "LocalEdit":
                prev_left = previous_transition_info[1]["local_variable_1_idx"]
                current_left = current_transition_info[1]["local_variable_1_idx"]
                prev_meta_field = previous_transition_info[1]["meta_field"]
                current_meta_field = current_transition_info[1]["meta_field"]
                if self.local_states["variables"][prev_left].value.current_value["id"] != self.local_states["variables"][current_left].value.current_value["id"]:
                    return True
                if prev_meta_field == current_meta_field:
                    if prev_meta_field != "data":
                        return False
                    else:
                        prev_field = previous_transition_info[1]["field"]
                        current_field = current_transition_info[1]["field"]
                        if prev_field == current_field:
                            return False
                        else:
                            return True
                else:
                    return True
        elif previous_transition_info[0] == "GetSessions" and (current_transition_info[0] == "GetSession" or current_transition_info[0] == "GetSessionByQuery"):
            # Getsessions already contain information needed for GetSession and GetSessionByQuery.
            # So we should not choose the pair of transitions.
            if previous_transition_info[1]["source"] == current_transition_info[1]["source"] and \
                previous_transition_info[1]["type"] == current_transition_info[1]["type"]:
                return False
            else:
                return True
        return True
    
    def postprocess_transitions(self, remaining_call: int) -> Tuple[bool, List[str]]:
        """
        Postprocess the transitions.
        """
        updated_list = []
        transitions = []
        for idx, local_variable in enumerate(self.local_states["variables"]):
            if local_variable.updated:
                updated_list.append(idx)
        
        updated_list = sorted(updated_list, reverse=True) # The latest updated variable should be submitted first.
        if len(updated_list) >= remaining_call:
            available_num = min(remaining_call, len(updated_list))
            for idx in updated_list:
                variable_id = self.local_states["variables"][idx].value.current_value["id"]
                if variable_id is None:
                    # AddSession should be done
                    target_parameters = {
                        "local_variable": self.local_states["variables"][idx],
                        "local_variable_idx": idx,
                    }
                    latest_call = self.local_states["variables"][idx].latest_call
                    whether_updated = self.local_states["variables"][idx].updated
                    producer_variable_idx = idx
                    transition_name = "AddSession"
                    transition_pairs = [self.form_pair_transition(self.local_states["variables"][idx].value, transition_name)]
                else:
                    # UpdateSession should be done
                    target_parameters = {
                        "source": self.local_states["variables"][idx].value.current_value["source"],
                        "type": self.local_states["variables"][idx].value.current_value["type"],
                        "id": self.local_states["variables"][idx].value.current_value["id"],
                        "data": self.local_states["variables"][idx].value.current_value["data"],
                    }
                    latest_call = self.local_states["variables"][idx].latest_call
                    whether_updated = self.local_states["variables"][idx].updated
                    producer_variable_idx = idx
                    transition_name = "UpdateSession"
                    transition_pairs = [self.form_pair_transition(self.local_states["variables"][idx].value, transition_name)]
                    
                transitions.append({
                        "required_parameters": target_parameters,
                        "latest_call": latest_call,
                        "whether_updated": whether_updated,
                        "producer_variable_idx": producer_variable_idx,
                        "transition_pairs": transition_pairs,
                        "transition_name": transition_name,
                })
                available_num -= 1
                if available_num == 0:
                    break
            return True, transitions
        else:
            return False, []
    
    def get_available_transitions(self, 
                                  random_generator: SessionRandomInitializer, 
                                  current_call: int, 
                                  max_call: int,
                                  duplicate_local_variable_map,
                                  previous_transition_info: Tuple = None,
                                  ) -> Dict[str, Transition]:
        available_transitions = {}
        self.get_latest_call_map()
        for transition in self.transitions:
            if transition.__name__ not in duplicate_local_variable_map:
                duplicate_local_variable_map[transition.__name__] = set([])
            if transition.__name__ == "LocalEdit":
                if len(self.local_states["variables"]) == 0:
                    continue
                latest_call = max(self.local_call_map.keys())
                local_candidate = [idx for idx in self.local_call_map[latest_call] if self.local_states["variables"][idx].exist]
                if len(local_candidate) == 0:
                    continue
                local_variable_1_idx = random.choice(local_candidate)
                local_variable = self.local_states["variables"][local_variable_1_idx]
                available_indices = [idx for idx in range(len(self.local_states["variables"])) 
                                                        if idx != local_variable_1_idx]
                # Create a new local variable for the edition
                # This will be the user-provided local variable.
                if len(available_indices) == 0 or (random.random() < 1 / (1 + len(self.local_states["variables"]))):
                    # Choose random parameter
                    meta_field = random.choice(["source", "type", "data"])
                    if meta_field == "data":
                        local_data = local_variable.value.current_value["data"]
                        if local_data is None or len(local_data.keys()) == 0:
                            field = "title"
                        else:
                            field = random.choice(list(local_variable.value.current_value["data"].keys()))
                        value = random_generator.random_generate_session_data(meta_field, field)
                    else:
                        field = None
                        value = random_generator.random_generate_session_data(meta_field)
                    # Remember to update the local variable when actually applying the transition.
                    local_variable_2_idx = None
                else:
                    # Choose from existing variables
                    local_variable_2_idx = random.choice(available_indices)
                    meta_field = [field for field in self.local_states["variables"][local_variable_2_idx].value.current_value.keys() 
                                 if field in ["source", "type", "data"] and self.local_states["variables"][local_variable_2_idx].value.current_value[field] is not None]
                    meta_field = random.choice(meta_field)
                    if meta_field == "data":
                        field = random.choice(list(self.local_states["variables"][local_variable_2_idx].value.current_value["data"].keys()))
                        value = self.local_states["variables"][local_variable_2_idx].value.current_value["data"][field]
                    else:
                        field = None
                        value = self.local_states["variables"][local_variable_2_idx].value.current_value[meta_field]
                

                if transition.__name__ not in available_transitions:
                    available_transitions[transition.__name__] = []
                    
                transition_pair = self.form_pair_transition(local_variable.value, transition.__name__)
                transition_pairs = [transition_pair]
                if local_variable_2_idx is None:
                    transition_pairs.append(("NONE", transition.__name__))
                else:
                    transition_pairs.append(self.form_pair_transition(self.local_states["variables"][local_variable_2_idx].value, transition.__name__))
                
                target_parameters = {
                        "local_variable_1_idx": local_variable_1_idx,
                        "local_variable_2_idx": local_variable_2_idx,
                        "meta_field": meta_field,
                        "field": field,
                        "value": value,
                    }
                if self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, target_parameters)):
                    duplicate_str = self.transform_parameters_to_str(target_parameters)
                    if duplicate_str not in duplicate_local_variable_map[transition.__name__]:
                        duplicate_local_variable_map[transition.__name__].add(duplicate_str)
                        available_transitions[transition.__name__].append({
                            "required_parameters": target_parameters,
                            "latest_call": local_variable.latest_call,
                            "whether_updated": local_variable.updated,
                            "producer_variable_idx": local_variable_2_idx,
                            "transition_pairs": transition_pairs,
                        })
            elif transition.__name__ == "AddSession":
                # AddSession should be done after LocalEdit.
                for idx, local_variable in enumerate(self.local_states["variables"]):
                    if local_variable.updated and local_variable.exist:
                        # This local variable is updated by LocalEdit.
                        # If not updated, we will not treat it as the candidate of AddSession.
                        if local_variable.value.current_value["data"] is None:
                            continue
                        if transition.__name__ not in available_transitions:
                            available_transitions[transition.__name__] = []
                        transition_pairs = [self.form_pair_transition(local_variable.value, transition.__name__)]
                        target_parameters = {
                            "local_variable": local_variable,
                            "local_variable_idx": idx,
                        }
                        if self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, target_parameters)):
                            duplicate_str = self.transform_parameters_to_str(target_parameters)
                            if duplicate_str not in duplicate_local_variable_map[transition.__name__]:
                                duplicate_local_variable_map[transition.__name__].add(duplicate_str)
                                available_transitions[transition.__name__].append({
                                    "required_parameters": target_parameters,
                                    "latest_call": local_variable.latest_call,
                                    "whether_updated": local_variable.updated,
                                    "producer_variable_idx": idx,
                                    "transition_pairs": transition_pairs,
                                })
                    else:
                        continue
            else:
                for idx, local_variable in enumerate(self.local_states["variables"]):
                    if local_variable.exist is False:
                        continue
                    for required_parameters in transition.get_required_parameters():
                        satisfied = True
                        # Avoid empty query return.
                        if transition.__name__ in QUERY_SET:
                            if len(local_variable.value.transitions) > 0 and local_variable.value.transitions[-1]['name'] in QUERY_SET:
                                # Already queried. Skip.
                                continue
                            has_implicit_variable = False
                            for key in self.implicit_states["sessions"]:
                                if self.implicit_states["sessions"][key].exist is False:
                                    continue
                                if self.implicit_states["sessions"][key].current_value["source"] == local_variable.value.current_value["source"] or \
                                    self.implicit_states["sessions"][key].current_value["type"] == local_variable.value.current_value["type"]:
                                    if transition.__name__ == "GetSession":
                                        if self.implicit_states["sessions"][key].current_value["id"] == local_variable.value.current_value["id"]:
                                            has_implicit_variable = True
                                            break
                                        else:
                                            continue
                                    elif transition.__name__ == "GetSessionByQuery":
                                        if local_variable.value.current_value["data"] is None:
                                            continue
                                        else:
                                            has_implicit_variable = True
                                            break
                                    else:
                                        has_implicit_variable = True
                                        break
                            if not has_implicit_variable:
                                satisfied = False
                                break
                        if satisfied:
                            target_parameters = {}
                            if transition.__name__ == "GetSessionByQuery":
                                field = random.choice(list(local_variable.value.current_value["data"].keys()))
                                target_parameters = {
                                    "source": local_variable.value.current_value["source"],
                                    "type": local_variable.value.current_value["type"],
                                    "field": field,
                                    "value": local_variable.value.current_value["data"][field],
                                }
                            else:
                                for parameter in required_parameters:
                                    if parameter not in local_variable.value.current_value or local_variable.value.current_value[parameter] is None:
                                        satisfied = False
                                        break
                                    target_parameters[parameter] = local_variable.value.current_value[parameter]
                        
                        if satisfied:
                            duplicate_str = self.transform_parameters_to_str(target_parameters)
                            if duplicate_str in duplicate_local_variable_map[transition.__name__]:
                                satisfied = False
                            
                        if satisfied:
                            if current_call < max_call and transition.__name__ == "DeleteSession":
                                # When this is not the last call, we should not delete the only local variable.
                                # Because it can sometimes cause the termination of the trace generation.
                                exist_local_variable = [variable for variable in self.local_states["variables"] if variable.exist]
                                exist_local_variable = set([i.value.current_value["id"] for i in exist_local_variable if i.value.current_value["id"] is not None])
                                if len(exist_local_variable) == 1:
                                    satisfied = False
                                    break
                            if transition.__name__ not in available_transitions:
                                available_transitions[transition.__name__] = []
                            transition_pairs = [self.form_pair_transition(local_variable.value, transition.__name__)]
                            if self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, target_parameters)):
                                available_transitions[transition.__name__].append({
                                    "required_parameters": target_parameters,
                                    "latest_call": local_variable.latest_call,
                                    "whether_updated": local_variable.updated,
                                    "producer_variable_idx": idx,
                                    "transition_pairs": transition_pairs,
                                })
                                duplicate_local_variable_map[transition.__name__].add(duplicate_str)
        return available_transitions
    
    def craft_transition(self, parameters, calling_timestamp, transition, producer="None"):
        if transition == "LocalEdit":
            if parameters["local_variable_2_idx"] is None:
                # This local variable is generaed randomly in the get_available_transitions function.
                # Because it is already used in LocalEdit,
                # we set the updated to False.
                new_local_session = Session(id=None, source=None, type=None)
                if parameters["meta_field"] == "data" and parameters["field"] is not None:
                    new_local_session.initial_value["data"] = {parameters["field"]: parameters["value"]}
                    new_local_session.current_value["data"] = {parameters["field"]: parameters["value"]}
                else:
                    new_local_session.initial_value[parameters["meta_field"]] = parameters["value"]
                    new_local_session.current_value[parameters["meta_field"]] = parameters["value"]
                
                new_local_variable = LocalVariable(
                    name=USER_FUNCTION_PARAM_FLAG+f"_{len(self.local_states['variables'])}",
                    value=new_local_session, 
                    updated=False,  # Because it is already used to edit another local variable.
                    latest_call=calling_timestamp, 
                    created_by=USER_FUNCTION_PARAM_FLAG
                )
                self.add_local_variable(new_local_variable)
                parameters["local_variable_2_idx"] = len(self.local_states["variables"]) - 1
                producer = copy.deepcopy(self.local_states["variables"][parameters["local_variable_2_idx"]])
        
        transition_class = globals()[transition]
        new_transition = transition_class(
            parameters=parameters, 
            calling_timestamp=calling_timestamp
            )
        new_transition.producer = producer
        return new_transition
    

class GetSessions(Transition):
    """
    Represents a query to the session-service API
    Get all sessions of a given type
    Parameters:
        - source: the source of the sessions
        - type: the type of the sessions
    Side Effects:
        - GET: get all sessions of a given type
    """
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        if isinstance(parameters["type"], str):
            parameters["type"] = SessionType(parameters["type"])
        assert "source" in parameters
        assert "type" in parameters
        super().__init__(name="GetSessions", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        
    def __str__(self):
        source = self.parameters["source"]
        type = self.parameters["type"]
        url = "[BASE_URL]/api/sessions/{source}/{type}"
        target_transition = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + f" = request.get(url)"
        return (
            f"source = {self.producer.name}['source']\n"
            f"type = {self.producer.name}['type']\n"
            f"url = f'{url}'\n"
            f"{target_transition}"
            f"    # FROM {self.producer.name}. source={source}, type={type}"
        )
    
    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["source", "type"]]
    
    def get_effected_states(self, variable_schema: SessionVariableSchema) -> List[str]:
        result = []
        for state_id in variable_schema.implicit_states["sessions"]:
            state = variable_schema.implicit_states["sessions"][state_id]
            if isinstance(state, Session):
                if self.parameters["source"] == state.current_value["source"] and \
                    state.exist and \
                    self.parameters["type"] == state.current_value["type"]:
                    result.append(state.current_value["id"])
        return result, None
    
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema: SessionVariableSchema):
        for idx, state_id in enumerate(implicit_states):
            state = variable_schema.implicit_states["sessions"][state_id]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            variable_schema.implicit_states["latest_call"][state_id] = self.calling_timestamp
            
            local_variable = LocalVariable(
                name=RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + f"[{idx}]",
                value=copy.deepcopy(state), 
                updated=False, 
                latest_call=self.calling_timestamp, 
                created_by=f"{self.name}@{self.calling_timestamp}"
            )
            variable_schema.add_local_variable(local_variable)

        return variable_schema

class AddSession(Transition):
    """
    Represents a query to the session-service API
    Add a new session to the session-service API
    Parameters:
        - source: the source of the session
        - type: the type of the session
        - data: the data of the session 
    Side Effects:
        - POST: add a new session to the session-service API
    """
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        #assert "source" in parameters
        #assert "type" in parameters
        #assert "data" in parameters
        assert "local_variable" in parameters
        assert "local_variable_idx" in parameters
        assert parameters["local_variable"].value.current_value["source"] is not None
        assert parameters["local_variable"].value.current_value["type"] is not None
        assert parameters["local_variable"].value.current_value["data"] is not None
        self.local_variable = parameters["local_variable"]
        self.local_variable_idx = parameters["local_variable_idx"]
        parameters = {
            "source": self.local_variable.value.current_value["source"],
            "type": self.local_variable.value.current_value["type"],
            "data": self.local_variable.value.current_value["data"],
        }
        super().__init__(name="AddSession", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.string_parameters = None
        
    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["local_variable", "local_variable_idx"]]
    
    def get_effected_states(self, variable_schema: SessionVariableSchema) -> List[str]:
        state_id_collection = [int(i) for i in variable_schema.implicit_states["sessions"].keys()]
        current_max_id = max(state_id_collection) + 1
        #new_session = Session(id=current_max_id + 1, source=self.parameters["source"], type=self.parameters["type"], data=self.parameters["data"])
        return [current_max_id], [self.local_variable_idx]
    
    def __str__(self):
        if self.string_parameters is None:
            return f"ADD Session with local_variable_idx {self.local_variable_idx}"
        else:
            source = self.string_parameters["source"]
            type = self.string_parameters["type"]
            url = "[BASE_URL]/api/sessions/{source}/{type}"
            target_transition = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + f" = request.post({url}"
            data = f"{self.producer.name}['data']"
            return (
                f"source = {self.producer.name}['source']\n"
                f"type = {self.producer.name}['type']\n"
                f"url = f'{url}'\n"
                f"{target_transition}"
                f", headers={{\"Content-Type\": \"application/json\"}}"
                f", data=json.dumps({data})) # Local_variable_idx {self.local_variable_idx} being modified. source={source}, type={type}, data={self.string_parameters['data']}"
            )
    
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema: SessionVariableSchema):
        assert len(implicit_states) == 1
        new_session = Session(id=str(implicit_states[0]), 
                              source=self.parameters["source"], 
                              type=self.parameters["type"], 
                              data=self.parameters["data"],
                              #created_by=f"{self.name}@{self.calling_timestamp}"
        )
        new_session.transitions.append({
            "name": self.name,
            "parameters": self.parameters,
        })
        variable_schema.add_implicit_variable(new_session, self.calling_timestamp)
        
        variable_schema.local_states["variables"][local_states[0]].value = new_session # Add session will return.
        variable_schema.local_states["variables"][local_states[0]].updated = False
        variable_schema.local_states["variables"][local_states[0]].latest_call = self.calling_timestamp
        self.string_parameters = {
            "source": self.parameters["source"],
            "type": self.parameters["type"],
            "data": self.parameters["data"],
        }
        return variable_schema

class GetSessionByQuery(Transition):
    """
    Represents a query to the session-service API
    Get a session by query
    Parameters:
        - source: the source of the session
        - type: the type of the session
        - field: the data field to query
        - value: the data value to query
    Side Effects:
        - GET: get a session by query
    """
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "source" in parameters
        assert "type" in parameters
        assert "field" in parameters
        assert "value" in parameters
        if isinstance(parameters["type"], str):
            parameters["type"] = SessionType(parameters["type"])
        super().__init__(name="GetSessionByQuery", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp

    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["source", "type", "field", "value"]]
    
    def get_effected_states(self, variable_schema: SessionVariableSchema) -> List[str]:
        result = []
        for state_id in variable_schema.implicit_states["sessions"]:
            state = variable_schema.implicit_states["sessions"][state_id]
            if isinstance(state, Session):
                if self.parameters["field"] not in state.current_value["data"]:
                    continue
                if state.current_value["source"] == self.parameters["source"] and state.exist and \
                    state.current_value["type"] == self.parameters["type"] and \
                    state.current_value["data"][self.parameters["field"]] == self.parameters["value"]: ## Match --> "=="
                    result.append(state.current_value["id"])
        return result, None
    
    def __str__(self):
        source = self.parameters["source"]
        type = self.parameters["type"]
        url = "[BASE_URL]/api/sessions/{source}/{type}/query"
        target_transition = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + f" = request.get({url}"
        return (
            f"source = {self.producer.name}['source']\n"
            f"type = {self.producer.name}['type']\n"
            f"field = {self.producer.name}['field']\n"
            f"value = {self.producer.name}['value']\n"
            f"url = f'{url}'\n"
            f"{target_transition}"
            f", params={{\"field\": field, \"value\": value}})"
            #f' params={{"field": "{self.parameters["field"]}", "value": "{self.parameters["value"]}"}})'
            f'  # source={source}, type={type}, field={self.parameters["field"]}, value={self.parameters["value"]}'
        )
    
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema: SessionVariableSchema):
        for idx, state_id in enumerate(implicit_states):
            state = variable_schema.implicit_states["sessions"][state_id]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            local_variable = LocalVariable(
                name=RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + f"[{idx}]",
                value=copy.deepcopy(state), 
                updated=False, 
                latest_call=self.calling_timestamp, 
                created_by=f"{self.name}@{self.calling_timestamp}"
            )
            variable_schema.add_local_variable(local_variable)
        return variable_schema

# We do not involve fetchSessionByQuery in the state transition graph. Because it has similar effect as getSessionByQuery with more complex MongoDB filter options.
# Adding it fine, but it will complicate the state transition graph.

class GetSession(Transition):
    """
    Represents a query to the session-service API
    Get a session by query
    Parameters:
        - source: the source of the session
        - type: the type of the session
        - id: the id of the session
    Side Effects:
        - GET: get a session by query
    """
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "source" in parameters
        assert "type" in parameters
        assert "id" in parameters
        if isinstance(parameters["type"], str):
            parameters["type"] = SessionType(parameters["type"])
        super().__init__(name="GetSession", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
    
    def get_effected_states(self, variable_schema: SessionVariableSchema) -> List[str]:
        result = []
        for state_id in variable_schema.implicit_states["sessions"]:
            state = variable_schema.implicit_states["sessions"][state_id]
            if isinstance(state, Session):
                if state.current_value["source"] == self.parameters["source"] and state.exist and \
                    state.current_value["type"] == self.parameters["type"] and state.current_value["id"] == self.parameters["id"]:
                    result.append(state.current_value["id"])
        return result, None

    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["source", "type", "id"]]
    
    def __str__(self):
        if self.producer is None:
            return (
                RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + " = request.get("
                f'"[BASE_URL]/api/sessions/{self.parameters["source"]}/{str(self.parameters["type"])}/{self.parameters["id"]})"'
            )
        else:
            source = self.parameters["source"]
            type = self.parameters["type"]
            id = self.parameters["id"]
            url = '[BASE_URL]/api/sessions/{source}/{type}/{id}'
            return (
                f"source = {self.producer.name}['source']\n"
                f"type = {self.producer.name}['type']\n"
                f"id = {self.producer.name}['id']\n"
                f"url = f'{url}'\n"
                f"{RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)} = request.get({url})"
                f'  # source={source}, type={type}, id={id}'
            )
    
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema: SessionVariableSchema):
        for state_id in implicit_states:
            state = variable_schema.implicit_states["sessions"][state_id]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            local_variable = LocalVariable(
                name=RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp),
                value=copy.deepcopy(state), 
                updated=False, 
                latest_call=self.calling_timestamp, 
                created_by=f"{self.name}@{self.calling_timestamp}"
                )
            variable_schema.add_local_variable(local_variable)
        return variable_schema

class UpdateSession(Transition):
    """
    Represents a query to the session-service API
    Update a session
    Parameters:
        - source: the source of the session
        - type: the type of the session
        - id: the id of the session
        - data: the data of the session
    Side Effects:
        - PUT: update a session
    """
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "source" in parameters
        assert "type" in parameters
        assert "id" in parameters
        assert "data" in parameters
        if isinstance(parameters["type"], str):
            parameters["type"] = SessionType(parameters["type"])
        super().__init__(name="UpdateSession", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
    
    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["source", "type", "id", "data"]]
    
    def get_effected_states(self, variable_schema: SessionVariableSchema) -> List[str]:
        return [self.parameters["id"]], None
    
    def __str__(self):
        if self.producer is None:
            return (
                RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + " = request.put("
                f'"[BASE_URL]/api/sessions/{self.parameters["source"]}/{str(self.parameters["type"])}/{self.parameters["id"]}',
                f'headers={{"Content-Type\": \"application/json\"}},'
                f'data=json.dumps({self.parameters["data"]}))'
        )
        else:
            source = self.parameters["source"]
            type = self.parameters["type"]
            id = self.parameters["id"]
            url = '[BASE_URL]/api/sessions/{source}/{type}/{id}'
            return (
                f"source = {self.producer.name}['source']\n"
                f"type = {self.producer.name}['type']\n"
                f"id = {self.producer.name}['id']\n"
                f"url = f'{url}'\n"
                f"{RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)} = request.put({url}, headers={{\"Content-Type\": \"application/json\"}}, data=json.dumps({self.producer.name}['data']))"
                f'  # source={source}, type={type}, id={id}, data={self.parameters["data"]}'
            )
    
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema: SessionVariableSchema):
        assert len(implicit_states) == 1
        if implicit_states[0] not in variable_schema.implicit_states["sessions"]:
            # Create a new session
            new_session = Session(id=implicit_states[0], 
                                  source=self.parameters["source"], 
                                  type=self.parameters["type"], 
                                  data=self.parameters["data"],
                                  #created_by=f"{self.name}@{self.calling_timestamp}"
                                  )
            new_session.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            variable_schema.add_implicit_variable(new_session, self.calling_timestamp)
        else:
            # check whether the side effect is valid
            state = variable_schema.implicit_states["sessions"][implicit_states[0]]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            state.current_value["data"] = self.parameters["data"]

        for local_variable in variable_schema.local_states["variables"]:
            if local_variable.value.current_value["id"] == implicit_states[0]:
                local_variable.updated = False # local variable update has been submitted to the remote.
        return variable_schema

class DeleteSession(Transition):
    """
    Represents a query to the session-service API
    Delete a session
    Parameters:
        - source: the source of the session
        - type: the type of the session
        - id: the id of the session
    Side Effects:
        - DELETE: delete a session
    """
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "source" in parameters
        assert "type" in parameters
        assert "id" in parameters
        if isinstance(parameters["type"], str):
            parameters["type"] = SessionType(parameters["type"])
        super().__init__(name="DeleteSession", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp

    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["source", "type", "id"]]
    
    def get_effected_states(self, variable_schema: SessionVariableSchema) -> List[str]:
        return [self.parameters["id"]], None
    
    def __str__(self):
        if self.producer is None:
            return (
                RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + " = request.delete("
                f'"[BASE_URL]/api/sessions/{self.parameters["source"]}/{str(self.parameters["type"])}/{self.parameters["id"]})'
            )
        else:
            source = self.parameters["source"]
            type = self.parameters["type"]
            id = self.parameters["id"]
            url = '[BASE_URL]/api/sessions/{source}/{type}/{id}'
            return (
                f"source = {self.producer.name}['source']\n"
                f"type = {self.producer.name}['type']\n"
                f"id = {self.producer.name}['id']\n"
                f"url = f'{url}'\n"
                f"{RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)} = request.delete({url})"
                f'  # source={source}, type={type}, id={id}'
            )
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema: SessionVariableSchema):
        assert len(implicit_states) == 1
        if implicit_states[0] not in variable_schema.implicit_states["sessions"]:
            raise ValueError(f"Session {implicit_states[0]} does not exist")
        state = variable_schema.implicit_states["sessions"][implicit_states[0]]
        assert state.exist
        state.transitions.append({
            "name": self.name,
            "parameters": self.parameters,
        })
        state.exist = False
        variable_schema.implicit_states["latest_call"][implicit_states[0]] = self.calling_timestamp
        
        for local_variable in variable_schema.local_states["variables"]:
            if local_variable.value.current_value["id"] == implicit_states[0]:
                local_variable.updated = False
                local_variable.exist = False
        return variable_schema

class LocalEdit(Transition):
    """
    Represents a edit operation of local variables.
    """
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        # local_variable_1 <- local_variable_2
        assert "local_variable_1_idx" in parameters # target
        assert "local_variable_2_idx" in parameters # source
        assert "meta_field" in parameters
        assert parameters["meta_field"] in ["source", "type", "data"]
        if parameters["meta_field"] == "data":
            assert "field" in parameters and parameters["field"] is not None
        assert "value" in parameters
        super().__init__(name="LocalEdit", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.string_parameters = None
        
    @staticmethod
    def get_required_parameters() -> List[List[str]]:
        return [["local_variable_1_idx", "local_variable_2_idx", "meta_field", "field", "value"]] # data
    
    def get_effected_states(self, variable_schema: SessionVariableSchema) -> List[str]:
        '''
        result = []
        last_call = -1 # Only the latest call will be considered.
        for idx, local_variable in enumerate(variable_schema.local_states["variables"]):
            if local_variable.value.current_value["id"] == self.parameters["session"].current_value["id"] and local_variable.exist:
                if local_variable.latest_call > last_call:
                    last_call = local_variable.latest_call
                    result = [idx]
                elif local_variable.latest_call == last_call:
                    result.append(idx)
        '''
        return None, [self.parameters["local_variable_1_idx"]]
    
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema: SessionVariableSchema):
        assert len(local_states) == 1
        local_variable = variable_schema.local_states["variables"][local_states[0]]
        if self.parameters["field"] is None:
            local_variable.value.current_value[self.parameters["meta_field"]] = self.parameters["value"]
        else:
            if local_variable.value.current_value["data"] is None:
                local_variable.value.current_value["data"] = {}
            local_variable.value.current_value["data"][self.parameters["field"]] = self.parameters["value"]
        local_variable.updated = True
        local_variable.latest_call = self.calling_timestamp
        variable_schema.local_states["variables"][self.parameters["local_variable_2_idx"]].updated = False
        self.string_parameters = {
            "left_variable": local_variable.name,
            "right_variable": variable_schema.local_states["variables"][self.parameters["local_variable_2_idx"]].name,
            "meta_field": self.parameters["meta_field"],
            "field": self.parameters["field"],
            "value": self.parameters["value"]
        }
        return variable_schema

    def __str__(self):
        if self.string_parameters is None:
            return (
                f"LOCAL EDIT {self.parameters['session'].current_value['id']} with field {self.parameters['field']} to {self.parameters['value']}"
            )
        else:
            all_fields = []
            all_fields.append(self.string_parameters["meta_field"])
            if self.string_parameters["field"] is not None:
                all_fields.append(self.string_parameters["field"])
            left_string = get_nested_path_string(self.string_parameters["left_variable"], all_fields)
            right_string = get_nested_path_string(self.string_parameters["right_variable"], all_fields)
            return (
                f"{left_string} = {right_string}"
                f'  # Local Variable Edit from {self.producer.name}'
            )




if __name__ == "__main__":
    
    # ======= Example Usage and Test Case =======
    
    # ==== Initialize the variable schema ====
    all_state = SessionVariableSchema()
    init_variable_1 = LocalVariable(
        name="local_1",
        value=Session(id="local_1", 
                        source="test", 
                        type="main_session", 
                        data={"title": "user_provided_data_1"},
                        created_by="init_variable_1"
                        ), 
                    updated=False, 
                    latest_call=0,
                    created_by="local_init_variable_1"
                    )

    init_variable_2 = LocalVariable(
        name="local_2",
        value = Session(id="local_2", 
                        source="test", 
                        type="virtual_study", 
                        data={"title": "user_provided_data_2"},
                        created_by="init_variable_2"
                        ),
        updated=False,
        latest_call=0,
        created_by="local_local_init_variable_2"
    )

    all_state.add_local_variable(init_variable_1)
    all_state.add_local_variable(init_variable_2)

    state_1 = Session(id=1, source="test", type="main_session", 
                    data={"title": "my main portal session",
                            "description": "this is another example"},
                    created_by="implicit_init_state_1"
    )


    state_2 = Session(id=2, source="test", type="main_session", 
                    data={"title": "my main portal session",
                            "description": "This is the main session"},
                    created_by="implicit_init_state_2"
    )

    all_state.implicit_states['sessions'][state_1.current_value["id"]] = state_1
    all_state.implicit_states['sessions'][state_2.current_value["id"]] = state_2

    # Get Session Transition
    get_session = GetSessions({"source": "test", "type": "main_session"}, calling_timestamp=1)
    implicit_effected_states, local_effected_states = get_session.get_effected_states(all_state)
    get_session.apply(implicit_effected_states, local_effected_states, all_state)

    # Add Session Transition
    add_session = AddSession(
        {
            "source": all_state.local_states['variables'][0].value.current_value["source"],
            "type": all_state.local_states['variables'][0].value.current_value["type"],
            "data": all_state.local_states['variables'][0].value.current_value["data"],
        },
        calling_timestamp=2
    )
    implicit_effected_states, local_effected_states = add_session.get_effected_states(all_state)
    add_session.apply(implicit_effected_states, local_effected_states, all_state)

    # Get Session Transition
    query_session = GetSession(
        parameters = {
            "source": "test",
            "type": "main_session",
            "id":1
        },
        calling_timestamp=3
    )

    implicit_effected_states, local_effected_states = query_session.get_effected_states(all_state)
    query_session.apply(implicit_effected_states, local_effected_states, all_state)

    # Update Session Transition
    update_session = UpdateSession(
        parameters = {
            "source": "test",
            "type": "main_session",
            "id":1,
            "data": {
                "descriptions": "this is updated session"
            }
        },
        calling_timestamp=4
    )
    implicit_effected_states, local_effected_states = update_session.get_effected_states(all_state)
    update_session.apply(implicit_effected_states, local_effected_states, all_state)

    # Delete Session Transition
    delete_session = DeleteSession(
        parameters = {
            "source": all_state.local_states['variables'][-1].value.current_value["source"],
            "type": all_state.local_states['variables'][-1].value.current_value["type"],
            "id": all_state.local_states['variables'][-1].value.current_value["id"]
        },
        calling_timestamp=5
    )
    implicit_effected_states, local_effected_states = delete_session.get_effected_states(all_state)
    delete_session.apply(implicit_effected_states, local_effected_states, all_state)

    # Local Edit Transition
    local_edit = LocalEdit(
        parameters = {
            "session": all_state.local_states["variables"][-1].value,
            "meta_field": "source",
            "value": "new_test"
        },
        calling_timestamp=6
    )
    implicit_effected_states, local_effected_states = local_edit.get_effected_states(all_state)
    local_edit.apply(implicit_effected_states, local_effected_states, all_state)
    
    
    
    assert len(all_state.local_states['variables']) == 5
    assert len(all_state.local_states['variables'][-1].value.transitions) == 2
    assert all_state.local_states['variables'][-1].value.current_value['source'] == "new_test"
    assert all_state.implicit_states['sessions'][1].current_value['data']['descriptions'] == "this is updated session"
    assert len(all_state.implicit_states['sessions']) == 3
    
    