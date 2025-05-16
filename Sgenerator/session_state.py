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
import requests
import json
import traceback
import typer
from typing import Annotated
from pathlib import Path
import yaml
import pickle
import os

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

from .utils import get_nested_path_string, get_added_changes
from .state import State, Transition, Schema, RandomInitializer, USER_FUNCTION_PARAM_FLAG, RESPONSE_VARIABLE_TEMP
from .state import INDENT, RESULT_NAME, TraceGenerator, generate_program

SESSION_SUMMARY_PROMPT = '''The below code is about a session service API provides CRUD operations for managing cBioPortal user sessions in MongoDB. 
It supports various session types such as main sessions and virtual studies, organized by source and type parameters. 
The API returns session objects containing ID, source, type, and data in JSON format.
API Functions:
Get Sessions (GET /api/sessions/{source}/{type}) - Retrieves all sessions of specified type/source
Add Session (POST /api/sessions/{source}/{type}) - Creates new session with JSON payload
Query Sessions (GET /api/sessions/{source}/{type}/query) - Finds sessions by field/value pair
Advanced Query (POST /api/sessions/{source}/{type}/query/fetch) - Complex searches using MongoDB-like filters
Session Management (GET/PUT/DELETE /api/sessions/{source}/{type}/{id}) - Get, update, or delete individual sessions
Service Info (GET /info) - Returns basic service information
For your descrptions, directly use {BASE_URL} to refer to the Query URL. It will be replaced later when the evaluation is performed. 
Please be careful when describing "updates" (it is ambiguous whether it is local variable updates or remote updates through APIs).
'''

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
POST_SET = set(["UpdateSession", "AddSession"])
LOOP_ITEM_NAME = "single_item"

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
                     "similarities": [str(i) for i in range(100)],
                     # The score is a string. This is because Session Backend seems to not support float search for Query.
                     "significantDifferences": [str(i/100) for i in range(100)],
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
        # 
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
        
    def random_generate_session_data(self, meta_field: str, field: str=None, exclude: str=None):
        assert meta_field in ["source", "type", "data"]
        if meta_field != "data":
            target_list = copy.deepcopy(self.parameter_space[meta_field])
            if exclude is not None and exclude in target_list:
                target_list.remove(exclude)
            return random.choice(target_list)
        else:
            assert field in self.parameter_space["data"]
            target_list = copy.deepcopy(self.parameter_space["data"][field])
            if exclude is not None and exclude in target_list:
                target_list.remove(exclude)
            return random.choice(target_list)
    

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
                return_str += f"'{key}': {value:.2f}, "
            else:
                return_str += f"'{key}': \"{value}\", "
        return_str = return_str[:-2] + "}"
        return return_str
    
    def get_current_value(self):
        return_value = copy.deepcopy(self.current_value)
        return_value["type"] = return_value["type"].value
        return dict(return_value)
    
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
        self.init_implict_dict = {}
        
    def normalize_data(self, data):
        data = dict(data)
        for key in data:
            if isinstance(data[key], SessionType):
                data[key] = data[key].value
        return data
    
    def add_local_variable(self, local_variable: LocalVariable):
        self.local_states["variables"].append(local_variable)
    
    def clear_state(self):
        self.local_states["variables"] = []
        self.implicit_states["sessions"] = {}
        self.implicit_states["latest_call"] = {}
        self.init_local_info = []
        self.local_call_map = {}
        self.implicit_call_map = {}
        self.init_implict_dict = {}
        
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
    
    def get_implicit_states(self, current_value: bool = True):
        result = {}
        if current_value is True:
            sessions = self.implicit_states["sessions"]
            for key, value in sessions.items():
                result[key] = dict()
                for field in value.current_value.keys():
                    if isinstance(value.current_value[field], SessionType):
                        result[key][field] = value.current_value[field].value
                    else:
                        result[key][field] = value.current_value[field]
        else:
            sessions = self.init_implict_dict
            for key, value in sessions.items():
                result[key] = dict()
                for field in value.keys():
                    if isinstance(value[field], SessionType):
                        result[key][field] = value[field].value
                    else:
                        result[key][field] = value[field]
        return result
    
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
        #replace_local_variable = False
        # Maximum number of local variables to be replaced is 2.
        #replace_local_variable_num = random.randint(0, min(2, len(self.local_states["variables"])))
        #if replace_local_variable_num > 0:
        #    replace_local_variable = True
        #    drop_idx = random.sample(range(len(self.local_states["variables"])), replace_local_variable_num)
        #    for idx in sorted(drop_idx, reverse=True):
        #        self.local_states["variables"].pop(idx)
        implicit_set = dict()
        has_more_than_one_path = False
        for key in self.implicit_states["sessions"]:
            s_source = self.implicit_states["sessions"][key].current_value["source"]
            s_type = self.implicit_states["sessions"][key].current_value["type"]
            if (s_source, s_type) not in implicit_set:
                implicit_set[(s_source, s_type)] = []
            implicit_set[(s_source, s_type)].append(key)
            if len(implicit_set[(s_source, s_type)]) > 1:
                has_more_than_one_path = True
        if not has_more_than_one_path and len(self.implicit_states["sessions"]) > 1:
            # manually modify to at least two variables have the same source and type.
            source_type_pair = random.sample(list(implicit_set.keys()), 2)
            left_idx = implicit_set[source_type_pair[0]][0]
            right_idx = implicit_set[source_type_pair[1]][0]
            self.implicit_states["sessions"][left_idx].initial_value["source"] = self.implicit_states["sessions"][right_idx].current_value["source"]
            self.implicit_states["sessions"][left_idx].initial_value["type"] = self.implicit_states["sessions"][right_idx].current_value["type"]
            self.implicit_states["sessions"][left_idx].current_value["source"] = self.implicit_states["sessions"][right_idx].current_value["source"]
            self.implicit_states["sessions"][left_idx].current_value["type"] = self.implicit_states["sessions"][right_idx].current_value["type"]
            implicit_set.pop(source_type_pair[0])
            implicit_set[source_type_pair[1]].append(left_idx)
            
        # ====== Local Variable Alignment ======
        if len(self.local_states["variables"]) != 0:
            has_one_corresponding = False
            for local_variable in self.local_states["variables"]:
                if (local_variable.value.current_value["source"], local_variable.value.current_value["type"]) in implicit_set:
                    has_one_corresponding = True
                    break
            # Align the initial state with the parameter space
            # so at least we can call GetSessions.
            if not has_one_corresponding:
                # Choose a random source and type from the parameter space.
                source, type = random.choice(list(implicit_set.keys()))
                chosen_local_variable = random.choice(self.local_states["variables"])
                chosen_local_variable.value.initial_value["source"] = source
                chosen_local_variable.value.initial_value["type"] = type
                chosen_local_variable.value.current_value["source"] = source
                chosen_local_variable.value.current_value["type"] = type
        
        ''' 
        # The original aim of this part is to replace the local variable with the implicit variable id.
        # However, it is not a good idea to do this because the variable id is assigned by the backend.
        # As such, We remove the following code.
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
        '''
        # ====== Local Variable Alignment ======
        
        for idx, local_variable in enumerate(self.local_states["variables"]):
            self.add_local_constant(self.normalize_data(local_variable.value.current_value), local_variable.name)
            #self.init_local_info.append([idx, local_variable.name, str(local_variable.value)])
        
        for idx, session in self.implicit_states["sessions"].items():
            self.init_implict_dict[idx] = copy.deepcopy(session.current_value)
    
    
    def obtain_if_condition(self):
        """
        Obtain the condition for the if-else transition.
        """
        if_condition = None
        for idx in range(len(self.local_states["variables"])-1, -1, -1):
            local_variable = self.local_states["variables"][idx]
            skip_flag = False
            for transition in local_variable.value.transitions:
                if transition['name'] == "AddSession" or transition['name'] == "UpdateSession" \
                    or transition['name'] == "LocalEdit" or transition['name'] == "DeleteSession":
                    # AddSession means we know everything about the local variable.
                    # So we do not need to consider the if-else transition.
                    skip_flag = True
                    break
            if skip_flag:
                continue
            if local_variable.created_by == USER_FUNCTION_PARAM_FLAG and len(self.local_call_map) == 0:
                # For user-provided local variable and the first line of the program,
                # source and type could be used as the condition.
                meta_field = random.choice([key for key in local_variable.value.current_value.keys() \
                    if key in ["source", "type", "data"] and local_variable.value.current_value[key] is not None])
            else:
                # For NON-user-provided local variable, the program must have its source and type.
                # So we only need to consider the data field.
                if local_variable.value.current_value["data"] is not None:
                    meta_field = "data"
                elif local_variable.value.current_value["id"] is not None:
                    # The id field in our state machine is
                    # usually not the same as the real id in the database, which
                    # is generated by the backend. As such, the program will
                    # always reach to the else block.
                    meta_field = "id"
                else:
                    continue
            if meta_field == "source" or meta_field == "type":
                if_condition = (local_variable.name, (meta_field, ), local_variable.value.current_value[meta_field])
                break
            elif meta_field == "id":
                if_condition = (local_variable.name, (meta_field, ), local_variable.value.current_value["id"])
                break
            elif meta_field == "data":
                field = random.choice(list(local_variable.value.current_value["data"].keys()))  
                if_condition = (local_variable.name, ("data", field), local_variable.value.current_value["data"][field])
                break
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
        elif previous_transition_info[0] == "UpdateSession" and current_transition_info[0] == "GetSession":
            if previous_transition_info[1]["id"] == current_transition_info[1]["id"] and \
                previous_transition_info[1]["source"] == current_transition_info[1]["source"] and \
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
            if local_variable.updated and local_variable.value.current_value["data"] is not None:
                if local_variable.value.current_value["source"] is None or local_variable.value.current_value["type"] is None:
                    continue
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
                    has_implicit_session = False
                    for session in self.implicit_states["sessions"].values():
                        if session.current_value["source"] == target_parameters["source"] and \
                            session.current_value["type"] == target_parameters["type"] and \
                            session.current_value["id"] == target_parameters["id"] and \
                            session.exist:
                            has_implicit_session = True
                            break
                    if not has_implicit_session:
                        continue
                        
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
            if len (transitions) == 0 and len(transitions) >= remaining_call:
                return False, []
            else:
                return True, transitions
        else:
            return False, []
    
    def find_transition_by_id(self, id: str) -> List[str]:
        result = []
        updated = None
        for item in self.local_states["variables"]:
            if item.value.current_value["id"] == id:
                if len(item.value.transitions) > len(result):
                    result = item.value.transitions
                    updated = item.updated
        return result, updated
    
    def postprocess_choose_result(self):
        result_str = None
        for idx in range(len(self.local_states["variables"])-1, -1, -1):
            local_variable = self.local_states["variables"][idx]
            transitions = local_variable.value.transitions
            if local_variable.updated == True or (len(transitions) > 0 and transitions[-1]['name'] in QUERY_SET):
                # Either being updated or being queried.
                result_str = f"{RESULT_NAME} = {local_variable.name}"
                break
        return result_str
    
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
                chosen_flag = False
                if len(available_indices) > 0 and (random.random() > 1 / (1 + len(self.local_states["variables"]))):
                    # Choose from existing variables
                    local_variable_2_idx = random.choice(available_indices)
                    while len(available_indices) > 0:
                        meta_field = [field for field in self.local_states["variables"][local_variable_2_idx].value.current_value.keys() 
                                    if field in ["source", "type", "data"] and self.local_states["variables"][local_variable_2_idx].value.current_value[field] is not None]
                        meta_field = [field for field in meta_field 
                                      if self.local_states["variables"][local_variable_2_idx].value.current_value[field] != local_variable.value.current_value[field]]
                        if len(meta_field) == 0:
                            available_indices.remove(local_variable_2_idx)
                            if len(available_indices) == 0:
                                break
                            local_variable_2_idx = random.choice(available_indices)
                            continue
                        meta_field = random.choice(meta_field)
                        if meta_field == "data":
                            if self.local_states["variables"][local_variable_2_idx].value.current_value["data"] is None:
                                available_indices.remove(local_variable_2_idx)
                                if len(available_indices) == 0:
                                    break
                                local_variable_2_idx = random.choice(available_indices)
                                continue
                            data_field = []
                            for field in self.local_states["variables"][local_variable_2_idx].value.current_value["data"].keys():
                                if local_variable.value.current_value["data"] is None or field not in local_variable.value.current_value["data"] or local_variable.value.current_value["data"][field] != self.local_states["variables"][local_variable_2_idx].value.current_value["data"][field]:
                                    data_field.append(field)
                                else:
                                    left = local_variable.value.current_value["data"][field]
                                    right = self.local_states["variables"][local_variable_2_idx].value.current_value["data"][field]
                                    if left == right:
                                        continue
                                    else:
                                        data_field.append(field)
                            if len(data_field) == 0:
                                available_indices.remove(local_variable_2_idx)
                                if len(available_indices) == 0:
                                    break
                                local_variable_2_idx = random.choice(available_indices)
                                continue
                            field = random.choice(data_field)
                            value = self.local_states["variables"][local_variable_2_idx].value.current_value["data"][field]
                            chosen_flag = True
                            break
                        else:
                            field = None
                            value = self.local_states["variables"][local_variable_2_idx].value.current_value[meta_field]
                            chosen_flag = True
                            break
        
                if chosen_flag is False:
                    # Choose random parameter
                    meta_field = random.choice(["source", "type", "data"])
                    if meta_field == "data":
                        local_data = local_variable.value.current_value["data"]
                        if local_data is None or len(local_data.keys()) == 0:
                            field = "title"
                            exclude = None
                        else:
                            field = random.choice(list(local_variable.value.current_value["data"].keys()))
                            exclude = local_data[field]
                        value = random_generator.random_generate_session_data(meta_field, field, exclude=exclude)
                    else:
                        field = None
                        value = random_generator.random_generate_session_data(meta_field, exclude=local_variable.value.current_value[meta_field])
                    # Remember to update the local variable when actually applying the transition.
                    local_variable_2_idx = None
                

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
                            "local_variable_idx": local_variable_1_idx,
                        })
            elif transition.__name__ == "AddSession":
                # AddSession should be done after LocalEdit.
                for idx, local_variable in enumerate(self.local_states["variables"]):
                    if local_variable.updated and local_variable.exist:
                        current_value = local_variable.value.current_value
                        if current_value["source"] is None or current_value["type"] is None:
                            continue
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
                                    "local_variable_idx": idx,
                                })
                    else:
                        continue
            else:
                for idx, local_variable in enumerate(self.local_states["variables"]):
                    # ====== Pre-check ======
                    if local_variable.exist is False:
                        continue
                    if local_variable.updated is False and transition.__name__ in POST_SET:
                        continue
                    id_transitions, updated = self.find_transition_by_id(local_variable.value.current_value["id"])
                    if len(id_transitions) > 0 and id_transitions[-1]['name'] in QUERY_SET and transition.__name__ in QUERY_SET:
                        # Already queried. Skip.
                        continue
                    if len(id_transitions) > 0 and id_transitions[-1]['name'] in POST_SET:
                        if transition.__name__ == "DeleteSession":
                            # If the session is updated or added, we should not delete it.
                            # Otherwise, the previous POST method makes no sense.
                            continue
                        if transition.__name__ == "GetSession" or transition.__name__ == "GetSessionByQuery":
                            # No need to query the session again. It is already returned by the previous transition.
                            continue
                    # ====== Pre-check ======
                    # Avoid empty query return.
                    satisfied = True
                    if transition.__name__ in QUERY_SET:
                        has_implicit_variable = False
                        getsession_implicit_variable = None
                        for key in self.implicit_states["sessions"]:
                            if self.implicit_states["sessions"][key].exist is False:
                                continue
                            # The local variable and the implicit variable should have the same source and type to enable valid query.
                            if self.implicit_states["sessions"][key].current_value["source"] == local_variable.value.current_value["source"] and \
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
                                        getsession_implicit_variable = self.implicit_states["sessions"][key]
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
                            field = random.choice(list(getsession_implicit_variable.current_value["data"].keys()))
                            value = getsession_implicit_variable.current_value["data"][field]
                            local_data = local_variable.value.current_value["data"]
                            if local_data is not None and field in local_data:
                                if local_data[field] == value:
                                    value_from_variable = True
                                else:
                                    value_from_variable = False
                            else:
                                value_from_variable = False
                            target_parameters = {
                                "source": local_variable.value.current_value["source"],
                                "type": local_variable.value.current_value["type"],
                                "field": field,
                                "value": value,
                                "value_from_variable": value_from_variable
                            }
                        else:
                            if transition.__name__ == "UpdateSession":
                                source = local_variable.value.current_value["source"]
                                type = local_variable.value.current_value["type"]
                                id = local_variable.value.current_value["id"]
                                if local_variable.value.current_value["data"] is None:
                                    satisfied = False
                                    break
                                satisfied = False
                                for session in self.implicit_states["sessions"].values():
                                    if session.current_value["source"] == source and \
                                        session.current_value["type"] == type and \
                                        session.current_value["id"] == id and \
                                        session.exist:
                                        satisfied = True
                                        # Find the session to be updated exists in the remote.
                                        break
                                if not satisfied:
                                    continue
                                
                            
                            for required_parameters in transition.get_required_parameters():
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
                                continue
                        if transition.__name__ == "DeleteSession":
                            source = target_parameters["source"]
                            type = target_parameters["type"]
                            id = target_parameters["id"]
                            has_corresponding = False
                            for session in self.implicit_states["sessions"].values():
                                if session.current_value["source"] == source and \
                                    session.current_value["type"] == type and \
                                    session.current_value["id"] == id and \
                                    session.exist:
                                    has_corresponding = True
                                    break
                            if not has_corresponding:
                                satisfied = False
                                continue
                                
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
                                "local_variable_idx": idx,
                            })
                            duplicate_local_variable_map[transition.__name__].add(duplicate_str)
        return available_transitions
    
    def craft_transition(self, transition_info, calling_timestamp, transition, producer="None"):
        parameters = transition_info["required_parameters"]
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
                #self.init_local_info.append([len(self.local_states["variables"]) - 1, new_local_variable.name, str(new_local_variable.value)])
                name, already_exist = self.add_local_constant(self.normalize_data(new_local_variable.value.current_value), new_local_variable.name)
                if not already_exist:
                    self.add_local_variable(new_local_variable)
                parameters["local_variable_2_idx"] = len(self.local_states["variables"]) - 1
                producer = copy.deepcopy(self.local_states["variables"][parameters["local_variable_2_idx"]])
        transition_class = globals()[transition]
        new_transition = transition_class(
            parameters=parameters, 
            calling_timestamp=calling_timestamp
            )
        new_transition.producer = producer
        if transition == "DeleteSession":
            idx = transition_info["local_variable_idx"]
            if self.local_states["variables"][idx].is_indexed:
                # If the local variable is indexed,
                # we need a for loop with a condition to delete the session.
                variable_name = self.local_states["variables"][idx].name
                variable_name = variable_name.split("[")[0]
                local_list = []
                for other_idx, other_variable in enumerate(self.local_states["variables"]):
                    if other_variable.is_indexed and other_variable.name.split("[")[0] == variable_name:
                        if other_idx == idx:
                            continue
                        # If the other local variable is also indexed and has the same name,
                        # we need to add a condition to the for loop.
                        local_list.append(other_variable)
                if len(local_list) == 0:
                    if_condition = None
                else:
                    if_condition = DeleteSession.get_if_condition(local_list, self.local_states["variables"][idx])
                    #if isinstance(if_condition[1], float) or isinstance(if_condition[1], int):
                    #    right_str = f"{if_condition[1]}"
                    #else:
                    #    right_str = f"\"{if_condition[1]}\""
                    init_variable_name, already_exist = self.add_local_constant(if_condition[1])
                    #self.init_local_info.append([str(idx) + "_single", init_variable_name, right_str])
                    if_condition[1] = init_variable_name
                new_transition.add_string_parameters(if_condition, variable_name)
        elif transition == "GetSessionByQuery":
            field = transition_info["required_parameters"]["field"]
            filed_name, already_exist = self.add_local_constant(field)
            if transition_info["required_parameters"]["value_from_variable"]:
                local_variable_idx = transition_info["local_variable_idx"]
                local_variable = self.local_states["variables"][local_variable_idx]
                value_name = local_variable.name + "['data'][field]"
            else:
                value = transition_info["required_parameters"]["value"]
                value_name, already_exist = self.add_local_constant(value)
            new_transition.string_parameters = {"field": filed_name, "value": value_name}
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
        url = "{BASE_URL}/api/sessions/{source}/{type}"
        variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        target_transition = variable_name + f" = requests.get(url)"
        return (
            f"source = {self.producer.name}['source']\n"
            f"type = {self.producer.name}['type']\n"
            f"url = f'{url}'\n"
            f"{target_transition}"
            f"    # FROM {self.producer.name}. source={source}, type={type}\n"
            f"{variable_name}.raise_for_status()\n"
            f"{variable_name} = {variable_name}.json()"
        )
    
    def get_program_str(self) -> Tuple[List[str], str]:
        """
        Return the program string in line and the indent.
        """
        source = self.parameters["source"]
        type = self.parameters["type"]
        url = "{BASE_URL}/api/sessions/{source}/{type}"
        variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        target_transition = variable_name + f" = requests.get(url)"
        result = [
            f"source = {self.producer.name}['source']\n",
            f"type = {self.producer.name}['type']\n",
            f"url = f'{url}'\n",
            f"{target_transition}"
            f"  # FROM {self.producer.name}. source={source}, type={type}\n",
            f"{variable_name}.raise_for_status()\n",
            f"{variable_name} = {variable_name}.json()\n",
            f"{variable_name} = sorted({variable_name}, key=lambda x: x['id'])"
        ]
        return result, ""
    
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
                created_by=f"{self.name}@{self.calling_timestamp}",
                is_indexed=True
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
            url = "{BASE_URL}/api/sessions/{source}/{type}"
            variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
            target_transition = variable_name + f" = requests.post(url"
            data = f"{self.producer.name}['data']"
            return (
                f"source = {self.producer.name}['source']\n"
                f"type = {self.producer.name}['type']\n"
                f"url = f'{url}'\n"
                f"{target_transition}"
                f", headers={{\"Content-Type\": \"application/json\"}}"
                f", data=json.dumps({data})) # Local_variable_idx {self.local_variable_idx} being modified. source={source}, type={type}, data={self.string_parameters['data']}\n"
                f"{variable_name}.raise_for_status()\n"
                f"{variable_name} = {variable_name}.json()"
            )
    
    def get_program_str(self) -> Tuple[List[str], str]:
        source = self.parameters["source"]
        type = self.parameters["type"]
        url = "{BASE_URL}/api/sessions/{source}/{type}"
        variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        target_transition = variable_name + f" = requests.post(url"
        data = f"{self.producer.name}['data']"
        result = [
            f"source = {self.producer.name}['source']\n",
            f"type = {self.producer.name}['type']\n",
            f"url = f'{url}'\n",
            f"{target_transition}"
            f", headers={{\"Content-Type\": \"application/json\"}}"
            f", data=json.dumps({data})) # Local_variable_idx {self.local_variable_idx} being modified. source={source}, type={type}, data={self.string_parameters['data']}\n",
            f"{variable_name}.raise_for_status()\n",
            f"{variable_name} = {variable_name}.json()"
        ]
        return result, ""
    
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema: SessionVariableSchema):
        assert len(implicit_states) == 1
        new_session = Session(id=str(implicit_states[0]), 
                              source=self.parameters["source"], 
                              type=self.parameters["type"], 
                              data=self.parameters["data"],
                              #created_by=f"{self.name}@{self.calling_timestamp}"
        )
        local_transitions = copy.deepcopy(variable_schema.local_states["variables"][local_states[0]].value.transitions)
        new_session.transitions = local_transitions
        new_session.transitions.append({
            "name": self.name,
            "parameters": self.parameters,
        })
        variable_schema.add_implicit_variable(new_session, self.calling_timestamp)
        
        #variable_schema.local_states["variables"][local_states[0]].value = new_session # Add session will return.
        variable_schema.local_states["variables"][local_states[0]].updated = False
        #variable_schema.local_states["variables"][local_states[0]].latest_call = self.calling_timestamp
        value = Session(id=str(implicit_states[0]), source=None, type=None, data=None)
        value.transitions.append({
            "name": self.name,
            "parameters": self.parameters,
        })
        local_variable = LocalVariable(
            name=RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp),
            value=value,  # The API only returns the id of the session.
            updated=False, 
            latest_call=self.calling_timestamp, 
            created_by=f"{self.name}@{self.calling_timestamp}"
        )
        variable_schema.add_local_variable(local_variable)
        
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
        self.string_parameters = None

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
        url = "{BASE_URL}/api/sessions/{source}/{type}/query"
        variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        target_transition = variable_name + f" = requests.get(url"
        return (
            f"source = {self.producer.name}['source']\n"
            f"type = {self.producer.name}['type']\n"
            f"field = {self.producer.name}['field']\n"
            f"value = {self.producer.name}['value']\n"
            f"url = f'{url}'\n"
            f"{target_transition}"
            f", params={{\"field\": field, \"value\": value}})"
            #f' params={{"field": "{self.parameters["field"]}", "value": "{self.parameters["value"]}"}})'
            f'  # source={source}, type={type}, field={self.parameters["field"]}, value={self.parameters["value"]}\n'
            f"{variable_name}.raise_for_status()\n"
            f"{variable_name} = {variable_name}.json()"
        )
    
    def get_program_str(self) -> Tuple[List[str], str]:
        source = self.parameters["source"]
        type = self.parameters["type"]
        url = "{BASE_URL}/api/sessions/{source}/{type}/query"
        variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        field = self.string_parameters["field"] # This field should be a constant variable name.
        value = self.string_parameters["value"] # This value should be a constant variable name.
        target_transition = variable_name + f" = requests.get(url"
        result = [
            f"source = {self.producer.name}['source']\n",
            f"type = {self.producer.name}['type']\n",
            f"field = {field}\n",
            f"value = {value}\n",
            f"url = f'{url}'\n",
            f"{target_transition}"
            f", params={{\"field\": 'data.' + field, \"value\": value}})"
            f'  # source={source}, type={type}, field={self.parameters["field"]}, value={self.parameters["value"]}\n',
            f"{variable_name}.raise_for_status()\n",
            f"{variable_name} = {variable_name}.json()",
        ]
        return result, ""
    
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
                created_by=f"{self.name}@{self.calling_timestamp}",
                is_indexed=True
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
                RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + " = requests.get("
                f'"{{BASE_URL}}/api/sessions/{self.parameters["source"]}/{str(self.parameters["type"])}/{self.parameters["id"]})"'
            )
        else:
            source = self.parameters["source"]
            type = self.parameters["type"]
            id = self.parameters["id"]
            url = '{BASE_URL}/api/sessions/{source}/{type}/{id}'
            variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
            target_transition = variable_name + f" = requests.get(url)"
            return (
                f"source = {self.producer.name}['source']\n"
                f"type = {self.producer.name}['type']\n"
                f"id = {self.producer.name}['id']\n"
                f"url = f'{url}'\n"
                f"{target_transition}"
                f'  # source={source}, type={type}, id={id}\n'
                f"{variable_name}.raise_for_status()\n"
                f"{variable_name} = {variable_name}.json()"
            )
            
    def get_program_str(self) -> Tuple[List[str], str]:
        source = self.parameters["source"]
        type = self.parameters["type"]
        id = self.parameters["id"]
        url = '{BASE_URL}/api/sessions/{source}/{type}/{id}'
        variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        target_transition = variable_name + f" = requests.get(url)"
        result = [
            f"source = {self.producer.name}['source']\n",
            f"type = {self.producer.name}['type']\n",
            f"id = {self.producer.name}['id']\n",
            f"url = f'{url}'\n",
            f"{target_transition}"
            f'  # source={source}, type={type}, id={id}\n',
            f"{variable_name}.raise_for_status()\n",
            f"{variable_name} = {variable_name}.json()"
        ]
        return result, ""
    
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
        local_variable_idx = None
        longest_transitions = -1
        for idx, local_variable in enumerate(variable_schema.local_states["variables"]):
            if local_variable.value.current_value["id"] == self.parameters["id"]:
                if len(local_variable.value.transitions) > longest_transitions:
                    longest_transitions = len(local_variable.value.transitions)
                    local_variable_idx = idx
        return [self.parameters["id"]], [local_variable_idx]
    
    def __str__(self):
        if self.producer is None:
            return (
                RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + " = requests.put("
                f'"{{BASE_URL}}/api/sessions/{self.parameters["source"]}/{str(self.parameters["type"])}/{self.parameters["id"]}',
                f'headers={{"Content-Type\": \"application/json\"}},'
                f'data=json.dumps({self.parameters["data"]}))'
        )
        else:
            source = self.parameters["source"]
            type = self.parameters["type"]
            id = self.parameters["id"]
            url = '{BASE_URL}/api/sessions/{source}/{type}/{id}'
            variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
            target_transition = variable_name + f" = requests.put(url, headers={{\"Content-Type\": \"application/json\"}}, data=json.dumps({self.producer.name}['data']))"
            return (
                f"source = {self.producer.name}['source']\n"
                f"type = {self.producer.name}['type']\n"
                f"id = {self.producer.name}['id']\n"
                f"url = f'{url}'\n"
                f"{target_transition}"
                f'  # source={source}, type={type}, id={id}, data={self.parameters["data"]}\n'
                f"{variable_name}.raise_for_status()\n"
                #f"{variable_name} = {variable_name}.json()" # Return None
            )
    
    def get_program_str(self) -> Tuple[List[str], str]:
        source = self.parameters["source"]
        type = self.parameters["type"]
        id = self.parameters["id"]
        url = '{BASE_URL}/api/sessions/{source}/{type}/{id}'
        variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        target_transition = variable_name + f" = requests.put(url, headers={{\"Content-Type\": \"application/json\"}}, data=json.dumps({self.producer.name}['data']))"
        result = [
            f"source = {self.producer.name}['source']\n",
            f"type = {self.producer.name}['type']\n",
            f"id = {self.producer.name}['id']\n",
            f"url = f'{url}'\n",
            f"{target_transition}"
            f'  # source={source}, type={type}, id={id}, data={self.parameters["data"]}\n',
            f"{variable_name}.raise_for_status()\n",
            #f"{variable_name} = {variable_name}.json()", # Return None
        ]
        return result, ""
    
    def apply(self, implicit_states: List[str], local_states: List[str], variable_schema: SessionVariableSchema):
        assert len(implicit_states) == 1
        assert len(local_states) == 1
        local_transitions = copy.deepcopy(variable_schema.local_states["variables"][local_states[0]].value.transitions)
        if implicit_states[0] not in variable_schema.implicit_states["sessions"]:
            # Create a new session
            new_session = Session(id=implicit_states[0], 
                                  source=self.parameters["source"], 
                                  type=self.parameters["type"], 
                                  data=self.parameters["data"],
                                  #created_by=f"{self.name}@{self.calling_timestamp}"
                                  )
            new_session.transitions = local_transitions
            new_session.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            variable_schema.add_implicit_variable(new_session, self.calling_timestamp)
        else:
            # check whether the side effect is valid
            state = variable_schema.implicit_states["sessions"][implicit_states[0]]
            if len(state.transitions) < len(local_transitions):
                state.transitions = local_transitions
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
        self.string_parameters = {
            "for_loop": "",
            "if_condition": "",
            "indent": ""
        }

    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["source", "type", "id"]]
    
    @staticmethod
    def get_if_condition(variable_list: List[LocalVariable], target_variable: LocalVariable) -> str:
        unique = dict()
        if "data" not in target_variable.value.current_value:
            return None
        for field in target_variable.value.current_value["data"].keys():
            unique[field] = [target_variable.value.current_value["data"][field], 0]
        for variable in variable_list:
            if "data" in variable.value.current_value:
                for field in variable.value.current_value["data"].keys():
                    if field not in unique:
                        continue
                    if variable.value.current_value["data"][field] == unique[field][0]:
                        unique[field][1] += 1
        unique_fields = [i for i in unique.keys() if unique[i][1] == 0]
        if len(unique_fields) == 0:
            return None
        chosen_field = random.choice(unique_fields)
        return [chosen_field, unique[chosen_field][0]]
        
    
    def add_string_parameters(self, if_condition, variable_name):
        if if_condition is not None:
            self.string_parameters = {
                "for_loop": f"for {LOOP_ITEM_NAME} in {variable_name}:\n",
                "indent": INDENT * 2,
                "if_condition": f"{INDENT}if '{if_condition[0]}' in {LOOP_ITEM_NAME} and {LOOP_ITEM_NAME}['{if_condition[0]}'] == {if_condition[1]}:\n"
            }
        else:
            self.string_parameters = {
                "for_loop": f"for {LOOP_ITEM_NAME} in {variable_name}:\n",
                "indent": INDENT,
                "if_condition": ""
            }
        
    def get_effected_states(self, variable_schema: SessionVariableSchema) -> List[str]:
        return [self.parameters["id"]], None
    
    def __str__(self):
        if self.producer is None:
            return (
                RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp) + " = requests.delete("
                f'"{{BASE_URL}}/api/sessions/{self.parameters["source"]}/{str(self.parameters["type"])}/{self.parameters["id"]})'
            )
        else:
            source = self.parameters["source"]
            type = self.parameters["type"]
            id = self.parameters["id"]
            url = '{BASE_URL}/api/sessions/{source}/{type}/{id}'
            if self.string_parameters["for_loop"] != "":
                item_name = LOOP_ITEM_NAME
            else:
                item_name = self.producer.name
            variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
            target_transition = variable_name + f" = requests.delete(url)"  
            return (
                f"{self.string_parameters['for_loop']}"
                f"{self.string_parameters['if_condition']}"
                f"{self.string_parameters['indent']}"
                f"source = {item_name}['source']\n"
                f"{self.string_parameters['indent']}"
                f"type = {item_name}['type']\n"
                f"{self.string_parameters['indent']}"
                f"id = {item_name}['id']\n"
                f"{self.string_parameters['indent']}"
                f"url = f'{url}'\n"
                f"{self.string_parameters['indent']}"
                f"{target_transition}"
                f'  # source={source}, type={type}, id={id}\n'
                f"{self.string_parameters['indent']}"
                f"{variable_name}.raise_for_status()\n"
                #f"{self.string_parameters['indent']}"
                #f"{variable_name} = {variable_name}.json()"
            )
    
    def get_program_str(self) -> Tuple[List[str], str]:
        source = self.parameters["source"]
        type = self.parameters["type"]
        id = self.parameters["id"]
        url = '{BASE_URL}/api/sessions/{source}/{type}/{id}'
        if self.string_parameters["for_loop"] != "":
            item_name = LOOP_ITEM_NAME
        else:
            item_name = self.producer.name
        variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        target_transition = variable_name + f" = requests.delete(url)"  
        result = [
            f"{self.string_parameters['for_loop']}",
            f"{self.string_parameters['if_condition']}",
            f"{self.string_parameters['indent']}"
            f"source = {item_name}['source']\n",
            f"{self.string_parameters['indent']}"
            f"type = {item_name}['type']\n",
            f"{self.string_parameters['indent']}"
            f"id = {item_name}['id']\n",
            f"{self.string_parameters['indent']}"
            f"url = f'{url}'\n",
            f"{self.string_parameters['indent']}"
            f"{target_transition}"
            f'  # source={source}, type={type}, id={id}\n',
            f"{self.string_parameters['indent']}"
            f"{variable_name}.raise_for_status()\n",
            #f"{self.string_parameters['indent']}"
            #f"{variable_name} = {variable_name}.json()"
        ]
        result = [item for item in result if item != ""]
        return result, ""
        
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
        local_variable.value.transitions.append(
            {
                "name": self.name,
                "parameters": self.parameters,
            }
        )
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
    
    def get_program_str(self) -> Tuple[List[str], str]:
        all_fields = []
        all_fields.append(self.string_parameters["meta_field"])
        if self.string_parameters["field"] is not None:
            all_fields.append(self.string_parameters["field"])
        left_string = get_nested_path_string(self.string_parameters["left_variable"], all_fields)
        right_string = get_nested_path_string(self.string_parameters["right_variable"], all_fields)
        result = [
            f"{left_string} = {right_string}"
            f'  # Local Variable Edit from {self.producer.name}',
        ]
        if len(all_fields) > 1:
            # It is nested data assignment
            # Perform if check for the dict.
            if_block = [
                f"if '{all_fields[0]}' not in {self.string_parameters['left_variable']} or {self.string_parameters['left_variable']}['{all_fields[0]}'] is None:\n",
                f"{INDENT}{self.string_parameters['left_variable']}['{all_fields[0]}'] = {{}}\n",
            ]
            result = if_block + result
        return result, ""

class SessionEvaluator:
    def __init__(self, 
                 #init_implicit_dict: Dict[str, Any],
                 #init_local_list,
                 config: Dict[str, Any],
                 ):
        #self.init_implicit_dict = None
        #self.init_local_list = None # Used to clean up local variables stored in the remote.
        self.config = config
        assert "base_url" in self.config
        self.test_cases = []
        self.local_environment = [
            "import requests",
            "import json",
        ]
        self.local_environment_str = "\n".join(self.local_environment)
        self.occ_book_diff = None
    @classmethod
    def load(cls, file_path: str, config: Dict[str, Any] = None):
        with open(file_path, "r") as f:
            saved_info = json.load(f)
        if config is None:
            config = saved_info["config"]
        created_cls =  cls(config)
        created_cls.test_cases = saved_info["test_cases"]
        created_cls.occ_book_diff = saved_info["occ_book_diff"]
        return created_cls

    def store(self, file_path: str):
        saved_info = {
            "test_cases": self.test_cases,
            "config": self.config,
        }
        with open(file_path, "w") as f:
            json.dump(saved_info, f)
            
    def prepare_environment(self, init_implicit_dict, init_local_info):
        source_type_pair = [set([]), set([])]
        for idx in init_implicit_dict:
            source_type_pair[0].add(init_implicit_dict[idx]["source"])
            source_type_pair[1].add(init_implicit_dict[idx]["type"])
        for item in init_local_info:
            item_data = item[1]
            if isinstance(item_data, str):
                # User-defined constants
                continue
            elif isinstance(item_data, float) or isinstance(item_data, int):
                continue
            else:
                if item_data["source"] is not None and item_data["type"] is not None:
                    source_type_pair[0].add(item_data["source"])
                    source_type_pair[1].add(item_data["type"])
            
        for source in list(source_type_pair[0]):
            for type in list(source_type_pair[1]):
                # It is O(n^2)
                # But it has to be done since in the middle of the program, the source and type might be changed.
                # And a pair that never exists in the init environment might be created.
                # So we have to clean up all the sessions with the recorded source and type.
                url = f"{self.config['base_url']}/api/sessions/{source}/{type}"
                response = requests.get(url)
                response.raise_for_status()
                response_json = response.json()
                if len(response_json) > 0:
                    # Delete all existing sessions with the recorded source and type
                    for session in response_json:
                        delete_session = requests.delete(
                            url = f"{url}/{session['id']}"
                        )
                        delete_session.raise_for_status()
                    
        for idx in init_implicit_dict:
            session = init_implicit_dict[idx]
            source = session["source"]
            type = session["type"]
            data = session["data"]
            url = f"{self.config['base_url']}/api/sessions/{source}/{type}"
            add_session = requests.post(
                url = url,
                headers={"Content-Type": "application/json"},
                data = json.dumps(data)
            )
            add_session.raise_for_status()


    def collect_test_case(self, program_info, program):
        '''
        program_info:
            "init_local_str": init_local_str
            "init_local_info": init_local_info
            --> Both items are returned by result["init_block"] of function generate_program
            "init_implicit_dict": result["init_implicit_dict"]
            --> All the three items above are related to the initial state of the program.
            "end_implict_list": [session_id: state] --> Ending implicit states
        program:
            program_str
        '''
        complete_program = program_info["init_local_str"] + program
        namespace = {}
        complete_program = complete_program.replace("{BASE_URL}", self.config['base_url'])
        complete_program = self.local_environment_str + "\n" + complete_program
        
        self.prepare_environment(program_info['init_implicit_dict'], program_info['init_local_info'])
        exec(complete_program, namespace)
        
        # Collect States
        result = None
        if f"{RESULT_NAME}" in namespace:
            result = namespace[RESULT_NAME]
        
        oracle = {}
        implict_list = program_info["end_implict_list"]
        source_type_pair = set([])
        for state in implict_list:
            source = state["source"]
            type = state["type"]
            source_type_pair.add(f"{source}||{type}")
            
        for source_type in source_type_pair:
            source, type = source_type.split("||")
            url = f"{self.config['base_url']}/api/sessions/{source}/{type}"
            response = requests.get(url)
            response.raise_for_status()
            response_json = response.json()
            for session in response_json:
                if source_type not in oracle:
                    oracle[source_type] = []
                oracle[source_type].append(session)
        
        if result is not None and "id" in result:
            result.pop("id")
        test_case = {
            "result": result,
            "state_oracle": oracle,
            "program_info": program_info,
            "program": program
        }
        self.test_cases.append(test_case)
        return test_case
    
    
    def evaluate(self, program: str, threshold: float=1e-4):
        pass_list = []
        test_case_pass_detail = []
        for idx, test_case in enumerate(self.test_cases):
            result_pass = True
            self.prepare_environment(test_case["program_info"]["init_implicit_dict"], test_case["program_info"]["init_local_info"])
            complete_program = test_case["program_info"]["init_local_str"] + program
            namespace = {}
            complete_program = complete_program.replace("{BASE_URL}", self.config['base_url'])
            complete_program = self.local_environment_str + "\n" + complete_program
            try:
                exec(complete_program, namespace)
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
            # ====== Evaluate variable oracle ======
            if f"{RESULT_NAME}" in namespace:
                result = namespace[RESULT_NAME]
                if result is not None and "id" in result:
                    result.pop("id")
                if isinstance(result, float) or isinstance(result, int):
                    if abs(result - test_case["result"]) > threshold:
                        result_pass = False
                elif isinstance(result, dict):
                    for key in result:
                        if isinstance(result[key], float) or isinstance(result[key], int):
                            if abs(result[key] - test_case["result"][key]) > threshold:
                                result_pass = False
                        else:
                            if result[key] != test_case["result"][key]:
                                result_pass = False
                elif result != test_case["result"]:
                    result_pass = False
            else:
                if test_case["result"] is not None:
                    result_pass = False
            # ====== Evaluate state oracle ======
            state_pass = True
            state_pass_detail = {}
            for source_type in test_case["state_oracle"]:
                source, type = source_type.split("||")
                url = f"{self.config['base_url']}/api/sessions/{source}/{type}"
                local_oracle_list = [0] * len(test_case["state_oracle"][source_type])
                response = requests.get(url)
                response.raise_for_status()
                response_json = response.json()
                has_corresponding = True
                for session in response_json:
                    source = session["source"]
                    type = session["type"]
                    data = session["data"]
                    has_corresponding = False
                    for idx, oracle_session in enumerate(test_case["state_oracle"][source_type]):
                        if data == oracle_session["data"]:
                            local_oracle_list[idx] += 1
                            has_corresponding = True
                    if not has_corresponding:
                        state_pass = False
                if not all(x == 1 for x in local_oracle_list):
                    state_pass = False
                state_pass_detail[source_type] = [local_oracle_list, has_corresponding]
            if state_pass and result_pass:
                pass_list.append(True)
            else:
                pass_list.append(False)
            test_case_pass_detail.append({
                "result_pass": result_pass,
                "state_pass": state_pass,
                "state_pass_detail": state_pass_detail,
                "error_info": None
            })
        return pass_list, test_case_pass_detail

def generate_and_collect_test_case(trace_config,
                                   base_url,
                                   num_of_apis=5,
                                   control_position_candidate=[3, 4],
                                   occurence_book={}):
    state_schema = SessionVariableSchema()
    random_init = SessionRandomInitializer()
    trace_generator = TraceGenerator(
        state_schema,
        random_init,
        trace_config,
        occurence_book
    )
    trace_generator.prepare_initial_state()
    result, is_success = generate_program(trace_generator, num_of_apis, control_position_candidate)
    added_changes = get_added_changes(occurence_book, result["occurence_book"])
    occurence_book = result["occurence_book"]
    if not is_success:
        return None, is_success, None, None
    evaluator = SessionEvaluator({"base_url": base_url})
    implict_list = list(result['main_trace'][1].values())
    if result["if_trace"] is not None:
        for key in result["if_trace"][1]:
            if key not in result['main_trace'][1]:
                implict_list.append(result["if_trace"][1][key])

    if result["else_trace"] is not None:
        for key in result["else_trace"][1]:
            if key not in result['main_trace'][1]:
                implict_list.append(result["else_trace"][1][key])
    
    program_info = {
            "init_local_str": result["init_block"][0],
            "init_local_info": result["init_block"][1],
            "init_implicit_dict": result['init_implict_dict'],
            "end_implict_list": implict_list
        }
    
    try:
        evaluator.collect_test_case(
            program_info = program_info,
            program = result['program']
        )

        pass_list, test_case_pass_detail = evaluator.evaluate(result['program'])
        assert pass_list[0] == True
    except Exception as e:
        logger.warning(f"Error in generating and collecting test case: {e}, skip.")
        return None, False, None, None
    
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
            "end_implict_list": implict_list
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

@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    save_path: Annotated[Path, typer.Option()],
):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    generation_config = config_dict["generation_config"]
    
    num_of_apis = generation_config["num_of_apis"]
    control_position_candidate = generation_config["control_position_candidate"]
    num_of_tests = generation_config["num_of_tests"]
    
    base_url = config_dict["env"]["base_url"]
    
    occurence_book = {}
    evaluator_book = {}
    occ_book_diff_recorder = {}
    idx = 0
    while idx < num_of_tests:
        evaluator, is_success, new_occurence_book, occ_diff = generate_and_collect_test_case(
            trace_config = generation_config["trace_config"],
            base_url = base_url,
            num_of_apis = num_of_apis,
            control_position_candidate = control_position_candidate,
            occurence_book=occurence_book
        )
        if is_success:
            occurence_book = new_occurence_book
            occ_book_diff_recorder[idx] = occ_diff
            evaluator_book[idx] = evaluator
            idx += 1
    
    for idx in evaluator_book:
        evaluator_save_path = os.path.join(save_path, f"evaluator_{idx}.json")
        evaluator_book[idx].store(evaluator_save_path)
    metadata_save_path = os.path.join(save_path, "metadata.pkl")
    with open(metadata_save_path, 'wb') as file:
        pickle.dump({
            "occurence_book": occurence_book,
            "config": config_dict,
            "occ_book_diff_recorder": occ_book_diff_recorder
        }, file)

if __name__ == "__main__":
    app()
'''
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
'''