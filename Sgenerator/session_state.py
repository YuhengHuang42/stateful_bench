from typing import Callable, Any, Dict, List, Optional, Iterable
from Sgenerator.state import State, Transition, Schema, RandomInitializer, USER_FUNCTION_PARAM_FLAG
from dataclasses import dataclass, field
from enum import Enum
import re
import copy
import random
from faker import Faker
from loguru import logger
from collections import defaultdict
from collections import OrderedDict

class SessionType(str, Enum):
    MAIN_SESSION = "main_session"
    VIRTUAL_STUDY = "virtual_study"
    GROUP = "group"
    COMPARISON_SESSION = "comparison_session"
    SETTINGS = "settings"
    CUSTOM_DATA = "custom_data"
    GENOMIC_CHART = "genomic_chart"
    CUSTOM_GENE_LIST = "custom_gene_list"

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
                 data: Dict[str, Any] = field(default_factory=dict)):
        if id is None:
            identifier = "local_variable"
        else:
            identifier = id
        super().__init__(identifier=identifier)
        self.id = id
        self.initial_value = OrderedDict([
            ("id", id),
            ("source", source),
            ("type", SessionType(type)),
            ("checksum", checksum),
            ("data", data)
        ])
        self.current_value = OrderedDict([
            ("id", id),
            ("source", source),
            ("type", SessionType(type)),
            ("checksum", checksum),
            ("data", data)
        ])
    
    def get_id(self):
        return self.id

@dataclass
class LocalVariable:
    value: Any
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
    
    def add_local_variable(self, local_variable: LocalVariable):
        self.local_states["variables"].append(local_variable)
    
    def clear_state(self):
        self.local_states["variables"] = []
        self.implicit_states["sessions"] = {}
        self.implicit_states["latest_call"] = {}
    
    def add_local_variable_using_state(self, state: Session, latest_call=0, updated=True, created_by=USER_FUNCTION_PARAM_FLAG):
        local_variable = LocalVariable(value=state,
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
        
        if random.random() < 0.5:
            # Add a new local variable with id available.
            chosen_implicit_variable_id = random.choice(list(self.implicit_states["sessions"].keys()))
            chosen_implicit_variable = self.implicit_states["sessions"][chosen_implicit_variable_id]
            new_session = Session(id=chosen_implicit_variable.current_value["id"],
                                  source=chosen_implicit_variable.current_value["source"],
                                  type=chosen_implicit_variable.current_value["type"],
                                  data=None)
            self.add_local_variable_using_state(new_session, latest_call=0, updated=True, created_by=USER_FUNCTION_PARAM_FLAG)
    

    def obtain_if_condition(self):
        """
        Obtain the condition for the if-else transition.
        """
        for idx in range(len(self.local_states["variables"]), 0, -1):
            local_variable = self.local_states["variables"][idx]
            if local_variable.created_by == USER_FUNCTION_PARAM_FLAG:
                meta_field = random.choice([key for key in local_variable.value.current_value.keys() \
                    if key in ["source", "type", "data"] and local_variable.value.current_value[key] is not None])
            else:
                meta_field = "data"
            if meta_field == "source" or meta_field == "type":
                if_condition = (idx, meta_field, local_variable.value.current_value[meta_field])
            elif meta_field == "data":
                if_condition = (idx, "data", random.choice(list(local_variable.value.current_value["data"].keys())))
            return if_condition
            
                
    
    def get_available_transitions(self, random_generator: SessionRandomInitializer) -> Dict[str, Transition]:
        available_transitions = {}
        duplicate_local_variable_map = {}
        self.get_latest_call_map()
        for transition in self.transitions:
            if transition.__name__ == "LocalEdit":
                if len(self.local_states["variables"]) == 0:
                    continue
                latest_call = max(self.local_call_map.keys())
                local_variable_1_idx = random.choice(self.local_call_map[latest_call])
                local_variable = self.local_states["variables"][local_variable_1_idx]
                # Create a new local variable for the edition
                # This will be the user-provided local variable.
                if random.random() < 1 / (1 + len(self.local_states["variables"])):
                    # Choose random parameter
                    meta_field = random.choice(["source", "type", "data"])
                    if meta_field == "data":
                        field = random.choice(list(local_variable.value.current_value["data"].keys()))
                        value = random_generator.random_generate_session_data(meta_field, field)
                    else:
                        field = None
                        value = random_generator.random_generate_session_data(meta_field)
                    # Remember to update the local variable when actually applying the transition.
                    local_variable_2_idx = None
                else:
                    # Choose from existing variables
                    available_indices = [idx for idx in range(len(self.local_states["variables"])) 
                                         if idx != local_variable_1_idx]
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
                
                available_transitions[transition.__name__].append({
                    "required_parameters": {
                        "local_variable_1_idx": local_variable_1_idx,
                        "local_variable_2_idx": local_variable_2_idx,
                        "meta_field": meta_field,
                        "field": field,
                        "value": value,
                    },
                    "latest_call": local_variable.latest_call,
                    "whether_updated": local_variable.updated,
                    "local_variable_idx": local_variable_1_idx,
                    "transition_pairs": transition_pairs,
                })
            elif transition.__name__ == "AddSession":
                # AddSession should be done after LocalEdit.
                for idx, local_variable in enumerate(self.local_states["variables"]):
                    if local_variable.updated:
                        # This local variable is updated by LocalEdit.
                        # If not updated, we will not treat it as the candidate of AddSession.
                        if transition.__name__ not in available_transitions:
                            available_transitions[transition.__name__] = []
                        transition_pairs = [self.form_pair_transition(local_variable.value, transition.__name__)]
                        available_transitions[transition.__name__].append({
                            "required_parameters": {
                                "local_variable": local_variable,
                                "local_variable_idx": idx,
                            },
                            "latest_call": local_variable.latest_call,
                            "whether_updated": local_variable.updated,
                            "local_variable_idx": idx,
                            "transition_pairs": transition_pairs,
                        })
                    else:
                        continue
            else:
                for idx, local_variable in enumerate(self.local_states["variables"]):
                    for required_parameters in transition.get_required_parameters():
                        satisfied = True
                        # Avoid empty query return.
                        if transition.__name__ in QUERY_SET:
                            has_implicit_variable = False
                            for key in self.implicit_states["sessions"]:
                                if self.implicit_states["sessions"][key].current_value["source"] == local_variable.value.current_value["source"] or \
                                    self.implicit_states["sessions"][key].current_value["type"] == local_variable.value.current_value["type"]:
                                    if transition.__name__ == "GetSession":
                                        if self.implicit_states["sessions"][key].current_value["id"] == local_variable.value.current_value["id"]:
                                            has_implicit_variable = True
                                            break
                                        else:
                                            continue
                                    else:
                                        has_implicit_variable = True
                                        break
                            if not has_implicit_variable:
                                satisfied = False
                                break
                        if satisfied:
                            for parameter in required_parameters:
                                if parameter not in local_variable.value.current_value or local_variable.value.current_value[parameter] is None:
                                    satisfied = False
                                    break
                        if transition.__name__ not in duplicate_local_variable_map:
                            duplicate_local_variable_map[transition.__name__] = set([])
                        if local_variable.value.current_value['data'] is None:
                            data_str = "None"
                        else:
                            data_str = sorted(local_variable.value.current_value['data'].items())
                        duplicate_str = (
                            f"source:{local_variable.value.current_value['source']}-"
                            f"type:{local_variable.value.current_value['type']}-"
                            f"data:{data_str}"
                        )
                        if duplicate_str in duplicate_local_variable_map[transition.__name__]:
                            satisfied = False
                        else:
                            duplicate_local_variable_map[transition.__name__].add(duplicate_str)
                            
                        if satisfied:
                            if transition.__name__ not in available_transitions:
                                available_transitions[transition.__name__] = []
                            transition_pairs = [self.form_pair_transition(local_variable.value, transition.__name__)]
                            available_transitions[transition.__name__].append({
                                "required_parameters": {parameter: local_variable.value.current_value[parameter] 
                                                        for parameter in required_parameters},
                                "latest_call": local_variable.latest_call,
                                "whether_updated": local_variable.updated,
                                "local_variable_idx": idx,
                                "transition_pairs": transition_pairs,
                            })
        return available_transitions
    
    def craft_transition(self, parameters, calling_timestamp, transition):
        if transition.__name__ == "LocalEdit":
            if parameters["local_variable_2_idx"] is None:
                # This local variable is generaed randomly in the get_available_transitions function.
                # Because it is already used in LocalEdit,
                # we set the updated to False.
                new_local_variable = LocalVariable(value=parameters["value"], 
                                                   updated=False, 
                                                   latest_call=calling_timestamp, 
                                                   created_by=USER_FUNCTION_PARAM_FLAG)
                self.add_local_variable(new_local_variable)
                
        new_transition = transition(
            parameters=parameters, 
            calling_timestamp=calling_timestamp
            )
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
        return f"GET {self.parameters['type']} sessions from {self.parameters['source']}"
    
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
    
    def apply(self, effect_states, variable_schema: SessionVariableSchema):
        for state_id in effect_states:
            state = variable_schema.implicit_states["sessions"][state_id]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            variable_schema.implicit_states["latest_call"][state_id] = self.calling_timestamp
            
            local_variable = LocalVariable(value=copy.deepcopy(state), 
                                           updated=False, 
                                           latest_call=self.calling_timestamp, 
                                           created_by=f"{self.name}@{self.calling_timestamp}")
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
        
    @staticmethod
    def get_required_parameters() -> List[str]:
        return [["local_variable", "local_variable_idx"]]
    
    def get_effected_states(self, variable_schema: SessionVariableSchema) -> List[str]:
        state_id_collection = [int(i) for i in variable_schema.implicit_states["sessions"].keys()]
        current_max_id = max(state_id_collection) + 1
        #new_session = Session(id=current_max_id + 1, source=self.parameters["source"], type=self.parameters["type"], data=self.parameters["data"])
        return [current_max_id], [self.local_variable_idx]
    
    def __str__(self):
        return f"POST {self.parameters['type']} session to {self.parameters['source']} with data {self.parameters['data']}"
    
    def apply(self, effect_states: List[str], variable_schema: SessionVariableSchema):
        assert len(effect_states) == 1
        new_session = Session(id=effect_states[0], 
                              source=self.parameters["source"], 
                              type=self.parameters["type"], 
                              data=self.parameters["data"],
                              created_by=f"{self.name}@{self.calling_timestamp}"
        )
        new_session.transitions.append({
            "name": self.name,
            "parameters": self.parameters,
        })
        variable_schema.add_implicit_variable(new_session, self.calling_timestamp)
        
        variable_schema.local_states["variables"][self.local_variable_idx].value = new_session # Add session will return.
        variable_schema.local_states["variables"][self.local_variable_idx].updated = False
        variable_schema.local_states["variables"][self.local_variable_idx].latest_call = self.calling_timestamp
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
        return f"GET {self.parameters['type']} session from {self.parameters['source']} with field {self.parameters['field']} and value {self.parameters['value']}"
    
    def apply(self, effect_states: List[str], variable_schema: SessionVariableSchema):
        for state_id in effect_states:
            state = variable_schema.implicit_states["sessions"][state_id]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            local_variable = LocalVariable(value=copy.deepcopy(state), 
                                           updated=False, 
                                           latest_call=self.calling_timestamp, 
                                           created_by=f"{self.name}@{self.calling_timestamp}")
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
        return f"GET {self.parameters['type']} session from {self.parameters['source']} with id {self.parameters['id']}"
    
    def apply(self, effect_states: List[str], variable_schema: SessionVariableSchema):
        for state_id in effect_states:
            state = variable_schema.implicit_states["sessions"][state_id]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            local_variable = LocalVariable(value=copy.deepcopy(state), 
                                           updated=False, 
                                           latest_call=self.calling_timestamp, 
                                           created_by=f"{self.name}@{self.calling_timestamp}")
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
        return f"PUT {self.parameters['type']} session from {self.parameters['source']} with id {self.parameters['id']} and data {self.parameters['data']}"
    
    def apply(self, effect_states: List[str], variable_schema: SessionVariableSchema):
        assert len(effect_states) == 1
        if effect_states[0] not in variable_schema.implicit_states["sessions"]:
            # Create a new session
            new_session = Session(id=effect_states[0], 
                                  source=self.parameters["source"], 
                                  type=self.parameters["type"], 
                                  data=self.parameters["data"],
                                  created_by=f"{self.name}@{self.calling_timestamp}"
                                  )
            new_session.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            variable_schema.add_implicit_variable(new_session, self.calling_timestamp)
        else:
            # check whether the side effect is valid
            state = variable_schema.implicit_states["sessions"][effect_states[0]]
            for local_variable in variable_schema.local_states["variables"]:
                if local_variable.value.current_value["id"] == state.current_value["id"]:
                    local_variable.updated = False # local variable update has been submitted to the remote.
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            state.current_value["data"] = self.parameters["data"]
        
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
        return f"DELETE {self.parameters['type']} session from {self.parameters['source']} with id {self.parameters['id']}"
    
    def apply(self, effect_states: List[str], variable_schema: SessionVariableSchema):
        assert len(effect_states) == 1
        if effect_states[0] not in variable_schema.implicit_states["sessions"]:
            raise ValueError(f"Session {effect_states[0]} does not exist")
        state = variable_schema.implicit_states["sessions"][effect_states[0]]
        assert state.exist
        state.transitions.append({
            "name": self.name,
            "parameters": self.parameters,
        })
        state.exist = False
        variable_schema.implicit_states["latest_call"][effect_states[0]] = self.calling_timestamp
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
    
    def apply(self, effect_states: List[str], variable_schema: SessionVariableSchema):
        assert len(effect_states) == 1
        local_variable = variable_schema.local_states["variables"][effect_states[0]]
        if self.parameters["field"] is None:
            local_variable.value.current_value[self.parameters["meta_field"]] = self.parameters["value"]
        else:
            local_variable.value.current_value["data"][self.parameters["field"]] = self.parameters["value"]
        local_variable.updated = True
        local_variable.latest_call = self.calling_timestamp
        return variable_schema

    def __str__(self):
        return f"LOCAL EDIT {self.parameters['session'].current_value['id']} with field {self.parameters['field']} to {self.parameters['value']}"




if __name__ == "__main__":
    
    # ======= Example Usage and Test Case =======
    
    # ==== Initialize the variable schema ====
    all_state = SessionVariableSchema()
    init_variable_1 = LocalVariable(value=Session(id="local_1", 
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
    get_session.apply(implicit_effected_states, all_state)

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
    add_session.apply(implicit_effected_states, all_state)

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
    query_session.apply(implicit_effected_states, all_state)

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
    update_session.apply(implicit_effected_states, all_state)

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
    delete_session.apply(implicit_effected_states, all_state)

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
    local_edit.apply(local_effected_states, all_state)
    
    
    
    assert len(all_state.local_states['variables']) == 5
    assert len(all_state.local_states['variables'][-1].value.transitions) == 2
    assert all_state.local_states['variables'][-1].value.current_value['source'] == "new_test"
    assert all_state.implicit_states['sessions'][1].current_value['data']['descriptions'] == "this is updated session"
    assert len(all_state.implicit_states['sessions']) == 3
    
    