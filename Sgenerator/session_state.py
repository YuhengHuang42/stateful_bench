from typing import Callable, Any, Dict, List, Optional, Iterable
from Sgenerator.state import State, Transition, StateSchema
from dataclasses import dataclass, field
from enum import Enum
import re
import copy

class SessionType(str, Enum):
    MAIN_SESSION = "main_session"
    VIRTUAL_STUDY = "virtual_study"
    GROUP = "group"
    COMPARISON_SESSION = "comparison_session"
    SETTINGS = "settings"
    CUSTOM_DATA = "custom_data"
    GENOMIC_CHART = "genomic_chart"
    CUSTOM_GENE_LIST = "custom_gene_list"
    
class Session(State):
    """
    Represents a session entity from the session-service API
    """
    def __init__(self, id: str, 
                 source: str, 
                 type: str, 
                 checksum: str = "", 
                 data: Dict[str, Any] = field(default_factory=dict)):
        super().__init__(identifier=f"session_{id}")
        self.id = id
        self.initial_value = {
            "id": id,
            "source": source,
            "type": type,
            "checksum": checksum,
            "data": data
        }
    
    def __post_init__(self):
        # Validate session type matches enum values
        if not isinstance(self.initial_value["type"], SessionType):
            try:
                self.type = SessionType(self.type)
            except ValueError as e:
                raise ValueError(
                    f"Invalid session type: {self.type}. "
                    f"Valid types: {[t.value for t in SessionType]}"
                ) from e
    
    def get_id(self):
        return self.id

class SessionProgram(State):
    def __init__(self):
        super().__init__(identifier=f"session_program")
        self.initial_value = {
            "sessions": {},
            "length": 0
        }
        self.current_value = self.initial_value
        self.transitions = []
        

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
    def __init__(self, parameters: Dict[str, Any]):
        if isinstance(parameters["type"], str):
            parameters["type"] = SessionType(parameters["type"])
        assert "source" in parameters
        assert "type" in parameters
        super().__init__(name="getSessions", parameters=parameters, func=None)
    
    def __str__(self):
        return f"GET {self.parameters['type']} sessions from {self.parameters['source']}"
    
    def get_required_parameters(self) -> List[str]:
        return ["source", "type"]
    
    def get_effected_states(self, state_list: Iterable[State], session_program: SessionProgram) -> List[str]:
        result = []
        for state in state_list:
            if isinstance(state, Session):
                if self.parameters["source"] == state.current_value["source"] and \
                    state.exist and \
                    self.parameters["type"] == state.current_value["type"]:
                    result.append(state.identifier)
        return result
    
    def apply(self, effect_states: List[str], session_program: SessionProgram):
        return_result = []
        for state in effect_states:
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            return_result.append(state)
            session_program.current_value["sessions"][state.identifier] = copy.deepcopy(state)
        
        session_program.current_value["length"] = len(session_program.current_value["sessions"])
        return return_result

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
    def __init__(self, parameters: Dict[str, Any]):
        assert "source" in parameters
        assert "type" in parameters
        assert "data" in parameters
        super().__init__(name="addSession", parameters=parameters, func=None)

    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "data"]
    
    def get_effected_states(self, state_list: Iterable[State], session_program: SessionProgram) -> List[str]:
        current_max_id = None
        for state in state_list:
            if isinstance(state, Session):
                id = state.get_id()
                if current_max_id is None or id > current_max_id:
                    current_max_id = id
        new_session = Session(id=current_max_id + 1, source=self.parameters["source"], type=self.parameters["type"], data=self.parameters["data"])
        return new_session
    
    def __str__(self):
        return f"POST {self.parameters['type']} session to {self.parameters['source']} with data {self.parameters['data']}"
    
    def apply(self, effect_states: List[str], session_program: SessionProgram):
        return_result = []
        for state in effect_states:
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            return_result.append(state)
        return return_result

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
    def __init__(self, parameters: Dict[str, Any]):
        assert "source" in parameters
        assert "type" in parameters
        assert "field" in parameters
        assert "value" in parameters
        super().__init__(name="getSessionByQuery", parameters=parameters, func=None)

    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "field", "value"]
    
    def get_effected_states(self, state_list: Iterable[State], session_program: SessionProgram) -> List[str]:
        result = []
        for state in state_list:
            if isinstance(state, Session):
                if state.current_value["source"] == self.parameters["source"] and state.exist and \
                    state.current_value["type"] == self.parameters["type"] and state.current_value["data"][self.parameters["field"]] == self.parameters["value"]: ## Match --> "=="
                    result.append(state.identifier)
        return result
    
    def __str__(self):
        return f"GET {self.parameters['type']} session from {self.parameters['source']} with field {self.parameters['field']} and value {self.parameters['value']}"
    
    def apply(self, effect_states: List[str], session_program: SessionProgram):
        return_result = []
        for state in effect_states:
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            return_result.append(state)
            session_program.current_value["sessions"][state.identifier] = copy.deepcopy(state)
        session_program.current_value["length"] = len(session_program.current_value["sessions"])
        return return_result

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
    def __init__(self, parameters: Dict[str, Any]):
        assert "source" in parameters
        assert "type" in parameters
        assert "id" in parameters
        super().__init__(name="getSession", parameters=parameters, func=None)
    
    def get_effected_states(self, state_list: Iterable[State], session_program: SessionProgram) -> List[str]:
        result = []
        for state in state_list:
            if isinstance(state, Session):
                if state.current_value["source"] == self.parameters["source"] and state.exist and \
                    state.current_value["type"] == self.parameters["type"] and state.current_value["id"] == self.parameters["id"]:
                    result.append(state.identifier)
        return result

    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "id"]
    
    def __str__(self):
        return f"GET {self.parameters['type']} session from {self.parameters['source']} with id {self.parameters['id']}"
    
    def apply(self, effect_states: List[str], session_program: SessionProgram):
        return_result = []
        for state in effect_states:
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            return_result.append(state)
            session_program.current_value["sessions"][state.identifier] = copy.deepcopy(state)
        session_program.current_value["length"] = len(session_program.current_value["sessions"])
        assert len(return_result) <= 1
        return return_result

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
    def __init__(self, parameters: Dict[str, Any]):
        assert "source" in parameters
        assert "type" in parameters
        assert "id" in parameters
        assert "data" in parameters
        super().__init__(name="updateSession", parameters=parameters, func=None)

    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "id", "data"]
    
    def get_effected_states(self, state_list: Iterable[State], session_program: SessionProgram) -> List[str]:
        result = []
        for state in state_list:
            if isinstance(state, Session):
                if state.current_value["source"] == self.parameters["source"] and state.exist and \
                    state.current_value["type"] == self.parameters["type"] and state.current_value["id"] == self.parameters["id"]:
                    result.append(state.identifier)
        assert len(result) <= 1
        return result
    
    def __str__(self):
        return f"PUT {self.parameters['type']} session from {self.parameters['source']} with id {self.parameters['id']} and data {self.parameters['data']}"
    
    def apply(self, effect_states: List[str], session_program: SessionProgram):
        return_result = []
        for state in effect_states:
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            state.current_value["data"] = self.parameters["data"]
            return_result.append(state)
        return return_result

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
    def __init__(self, parameters: Dict[str, Any]):
        assert "source" in parameters
        assert "type" in parameters
        assert "id" in parameters
        super().__init__(name="deleteSession", parameters=parameters, func=None)

    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "id"]
    
    def get_effected_states(self, state_list: Iterable[State], session_program: SessionProgram) -> List[str]:
        result = []
        for state in state_list:
            if isinstance(state, Session):
                if state.current_value["source"] == self.parameters["source"] and state.exist and \
                    state.current_value["type"] == self.parameters["type"] and state.current_value["id"] == self.parameters["id"]:
                    result.append(state.identifier)
        assert len(result) <= 1
        return result
    
    def __str__(self):
        return f"DELETE {self.parameters['type']} session from {self.parameters['source']} with id {self.parameters['id']}"
    
    def apply(self, effect_states: List[str], session_program: SessionProgram):
        return_result = []
        for state in effect_states:
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            state.exist = False
            return_result.append(state)
        return return_result

class GetInfo(Transition):
    """
    Represents a query to the session-service API
    Get info
    """
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(name="getInfo", parameters=parameters, func=None)

    def get_required_parameters(self) -> List[str]:
        return []
    
    def get_effected_states(self, state_list: Iterable[State], session_program: SessionProgram) -> List[str]:
        result = []
        for state in state_list:
            if isinstance(state, Session):