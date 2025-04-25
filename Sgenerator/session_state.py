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
        self.initial_value = {
            "id": id,
            "source": source,
            "type": SessionType(type),
            "checksum": checksum,
            "data": data
        }
        self.current_value = {
            "id": id,
            "source": source,
            "type": SessionType(type),
            "checksum": checksum,
            "data": data
        }
    
    def get_id(self):
        return self.id

@dataclass
class LocalVariable:
    value: Any
    updated: bool = False
    latest_call: int = 0
    exist: bool = True
    
class VariableSchema():
    def __init__(self):
        self.local_states = {
            "variables": [],
        }
        self.global_states = {
            "sessions": {},
            "latest_call": {},
        }

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
        super().__init__(name="getSessions", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
    def __str__(self):
        return f"GET {self.parameters['type']} sessions from {self.parameters['source']}"
    
    def get_required_parameters(self) -> List[str]:
        return ["source", "type"]
    
    def get_effected_states(self, variable_schema: VariableSchema) -> List[str]:
        result = []
        for state_id in variable_schema.global_states["sessions"]:
            state = variable_schema.global_states["sessions"][state_id]
            if isinstance(state, Session):
                if self.parameters["source"] == state.current_value["source"] and \
                    state.exist and \
                    self.parameters["type"] == state.current_value["type"]:
                    result.append(state.current_value["id"])
        return result, None
    
    def apply(self, effect_states, variable_schema: VariableSchema):
        for state_id in effect_states:
            state = variable_schema.global_states["sessions"][state_id]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            variable_schema.global_states["latest_call"][state_id] = self.calling_timestamp
            
            local_variable = LocalVariable(value=copy.deepcopy(state), updated=False, latest_call=self.calling_timestamp)
            variable_schema.local_states["variables"].append(local_variable)

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
        assert "source" in parameters
        assert "type" in parameters
        assert "data" in parameters
        if isinstance(parameters["type"], str):
            parameters["type"] = SessionType(parameters["type"])
        super().__init__(name="addSession", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        
    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "data"]
    
    def get_effected_states(self, variable_schema: VariableSchema) -> List[str]:
        state_id_collection = [int(i) for i in variable_schema.global_states["sessions"].keys()]
        current_max_id = max(state_id_collection) + 1
        #new_session = Session(id=current_max_id + 1, source=self.parameters["source"], type=self.parameters["type"], data=self.parameters["data"])
        return [current_max_id], None
    
    def __str__(self):
        return f"POST {self.parameters['type']} session to {self.parameters['source']} with data {self.parameters['data']}"
    
    def apply(self, effect_states: List[str], variable_schema: VariableSchema):
        assert len(effect_states) == 1
        new_session = Session(id=effect_states[0], source=self.parameters["source"], type=self.parameters["type"], data=self.parameters["data"])
        new_session.transitions.append({
            "name": self.name,
            "parameters": self.parameters,
        })
        variable_schema.global_states["sessions"][new_session.current_value["id"]] = new_session
        variable_schema.global_states["latest_call"][new_session.current_value["id"]] = self.calling_timestamp
            
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
        super().__init__(name="getSessionByQuery", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp

    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "field", "value"]
    
    def get_effected_states(self, variable_schema: VariableSchema) -> List[str]:
        result = []
        for state_id in variable_schema.global_states["sessions"]:
            state = variable_schema.global_states["sessions"][state_id]
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
    
    def apply(self, effect_states: List[str], variable_schema: VariableSchema):
        for state_id in effect_states:
            state = variable_schema.global_states["sessions"][state_id]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            local_variable = LocalVariable(value=copy.deepcopy(state), updated=False, latest_call=self.calling_timestamp)
            variable_schema.local_states["variables"].append(local_variable)
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
        super().__init__(name="getSession", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
    
    def get_effected_states(self, variable_schema: VariableSchema) -> List[str]:
        result = []
        for state_id in variable_schema.global_states["sessions"]:
            state = variable_schema.global_states["sessions"][state_id]
            if isinstance(state, Session):
                if state.current_value["source"] == self.parameters["source"] and state.exist and \
                    state.current_value["type"] == self.parameters["type"] and state.current_value["id"] == self.parameters["id"]:
                    result.append(state.current_value["id"])
        return result, None

    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "id"]
    
    def __str__(self):
        return f"GET {self.parameters['type']} session from {self.parameters['source']} with id {self.parameters['id']}"
    
    def apply(self, effect_states: List[str], variable_schema: VariableSchema):
        for state_id in effect_states:
            state = variable_schema.global_states["sessions"][state_id]
            state.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            local_variable = LocalVariable(value=copy.deepcopy(state), updated=False, latest_call=self.calling_timestamp)
            variable_schema.local_states["variables"].append(local_variable)
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
        super().__init__(name="updateSession", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "id", "data"]
    
    def get_effected_states(self, variable_schema: VariableSchema) -> List[str]:
        return [self.parameters["id"]], None
    
    def __str__(self):
        return f"PUT {self.parameters['type']} session from {self.parameters['source']} with id {self.parameters['id']} and data {self.parameters['data']}"
    
    def apply(self, effect_states: List[str], variable_schema: VariableSchema):
        assert len(effect_states) == 1
        if effect_states[0] not in variable_schema.global_states["sessions"]:
            # Create a new session
            new_session = Session(id=effect_states[0], source=self.parameters["source"], type=self.parameters["type"], data=self.parameters["data"])
            new_session.transitions.append({
                "name": self.name,
                "parameters": self.parameters,
            })
            variable_schema.global_states["sessions"][effect_states[0]] = new_session
            variable_schema.global_states["latest_call"][effect_states[0]] = self.calling_timestamp
        else:
            # check whether the side effect is valid
            state = variable_schema.global_states["sessions"][effect_states[0]]
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
        super().__init__(name="deleteSession", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp

    def get_required_parameters(self) -> List[str]:
        return ["source", "type", "id"]
    
    def get_effected_states(self, variable_schema: VariableSchema) -> List[str]:
        return [self.parameters["id"]], None
    
    def __str__(self):
        return f"DELETE {self.parameters['type']} session from {self.parameters['source']} with id {self.parameters['id']}"
    
    def apply(self, effect_states: List[str], variable_schema: VariableSchema):
        assert len(effect_states) == 1
        if effect_states[0] not in variable_schema.global_states["sessions"]:
            raise ValueError(f"Session {effect_states[0]} does not exist")
        state = variable_schema.global_states["sessions"][effect_states[0]]
        assert state.exist
        state.transitions.append({
            "name": self.name,
            "parameters": self.parameters,
        })
        state.exist = False
        variable_schema.global_states["latest_call"][effect_states[0]] = self.calling_timestamp
        return variable_schema

class LocalEdit(Transition):
    """
    Represents a edit operation of local variables.
    """
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        assert "session" in parameters
        assert "meta_field" in parameters
        assert parameters["meta_field"] in ["source", "type", "data"]
        assert "value" in parameters
        super().__init__(name="localEdit", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        
    def get_required_parameters(self) -> List[str]:
        return ["session", "meta_field", "field", "value"]
    
    def get_effected_states(self, variable_schema: VariableSchema) -> List[str]:
        result = []
        last_call = -1 # Only the latest call will be considered.
        for idx, local_variable in enumerate(variable_schema.local_states["variables"]):
            if local_variable.value.current_value["id"] == self.parameters["session"].current_value["id"] and local_variable.exist:
                if local_variable.latest_call > last_call:
                    last_call = local_variable.latest_call
                    result = [idx]
                elif local_variable.latest_call == last_call:
                    result.append(idx)
        return None, result
    
    def apply(self, effect_states: List[str], variable_schema: VariableSchema):
        assert len(effect_states) == 1
        local_variable = variable_schema.local_states["variables"][effect_states[0]]
        if "field" not in self.parameters:
            local_variable.value.current_value[self.parameters["meta_field"]] = self.parameters["value"]
        else:
            local_variable.value.current_value["data"][self.parameters["field"]] = self.parameters["value"]
        local_variable.updated = True
        local_variable.latest_call = self.calling_timestamp
        return variable_schema

    def __str__(self):
        return f"LOCAL EDIT {self.parameters['session'].current_value['id']} with field {self.parameters['field']} to {self.parameters['value']}"



def find_valid_candidates(state_list: Iterable[State]) -> List[str]:
    pass
    

if __name__ == "__main__":
    # ==== Initialize the variable schema ====
    
    # ==== Test Case ====
    
    all_state = VariableSchema()
    init_variable_1 = LocalVariable(value=Session(id="local_1", source="test", type="main_session", data={"title": "user_provided_data_1"}), 
                                updated=False, 
                                latest_call=0
                                )

    init_variable_2 = LocalVariable(
        value = Session(id="local_2", source="test", type="virtual_study", data={"title": "user_provided_data_2"}),
        updated=False,
        latest_call=0
    )

    all_state.local_states["variables"].append(init_variable_1)
    all_state.local_states["variables"].append(init_variable_2)

    state_1 = Session(id=1, source="test", type="main_session", 
                    data={"title": "my main portal session",
                            "description": "this is another example"}
    )


    state_2 = Session(id=2, source="test", type="main_session", 
                    data={"title": "my main portal session",
                            "description": "This is the main session"}
    )

    all_state.global_states['sessions'][state_1.current_value["id"]] = state_1
    all_state.global_states['sessions'][state_2.current_value["id"]] = state_2

    # Get Session Transition
    get_session = GetSessions({"source": "test", "type": "main_session"}, calling_timestamp=1)
    global_effected_states, local_effected_states = get_session.get_effected_states(all_state)
    get_session.apply(global_effected_states, all_state)

    # Add Session Transition
    add_session = AddSession(
        {
            "source": all_state.local_states['variables'][0].value.current_value["source"],
            "type": all_state.local_states['variables'][0].value.current_value["type"],
            "data": all_state.local_states['variables'][0].value.current_value["data"],
        },
        calling_timestamp=2
    )
    global_effected_states, local_effected_states = add_session.get_effected_states(all_state)
    add_session.apply(global_effected_states, all_state)

    # Get Session Transition
    query_session = GetSession(
        parameters = {
            "source": "test",
            "type": "main_session",
            "id":1
        },
        calling_timestamp=3
    )

    global_effected_states, local_effected_states = query_session.get_effected_states(all_state)
    query_session.apply(global_effected_states, all_state)

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
    global_effected_states, local_effected_states = update_session.get_effected_states(all_state)
    update_session.apply(global_effected_states, all_state)

    # Delete Session Transition
    delete_session = DeleteSession(
        parameters = {
            "source": all_state.local_states['variables'][-1].value.current_value["source"],
            "type": all_state.local_states['variables'][-1].value.current_value["type"],
            "id": all_state.local_states['variables'][-1].value.current_value["id"]
        },
        calling_timestamp=5
    )
    global_effected_states, local_effected_states = delete_session.get_effected_states(all_state)
    delete_session.apply(global_effected_states, all_state)

    # Local Edit Transition
    local_edit = LocalEdit(
        parameters = {
            "session": all_state.local_states["variables"][-1].value,
            "meta_field": "source",
            "value": "new_test"
        },
        calling_timestamp=6
    )
    global_effected_states, local_effected_states = local_edit.get_effected_states(all_state)
    local_edit.apply(local_effected_states, all_state)
    
    
    
    assert len(all_state.local_states['variables']) == 5
    assert len(all_state.local_states['variables'][-1].value.transitions) == 2
    assert all_state.local_states['variables'][-1].value.current_value['source'] == "new_test"
    assert all_state.global_states['sessions'][1].current_value['data']['descriptions'] == "this is updated session"
    assert len(all_state.global_states['sessions']) == 3
    
    