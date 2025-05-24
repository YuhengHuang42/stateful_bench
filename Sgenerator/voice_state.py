from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional
import copy
from enum import Enum
from faker import Faker
import random

from Sgenerator.utils import get_nested_path_string
from Sgenerator.state import State, Transition, Schema, RandomInitializer, USER_FUNCTION_PARAM_FLAG, RESPONSE_VARIABLE_TEMP
from Sgenerator.state import INDENT, RESULT_NAME, ProgramEvaluator, LocalVariable
from Sgenerator import voice_lab
from Sgenerator.voice_lab import Speech

INIT_VOICE = "user_voice"

class VoiceLocalVariableType(Enum):
    STATE = "state"
    VOICE_TYPE = "voice_type" # Dict Combination of voice_name, voice_category, voice_gender
    VOICE_NAME = "voice_name"
    VOICE_CATEGORY = "voice_category"
    VOICE_GENDER = "voice_gender"
    VOICE_TEXT = "voice_text"
    VOICE_TRANSCRIPTION = "voice_transcription" # Dict Combination of text, diarize

class VoiceRandomInitializer(RandomInitializer):
    def __init__(self):
        super().__init__()
        self.fake = Faker()
    
    def random_generate_state(self):
        # Generate voice
        return VoiceState(
            name=USER_FUNCTION_PARAM_FLAG,
            voice_value=Speech(
                text=self.fake.paragraph(nb_sentences=5),
                voice_name=INIT_VOICE,
                stability=None,
                similarity_boost=None,
                style=None,
                duration=None
            ),
            outbound=False,
            created_by=USER_FUNCTION_PARAM_FLAG
        )
    
    def random_generate_text(self):
        lvar = LocalVariable(
            name=USER_FUNCTION_PARAM_FLAG,
            value=self.fake.paragraph(nb_sentences=5),
            latest_call=0,
            updated=True,
            created_by=USER_FUNCTION_PARAM_FLAG,
            variable_type=VoiceLocalVariableType.VOICE_TEXT
        )
        return lvar
    
    def random_generate_voice_term(self):
        lvar = LocalVariable(
            name=USER_FUNCTION_PARAM_FLAG,
            value={},
            latest_call=0,
            updated=True,
            created_by=USER_FUNCTION_PARAM_FLAG,
            variable_type=None
        )
        choice = random.random()
        if choice < 1/4:
            name = random.choice(voice_lab.voice_library.obtain_all_names())
            lvar.variable_type = VoiceLocalVariableType.VOICE_NAME
            lvar.value = {"voice_name": name}
        elif choice < 2/4:
            name = None
            category = random.choice(voice_lab.voice_library.obtain_all_categories())
            gender = None
            lvar.variable_type = VoiceLocalVariableType.VOICE_CATEGORY
            lvar.value = {"voice_category": category}
        elif choice < 3/4:
            name = None
            category = None
            gender = random.choice(voice_lab.voice_library.obtain_all_genders())
            lvar.variable_type = VoiceLocalVariableType.VOICE_GENDER
            lvar.value = {"voice_gender": gender}
        else:
            item = random.choice([(key, value) for (key, value) in voice_lab.voice_library.voices.items()])
            lvar.variable_type = VoiceLocalVariableType.VOICE_TYPE
            lvar.value = {"voice_name": item[0], "voice_category": item[1]["category"], "voice_gender": item[1]["gender"]}
        return lvar
    
    def random_generate_voice_state(self):
        lvar = LocalVariable(
            name=USER_FUNCTION_PARAM_FLAG,
            value=self.random_generate_state(),
        )

class VoiceState(State):
    def __init__(self, 
                 name: str,
                 voice_value: Speech,
                 outbound: bool = False
                 ):
        super().__init__(name)
        self.initial_value = OrderedDict([
            ("voice_value", voice_value),
            ("outbound", outbound)
        ])
        self.current_value = copy.deepcopy(self.initial_value)

    def __str__(self):
        return f"VoiceState(name={self.name}, voice_value={self.current_value['voice_value']}, outbound={self.current_value['outbound']})"

    def get_current_value(self):
        return self.current_value

class VoiceVariableSchema(Schema):
    def __init__(self):
        super().__init__()
        self.local_states = {
             "variables": [],
        }
        self.implicit_states = {
            "voice_info": {},
            "latest_call": {},
        }
        self.transitions = [
            SearchVoiceTransition,
            TextToSpeechTransition,
            SpeechToSpeechTransition,
            SpeechToTextTransition,
            IsolateAudioTransition,
            MakeOutboundCallTransition
        ]
        self.local_call_map = {}
        self.implicit_call_map = {}
        self.init_implict_dict = {}

    def add_implicit_variable(self, implicit_variable: Any, latest_call: int):
        assert implicit_variable.identifier not in self.implicit_states["voice_info"]
        self.implicit_states["voice_info"][implicit_variable.identifier] = implicit_variable
        self.implicit_states["latest_call"][implicit_variable.identifier] = latest_call
        self.init_implict_dict[implicit_variable.identifier] = implicit_variable

    def add_local_variable(self, local_variable: Any, update_implicit=False):
        self.local_states["variables"].append(local_variable)

    def prepare_initial_state(self, random_generator: VoiceRandomInitializer, config: Dict[str, Any], random_generate_config: Dict[str, Any]):
        self.clear_state()

    def clear_state(self):
        self.local_states["variables"] = []
        self.implicit_states["tensor_info"] = {}
        self.implicit_states["latest_call"] = {}
        self.init_implict_dict = {}
        self.implicit_call_map = {}
        self.local_call_map = {}
        self.init_load_info = {}
        self.init_local_info = []
        self.init_tensor_counter = 0
        self.init_weight_counter = 0
        voice_lab.voice_library.reset()
        voice_lab.call_recorder.reset()
    
    def get_implicit_states(self, current_value: bool = True):
        pass
    
    def align_initial_state(self):
        pass
    
    def get_latest_call_map(self):
        local_call_map = defaultdict(list)
        for idx, var in enumerate(self.local_states["variables"]):
            local_call_map[var.latest_call].append(idx)
        implicit_call_map = defaultdict(list)
        for tensor_name in self.implicit_states["voice_info"]:
            latest_call = self.implicit_states["latest_call"][tensor_name]
            implicit_call_map[latest_call].append(tensor_name)
    
    def determine_whether_to_keep_pair(self, previous_transition_info: Tuple, current_transition_info: Tuple):
        #  transition_info = (Transition, parameters)
        # 1. isolate_audio can only be selected if it is not selected before, or there is a speech_to_speech in between.
        # 2. make_outbound_call with the same parameters should not be selected.
        # 3. text_to_speech with the same parameters should not be selected.
        # 4. 
        pass
    
    def transform_parameters_to_str(self, parameters: Dict[str, Any]):
        result = ""
        for key in sorted(parameters.keys()):
            value = parameters[key]
            if isinstance(value, LocalVariable):
                result += f"{key}={value.name}, "
            elif isinstance(value, dict):
                result += f"{key}={{"
                for k, v in sorted(value.items()):
                    if isinstance(v, dict):
                        result += f"{k}="
                        result += self.transform_parameters_to_str(v)
                    elif isinstance(v, list):
                        result += f"{k}=["
                        for i, item in enumerate(v):
                            if isinstance(item, dict):
                                result += self.transform_parameters_to_str(item)
                            else:
                                result += f"{item}"
                            if i < len(v) - 1:
                                result += ", "
                        result += "]"
                    else:
                        result += f"{k}={v}"
                    result += ", "
                result = result.rstrip(", ") + "}, "
            elif isinstance(value, list) or isinstance(value, tuple):
                result += f"{key}=["
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        result += self.transform_parameters_to_str(item)
                    else:
                        result += f"{item}"
                    if i < len(value) - 1:
                        result += ", "
                result += "], "
            else:
                result += f"{key}={value}, "
        return result.rstrip(", ")
    
    def get_available_transitions(self, random_generator: Any, current_call: int, max_call: int, duplicate_local_variable_map: Dict[str, Set[str]], previous_transition_info: Tuple):
        """
        Return:
            {
                "required_parameters": target_parameters, # Dict
                "latest_call": local_variable.latest_call,
                "whether_updated": local_variable.updated,
                "producer_variable_idx": idx,
                "transition_pairs": transition_pairs,
                "local_variable_idx": idx,
            }
        """
        available_transitions = {}
        self.get_latest_call_map()
        # Outer loop: transition
        # Inner loop: variables
        # Check:
        # 1. Whether the transition is valid.
        # 2. whether it passes determine_whether_to_keep_pair
        # 3. Whether the transition passes duplicate_local_variable_map
        # 4. For the same transition, if there is updated variable as its parameter, we should choose it.
        for transition in self.transitions:
            if transition.__name__ not in duplicate_local_variable_map:
                duplicate_local_variable_map[transition.__name__] = set([])
            if transition.__name__ not in available_transitions:
                available_transitions[transition.__name__] = []
            # Different from other scenarios
            # Here we directly iterate over local variables
            # 
            for idx, lvar in enumerate(self.local_states["variables"]):
                if transition.__name__ == "SearchVoiceTransition":
                    # ["search_name", "search_category", "search_gender"]
                    if lvar.variable_type == VoiceLocalVariableType.VOICE_NAME:
                        required_parameters = {
                            "search_name": lvar,
                            "search_category": None,
                            "search_gender": None,
                        }
                    elif lvar.variable_type == VoiceLocalVariableType.VOICE_CATEGORY:
                        required_parameters = {
                            "search_category": lvar,
                            "search_name": None,
                            "search_gender": None,
                        }
                    elif lvar.variable_type == VoiceLocalVariableType.VOICE_GENDER:
                        required_parameters = {
                            "search_gender": lvar,
                            "search_name": None,
                            "search_category": None,
                        }
                    elif lvar.variable_type == VoiceLocalVariableType.VOICE_TYPE:
                        required_parameters = {
                            "search_name": lvar, # We deal this situation inside SearchVoiceTransition
                        }
                    else:
                        continue
                    if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, required_parameters)):
                        continue
                    str_required_parameters = self.transform_parameters_to_str(required_parameters)
                    if str_required_parameters in duplicate_local_variable_map[transition.__name__]:
                        continue
                    target_parameters = {
                        "required_parameters": required_parameters,
                        "latest_call": lvar.latest_call,
                        "whether_updated": lvar.updated,
                        "producer_variable_idx": idx,
                        "transition_pairs": [self.form_pair_transition(lvar, transition.__name__)],
                        "local_variable_idx": idx,
                    }
                    available_transitions[transition.__name__].append(target_parameters)
                elif transition.__name__ == "TextToSpeechTransition":
                    if lvar.variable_type == VoiceLocalVariableType.VOICE_TEXT:
                        # As long as we can get text, we will be able to call text_to_speech
                        for idx in range(len(self.local_states["variables"]), -1, -1):
                            variable_type = self.local_states["variables"][idx].variable_type
                            if variable_type == VoiceLocalVariableType.VOICE_NAME or variable_type == VoiceLocalVariableType.VOICE_TYPE:
                                voice_name = self.local_states["variables"][idx]
                                break
                        else:
                            voice_name = None
                        required_parameters = {
                            "text": lvar,
                            "voice_name": voice_name,
                        }
                        for param in ["stability", "similarity_boost", "style"]:
                            if random.random() < 0.5:  # 50% chance to include each parameter
                                required_parameters[param] = round(random.random(), 2)
                            else:
                                required_parameters[param] = None
                        if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, required_parameters)):
                            continue
                        skip_flag = False
                        for transition in lvar.transitions:
                            if transition[0] == transition.__name__:
                                # The text has already been modified. 
                                # In order to make the generated program more realistic, 
                                # we try in-place modification
                                text = lvar.value
                                sentences = text.split(".")
                                if len(sentences) > 1:
                                    text = sentences[0].strip() + "."
                                    required_parameters["text_modification"] = "split"
                                else:
                                    # Already modified once, skip.
                                    skip_flag = True
                        if skip_flag:
                            continue
                        str_required_parameters = self.transform_parameters_to_str(required_parameters)
                        if str_required_parameters in duplicate_local_variable_map[transition.__name__]:
                            continue  
                        target_parameters = {
                            "required_parameters": required_parameters,
                            "latest_call": lvar.latest_call,
                            "whether_updated": lvar.updated,
                            "producer_variable_idx": idx,
                            "transition_pairs": [self.form_pair_transition(lvar, transition.__name__)],
                            "local_variable_idx": idx,
                        }
                        available_transitions[transition.__name__].append(target_parameters)
                elif transition.__name__ == "SpeechToSpeechTransition":
                    # input_speech, voice_name
                    if lvar.variable_type == VoiceLocalVariableType.STATE:
                        find_voice = False
                        for idx in range(len(self.local_states["variables"]), -1, -1):
                            variable_type = self.local_states["variables"][idx].variable_type
                            if variable_type == VoiceLocalVariableType.VOICE_NAME or variable_type == VoiceLocalVariableType.VOICE_TYPE:
                                voice_name = self.local_states["variables"][idx]
                                find_voice = True
                                break
                        if not find_voice:
                            # Generate a new voice_name here
                            voice_name = random.choice(voice_lab.voice_library.obtain_all_names())
                        required_parameters = {
                            "input_speech": lvar,
                            "voice_name": voice_name,
                        }
                        if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, required_parameters)):
                            continue
                        str_required_parameters = self.transform_parameters_to_str(required_parameters)
                        if str_required_parameters in duplicate_local_variable_map[transition.__name__]:
                            continue
                        target_parameters = {
                            "required_parameters": required_parameters,
                            "latest_call": lvar.latest_call,
                            "whether_updated": lvar.updated,
                            "producer_variable_idx": idx,
                            "transition_pairs": [self.form_pair_transition(lvar, transition.__name__)],
                            "local_variable_idx": idx,
                        }
                        available_transitions[transition.__name__].append(target_parameters)
                elif transition.__name__ == "SpeechToTextTransition":
                    # input_speech, diarize
                    if lvar.variable_type == VoiceLocalVariableType.STATE:
                        required_parameters = {
                            "input_speech": lvar,
                            "diarize": random.random() < 0.5,
                        }
                        if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, required_parameters)):
                            continue
                        str_required_parameters = self.transform_parameters_to_str(required_parameters)
                        if str_required_parameters in duplicate_local_variable_map[transition.__name__]:
                            continue
                        target_parameters = {
                            "required_parameters": required_parameters,
                            "latest_call": lvar.latest_call,
                            "whether_updated": lvar.updated,
                            "producer_variable_idx": idx,
                            "transition_pairs": [self.form_pair_transition(lvar, transition.__name__)],
                            "local_variable_idx": idx,
                        }
                        available_transitions[transition.__name__].append(target_parameters)
                elif transition.__name__ == "IsolateAudioTransition":
                    # input_speech
                    if lvar.variable_type == VoiceLocalVariableType.STATE:
                        required_parameters = {
                            "input_speech": lvar,
                        }
                        if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, required_parameters)):
                            continue
                        str_required_parameters = self.transform_parameters_to_str(required_parameters)
                        if str_required_parameters in duplicate_local_variable_map[transition.__name__]:
                            continue
                        target_parameters = {
                            "required_parameters": required_parameters,
                            "latest_call": lvar.latest_call,
                            "whether_updated": lvar.updated,
                            "producer_variable_idx": idx,
                            "transition_pairs": [self.form_pair_transition(lvar, transition.__name__)],
                            "local_variable_idx": idx,
                        }
                        available_transitions[transition.__name__].append(target_parameters)
                elif transition.__name__ == "MakeOutboundCallTransition":
                    # input_speech, voice_name
                    if lvar.variable_type == VoiceLocalVariableType.STATE:
                        for idx in range(len(self.local_states["variables"]), -1, -1):
                            variable_type = self.local_states["variables"][idx].variable_type
                            if variable_type == VoiceLocalVariableType.VOICE_NAME or variable_type == VoiceLocalVariableType.VOICE_TYPE:
                                voice_name = self.local_states["variables"][idx]
                                break
                        else:
                            voice_name = None
                        required_parameters = {
                            "input_speech": lvar,
                        }
                        if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, required_parameters)):
                            continue
                        str_required_parameters = self.transform_parameters_to_str(required_parameters)
                        if str_required_parameters in duplicate_local_variable_map[transition.__name__]:
                            continue
                        target_parameters = {
                            "required_parameters": required_parameters,
                            "latest_call": lvar.latest_call,
                            "whether_updated": lvar.updated,
                            "producer_variable_idx": idx,
                            "transition_pairs": [self.form_pair_transition(lvar, transition.__name__)],
                            "local_variable_idx": idx,
                        }
                        available_transitions[transition.__name__].append(target_parameters)
                        
        # For the same transition, if there is updated variable as its parameter, we should choose it.
        for transition_name in available_transitions:
            this_transition_updated = []
            for target_parameters in available_transitions[transition_name]:
                if target_parameters["whether_updated"]:
                    this_transition_updated.append(target_parameters)
            if len(this_transition_updated) > 0:
                available_transitions[transition_name] = this_transition_updated
                
        return available_transitions
    
    def craft_transition(self, transition_info, calling_timestamp, transition, producer="None"):
        if transition == "TextToSpeechTransition":
            if transition_info["required_parameters"]["text_modification"] == "split":
                local_state = transition_info["required_parameters"]["text"]
                local_state.value = local_state.value.split(".")[0] + "."
        elif transition == "SpeechToSpeechTransition":
            if isinstance(transition_info["required_parameters"]["voice_name"], str):
                new_name = self.get_new_local_constant_name()
                new_local_variable = LocalVariable(name=new_name,
                                                   value=transition_info["required_parameters"]["voice_name"],
                                                   variable_type=VoiceLocalVariableType.VOICE_NAME,
                                                   latest_call=calling_timestamp,
                                                   updated=False, # This variable will soon be used
                                                   created_by=USER_FUNCTION_PARAM_FLAG,
                                                   transitions=[transition, transition_info["required_parameters"]]
                                                   )
                self.add_local_constant(new_local_variable)
                self.add_local_variable(new_local_variable)
                transition_info["required_parameters"]["voice_name"] = new_local_variable
        parameters = transition_info["required_parameters"]
        transition_class = globals()[transition]
        new_transition = transition_class(
            parameters=parameters, 
            calling_timestamp=calling_timestamp
        )
        new_transition.producer = producer
        return new_transition
        
    
    # Functions in voice_lab.py
    def search_voice_library(self, search_name: str, search_category: str, search_gender: str):
        return voice_lab.search_voice_library(search_name, search_category, search_gender)
    
    # Functions in voice_lab.py
    def text_to_speech(self, text: str, voice_name: str=None, stability: float=None, similarity_boost: float=None, style: float=None):
        return voice_lab.text_to_speech(text, voice_name, stability, similarity_boost, style)
    
    # Functions in voice_lab.py
    def speech_to_speech(self, input_speech: Speech, voice_name: str):
        return voice_lab.speech_to_speech(input_speech, voice_name)
    
    # Functions in voice_lab.py
    def speech_to_text(self, input_speech: Speech, diarize: bool):
        return voice_lab.speech_to_text(input_speech, diarize)
    
    # Functions in voice_lab.py
    def isolate_audio(self, input_speech: Speech):
        return voice_lab.isolate_audio(input_speech)
    
    # Functions in voice_lab.py
    def make_outbound_call(self, input_speech: Speech, voice_name: str):
        return voice_lab.make_outbound_call(input_speech, voice_name)
    

class SearchVoiceTransition(Transition):
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        super().__init__("SearchVoiceTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {}
        
    def get_required_parameters(self) -> List[str]:
        return ["search_name", "search_category", "search_gender"]
    
    def get_effected_states(self, variable_schema: VoiceVariableSchema) -> List[str]:
        """
        Search is a READ operation. No implicit states are affected.
        """
        implicit_states = []
        local_states = []
        search_name = self.parameters["search_name"]
        search_category = self.parameters["search_category"]
        search_gender = self.parameters["search_gender"]
        variable_list = []
        if search_name is not None and isinstance(search_name, LocalVariable):
            if search_name.variable_type == VoiceLocalVariableType.VOICE_TYPE:
                self.string_parameters["search_name"] = search_name.name + "['voice_name']"
                self.string_parameters["search_category"] = search_name.name + "['voice_category']"
                self.string_parameters["search_gender"] = search_name.name + "['voice_gender']"
            else:
                self.string_parameters["search_name"] = search_name.name
            variable_list.append(search_name.name)
        elif isinstance(search_name, str):
            self.string_parameters["search_name"] = f'"{search_name}"'
        elif search_name is not None:
            raise Exception(f"search_name must be a LocalVariable or a string, but got {type(search_name)}")
        
        if search_category is not None and isinstance(search_category, LocalVariable):
            self.string_parameters["search_category"] = search_category.name
            variable_list.append(search_category.name)
        elif isinstance(search_category, str):
            self.string_parameters["search_category"] = f'"{search_category}"'
        elif search_category is not None:
            raise Exception(f"search_category must be a LocalVariable or a string, but got {type(search_category)}")
        if search_gender is not None and isinstance(search_gender, LocalVariable):
            self.string_parameters["search_gender"] = search_gender.name
            variable_list.append(search_gender.name)
        elif isinstance(search_gender, str):
            self.string_parameters["search_gender"] = f'"{search_gender}"'
        elif search_gender is not None:
            raise Exception(f"search_gender must be a LocalVariable or a string, but got {type(search_gender)}")
        
        for name in variable_list:
            for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
                if variable_schema.local_states["variables"][local_idx].name == name:
                    local_states.append(local_idx)
                    break
        
        return implicit_states, local_states
    
    def apply(self, implicit_states, local_states, variable_schema: VoiceVariableSchema):
        longest_transitions = []
        for l_state in local_states:
            variable_schema.local_states["variables"][l_state].latest_call = self.calling_timestamp
            variable_schema.local_states["variables"][l_state].updated = False
            if len(variable_schema.local_states["variables"][l_state].transitions) > len(longest_transitions):
                longest_transitions = copy.deepcopy(variable_schema.local_states["variables"][l_state].transitions)
        pass_parameters = {}
        if self.parameters["search_name"] is not None:
            if isinstance(self.parameters["search_name"], LocalVariable):
                if self.parameters["search_name"].variable_type == VoiceLocalVariableType.VOICE_TYPE:
                    pass_parameters["search_name"] = self.parameters["search_name"].value["voice_name"]
                    pass_parameters["search_category"] = self.parameters["search_name"].value["voice_category"]
                    pass_parameters["search_gender"] = self.parameters["search_name"].value["voice_gender"]
        if len(pass_parameters) == 0:
            for key, value in self.parameters.items():
                if isinstance(value, LocalVariable):
                    pass_parameters[key] = value.value
                else:
                    pass_parameters[key] = value
        result = variable_schema.search_voice_library(**pass_parameters)
        for idx, item in enumerate(result):
            new_local_variable = LocalVariable(name = f"{self.new_variable_name}[{idx}]",
                                                value = item,
                                                latest_call = self.calling_timestamp,
                                                updated = True,
                                                created_by = f"{self.name}@{self.calling_timestamp}",
                                                variable_type = VoiceLocalVariableType.VOICE_TYPE)
            new_local_variable.is_indexed = True
            new_local_variable.transitions = copy.deepcopy(longest_transitions)
            new_local_variable.transitions.append([self.__class__.__name__, 
                                                   {'search_name': self.parameters["search_name"], 
                                                    "search_category": self.parameters["search_category"],
                                                    "search_gender": self.parameters["search_gender"]}])
            variable_schema.add_local_variable(new_local_variable)
        
    def __str__(self):
        return f"SearchVoiceTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name})"

    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        if self.parameters["search_name"] is not None:
            if isinstance(self.parameters["search_name"], LocalVariable):
                parameter_str += f"search_name={self.parameters['search_name'].name}, "
            else:
                parameter_str += f"search_name={self.parameters['search_name']}, "
        if self.parameters["search_category"] is not None:
            if isinstance(self.parameters["search_category"], LocalVariable):
                parameter_str += f"search_category={self.parameters['search_category'].name}, "
            else:
                parameter_str += f"search_category={self.parameters['search_category']}, "
        if self.parameters["search_gender"] is not None:
            if isinstance(self.parameters["search_gender"], LocalVariable):
                parameter_str += f"search_gender={self.parameters['search_gender'].name}, "
            else:
                parameter_str += f"search_gender={self.parameters['search_gender']}, "
        parameter_str = parameter_str.rstrip(", ")
        result = [
            f"{self.new_variable_name} = search_voice_library({parameter_str})\n",
        ]
        return result, ""
    
class TextToSpeechTransition(Transition):
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        super().__init__("TextToSpeechTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {}
        if "text_modification" in parameters:
            assert parameters["text_modification"] == "split", "text_modification must be split"
            assert isinstance(parameters["text"], LocalVariable), "text must be a LocalVariable"

    def get_required_parameters(self) -> List[str]:
        return ["text", "voice_name", "stability", "similarity_boost", "style"]
    
    def get_effected_states(self, variable_schema: VoiceVariableSchema) -> List[str]:
        """
        Text to speech will create a new voice state.
        """
        implicit_states = []
        local_states = []
        local_variable_list = []
        for key, value in self.parameters.items():
            if value is None:
                continue
            if isinstance(value, LocalVariable):
                local_variable_list.append(value.name)
        
        for name in local_variable_list:
            for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
                if variable_schema.local_states["variables"][local_idx].name == name:
                    local_states.append(local_idx)
                    break
        
        return implicit_states, local_states
    
    def apply(self, implicit_states, local_states, variable_schema: VoiceVariableSchema):
        for l_state in local_states:
            variable_schema.local_states["variables"][l_state].latest_call = self.calling_timestamp
            variable_schema.local_states["variables"][l_state].updated = False
        pass_parameters = {}
        longest_transitions = []
        for key, value in self.parameters.items():
            if key not in self.get_required_parameters():
                continue
            if isinstance(value, LocalVariable):
                if value.variable_type == VoiceLocalVariableType.VOICE_TYPE:
                    temp_name = value.name
                    pass_parameters[key] = value.value['voice_name']
                    self.string_parameters[key] = f"{temp_name}['voice_name']"
                else:
                    pass_parameters[key] = value.value
                    self.string_parameters[key] = value.name
                if len(value.transitions) > len(longest_transitions):
                    longest_transitions = copy.deepcopy(value.transitions)
            elif value is None:
                continue
            else:
                pass_parameters[key] = value
                if value is not None:
                    if isinstance(value, str):
                        self.string_parameters[key] = f'"{value}"'
                    else:
                        self.string_parameters[key] = value
        result = variable_schema.text_to_speech(**pass_parameters)
        new_voice_state = VoiceState(name=self.new_variable_name, 
                                     voice_value=result,
                                     outbound=False)
        new_voice_state.transitions = copy.deepcopy(longest_transitions)
        new_voice_state.transitions.append([self.__class__.__name__, {'text': self.parameters["text"], 
                                                               "voice_name": self.parameters["voice_name"],
                                                               "stability": self.parameters["stability"],
                                                               "similarity_boost": self.parameters["similarity_boost"],
                                                               "style": self.parameters["style"]}])
        new_local_variable = LocalVariable(name=self.new_variable_name,
                                          value=new_voice_state,
                                          latest_call=self.calling_timestamp,
                                          updated=True,
                                          created_by=f"{self.name}@{self.calling_timestamp}",
                                          variable_type=VoiceLocalVariableType.STATE)
        new_local_variable.transitions = copy.deepcopy(new_voice_state.transitions)
        variable_schema.add_implicit_variable(new_voice_state, self.calling_timestamp)
        variable_schema.add_local_variable(new_local_variable)

    def __str__(self):
        return f"TextToSpeechTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        for key, value in self.string_parameters.items():
            parameter_str += f"{key}={value}, "
        parameter_str = parameter_str.rstrip(", ")
        result = []
        if "text_modification" in self.parameters and self.parameters["text_modification"] == "split":
            result.append(f"{self.parameters['text'].name} = {self.parameters['text'].name}.split('.')[0] + '.'\n")
        result.append(f"{self.new_variable_name} = text_to_speech({parameter_str})\n")
        return result, ""


class SpeechToSpeechTransition(Transition):
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        super().__init__("SpeechToSpeechTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {}
        assert "input_speech" in self.parameters, "input_speech is required"
        assert isinstance(self.parameters["input_speech"], LocalVariable), "input_speech must be a LocalVariable"
        assert self.parameters["voice_name"] is not None, "voice_name is required"

    def get_required_parameters(self) -> List[str]:
        return ["input_speech", "voice_name"]
    
    def get_effected_states(self, variable_schema: VoiceVariableSchema) -> List[str]:
        """
        Speech to speech will create a new voice state.
        """
        implicit_states = []
        local_states = []
        local_variable_list = []
        input_speech_name = self.parameters["input_speech"].name
        self.string_parameters["input_speech"] = input_speech_name
        voice_name = None
        if isinstance(self.parameters["voice_name"], LocalVariable):
            if self.parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_NAME:
                voice_name = self.parameters["voice_name"].name
                self.string_parameters["voice_name"] = voice_name
            elif self.parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_TYPE:
                temp_name = self.parameters['voice_name'].name
                voice_name = f"{temp_name}['voice_name']" # Return of search_voice_library
                self.string_parameters["voice_name"] = voice_name
        else:
            self.string_parameters["voice_name"] = f'"{self.parameters["voice_name"]}"'
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == input_speech_name:
                local_states.append(local_idx)
                break
            if voice_name is not None:
                if variable_schema.local_states["variables"][local_idx].name == voice_name:
                    local_states.append(local_idx)
                    break
        
        implicit_states.append(input_speech_name)
        return implicit_states, local_states
    
    def apply(self, implicit_states, local_states, variable_schema: VoiceVariableSchema):
        for l_state in local_states:
            variable_schema.local_states["variables"][l_state].latest_call = self.calling_timestamp
            variable_schema.local_states["variables"][l_state].updated = False
        for i_state in implicit_states:
            variable_schema.implicit_states["latest_call"][i_state] = self.calling_timestamp
            #variable_schema.implicit_states["voice_info"][i_state].updated = False
        if isinstance(self.parameters["voice_name"], str):
            voice_name = self.parameters["voice_name"]
        else:
            if self.parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_NAME:
                voice_name = self.parameters["voice_name"].value
            elif self.parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_TYPE:
                voice_name = self.parameters["voice_name"].value["voice_name"] # Return of search_voice_library
        pass_parameters = {"input_speech": self.parameters["input_speech"].value.current_value["voice_value"], "voice_name": voice_name}
        result = variable_schema.speech_to_speech(**pass_parameters)
        
        
        
        new_voice_state = VoiceState(name=self.new_variable_name, 
                                     voice_value=result,
                                     outbound=False
                                     )
        new_voice_state.transitions = copy.deepcopy(self.parameters["input_speech"].value.transitions)
        new_voice_state.transitions.append([self.__class__.__name__, {'input_speech': self.parameters["input_speech"].name, "voice_name": voice_name}])
        variable_schema.add_implicit_variable(new_voice_state, self.calling_timestamp)
        new_local_variable = LocalVariable(name=self.new_variable_name,
                                          value=new_voice_state,
                                          latest_call=self.calling_timestamp,
                                          updated=True,
                                          created_by=f"{self.name}@{self.calling_timestamp}",
                                          variable_type=VoiceLocalVariableType.STATE)
        new_local_variable.transitions = copy.deepcopy(new_voice_state.transitions)
        variable_schema.add_local_variable(new_local_variable)
    
    def __str__(self):
        return f"SpeechToSpeechTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        for key, value in self.string_parameters.items():
            parameter_str += f"{key}={value}, "
        parameter_str = parameter_str.rstrip(", ")
        result = [
            f"{self.new_variable_name} = speech_to_speech({parameter_str})",
        ] 
        return result, ""

class SpeechToTextTransition(Transition):
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        super().__init__("SpeechToTextTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {}
        assert "input_speech" in self.parameters, "input_speech is required"
        assert isinstance(self.parameters["input_speech"], LocalVariable), "input_speech must be a LocalVariable"

    def get_required_parameters(self) -> List[str]:
        return ["input_speech", "diarize"]
    
    def get_effected_states(self, variable_schema: VoiceVariableSchema) -> List[str]:
        implicit_states = []
        local_states = []
        input_speech_name = self.parameters["input_speech"].name
        self.string_parameters["input_speech"] = input_speech_name
        self.string_parameters["diarize"] = self.parameters["diarize"]
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == input_speech_name:
                local_states.append(local_idx)
                break
        implicit_states.append(input_speech_name)
        return implicit_states, local_states
    
    def apply(self, implicit_states, local_states, variable_schema: VoiceVariableSchema):
        for l_state in local_states:
            variable_schema.local_states["variables"][l_state].latest_call = self.calling_timestamp
            variable_schema.local_states["variables"][l_state].updated = False
        assert len(implicit_states) == 1, "There should be only one implicit state for speech_to_text"
        implicit_state = variable_schema.implicit_states["voice_info"][implicit_states[0]]
        variable_schema.implicit_states["latest_call"][implicit_state.identifier] = self.calling_timestamp
        #implicit_state.latest_call = self.calling_timestamp
        #implicit_state.updated = False
        #implicit_state.transitions.append(["speech_to_text", 
        #                                       {'input_speech': self.parameters["input_speech"].name, 
        #                                        "diarize": self.parameters["diarize"]}]
        #                                      )
        
        pass_parameters = {"input_speech": self.parameters["input_speech"].value.current_value["voice_value"], "diarize": self.parameters["diarize"]}
        result = variable_schema.speech_to_text(**pass_parameters)
        new_local_variable = LocalVariable(name=self.new_variable_name,
                                          value=result,
                                          latest_call=self.calling_timestamp,
                                          updated=True,
                                          created_by=f"{self.name}@{self.calling_timestamp}",
                                          variable_type=VoiceLocalVariableType.VOICE_TRANSCRIPTION)
        new_local_variable.transitions = copy.deepcopy(implicit_state.transitions)
        new_local_variable.transitions.append([self.__class__.__name__, 
                                               {'input_speech': self.parameters["input_speech"].name, 
                                                "diarize": self.parameters["diarize"]}])
        variable_schema.add_local_variable(new_local_variable)
    
    def __str__(self):
        return f"SpeechToTextTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        for key, value in self.string_parameters.items():
            parameter_str += f"{key}={value}, "
        parameter_str = parameter_str.rstrip(", ")
        result = [
            f"{self.new_variable_name} = speech_to_text({parameter_str})",
        ] 
        return result, ""
        

class IsolateAudioTransition(Transition):
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        super().__init__("IsolateAudioTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {}
        assert "input_speech" in self.parameters, "input_speech is required"
        assert isinstance(self.parameters["input_speech"], LocalVariable), "input_speech must be a LocalVariable"

    def get_required_parameters(self) -> List[str]:
        return ["input_speech"]
    
    def get_effected_states(self, variable_schema: VoiceVariableSchema) -> List[str]:
        implicit_states = []
        local_states = []
        input_speech_name = self.parameters["input_speech"].name
        self.string_parameters["input_speech"] = input_speech_name
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == input_speech_name:
                local_states.append(local_idx)
                break
        implicit_states.append(input_speech_name)
        return implicit_states, local_states
    
    def apply(self, implicit_states, local_states, variable_schema: VoiceVariableSchema):
        assert len(implicit_states) == 1, "There should be only one implicit state for isolate_audio"
        assert len(local_states) == 1, "There should be only one local state for isolate_audio"
        
        implicit_state = variable_schema.implicit_states["voice_info"][implicit_states[0]]
        variable_schema.implicit_states["latest_call"][implicit_state.identifier] = self.calling_timestamp
        #implicit_state.latest_call = self.calling_timestamp
        #implicit_state.updated = False
        local_state = variable_schema.local_states["variables"][local_states[0]]
        local_state.latest_call = self.calling_timestamp
        local_state.updated = False
        
        pass_parameters = {"input_speech": self.parameters["input_speech"].value.current_value["voice_value"]}
        result = variable_schema.isolate_audio(**pass_parameters)
        new_implicit_variable = VoiceState(name=self.new_variable_name,
                                          voice_value=result,
                                          outbound=False)
        new_implicit_variable.transitions = copy.deepcopy(implicit_state.transitions)
        new_implicit_variable.transitions.append([self.__class__.__name__, {'input_speech': self.parameters["input_speech"].name}])
        variable_schema.add_implicit_variable(new_implicit_variable, self.calling_timestamp)
        
        #implicit_state.transitions.append(["isolate_audio", {'input_speech': self.parameters["input_speech"].name}])
        new_local_variable = LocalVariable(name=self.new_variable_name,
                                          value=new_implicit_variable,
                                          latest_call=self.calling_timestamp,
                                          updated=True,
                                          created_by=f"{self.name}@{self.calling_timestamp}",
                                          variable_type=VoiceLocalVariableType.STATE)
        new_local_variable.transitions = copy.deepcopy(new_implicit_variable.transitions)
        variable_schema.add_local_variable(new_local_variable)
    
    def __str__(self):
        return f"IsolateAudioTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        for key, value in self.string_parameters.items():
            parameter_str += f"{key}={value}, "
        parameter_str = parameter_str.rstrip(", ")
        result = [
            f"{self.new_variable_name} = isolate_audio({parameter_str})",
        ] 
        return result, ""

class MakeOutboundCallTransition(Transition):
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        super().__init__("MakeOutboundCallTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {}
        assert "input_speech" in self.parameters, "input_speech is required"
        assert isinstance(self.parameters["input_speech"], LocalVariable), "input_speech must be a LocalVariable"
        assert self.parameters["voice_name"] is not None, "voice_name is required"

    def get_required_parameters(self) -> List[str]:
        return ["input_speech", "voice_name"]
    
    def get_effected_states(self, variable_schema: VoiceVariableSchema) -> List[str]:
        implicit_states = []
        local_states = []
        input_speech_name = self.parameters["input_speech"].name
        self.string_parameters["input_speech"] = input_speech_name
        if isinstance(self.parameters["voice_name"], str):
            voice_name = f'"{self.parameters["voice_name"]}"'
        else:
            if self.parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_NAME:
                voice_name = self.parameters["voice_name"].name
            elif self.parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_TYPE:
                temp_name = self.parameters['voice_name'].name
                voice_name = f"{temp_name}['voice_name']" # Return of search_voice_library
        self.string_parameters["voice_name"] = voice_name
        
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == input_speech_name:
                local_states.append(local_idx)
                break
        implicit_states.append(input_speech_name)
        return implicit_states, local_states
    
    def apply(self, implicit_states, local_states, variable_schema: VoiceVariableSchema):
        assert len(implicit_states) == 1, "There should be only one implicit state for make_outbound_call"
        assert len(local_states) == 1, "There should be only one local state for make_outbound_call"
        implicit_state = variable_schema.implicit_states["voice_info"][implicit_states[0]]
        variable_schema.implicit_states["latest_call"][implicit_state.identifier] = self.calling_timestamp
        implicit_state.current_value["outbound"] = True
        implicit_state.transitions.append([self.__class__.__name__, 
                                           {'input_speech': self.parameters["input_speech"].name,
                                            "voice_name": self.parameters["voice_name"]}]
                                          )
        pass_parameters = {"input_speech": self.parameters["input_speech"].value.current_value["voice_value"], "voice_name": self.parameters["voice_name"]}
        result = variable_schema.make_outbound_call(**pass_parameters)
        assert result, "make_outbound_call failed"
        assert variable_schema.local_states["variables"][local_states[0]].variable_type == VoiceLocalVariableType.STATE, "local_state should be a VoiceState"
        variable_schema.local_states["variables"][local_states[0]].latest_call = self.calling_timestamp
        variable_schema.local_states["variables"][local_states[0]].updated = False
        variable_schema.local_states["variables"][local_states[0]].value = copy.deepcopy(implicit_state)
        variable_schema.local_states["variables"][local_states[0]].transitions = copy.deepcopy(implicit_state.transitions)
    
    def __str__(self):
        return f"MakeOutboundCallTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        for key, value in self.string_parameters.items():
            parameter_str += f"{key}={value}, "
        parameter_str = parameter_str.rstrip(", ")
        result = [
            f"{self.new_variable_name} = make_outbound_call({parameter_str})",
        ] 
        return result, ""


def test_search_voice_library():
    schema = VoiceVariableSchema()
    new_local_variable = LocalVariable(
        name="user_variable_1",
        value="Emma",
        latest_call=0,
        updated=True,
        created_by="user",
        variable_type=VoiceLocalVariableType.VOICE_NAME
    )
    transition = SearchVoiceTransition({"search_name": new_local_variable, "search_category": None, "search_gender": None}, 1)
    implicit_states, local_states = transition.get_effected_states(schema)
    transition.apply(implicit_states, local_states, schema)
    assert len(schema.local_states['variables']) == 1

    transition = SearchVoiceTransition({"search_name": None, "search_category": "conversational", "search_gender": None}, 2)
    implicit_states, local_states = transition.get_effected_states(schema)
    transition.apply(implicit_states, local_states, schema)
    assert schema.local_states['variables'][-1].name == 'response_2[2]'

def test_text_to_speech():
    schema = VoiceVariableSchema()
    new_local_variable = LocalVariable(
        name="user_variable_1",
        value="hello world",
        latest_call=0,
        updated=True,
        created_by="user",
        variable_type=VoiceLocalVariableType.VOICE_TEXT
    )
    transition = TextToSpeechTransition({"text": new_local_variable, 
                                        "voice_name": None, 
                                        "stability": None,
                                        "similarity_boost": None,
                                        "style": None
                                        }, 1
                                    )

    implicit_states, local_states = transition.get_effected_states(schema)
    transition.apply(implicit_states, local_states, schema)
    assert len(schema.local_states['variables']) == 1
    assert "user_variable_1" in transition.get_program_str()[0][0]

    transition = TextToSpeechTransition({"text": "hello", 
                                        "voice_name": "Emma", 
                                        "stability": None,
                                        "similarity_boost": None,
                                        "style": None
                                        }, 2
                                    )
    implicit_states, local_states = transition.get_effected_states(schema)
    transition.apply(implicit_states, local_states, schema)
    assert schema.local_states['variables'][-1].name == 'response_2'
    assert 'response_2 = text_to_speech(text="hello", voice_name="Emma")' in transition.get_program_str()[0][0]

def test_speech_to_speech():
    schema = VoiceVariableSchema()
    voice_state = VoiceState(
        name="user_voice_1",
        voice_value=Speech(
            "hello_world"
        )
    )
    new_local_variable = LocalVariable(
        name="user_voice_1",
        value=voice_state,
        latest_call=0,
        updated=True,
        created_by="user",
        variable_type=VoiceLocalVariableType.STATE
    )
    schema.add_implicit_variable(voice_state, 0)
    schema.add_local_variable(new_local_variable)
    transition = SpeechToSpeechTransition({"input_speech": new_local_variable, 
                                        "voice_name": "Emma"
                                        }, 1
                                    )

    implicit_states, local_states = transition.get_effected_states(schema)
    transition.apply(implicit_states, local_states, schema)
    assert len(schema.local_states['variables']) == 2
    assert isinstance(schema.local_states['variables'][-1].value, VoiceState)
    assert "response_1" in schema.implicit_states['voice_info']

def test_speech_to_text():
    schema = VoiceVariableSchema()
    voice_state = VoiceState(
        name="user_voice_1",
        voice_value=Speech(
            "hello_world"
        )
    )
    new_local_variable = LocalVariable(
        name="user_voice_1",
        value=voice_state,
        latest_call=0,
        updated=True,
        created_by="user",
        variable_type=VoiceLocalVariableType.STATE
    )
    schema.add_implicit_variable(voice_state, 0)
    schema.add_local_variable(new_local_variable)
    transition = SpeechToTextTransition({"input_speech": new_local_variable, 
                                        "diarize": True
                                        }, 1
                                    )

    implicit_states, local_states = transition.get_effected_states(schema)
    transition.apply(implicit_states, local_states, schema)
    assert schema.local_states['variables'][-1].variable_type == VoiceLocalVariableType.VOICE_TRANSCRIPTION
    assert 'response_1 = speech_to_text(input_speech=user_voice_1, diarize=True)' in transition.get_program_str()[0][0]

def test_isolate_audio():
    schema = VoiceVariableSchema()
    voice_state = VoiceState(
        name="user_voice_1",
        voice_value=Speech(
            "hello_world"
        )
    )
    new_local_variable = LocalVariable(
        name="user_voice_1",
        value=voice_state,
        latest_call=0,
        updated=True,
        created_by="user",
        variable_type=VoiceLocalVariableType.STATE
    )
    schema.add_implicit_variable(voice_state, 0)
    schema.add_local_variable(new_local_variable)
    transition = IsolateAudioTransition({"input_speech": new_local_variable, 
                                        }, 1
                                    )

    implicit_states, local_states = transition.get_effected_states(schema)
    transition.apply(implicit_states, local_states, schema)
    assert schema.local_states['variables'][-1].variable_type == VoiceLocalVariableType.STATE
    assert (schema.local_states['variables'][-1].value.current_value["voice_value"].duration - 8.0) < 1e-6
    assert 'response_1 = isolate_audio(input_speech=user_voice_1)' in transition.get_program_str()[0][0]

def test_make_outbound_call():
    schema = VoiceVariableSchema()
    voice_state = VoiceState(
        name="user_voice_1",
        voice_value=Speech(
            "hello_world"
        )
    )
    new_local_variable = LocalVariable(
        name="user_voice_1",
        value=voice_state,
        latest_call=0,
        updated=True,
        created_by="user",
        variable_type=VoiceLocalVariableType.STATE
    )
    schema.add_implicit_variable(voice_state, 0)
    schema.add_local_variable(new_local_variable)
    transition = MakeOutboundCallTransition({"input_speech": new_local_variable, 
                                            "voice_name": "Emma"
                                        }, 1
                                    )

    implicit_states, local_states = transition.get_effected_states(schema)
    transition.apply(implicit_states, local_states, schema)
    assert schema.local_states['variables'][-1].variable_type == VoiceLocalVariableType.STATE
    assert schema.local_states["variables"][-1].value.current_value["outbound"]
    assert schema.implicit_states["voice_info"]['user_voice_1'].current_value['outbound']
    assert 'response_1 = make_outbound_call(input_speech=user_voice_1, voice_name="Emma")' in transition.get_program_str()[0][0]