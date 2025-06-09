from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional
import copy
from enum import Enum
from faker import Faker
import random
from loguru import logger
import traceback
import json

from Sgenerator.utils import get_nested_path_string
from Sgenerator.state import State, Transition, Schema, RandomInitializer, USER_FUNCTION_PARAM_FLAG, RESPONSE_VARIABLE_TEMP
from Sgenerator.state import INDENT, RESULT_NAME, ProgramEvaluator, LocalVariable
from Sgenerator import voice_lab
from Sgenerator.voice_lab import Speech

from Sgenerator.voice_lab import (
    speech_to_speech,
    text_to_speech,
    speech_to_text,
    isolate_audio,
    make_outbound_call,
    search_voice_library,
    Speech,
)

INIT_VOICE = "user_voice"
LOOP_VARIABLE = "voice_item"

VOICE_SUMMARY_PROMPT = '''The below code is about a set of SpeechLab APIs that provide advanced speech and audio processing capabilities, 
including voice search, text-to-speech, speech transformation, transcription, and telephony integration. 
These APIs enable flexible manipulation and conversion of audio data for a variety of applications such as voice synthesis, voice conversion, and automated calling.

API Functions:
search_voice_library - Search for voices in the ElevenLabs voice library by name, category, or gender, returning matching voice details.
text_to_speech - Convert text into speech audio using a specified or default voice, with options to control stability, similarity, and style of the generated voice.
speech_to_speech - Transform audio from one voice to another, enabling voice conversion using provided audio and target voice.
speech_to_text - Transcribe spoken audio into text, with optional speaker diarization to identify who is speaking.
isolate_audio - Isolate and extract audio from a given input, returning the processed audio.
make_outbound_call - Initiate an outbound phone call using a specified voice and audio via Twilio and ElevenLabs integration.
'''

VOICE_GENERATION_PROMPT = '''All APIs, user-related variables, and constants have been preloaded into memory and are available for direct use.
Please begin your Python code generation with a code block (with ```), for example:
```
response_1 = text_to_speech(text=user_variable)
```
Your code:
'''

class VoiceLocalVariableType(Enum):
    STATE = "state"
    VOICE_NAME = "voice_name"
    VOICE_SEARCH_CONTENT = "voice_search_content" # used for search_voice_library
    VOICE_RETURN_CONTENT = "voice_return_content" # used for search_voice_library
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
                duration=voice_lab.DEFAULT_DURATION
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
            lvar.value = name
        elif choice < 2/4:
            name = None
            category = random.choice(voice_lab.voice_library.obtain_all_categories())
            gender = None
            lvar.variable_type = VoiceLocalVariableType.VOICE_SEARCH_CONTENT
            lvar.value = {"search_category": category}
        elif choice < 3/4:
            name = None
            category = None
            gender = random.choice(voice_lab.voice_library.obtain_all_genders())
            lvar.variable_type = VoiceLocalVariableType.VOICE_SEARCH_CONTENT
            lvar.value = {"search_gender": gender}
        else:
            item = random.choice([(key, value) for (key, value) in voice_lab.voice_library.voices.items()])
            lvar.variable_type = VoiceLocalVariableType.VOICE_SEARCH_CONTENT
            lvar.value = {"search_category": item[1]["category"], "search_gender": item[1]["gender"]}
        return lvar

class VoiceState(State):
    def __init__(self, 
                 name: str,
                 voice_value: Speech,
                 outbound: bool = False,
                 created_by: str = None
                 ):
        super().__init__(name, created_by=created_by)
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
        if local_variable.variable_type == VoiceLocalVariableType.VOICE_NAME:
            assert not isinstance(local_variable.value, VoiceState), "VoiceName should not be a VoiceState"
        self.local_states["variables"].append(local_variable)

    def get_load_info(self, init_load_info=None):
        # Return: program_string, list of (variable_name, variable_value)
        if init_load_info is None:
            init_load_info = self.init_load_info
        if init_load_info is None or len(init_load_info) == 0:
            return None, None
        init_program = "# == The variables below will be pre-loaded into the memory. Here we only show the shape of the variables. ==\n"
        all_init_pairs = []
        for name, value in init_load_info.items():
            init_program += f"{name} # This is user-provided voice variable.\n"
            all_init_pairs.append((name, value))
        init_program += "# == The variables above will be pre-loaded into the memory at evaluation runtime. ==\n"
        return init_program, all_init_pairs
    
    def prepare_initial_state(self, 
                              random_generator: VoiceRandomInitializer, 
                              config: Dict[str, Any], 
                              random_generate_config: Dict[str, Any]):
        self.clear_state()
        which_as_input = random.choice([0, 1, 2]) # 0: text, 1: voice, 2: text and voice
        text_range = random.randint(config["init_text_range"][0], config["init_text_range"][1])
        voice_range = random.randint(config["init_voice_range"][0], config["init_voice_range"][1])
        if "max_init_state_num" in config:
            max_init_state_num = config["max_init_state_num"]
            self.max_init_state_num = max_init_state_num
        else:
            self.max_init_state_num = None
            max_init_state_num = None
        if which_as_input == 0:
            for i in range(text_range):
                text = random_generator.random_generate_text()
                text.name = f"{text.name}_{len(self.init_local_info)}"
                if isinstance(text.value, str):
                    local_right = f"\"{text.value}\""
                else:
                    local_right = text.value
                self.init_local_info.append([text.name, local_right])
                self.add_local_variable(text)
            voice = None
        elif which_as_input == 1:
            text = None
            for i in range(voice_range):
                voice = random_generator.random_generate_state()
                voice.identifier = f"{voice.identifier}_{len(self.init_load_info)}_load"
                self.init_load_info[voice.identifier] = voice.initial_value["voice_value"]
                new_local_variable = LocalVariable(name=voice.identifier,
                                                value=voice,
                                                latest_call=0,
                                                updated=True,
                                                created_by=USER_FUNCTION_PARAM_FLAG,
                                                variable_type=VoiceLocalVariableType.STATE
                                                )
                self.add_local_variable(new_local_variable)
                self.add_implicit_variable(voice, 0)
        elif which_as_input == 2:
            if max_init_state_num is not None and text_range + voice_range > max_init_state_num:
                # Reduce both ranges proportionally to fit within max_state_num
                total = text_range + voice_range
                text_range = int((text_range / total) * max_init_state_num)
                text_range = max(text_range, 1)
                voice_range = max_init_state_num - text_range
                voice_range = max(voice_range, 1)
            for i in range(text_range):
                text = random_generator.random_generate_text()
                text.name = f"{text.name}_{len(self.init_local_info)}"
                if isinstance(text.value, str):
                    local_right = f"\"{text.value}\""
                else:
                    local_right = text.value
                self.init_local_info.append([text.name, local_right])
                self.add_local_variable(text)
            for i in range(voice_range):
                voice = random_generator.random_generate_state()
                voice.identifier = f"{voice.identifier}_{len(self.init_load_info)}_load"
                new_local_variable = LocalVariable(name=voice.identifier,
                                                value=voice,
                                                latest_call=0,
                                                updated=True,
                                                created_by=USER_FUNCTION_PARAM_FLAG,
                                                variable_type=VoiceLocalVariableType.STATE
                                                )
                self.init_load_info[voice.identifier] = new_local_variable.value.initial_value["voice_value"]
                self.add_local_variable(new_local_variable)
                self.add_implicit_variable(voice, 0)
        voice_name = random_generator.random_generate_voice_term()
        voice_name.name = f"{voice_name.name}_{len(self.init_local_info)}"
        if isinstance(voice_name.value, Dict) and len(voice_name.value) == 1:
            single_value = list(voice_name.value.values())[0]
            if isinstance(single_value, str):
                local_right = f"\"{single_value}\""
            else:
                local_right = single_value
            self.init_local_info.append([voice_name.name, local_right])
        else:
            if isinstance(voice_name.value, str):
                local_right = f"\"{voice_name.value}\""
            else:
                local_right = voice_name.value
            self.init_local_info.append([voice_name.name, local_right])
        self.add_local_variable(voice_name)
        self.align_initial_state()

    def clear_state(self):
        self.local_states["variables"] = []
        self.implicit_states["voice_info"] = {}
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
    
    def postprocess_choose_result(self):
        result_var_list = []
        index_book = set()
        for idx in range(len(self.local_states["variables"])-1, -1, -1):
            local_variable = self.local_states["variables"][idx]
            #transitions = local_variable.value.transitions
            if local_variable.updated == True:
                # Either being updated or being queried.
                #result_str = f"{RESULT_NAME} = {local_variable.name}"
                if local_variable.is_indexed:
                    local_variable_name = local_variable.name.split("[")[0]
                    if local_variable_name in index_book:
                        continue
                    index_book.add(local_variable_name)
                    result_var_list.append(local_variable_name)
                else:
                    result_var_list.append(local_variable)
        if len(result_var_list) == 0:
            return None
        elif len(result_var_list) == 1:
            if isinstance(result_var_list[0], LocalVariable):
                return f"{RESULT_NAME} = {result_var_list[0].name}"
            else:
                return f"{RESULT_NAME} = {result_var_list[0]}"
        else:
            result_str = f"{RESULT_NAME} = ("
            for idx, result_var in enumerate(result_var_list):
                if isinstance(result_var, LocalVariable):
                    result_str += f"{result_var.name}, "
                else:
                    result_str += f"{result_var}, "
            result_str = result_str[:-2]
            result_str += ")"
            return result_str
    
    def postprocess_transitions(self, remaining_call: int) -> Tuple[bool, List[str]]:
        '''
        No need to postprocess. We do not have remote database to be updated.
        '''
        return False, []
    
    def get_implicit_states(self, current_value: bool = True):
        result = {}
        if current_value is True:
            voice_info = self.implicit_states["voice_info"]
            for key, value in voice_info.items():
                result[key] = value.current_value
        else:
            voice_info = self.init_implict_dict
            for key, value in voice_info.items():
                result[key] = value
        return result
    
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
        # 2. make_outbound_call with the same parameters should not be selected. --> Be managed by duplicate_local_variable_map
        # 3. text_to_speech with the same parameters should not be selected. --> Be managed by duplicate_local_variable_map
        # 4. 
        if current_transition_info[0] == "IsolateAudioTransition":
            state = current_transition_info[1]["input_speech"]
            if isinstance(state, LocalVariable):
                name = state.name
            elif isinstance(state, VoiceState):
                name = state.identifier
            else:
                raise ValueError(f"Invalid state type: {type(state)}")
            transition_list = self.implicit_states["voice_info"][name].transitions
            for idx in range(len(transition_list)-1, -1, -1):
                if transition_list[idx]["name"] == "IsolateAudioTransition":
                    # Check if there's a speech_to_speech transition between this and current
                    found_speech_to_speech = False
                    for t in transition_list[idx:]:
                        if t["name"] == "SpeechToSpeechTransition":
                            found_speech_to_speech = True
                            break
                    if not found_speech_to_speech:
                        return False
        
        return True
    
    
    def transform_parameters_to_str(self, parameters: Dict[str, Any]):
        result = ""
        for key in sorted(parameters.keys()):
            value = parameters[key]
            if isinstance(value, LocalVariable):
                if isinstance(value.value, VoiceState):
                    result += f"{key}={value.value.identifier}, "
                else:
                    if value.variable_type == VoiceLocalVariableType.VOICE_SEARCH_CONTENT:
                        if "search_category" in value.value:
                            result += f"{key}={value.value['search_category']}, "
                        elif "search_gender" in value.value:
                            result += f"{key}={value.value['search_gender']}, "
                    else:
                        result += f"{key}={value.value}, "
            elif isinstance(value, VoiceState):
                result += f"{key}={value.identifier}, "
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
    
    def obtain_if_condition(self):
        """
        Obtain the condition for the if-else transition.
         (left_variable_name, index, right_value)
         Return:
            if_condition: The condition for the if-else transition.
            whether_replace_by_variable: Whether the if-else transition is replaced by a variable.
            additional_content: Additional statement for the if-else transition.
        """
        whether_replace_by_variable = True
        additional_content = None
        found_flag = False
        for idx in range(len(self.local_states["variables"])-1, -1, -1):
            local_variable = self.local_states["variables"][idx]
            if local_variable.variable_type == VoiceLocalVariableType.VOICE_NAME:
                if isinstance(local_variable.value, dict) and len(local_variable.value) == 1:
                    right_value = list(local_variable.value.values())[0]
                else:
                    right_value = local_variable.value
                if_condition = (local_variable.name, None, right_value)
                found_flag = True
                break
            elif local_variable.variable_type == VoiceLocalVariableType.VOICE_RETURN_CONTENT:
                assert local_variable.is_indexed
                search_statements = ["found_flag = False\n"]
                local_variable_name = local_variable.name
                local_variable_name = local_variable_name.split("[")[0]
                search_statements.append(f"for i in {local_variable_name}:\n")
                search_statements.append(f"{INDENT}if i['category'] == \"{local_variable.value['category']}\" and i['gender'] == \"{local_variable.value['gender']}\":\n")
                search_statements.append(f"{INDENT}{INDENT}found_flag = True\n")
                search_statements.append(f"{INDENT}{INDENT}break\n")
                if_condition = ("found_flag", None, True)
                additional_content = [search_statements, ""]
                found_flag = True
                break
            elif local_variable.variable_type == VoiceLocalVariableType.VOICE_TEXT:
                if_condition = (f"len({local_variable.name}.split('.'))", None, len(local_variable.value.split(".")))
                found_flag = True
                break
            elif local_variable.variable_type == VoiceLocalVariableType.VOICE_SEARCH_CONTENT:
                if len(local_variable.value) == 2:
                    if_condition = (f"{local_variable.name}['search_category']", None, local_variable.value["search_category"])
                    found_flag = True
                    break
                else:
                    if "search_category" in local_variable.value:
                        if_condition = (f"{local_variable.name}", None, local_variable.value["search_category"]) # Initialized value is without dict
                        found_flag = True
                        break
                    else:
                        if_condition = (f"{local_variable.name}", None, local_variable.value["search_gender"])  # Initialized value is without dict
                        found_flag = True
                        break
        if not found_flag:
            raise Exception("No if-else condition can be found.")
                
        return if_condition, whether_replace_by_variable, additional_content
    
    def get_available_transitions(self, 
                                  random_generator: Any, 
                                  current_call: int, 
                                  max_call: int, 
                                  duplicate_local_variable_map: Dict[str, Set[str]], 
                                  previous_transition_info: Tuple):
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
                    # ["search_category", "search_gender"]
                    if lvar.variable_type == VoiceLocalVariableType.VOICE_SEARCH_CONTENT:
                        if "search_category" in lvar.value:
                            if "search_gender" in lvar.value:
                                required_parameters = {
                                    "search_category": lvar,
                                    "search_gender": lvar,
                                }
                            else:
                                required_parameters = {
                                    "search_category": lvar,
                                    "search_gender": None,
                                }
                        elif "search_gender" in lvar.value:
                            required_parameters = {
                                "search_category": None,
                                "search_gender": lvar,
                            }
                        else:
                            continue
                    else:
                        continue
                    if not self.determine_whether_to_keep_pair(previous_transition_info, (transition.__name__, required_parameters)):
                        continue
                    
                    str_required_parameters = self.transform_parameters_to_str(required_parameters)
                    if str_required_parameters in duplicate_local_variable_map[transition.__name__]:
                        #print(f"Duplicate local variable: {str_required_parameters}")
                        #print(duplicate_local_variable_map[transition.__name__])
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
                        candidate_voice_name = []
                        for idx in range(len(self.local_states["variables"])-1, -1, -1):
                            variable_type = self.local_states["variables"][idx].variable_type
                            if variable_type == VoiceLocalVariableType.VOICE_NAME or variable_type == VoiceLocalVariableType.VOICE_RETURN_CONTENT:
                                voice_name = self.local_states["variables"][idx]
                                candidate_voice_name.append((voice_name, voice_name.updated))
                        if len(candidate_voice_name) == 0:
                            voice_name = None
                        else:
                            updated_candidate_voice_name = [voice_name for voice_name, updated in candidate_voice_name if updated]
                            if len(updated_candidate_voice_name) > 0:
                                voice_name = random.choice(updated_candidate_voice_name)
                            else:
                                voice_name = random.choice(candidate_voice_name)[0]
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
                        for transition_info in lvar.transitions:
                            if transition_info["name"] == transition.__name__:
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
                        candidate_voice_name = []
                        for idx in range(len(self.local_states["variables"])-1, -1, -1):
                            variable_type = self.local_states["variables"][idx].variable_type
                            if variable_type == VoiceLocalVariableType.VOICE_NAME or variable_type == VoiceLocalVariableType.VOICE_RETURN_CONTENT:
                                voice_name = self.local_states["variables"][idx]
                                candidate_voice_name.append((voice_name, voice_name.updated))
                        if len(candidate_voice_name) == 0:
                            voice_name = random.choice(voice_lab.voice_library.obtain_all_names())
                        else:
                            updated_candidate_voice_name = [voice_name for voice_name, updated in candidate_voice_name if updated]
                            if len(updated_candidate_voice_name) > 0:
                                voice_name = random.choice(updated_candidate_voice_name)
                            else:
                                voice_name = random.choice(candidate_voice_name)[0]
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
                    candidate_voice_name = []
                    if lvar.variable_type == VoiceLocalVariableType.STATE:
                        for idx in range(len(self.local_states["variables"])-1, -1, -1):
                            variable_type = self.local_states["variables"][idx].variable_type
                            if variable_type == VoiceLocalVariableType.VOICE_NAME or variable_type == VoiceLocalVariableType.VOICE_RETURN_CONTENT:
                                voice_name = self.local_states["variables"][idx]
                                candidate_voice_name.append((voice_name, voice_name.updated))
                        if len(candidate_voice_name) == 0:
                            voice_name = random.choice(voice_lab.voice_library.obtain_all_names())
                        else:
                            updated_candidate_voice_name = [voice_name for voice_name, updated in candidate_voice_name if updated]
                            if len(updated_candidate_voice_name) > 0:
                                voice_name = random.choice(updated_candidate_voice_name)
                            else:
                                voice_name = random.choice(candidate_voice_name)[0]
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
            if "text_modification" in transition_info["required_parameters"]:
                if transition_info["required_parameters"]["text_modification"] == "split":
                    local_state = transition_info["required_parameters"]["text"]
                    local_state.value = local_state.value.split(".")[0] + "."
        elif transition == "SpeechToSpeechTransition":
            if isinstance(transition_info["required_parameters"]["voice_name"], str):
                new_name = self.get_new_local_constant_name()
                pass_parameters = SpeechToSpeechTransition.process_parameters(transition_info["required_parameters"])
                new_local_variable = LocalVariable(name=new_name,
                                                   value={"voice_name": transition_info["required_parameters"]["voice_name"]},
                                                   variable_type=VoiceLocalVariableType.VOICE_NAME,
                                                   latest_call=calling_timestamp,
                                                   updated=False, # This variable will soon be used
                                                   created_by=USER_FUNCTION_PARAM_FLAG,
                                                   transitions=[{"name": transition, 
                                                                 "parameters": pass_parameters}]
                                                   )
                local_constant_value = new_local_variable.value
                if isinstance(local_constant_value, str):
                    pass
                elif isinstance(local_constant_value, dict):
                    if len(local_constant_value) == 1:
                        local_constant_value = list(local_constant_value.values())[0]
                    else:
                        raise Exception(f"Voice name must be a string or a dict with one key, but got {type(local_constant_value)}")
                self.add_local_constant(local_constant_value)
                self.add_local_variable(new_local_variable)
                transition_info["required_parameters"]["voice_name"] = new_local_variable
        elif transition == "MakeOutboundCallTransition":
            if isinstance(transition_info["required_parameters"]["voice_name"], str):
                new_name = self.get_new_local_constant_name()
                pass_parameters = MakeOutboundCallTransition.process_parameters(transition_info["required_parameters"])
                new_local_variable = LocalVariable(name=new_name,
                                                   value={"voice_name": transition_info["required_parameters"]["voice_name"]},
                                                   variable_type=VoiceLocalVariableType.VOICE_NAME,
                                                   latest_call=calling_timestamp,
                                                   updated=False,
                                                   created_by=USER_FUNCTION_PARAM_FLAG,
                                                   transitions=[{"name": transition, 
                                                                 "parameters": pass_parameters}]
                                                   )
                local_constant_value = new_local_variable.value
                if isinstance(local_constant_value, str):
                    pass
                elif isinstance(local_constant_value, dict):
                    if len(local_constant_value) == 1:
                        local_constant_value = list(local_constant_value.values())[0]
                    else:
                        raise Exception(f"Voice name must be a string or a dict with one key, but got {type(local_constant_value)}")
                self.add_local_constant(local_constant_value)
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
    def search_voice_library(self, search_category: str, search_gender: str, search_name: str=None):
        return voice_lab.search_voice_library(search_category=search_category, search_gender=search_gender, search_name=search_name)
    
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
        return ["search_category", "search_gender"]
    
    def get_effected_states(self, variable_schema: VoiceVariableSchema) -> List[str]:
        """
        Search is a READ operation. No implicit states are affected.
        """
        implicit_states = []
        local_states = []
        search_category = self.parameters["search_category"] if "search_category" in self.parameters else None
        search_gender = self.parameters["search_gender"] if "search_gender" in self.parameters else None
        variable_list = []
        
        if search_category is not None and search_gender is not None:
            # 1. The input are both str.
            # 2. The input is dict local variable.
            if isinstance(search_category, str) and isinstance(search_gender, str):
                self.string_parameters["search_category"] = f'"{search_category}"'
                self.string_parameters["search_gender"] = f'"{search_gender}"'
            elif isinstance(search_category, LocalVariable) and isinstance(search_gender, LocalVariable):
                self.string_parameters["search_category"] = f"{search_category.name}['search_category']"
                self.string_parameters["search_gender"] = f"{search_gender.name}['search_gender']"
                variable_list.append(search_category.name)
                variable_list.append(search_gender.name)
            else:
                raise Exception(f"search_category and search_gender must be a LocalVariable or a string, but got {type(search_category)} and {type(search_gender)}")
        elif search_category is not None:
            if isinstance(search_category, str):
                self.string_parameters["search_category"] = f'"{search_category}"'
            elif isinstance(search_category, LocalVariable):
                self.string_parameters["search_category"] = search_category.name
                variable_list.append(search_category.name)
            else:
                raise Exception(f"search_category must be a LocalVariable or a string, but got {type(search_category)}")
        elif search_gender is not None:
            if isinstance(search_gender, str):
                self.string_parameters["search_gender"] = f'"{search_gender}"'
            elif isinstance(search_gender, LocalVariable):
                self.string_parameters["search_gender"] = search_gender.name
                variable_list.append(search_gender.name)
            else:
                raise Exception(f"search_gender must be a LocalVariable or a string, but got {type(search_gender)}")
        
        for name in variable_list:
            for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
                if variable_schema.local_states["variables"][local_idx].name == name:
                    local_states.append(local_idx)
                    break
        
        return implicit_states, local_states
    
    @staticmethod
    def process_parameters(parameters: Dict[str, Any]):
        pass_parameters = {}
        both_flag = False
        if parameters["search_category"] is not None and parameters["search_gender"] is not None:
            if isinstance(parameters["search_category"], LocalVariable) and isinstance(parameters["search_gender"], LocalVariable):
                both_flag = True
        if both_flag:
            pass_parameters["search_category"] = parameters["search_category"].value["search_category"]
            pass_parameters["search_gender"] = parameters["search_gender"].value["search_gender"]
        else:
            for key, value in parameters.items():
                if isinstance(value, LocalVariable):
                    if isinstance(value.value, dict):
                        dict_value = list(value.value.values())
                        assert len(dict_value) == 1, "search_category and search_gender must be a dict with one key"
                        pass_parameters[key] = dict_value[0]
                    else:
                        pass_parameters[key] = value.value
                else:
                    pass_parameters[key] = value
        return pass_parameters
                
    def apply(self, implicit_states, local_states, variable_schema: VoiceVariableSchema):
        longest_transitions = []
        for l_state in local_states:
            variable_schema.local_states["variables"][l_state].latest_call = self.calling_timestamp
            variable_schema.local_states["variables"][l_state].updated = False
            if len(variable_schema.local_states["variables"][l_state].transitions) > len(longest_transitions):
                longest_transitions = copy.deepcopy(variable_schema.local_states["variables"][l_state].transitions)
        pass_parameters = SearchVoiceTransition.process_parameters(self.parameters)
        '''
        if len(self.string_parameters) == 2:
            pass_parameters["search_category"] = self.parameters["search_category"].value["search_category"]
            pass_parameters["search_gender"] = self.parameters["search_gender"].value["search_gender"]
        else:
            for key, value in self.parameters.items():
                if isinstance(value, LocalVariable):
                    if isinstance(value.value, dict):
                        dict_value = list(value.value.values())
                        assert len(dict_value) == 1, "search_category and search_gender must be a dict with one key"
                        pass_parameters[key] = dict_value[0]
                    else:
                        pass_parameters[key] = value.value
                else:
                    pass_parameters[key] = value
        '''
        result = variable_schema.search_voice_library(**pass_parameters)
        for idx, item in enumerate(result):
            new_local_variable = LocalVariable(name = f"{self.new_variable_name}[{idx}]",
                                                value = item,
                                                latest_call = self.calling_timestamp,
                                                updated = True,
                                                created_by = f"{self.name}@{self.calling_timestamp}",
                                                variable_type = VoiceLocalVariableType.VOICE_RETURN_CONTENT)
            new_local_variable.is_indexed = True
            new_local_variable.transitions = copy.deepcopy(longest_transitions)
            new_local_variable.transitions.append({"name": self.__class__.__name__, 
                                                   "parameters": {
                                                    "search_category": pass_parameters["search_category"],
                                                    "search_gender": pass_parameters["search_gender"]}})
            variable_schema.add_local_variable(new_local_variable)
        
    def __str__(self):
        return f"SearchVoiceTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name})"

    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        '''
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
        '''
        for key, value in self.string_parameters.items():
            parameter_str += f"{key}={value}, "
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
        self.search_index = False
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
                if isinstance(self.parameters["voice_name"], LocalVariable):
                    if variable_schema.local_states["variables"][local_idx].name == self.parameters["voice_name"].name:
                        local_states.append(local_idx)
        
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
                if value.variable_type == VoiceLocalVariableType.VOICE_RETURN_CONTENT:
                    temp_name = value.name
                    assert key == "voice_name", "voice_name must be a LocalVariable"
                    assert value.is_indexed, "voice_name must be indexed"
                    pass_parameters[key] = value.value['voice_name']
                    if "[0]" in temp_name:
                        self.string_parameters[key] = f"{temp_name}['voice_name']"
                    else:
                        self.search_index = True
                        self.string_parameters[key] = LOOP_VARIABLE
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
        new_voice_state.transitions.append({"name": self.__class__.__name__, 
                                            "parameters": {'text': self.string_parameters["text"], 
                                                               "voice_name": self.string_parameters["voice_name"] if "voice_name" in self.string_parameters else None,
                                                               "stability": self.parameters["stability"],
                                                               "similarity_boost": self.parameters["similarity_boost"],
                                                               "style": self.parameters["style"]}})
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
        if self.search_index:
            loop_body = self.parameters["voice_name"].name.split("[")[0]
            loop_pointer = self.string_parameters["voice_name"]
            voice_name_category = self.parameters["voice_name"].value["category"]
            voice_name_gender = self.parameters["voice_name"].value["gender"]
            result.append(f"for i in {loop_body}:\n")
            result.append(f"{INDENT}if i[\"category\"] == \"{voice_name_category}\" and i[\"gender\"] == \"{voice_name_gender}\":\n")
            result.append(f"{INDENT}{INDENT}{loop_pointer} = i[\"voice_name\"]\n")
            result.append(f"{INDENT}{INDENT}break\n")
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
        self.search_index = False
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
            elif self.parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_RETURN_CONTENT:
                temp_name = self.parameters['voice_name'].name
                if "[0]" in temp_name:
                    voice_name = f"{temp_name}['voice_name']" # Return of search_voice_library
                    self.string_parameters["voice_name"] = voice_name
                else:
                    self.search_index = True
                    self.string_parameters["voice_name"] = LOOP_VARIABLE
        else:
            self.string_parameters["voice_name"] = f'"{self.parameters["voice_name"]}"'
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == input_speech_name:
                local_states.append(local_idx)
                break
            if voice_name is not None:
                if variable_schema.local_states["variables"][local_idx].name == self.parameters["voice_name"].name:
                    local_states.append(local_idx)
                    break
        
        implicit_states.append(input_speech_name)
        return implicit_states, local_states
    
    @staticmethod
    def process_parameters(parameters: Dict[str, Any]):
        if isinstance(parameters["voice_name"], str):
            voice_name = parameters["voice_name"]
        else:
            if parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_NAME:
                if isinstance(parameters["voice_name"].value, dict):
                    dict_value = list(parameters["voice_name"].value.values())
                    assert len(dict_value) == 1, "voice_name must be a dict with one key"
                    voice_name = dict_value[0]
                else:
                    voice_name = parameters["voice_name"].value
            elif parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_RETURN_CONTENT:
                voice_name = parameters["voice_name"].value["voice_name"] # Return of search_voice_library
        pass_parameters = {"input_speech": parameters["input_speech"].value.current_value["voice_value"], "voice_name": voice_name}
        return pass_parameters
    
    def apply(self, implicit_states, local_states, variable_schema: VoiceVariableSchema):
        for l_state in local_states:
            variable_schema.local_states["variables"][l_state].latest_call = self.calling_timestamp
            variable_schema.local_states["variables"][l_state].updated = False
        for i_state in implicit_states:
            variable_schema.implicit_states["latest_call"][i_state] = self.calling_timestamp
            #variable_schema.implicit_states["voice_info"][i_state].updated = False

        pass_parameters = SpeechToSpeechTransition.process_parameters(self.parameters)
        result = variable_schema.speech_to_speech(**pass_parameters)
        
        
        
        new_voice_state = VoiceState(name=self.new_variable_name, 
                                     voice_value=result,
                                     outbound=False
                                     )
        new_voice_state.transitions = copy.deepcopy(self.parameters["input_speech"].value.transitions)
        new_voice_state.transitions.append({"name": self.__class__.__name__, 
                                            "parameters": pass_parameters})
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
        return f"SpeechToSpeechTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name}, string_parameters={self.string_parameters})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        for key, value in self.string_parameters.items():
            parameter_str += f"{key}={value}, "
        parameter_str = parameter_str.rstrip(", ")
        if self.search_index:
            loop_body = self.parameters["voice_name"].name.split("[")[0]
            loop_pointer = self.string_parameters["voice_name"]
            voice_name_category = self.parameters["voice_name"].value["category"]
            voice_name_gender = self.parameters["voice_name"].value["gender"]
            result = [
                f"for i in {loop_body}:\n",
                f"{INDENT}if i[\"category\"] == \"{voice_name_category}\" and i[\"gender\"] == \"{voice_name_gender}\":\n",
                f"{INDENT}{INDENT}{loop_pointer} = i[\"voice_name\"]\n",
                f"{INDENT}{INDENT}break\n",
            ]
        else:
            result = []
        result.append(f"{self.new_variable_name} = speech_to_speech({parameter_str})\n")
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
            if variable_schema.local_states["variables"][local_idx].name == self.parameters["input_speech"].name:
                local_states.append(local_idx)
                break
        implicit_states.append(input_speech_name)
        return implicit_states, local_states
    
    @staticmethod
    def process_parameters(parameters: Dict[str, Any]):
        pass_parameters = {"input_speech": parameters["input_speech"].value.current_value["voice_value"], 
                           "diarize": parameters["diarize"]}
        return pass_parameters
    
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
        
        pass_parameters = SpeechToTextTransition.process_parameters(self.parameters)
        result = variable_schema.speech_to_text(**pass_parameters)
        new_local_variable = LocalVariable(name=self.new_variable_name,
                                          value=result,
                                          latest_call=self.calling_timestamp,
                                          updated=True,
                                          created_by=f"{self.name}@{self.calling_timestamp}",
                                          variable_type=VoiceLocalVariableType.VOICE_TRANSCRIPTION)
        new_local_variable.transitions = copy.deepcopy(implicit_state.transitions)
        new_local_variable.transitions.append({"name": self.__class__.__name__, 
                                               "parameters": pass_parameters})
        variable_schema.add_local_variable(new_local_variable)
    
    def __str__(self):
        return f"SpeechToTextTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        for key, value in self.string_parameters.items():
            parameter_str += f"{key}={value}, "
        parameter_str = parameter_str.rstrip(", ")
        result = [
            f"{self.new_variable_name} = speech_to_text({parameter_str})\n",
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
        new_implicit_variable.transitions.append({"name": self.__class__.__name__, 
                                                 "parameters": {'input_speech': self.parameters["input_speech"].name}})
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
            f"{self.new_variable_name} = isolate_audio({parameter_str})\n",
        ] 
        return result, ""

class MakeOutboundCallTransition(Transition):
    def __init__(self, parameters: Dict[str, Any], calling_timestamp: int):
        super().__init__("MakeOutboundCallTransition", parameters=parameters, func=None)
        self.calling_timestamp = calling_timestamp
        self.new_variable_name = RESPONSE_VARIABLE_TEMP.format(self.calling_timestamp)
        self.string_parameters = {}
        self.search_index = False
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
            elif self.parameters["voice_name"].variable_type == VoiceLocalVariableType.VOICE_RETURN_CONTENT:
                temp_name = self.parameters['voice_name'].name
                if "[0]" in temp_name:
                    voice_name = f"{temp_name}['voice_name']" # Return of search_voice_library
                else:
                    self.search_index = True
                    self.string_parameters["voice_name"] = LOOP_VARIABLE
                    voice_name = LOOP_VARIABLE
        self.string_parameters["voice_name"] = voice_name
        
        if isinstance(self.parameters["voice_name"], LocalVariable):
            local_states = [None, None]
        else:
            local_states = [None]
        for local_idx in range(len(variable_schema.local_states["variables"])-1, -1, -1):
            if variable_schema.local_states["variables"][local_idx].name == input_speech_name and variable_schema.local_states["variables"][local_idx].variable_type == VoiceLocalVariableType.STATE:
                local_states[0] = local_idx
            if isinstance(self.parameters["voice_name"], LocalVariable) and variable_schema.local_states["variables"][local_idx].name == self.parameters["voice_name"].name:
                local_states[1] = local_idx
        implicit_states.append(input_speech_name)
        return implicit_states, local_states
    
    @staticmethod
    def process_parameters(parameters: Dict[str, Any]):
        voice_name = parameters["voice_name"]
        if isinstance(voice_name, str):
            pass
        elif isinstance(voice_name, LocalVariable) and voice_name.variable_type == VoiceLocalVariableType.VOICE_NAME:
            if isinstance(voice_name.value, dict):
                dict_value = list(voice_name.value.values())
                assert len(dict_value) == 1, "voice_name must be a dict with one key"
                voice_name = dict_value[0]
            else:
                voice_name = voice_name.value
        elif isinstance(voice_name, LocalVariable) and voice_name.variable_type == VoiceLocalVariableType.VOICE_RETURN_CONTENT:
            voice_name = voice_name.value["voice_name"]
        else:
            raise ValueError(f"Invalid voice_name type: {type(voice_name)}")
        pass_parameters = {"input_speech": parameters["input_speech"].value.current_value["voice_value"], "voice_name": voice_name}
        assert isinstance(pass_parameters["input_speech"], Speech), "input_speech must be a Speech, but got {type(pass_parameters['input_speech'])}"
        assert isinstance(pass_parameters["voice_name"], str), "voice_name must be a str, but got {type(pass_parameters['voice_name'])}"
        return pass_parameters
    
    def apply(self, implicit_states, local_states, variable_schema: VoiceVariableSchema):
        assert len(implicit_states) == 1, "There should be only one implicit state for make_outbound_call"
        #assert len(local_states) == 1, "There should be only one local state for make_outbound_call"
        implicit_state = variable_schema.implicit_states["voice_info"][implicit_states[0]]
        variable_schema.implicit_states["latest_call"][implicit_state.identifier] = self.calling_timestamp
        implicit_state.current_value["outbound"] = True

        pass_parameters = MakeOutboundCallTransition.process_parameters(self.parameters)
        implicit_state.transitions.append({"name": self.__class__.__name__, 
                                           "parameters": pass_parameters}
                                          )

        result = variable_schema.make_outbound_call(**pass_parameters)
        assert result, "make_outbound_call failed"
        #assert variable_schema.local_states["variables"][local_states[0]].variable_type == VoiceLocalVariableType.STATE, "local_state should be a VoiceState"
        variable_schema.local_states["variables"][local_states[0]].latest_call = self.calling_timestamp
        variable_schema.local_states["variables"][local_states[0]].updated = False
        variable_schema.local_states["variables"][local_states[0]].value = copy.deepcopy(implicit_state)
        variable_schema.local_states["variables"][local_states[0]].transitions = copy.deepcopy(implicit_state.transitions)
        
        if len(local_states) == 2:
            variable_schema.local_states["variables"][local_states[1]].latest_call = self.calling_timestamp
            variable_schema.local_states["variables"][local_states[1]].updated = False
            variable_schema.local_states["variables"][local_states[1]].transitions.append({"name": self.__class__.__name__, 
                                                                                           "parameters": pass_parameters}
                                                                                          )
    
    def __str__(self):
        return f"MakeOutboundCallTransition(parameters={self.parameters}, calling_timestamp={self.calling_timestamp}, new_variable_name={self.new_variable_name})"
    
    def get_program_str(self) -> Tuple[List[str], str]:
        parameter_str = ""
        for key, value in self.string_parameters.items():
            parameter_str += f"{key}={value}, "
        parameter_str = parameter_str.rstrip(", ")
        if self.search_index:
            loop_body = self.parameters["voice_name"].name.split("[")[0]
            voice_name_category = self.parameters["voice_name"].value["category"]
            voice_name_gender = self.parameters["voice_name"].value["gender"]
            loop_pointer = self.string_parameters["voice_name"]
            result = [
                f"for i in {loop_body}:\n",
                f"{INDENT}if i[\"category\"] == \"{voice_name_category}\" and i[\"gender\"] == \"{voice_name_gender}\":\n",
                f"{INDENT}{INDENT}{loop_pointer} = i[\"voice_name\"]\n",
                f"{INDENT}{INDENT}break\n",
            ]
        else:
            result = []
        result.append(f"{self.new_variable_name} = make_outbound_call({parameter_str})\n")
        return result, ""


class VoiceEvaluator(ProgramEvaluator):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_cases = []
        self.preload_info = {
            "speech_to_speech": speech_to_speech,
            "text_to_speech": text_to_speech,
            "speech_to_text": speech_to_text,
            "isolate_audio": isolate_audio,
            "make_outbound_call": make_outbound_call,
            "Speech": Speech,
            "search_voice_library": search_voice_library
        }
    def store(self, file_path: str):
        """Save test cases to disk. Handles Speech objects by converting them to JSON."""
        
        def speech_to_json(obj):
            """Recursively convert Speech objects to JSON format."""
            if isinstance(obj, Speech):
                return obj.to_json()
            elif isinstance(obj, dict):
                return {k: speech_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [speech_to_json(item) for item in obj]
            return obj

        # Convert test cases to serializable format
        serializable_test_cases = []
        for test_case in self.test_cases:
            serializable_case = {
                "result": speech_to_json(test_case["result"]),
                "state_oracle": test_case["state_oracle"],
                "program_info": {
                    "init_local_str": test_case["program_info"]["init_local_str"],
                    "init_local_info": test_case["program_info"]["init_local_info"],
                    "init_implicit_dict": None,
                    "end_implict_list": None,
                    "init_load_str": test_case["program_info"]["init_load_str"],
                    "init_load_info": speech_to_json(test_case["program_info"]["init_load_info"])
                },
                "program": test_case["program"]
            }
            serializable_test_cases.append(serializable_case)

        # Save to file
        saved_info = {
            "test_cases": serializable_test_cases,
            "config": self.config
        }
        with open(file_path, 'w') as f:
            json.dump(saved_info, f, indent=2)
    
    @classmethod
    def load(cls, file_path: str, config: Dict[str, Any] = None):
        """Load test cases from disk. Converts JSON back to Speech objects."""
        
        def json_to_speech(obj):
            """Recursively convert JSON format back to Speech objects."""
            if isinstance(obj, dict):
                if "is_speech" in obj and obj["is_speech"]:
                    return Speech.from_json(obj)
                else:
                    return {k: json_to_speech(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [json_to_speech(item) for item in obj]
            return obj

        # Load from file
        with open(file_path, 'r') as f:
            saved_info = json.load(f)
        
        if config is None:
            config = saved_info["config"]
        
        created_cls = cls(config)
        test_cases = saved_info["test_cases"]
        
        # Convert test cases back to proper format
        for test_case in test_cases:
            # Handle init_load_info conversion
            init_load_info = test_case["program_info"]["init_load_info"]
            if init_load_info is not None:
                converted_init_load_info = []
                for (name, value) in init_load_info:
                    converted_init_load_info.append((name, json_to_speech(value)))
                init_load_info = converted_init_load_info
            
            converted_case = {
                "result": json_to_speech(test_case["result"]),
                "state_oracle": test_case["state_oracle"],
                "program_info": {
                    "init_local_str": test_case["program_info"]["init_local_str"],
                    "init_local_info": test_case["program_info"]["init_local_info"],
                    "init_implicit_dict": test_case["program_info"]["init_implicit_dict"],
                    "end_implict_list": test_case["program_info"]["end_implict_list"],
                    "init_load_str": test_case["program_info"]["init_load_str"],
                    "init_load_info": init_load_info
                },
                "program": test_case["program"]
            }
            created_cls.test_cases.append(converted_case)

        return created_cls
    
    def prepare_environment(self, init_implicit_dict, init_local_info, init_load_info=None):
        # 
        voice_lab.voice_library.reset()
        voice_lab.call_recorder.reset()
        
    
    def collect_test_case(self, program_info, program):
        self.prepare_environment(program_info['init_implicit_dict'], program_info['init_local_info'], program_info["init_load_info"])
        init_load_info = program_info["init_load_info"]
        complete_program = program_info["init_local_str"] + program
        namespace = {}
        if init_load_info is not None:
            for (name, value) in init_load_info:
                namespace[name] = copy.deepcopy(value)
                
        for (name, value) in self.preload_info.items():
            namespace[name] = value
            
        try:
            exec(complete_program, namespace)
        except Exception as e:
            error_info = traceback.format_exc()
            logger.warning(f"Error in executing the program: {error_info}")
            return None
        
        if f"{RESULT_NAME}" in namespace:
            result = namespace[RESULT_NAME]
        else:
            result = None
        
        if isinstance(result, Speech):
            result = result.to_json()
        
        if isinstance(result, Tuple):
            result = list(result)
            for idx in range(len(result)):
                if isinstance(result[idx], Speech):
                    result[idx] = result[idx].to_json()
        
        state_oracle = voice_lab.call_recorder.get_outbound_call_info()
        test_case = {
            "result": result,
            "state_oracle": state_oracle,
            "program_info": program_info,
            "program": program
        }
        self.test_cases.append(test_case)
        return test_case
            
    def evaluate(self, program: str):
        pass_list = []
        test_case_pass_detail = []
        for test_idx, test_case in enumerate(self.test_cases):
            self.prepare_environment(test_case["program_info"]["init_implicit_dict"], test_case["program_info"]["init_local_info"])
            namespace = {}
            complete_program = test_case["program_info"]["init_local_str"] + program
            init_load_info = test_case["program_info"]["init_load_info"]
            if init_load_info is not None:
                for (name, value) in init_load_info:
                    namespace[name] = copy.deepcopy(value)
                    
            for (name, value) in self.preload_info.items():
                namespace[name] = value
            
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
            state_info = voice_lab.call_recorder.get_outbound_call_info()
            # ====== Evaluate variable oracle ======
            if f"{RESULT_NAME}" in namespace:
                result = namespace[RESULT_NAME]
                gt_result = test_case["result"]
                if gt_result is None:
                    if result is not None:
                        pass_list.append(False)
                        test_case_pass_detail.append({
                            "result_pass": False,
                            "error_info": None,
                            "state_pass": None,
                            "state_pass_detail": "There should be no return result"
                        })
                        continue
                if isinstance(test_case["result"], list):
                    gt_oracle_list = [0] * len(test_case["result"])
                    if len(test_case["result"]) != len(result):
                        pass_list.append(False)
                        test_case_pass_detail.append({
                            "result_pass": False,
                            "error_info": None,
                            "state_pass": None,
                            "state_pass_detail": "Return result length mismatch"
                        })
                        continue
                    else:
                        # The order of result and gt_result is not guaranteed.
                        fail = False
                        has_corresponding = False
                        for item in result:
                            left = item
                            left_is_speech = False
                            if isinstance(item, dict):
                                if "is_speech" in item and item["is_speech"]:
                                    left = Speech.from_json(item)
                                    left_is_speech = True
                            if isinstance(left, Speech):
                                left_is_speech = True
                            for gt_idx, gt_item in enumerate(gt_result):
                                if gt_oracle_list[gt_idx] == 1:
                                    continue
                                right = gt_item
                                if isinstance(right, dict) and "is_speech" in right and right["is_speech"]:
                                    if not left_is_speech:
                                        continue
                                    right = Speech.from_json(right)
                                if left == right:
                                    has_corresponding = True
                                    gt_oracle_list[gt_idx] = 1
                                    break
                            if not has_corresponding:
                                pass_list.append(False)
                                test_case_pass_detail.append({
                                    "result_pass": False,
                                    "error_info": None,
                                    "state_pass": None,
                                    "state_pass_detail": ["Return result mismatch: No corresponding result", gt_result, result]
                                })
                                fail = True
                                break
                    if fail:
                        continue
                    if not all(x == 1 for x in gt_oracle_list):
                        pass_list.append(False)
                        test_case_pass_detail.append({
                            "result_pass": False,
                            "error_info": None,
                            "state_pass": None,
                            "state_pass_detail": ["Return result mismatch", gt_result, result]
                        })
                        continue
                else:
                    # Result is not List
                    right = test_case["result"]
                    if isinstance(right, dict) and "is_speech" in right and right["is_speech"]:
                        right = Speech.from_json(right)
                    if result != right:
                        pass_list.append(False)
                        test_case_pass_detail.append({
                            "result_pass": False,
                            "error_info": None,
                            "state_pass": None,
                            "state_pass_detail": ["Return result mismatch", gt_result, result]
                        })
                        continue
            else:
                if test_case["result"] is not None:
                    pass_list.append(False)
                    test_case_pass_detail.append({
                        "result_pass": False,
                        "error_info": None,
                        "state_pass": None,
                        "state_pass_detail": "Return result not found"
                    })
                    continue
            # ====== Evaluate state oracle ======
            if state_info is not None:
                if test_case["state_oracle"] is None:
                    test_case_pass_detail.append({
                        "result_pass": True,
                        "error_info": None,
                        "state_pass": False,
                        "state_pass_detail": "No call should be made"
                    })
                if len(state_info) != len(test_case["state_oracle"]):
                    test_case_pass_detail.append({
                        "result_pass": True,
                        "error_info": None,
                        "state_pass": False,
                        "state_pass_detail": "Call length mismatch"
                    })
                    pass_list.append(False)
                    continue
                state_oracle_list = [0] * len(test_case["state_oracle"])
                fail = False
                for state_item in state_info:
                    has_corresponding = False
                    for state_oracle_idx, state_oracle_item in enumerate(test_case["state_oracle"]):
                        if state_oracle_list[state_oracle_idx] == 1:
                            # Already matched
                            continue
                        if voice_lab.call_recorder.compare_outbound_info(state_item, state_oracle_item):
                            state_oracle_list[state_oracle_idx] = 1
                            has_corresponding = True
                            break
                    if not has_corresponding:
                        test_case_pass_detail.append({
                            "result_pass": True,
                            "error_info": None,
                            "state_pass": False,
                            "state_pass_detail": "Call mismatch: more calls than expected"
                        })
                        pass_list.append(False)
                        fail = True
                        break
                if fail:
                    continue
                if not all(x == 1 for x in state_oracle_list):
                    test_case_pass_detail.append({
                        "result_pass": True,
                        "error_info": None,
                        "state_pass": False,
                        "state_pass_detail": "Call mismatch: less calls than expected"
                    })
                    pass_list.append(False)
                    continue
            else:
                if test_case["state_oracle"] is not None:
                    test_case_pass_detail.append({
                        "result_pass": True,
                        "error_info": None,
                        "state_pass": False,
                        "state_pass_detail": "Call missing"
                    })
                    pass_list.append(False)
                    continue
                
                
            test_case_pass_detail.append({
                "result_pass": True,
                "error_info": None,
                "state_pass": True,
                "state_pass_detail": None
            })       
            pass_list.append(True)
        return pass_list, test_case_pass_detail
        

def test_search_voice_library():
    schema = VoiceVariableSchema()
    new_local_variable = LocalVariable(
        name="user_variable_1",
        value="conversational",
        latest_call=0,
        updated=True,
        created_by="user",
        variable_type=VoiceLocalVariableType.VOICE_SEARCH_CONTENT
    )
    transition = SearchVoiceTransition({"search_category": new_local_variable, "search_gender": None}, 1)
    implicit_states, local_states = transition.get_effected_states(schema)
    transition.apply(implicit_states, local_states, schema)
    assert len(schema.local_states['variables']) == 1

    transition = SearchVoiceTransition({"search_category": "conversational", "search_gender": None}, 2)
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