## Mock Testing Backend Library

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random
import os
import copy

@dataclass
class Speech:
    """Mock speech object containing audio data"""
    text: str
    voice_name: str = ""
    stability: float = 0
    similarity_boost: float = 0
    style: float = 0
    duration: float = 16.0
    transitions: List[Tuple[str, dict]] = field(default_factory=list)

    def __eq__(self, other: "Speech") -> bool:
        """Compare two Speech objects for equality"""
        if not isinstance(other, Speech):
            return False
        return (
            self.text == other.text
            and self.voice_name == other.voice_name
            and abs(self.stability - other.stability) < 1e-6
            and abs(self.similarity_boost - other.similarity_boost) < 1e-6
            and abs(self.style - other.style) < 1e-6
            and abs(self.duration - other.duration) < 1e-6
            and len(self.transitions) == len(other.transitions)
            and all(
                self.transitions[i][0] == other.transitions[i][0] 
                and self.transitions[i][1] == other.transitions[i][1]
                for i in range(len(self.transitions))
            )
        )

class OutboundCallRecorder:
    def __init__(self):
        self.outbound_call_info = []
    
    def add_outbound_call(self, speech: Speech, voice_name: str):
        self.outbound_call_info.append([speech, voice_name])
    
    def compare(self, other: "OutboundCallRecorder") -> bool:
        if len(self.outbound_call_info) != len(other.outbound_call_info):
            return False
        checked = set([])
        for i in range(len(self.outbound_call_info)):
            if i in checked:
                continue
            for j in range(len(other.outbound_call_info)):
                if self.outbound_call_info[i] == other.outbound_call_info[j]:
                    checked.add(j)
                    break
        return len(checked) == len(self.outbound_call_info)

    def reset(self):
        self.__init__()
    

class VoiceLibrary:
    """Voice library containing predefined voices with enhanced metadata"""
    def __init__(self):
        self.voices = {
            "Emma": {"id": "v1", "category": "conversational", "gender": "feminine"},
            "James": {"id": "v2", "category": "professional", "gender": "masculine"},
            "Sarah": {"id": "v3", "category": "casual", "gender": "feminine"},
            "Michael": {"id": "v4", "category": "professional", "gender": "masculine"},
            "Sofia": {"id": "v5", "category": "casual", "gender": "feminine"},
            "Alexander": {"id": "v6", "category": "professional", "gender": "masculine"},
            "Olivia": {"id": "v7", "category": "conversational", "gender": "feminine"},
            "William": {"id": "v8", "category": "professional", "gender": "masculine"},
            "Isabella": {"id": "v9", "category": "casual", "gender": "feminine"},
            "Daniel": {"id": "v10", "category": "professional", "gender": "masculine"},
            "Ava": {"id": "v11", "category": "casual", "gender": "feminine"},
            "Ethan": {"id": "v12", "category": "professional", "gender": "masculine"},
            "Charlotte": {"id": "v13", "category": "conversational", "gender": "feminine"},
            "Lucas": {"id": "v14", "category": "professional", "gender": "masculine"},
            "Mia": {"id": "v15", "category": "casual", "gender": "feminine"},
            "Henry": {"id": "v16", "category": "professional", "gender": "masculine"},
            "Sophia": {"id": "v17", "category": "conversational", "gender": "feminine"},
            "Benjamin": {"id": "v18", "category": "professional", "gender": "masculine"},
            "Amelia": {"id": "v19", "category": "casual", "gender": "feminine"},
            "Sebastian": {"id": "v20", "category": "professional", "gender": "masculine"}
        }
        self.current_id = max(int(voice_info["id"].replace("v", "")) for voice_info in self.voices.values()) + 1
    
    def add_voice(self, voice_name: str, category: str, gender: str):
        self.voices[voice_name] = {"id": str(self.current_id), "category": category, "gender": gender}
        self.current_id += 1
    
    def get_voice_by_name(self, voice_name: str) -> List[dict]:
        if voice_name not in self.voices:
            return None
        item = {key: self.voices[voice_name][key] for key in ["category", "gender"]}
        item["voice_name"] = voice_name
        return [item]
    
    def get_voice_by_info(self, category: str=None, gender: str=None) -> List[dict]:
        result = []
        for voice_name, voice_info in self.voices.items():
            if (category is None or voice_info["category"] == category) and (gender is None or voice_info["gender"] == gender):
                item = {key: voice_info[key] for key in ["category", "gender"]}
                item["voice_name"] = voice_name
                result.append(item)
        return result
    
    def reset(self):
        self.__init__()
    
    
voice_library = VoiceLibrary()
call_recorder = OutboundCallRecorder()

def search_voice_library(search_name: Optional[str] = None, 
                         search_category: Optional[str] = None, 
                         search_gender: Optional[str] = None
                         ) -> List[str]:
    """Mock implementation of voice library search"""
    assert not((search_name is None) and (search_category is None) and (search_gender is None)), "At least one of search_name, search_category, or search_gender must be provided"
    global voice_library
    if search_name is not None:
        return voice_library.get_voice_by_name(search_name)
    else:
        return voice_library.get_voice_by_info(category=search_category, gender=search_gender)
    

def text_to_speech(
    text: str,
    voice_name: Optional[str] = None,
    stability: float = 0.5,
    similarity_boost: float = 0.5,
    style: float = 0.0
) -> Speech:
    """Mock implementation of text to speech conversion"""
    # Generate mock audio data
    audio = Speech(text=text, voice_name=voice_name, stability=stability, similarity_boost=similarity_boost, style=style)
    audio.transitions.append(["text_to_speech", 
                              {"voice_name": voice_name, "stability": stability, 
                               "similarity_boost": similarity_boost, "style": style}])
    return audio

def speech_to_speech(input_speech: Speech, voice_name: str) -> Speech:
    """Mock implementation of speech to speech conversion"""
    # Return new speech object with modified audio data
    global voice_library
    assert voice_name in voice_library.voices, "Voice name not found in voice library"
    new_item = copy.deepcopy(input_speech)
    new_item.voice_name = voice_name
    new_item.transitions.append(["speech_to_speech", {"voice_name": voice_name}])
    return new_item

def speech_to_text(input_speech: Speech, diarize: bool = False) -> str:
    """Mock implementation of speech to text conversion"""
    # Generate mock transcription
    text = copy.deepcopy(input_speech.text)
    input_speech.transitions.append(["speech_to_text", {"diarize": diarize}])
    return text

def isolate_audio(input_speech: Speech) -> Speech:
    """Mock implementation of audio isolation"""
    # Return new speech object with "isolated" audio
    new_item = copy.deepcopy(input_speech)
    new_item.transitions.append(["isolate_audio", {}])
    new_item.duration = new_item.duration / 2
    return new_item

def make_outbound_call(input_speech: Speech, voice_name: Optional[str] = None) -> bool:
    """Mock implementation of making outbound calls"""
    # Always return success in mock implementation
    global call_recorder
    call_recorder.add_outbound_call(input_speech, voice_name)
    assert voice_name in voice_library.voices, "Voice name not found in voice library"
    return True


