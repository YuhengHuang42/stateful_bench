# Tools
search_voice_library: Search for a voice across the entire ElevenLabs voice library.

    Args:
        search_name (str, optional): Search term to filter voices by.
        search_category (str, optional): Category to filter voices by.
        search_gender (str, optional): Gender to filter voices by.

    Returns:
        List[Dict] for the voice matching the search term. Each dict contains the voice_name, category, and gender.
 
---
text_to_speech: Convert text to speech with a given voice and save the output audio file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.
    Only one of voice_id or voice_name can be provided. If none are provided, the default voice will be used.

    Args:
        text (str): The text to convert to speech.
        voice_name (str, optional): The name of the voice to use.
        stability (float, optional): Stability of the generated audio. Determines how stable the voice is and the randomness between each generation. Lower values introduce broader emotional range for the voice. Higher values can result in a monotonous voice with limited emotion. Range is 0 to 1.
        similarity_boost (float, optional): Similarity boost of the generated audio. Determines how closely the AI should adhere to the original voice when attempting to replicate it. Range is 0 to 1.
        style (float, optional): Style of the generated audio. Determines the style exaggeration of the voice. This setting attempts to amplify the style of the original speaker. It does consume additional computational resources and might increase latency if set to anything other than 0. Range is 0 to 1.
		
    Returns:
        A speech object containing the audio
---
speech_to_speech: Transform audio from one voice to another using provided audio files.

    Args:
        input_speech: A speech object containing the audio
        voice_name (str): The name of the voice to use.

    Returns:
        A speech object containing the audio
---      
speech_to_text: Transcribe speech from an audio.

    Args:
        input_speech: A speech object containing the audio
        diarize: Whether to diarize the audio file. If True, which speaker is currently speaking will be annotated in the transcription.
    Returns:
        A dict containing the transcription (key: "text") and diarization information (key: "diarize").
---
isolate_audio: Isolate audio from the given audio and return the output.

    Args:
        input_speech: A speech object containing the audio
    
    Returns:
        A speech object containing the isolated audio.
---
make_outbound_call: Make an outbound call via Twilio using an ElevenLabs agent.

    Args:
        input_speech: A speech object containing the audio
        voice_name (str, optional): The name of the voice to use.
    Returns:
        Boolean indicating whether the call was made successfully.







