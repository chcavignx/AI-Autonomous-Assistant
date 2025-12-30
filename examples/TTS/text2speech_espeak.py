#!/usr/bin/env python3
"""
Example of text to speech using eSpeak
"""

import os

import pyttsx3

from src.utils.config import config

# Paths to the model and config files for French and English voices
DATA_DIR = str(config.paths.data_path)
TEST_FILE_NAME = "test_espeak_voice.wav"

RATE = 180
VOLUME = 0.9
# defined voices to use
DEFAULT_VOICE_ID = 1
VOICE_GENDER_MALE = "VoiceGenderMale"
VOICE_GENDER_FEMALE = "VoiceGenderFemale"


# Service to create unique filenames from base name, suffix and counter if needed
def setup_output_filename(base_path, base_name, suffix, counter=1) -> str:
    """Generates a unique filename by appending a counter if needed."""
    counter = 1
    file_name, file_extension = os.path.splitext(base_name)
    output_file = os.path.join(base_path, f"{file_name}_{suffix}{file_extension}")

    while os.path.exists(output_file):
        output_file = os.path.join(
            base_path, f"{file_name}_{suffix}_{counter}{file_extension}"
        )
        counter += 1

    return output_file


def main() -> None:
    """Main function to demonstrate text-to-speech using eSpeak."""
    # Setup the TTS engine
    engine = pyttsx3.init()
    # setting up new voice rate
    engine.setProperty("rate", RATE)
    # setting up volume level between 0 and 1
    engine.setProperty("volume", VOLUME)
    # getting details of current available voices
    voices = engine.getProperty("voices")
    # setting up voice by id
    selected_voice_id = DEFAULT_VOICE_ID
    for voice in voices:
        # The way to find the voice may vary depending on the installation
        if (voice.languages == "en" and "GB") and (
            voice.gender == VOICE_GENDER_MALE
        ) in voice.id:
            engine.setProperty("voice", voice.id)
            # Find the index of the selected voice in the voices list
            for i, v in enumerate(voices):
                if v.id == voice.id:
                    selected_voice_id = i
                    break
            break
    # Queue the text to be spoken
    engine.say("Hello, How are you")
    engine.say("My name is JARVIS.")
    # Runs for small duration of time otherwise we may not be able to hear
    engine.runAndWait()
    # Save the speech to a file
    engine.save_to_file(
        "Hello, How are you. My name is JARVIS.",
        setup_output_filename(
            DATA_DIR,
            TEST_FILE_NAME,
            f"{voices[selected_voice_id].name}_{voices[selected_voice_id].languages}",
        ),
    )
    # Clean up and release resources
    engine.stop()
    print("Speech synthesis complete.")
    print(
        f"Audio file saved to: {setup_output_filename(DATA_DIR, TEST_FILE_NAME, f'{voices[selected_voice_id].name}_{voices[selected_voice_id].languages}')}"
    )


if __name__ == "__main__":
    main()
