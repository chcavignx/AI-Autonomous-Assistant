import pyttsx3
import os

# Paths to the model and config files for French and English voices
DATA_DIR = "../../data/"
TEST_FILE_NAME = "test_espeak_voice.wav"

rate = 180
volume = 0.9
# defined voices to use
voice_id = 1
VoiceGenderMale = "VoiceGenderMale"
VoiceGenderFemale = "VoiceGenderFemale"


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


# Setup the TTS engine
engine = pyttsx3.init()
# setting up new voice rate
engine.setProperty("rate", rate)
# setting up volume level  between 0 and 1
engine.setProperty("volume", volume)
# getting details of current available voices
voices = engine.getProperty("voices")
# setting up voice by id
for voice in voices:
    # The way to find the voice may vary depending on the installation
    if (voice.languages == "en" and "GB") and (
        voice.gender == VoiceGenderMale
    ) in voice.id:
        voice_id = voice
        engine.setProperty("voice", voice.id)
        break
# Queue the text to be spoken
engine.say("Hello, How are you")
engine.say("My name is JARVIS.")
# Runs for small duration of time otherwise we may not be able to hear
engine.runAndWait()
# Save# the speech to a file
engine.save_to_file(
    "Hello, How are you. My name is JARVIS.",
    setup_output_filename(
        DATA_DIR,
        TEST_FILE_NAME,
        f"{voices[voice_id].name}_{voices[voice_id].languages}",
    ),
)

# Clean up and release resources
engine.stop()
print("Speech synthesis complete.")
print(
    f"Audio file saved to: {setup_output_filename(DATA_DIR, TEST_FILE_NAME, f'{voices[voice_id].name}_{voices[voice_id].languages}')}"
)
