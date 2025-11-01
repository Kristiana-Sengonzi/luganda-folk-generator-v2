from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from app.models_manager import models_manager
from app.vllm_manager import vllm_manager 

# Create translation pipeline
translator = models_manager.load_translator()
   

def translate_lyrics_to_luganda(lyrics: str) -> dict:
    """
    Translates English lyrics line by line into Luganda.
    Keeps section headers (e.g. [Chorus], [Verse 1]) in English.
    
    Returns:
        dict with 'english' and 'luganda' keys (both as joined strings)
    """
    lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
    
    english_lines = []
    luganda_lines = []

    for line in lines:
        # Keep section headers (e.g. [Verse 1]) un-translated
        if line.startswith('[') and line.endswith(']'):
            english_lines.append(line)
            luganda_lines.append(line)
        else:
            english_lines.append(line)
            translated = translator(line)[0]['translation_text']
            luganda_lines.append(translated)

    return {
        "english": "\n".join(english_lines),
        "luganda": "\n".join(luganda_lines)
    }