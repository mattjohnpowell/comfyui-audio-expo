from .src.nodes import SaveAudioWithTags, LyricsVideoGenerator, OpenAudioInPlayer

# Traditional exports for ComfyUI and Registry compatibility
NODE_CLASS_MAPPINGS = {
    "SaveAudioWithTags": SaveAudioWithTags,
    "LyricsVideoGenerator": LyricsVideoGenerator,
    "OpenAudioInPlayer": OpenAudioInPlayer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAudioWithTags": "Save Audio (MP3 w/ Tags)",
    "LyricsVideoGenerator": "Lyrics Video Generator",
    "OpenAudioInPlayer": "Open Audio in Player",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
