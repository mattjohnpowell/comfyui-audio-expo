try:
    from comfy_api.latest import ComfyExtension, io
    from .src.nodes import SaveAudioWithTags, LyricsVideoGenerator

    class AudioExpoExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [
                SaveAudioWithTags,
                LyricsVideoGenerator
            ]

    async def comfy_entrypoint() -> ComfyExtension:
        return AudioExpoExtension()

except ImportError:
    # Fallback or error handling if V3 API is not present
    # For now we assume the V3 environment as requested
    print("ComfyUI V3 API not found or failed to import from src.nodes.")
    import traceback
    traceback.print_exc()
