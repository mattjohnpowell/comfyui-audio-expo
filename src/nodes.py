import os
import torch
import torchaudio
import numpy as np
import folder_paths
import wave
from PIL import Image, ImageDraw, ImageFont

# Try to import mutagen, handle if missing
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TYER, TCON, USLT, TBPM, ID3NoHeaderError
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


# Try to import moviepy, handle if missing
try:
    # MoviePy 2.0+ uses direct imports, 1.x uses .editor
    try:
        from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, VideoClip
    except ImportError:
        from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, VideoClip
        
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    from proglog import ProgressBarLogger
except ImportError:
    ProgressBarLogger = object

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

from comfy_api.latest import io, ui

class ComfyMoviePyLogger(ProgressBarLogger):
    def __init__(self, node_unique_id=None):
        super().__init__(init_state=None, bars=None, ignored_bars=None, logged_bars='all', min_time_interval=0, ignore_bars_under=0)
        self.node_unique_id = node_unique_id
        
    def bars_callback(self, bar, attr, value, old_value=None):
        if bar == 't' and self.node_unique_id and PromptServer:
            total = self.bars[bar]['total']
            if total > 0:
                PromptServer.instance.send_sync("progress", {"value": value, "max": total, "node": self.node_unique_id})

def save_wav_native(filepath, waveform, sample_rate):
    """
    Saves audio waveform to a WAV file using the standard 'wave' module.
    Avoids dependencies on external codecs or complex torchaudio backends.
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    # Ensure shape is [Channels, Samples]
    if waveform.ndim == 1:
        # [Samples] -> [1, Samples]
        waveform = waveform[np.newaxis, :]
        
    channels, samples = waveform.shape
    
    # If channels > samples, it might be transposed, but in ComfyUI AUDIO is usually [C, S]
    # We proceed assuming [C, S]
    
    # Transpose to [Samples, Channels] for interleaving
    data = waveform.T
    
    # Scale float32 [-1, 1] to int16
    data = np.clip(data, -1.0, 1.0)
    data = (data * 32767).astype(np.int16)
    
    # Flatten array to bytes (C-order: row-major)
    # If data is [Samples, Channels], C-order flatten gives [S1C1, S1C2, S2C1, S2C2...] 
    # which is interleaved L R L R
    
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2) # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


class SaveAudioWithTags(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveAudioWithTags",
            display_name="Save Audio (MP3 w/ Tags)",
            category="Audio/Expo",
            description="Saves audio as MP3 and adds metadata tags like artist, title, lyrics.",
            inputs=[
                io.Custom("AUDIO").Input("audio"),
                io.String.Input("filename_prefix", default="audio/ComfyUI"),
                io.String.Input("artist", default="Unknown Artist"),
                io.String.Input("title", default="Unknown Title"),
                io.String.Input("album", default="Unknown Album"),
                io.String.Input("year", default="2024"),
                io.String.Input("genre", default="Experimental"),
                io.String.Input("bpm", default=""), 
                io.String.Input("lyrics", default="", multiline=True, optional=True),
            ],
            outputs=[],  # Output node usually doesn't return data to the flow, but we could return path
            is_output_node=True
        )

    @classmethod
    def execute(cls, audio, filename_prefix, artist, title, album, year, genre, bpm, lyrics=None) -> io.NodeOutput:
        if not MUTAGEN_AVAILABLE:
            raise ImportError("Mutagen is required for tagging. Please install it: pip install mutagen")

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), 1, 1)
        
        # Audio object from ComfyUI is usually {"waveform": tensor, "sample_rate": int}
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Ensure waveform is on CPU
        if waveform.is_cuda:
            waveform = waveform.cpu()

        # Handle batch dimension if present [Batch, Channel, Samples] vs [Channel, Samples]
        if waveform.dim() == 3:
            # Take the first in batch
            waveform = waveform[0]
        
        file_name_with_ext = f"{filename}_{counter:05}_"
        mp3_file_path = os.path.join(full_output_folder, f"{file_name_with_ext}.mp3")
        wav_temp_path = os.path.join(folder_paths.get_temp_directory(), f"{file_name_with_ext}_temp.wav")

        # Save as WAV first using native python wave module
        try:
            save_wav_native(wav_temp_path, waveform, sample_rate)
        except Exception as e:
            # Fallback to torchaudio if native fails for some reason (e.g. funny shapes)
            print(f"Native WAV save failed: {e}. Trying torchaudio...")
            torchaudio.save(wav_temp_path, waveform, sample_rate, format="wav")

        # Convert to MP3 using external FFMPEG (via imageio_ffmpeg)
        try:
            import imageio_ffmpeg
            import subprocess
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # minimal ffmpeg command: -i input -y (overwrite) -codec:a libmp3lame -qscale:a 2 output
            # -map_metadata -1 clears existing metadata so we can write our own cleanly
            cmd = [
                ffmpeg_exe,
                '-y',
                '-i', wav_temp_path,
                '-codec:a', 'libmp3lame',
                '-qscale:a', '2',
                '-map_metadata', '-1',
                mp3_file_path
            ]
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        except Exception as e:
            if os.path.exists(wav_temp_path):
                os.remove(wav_temp_path)
            raise RuntimeError(f"Could not save MP3. FFmpeg conversion failed: {e}")

        # Clean up temp wav
        if os.path.exists(wav_temp_path):
            os.remove(wav_temp_path)

        # Add Tags
        try:
            tags = ID3(mp3_file_path)
        except ID3NoHeaderError:
            tags = ID3()
        
        tags.add(TIT2(encoding=3, text=title))
        tags.add(TPE1(encoding=3, text=artist))
        tags.add(TALB(encoding=3, text=album))
        tags.add(TYER(encoding=3, text=str(year)))
        tags.add(TCON(encoding=3, text=genre))
        
        if bpm and str(bpm).strip():
             tags.add(TBPM(encoding=3, text=str(bpm)))

        if lyrics and str(lyrics).strip():
            # USLT frame is used for unsynchronized lyrics.
            # desc='' sets the description to empty, often required for broad player support
            tags.add(USLT(encoding=3, lang='eng', desc='', text=lyrics))

        tags.save(mp3_file_path)

        return io.NodeOutput()


class LyricsVideoGenerator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LyricsVideoGenerator",
            display_name="Lyrics Video Generator",
            category="Audio/Expo",
            description="Generates an MP4 video with scrolling lyrics over a static image.",
            inputs=[
                io.Image.Input("image", tooltip="Background image"),
                io.Custom("AUDIO").Input("audio", tooltip="Audio track"),
                io.String.Input("lyrics", multiline=True, tooltip="Lyrics to scroll"),
                io.Int.Input("font_size", default=40, min=10, max=200),
                io.String.Input("font_color", default="white"),
                io.Int.Input("scroll_speed", default=50, min=1, max=500, tooltip="Speed of scrolling in pixels per second (override)"),
                io.Boolean.Input("auto_speed", default=True, tooltip="Calculate speed to fit lyrics to audio duration"),
                io.String.Input("filename_prefix", default="video/LyricsVideo")
            ],
            outputs=[], # Save to file
            is_output_node=True 
        )

    @classmethod
    def execute(cls, image, audio, lyrics, font_size, font_color, scroll_speed, auto_speed, filename_prefix) -> io.NodeOutput:
        # Get the unique_id if possible? 
        # The execute method in V3 API doesn't pass hidden prompt_id/unique_id by default in args unless requested in define_schema?
        # Actually in V3, we don't have easy access to unique_id in execute unless we ask for it.
        # But wait, we can add hidden inputs for it.
        pass

    @classmethod
    def execute(cls, image, audio, lyrics, font_size, font_color, scroll_speed, auto_speed, filename_prefix, unique_id=None) -> io.NodeOutput:
        if not MOVIEPY_AVAILABLE:
             raise ImportError("MoviePy is required. Please install it: pip install moviepy")
        
        # 1. Prepare Audio
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.is_cuda:
            waveform = waveform.cpu()
        if waveform.dim() == 3:
            waveform = waveform[0]
            
        # Save temp audio file for MoviePy using native wave
        temp_audio_path = os.path.join(folder_paths.get_temp_directory(), "temp_lyrics_audio.wav")
        save_wav_native(temp_audio_path, waveform, sample_rate)
        
        audio_clip = AudioFileClip(temp_audio_path)
        duration = audio_clip.duration

        # 2. Prepare Image
        # ComfyUI Image is [Batch, H, W, C] usually. 
        # Convert to numpy uint8
        img_tensor = image[0] # Take first image
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
        
        # MoviePy 2.0+ uses with_duration instead of set_duration
        try:
             background_clip = ImageClip(img_np).with_duration(duration)
        except AttributeError:
             # Fallback for MoviePy 1.x
             background_clip = ImageClip(img_np).set_duration(duration)

        # 3. Create Scrolling Text
        # We will create a VideoClip using a Make Frame function
        
        w, h = background_clip.size
        
        # Create a large image with the text
        # Using PIL to render text
        # simple estimation of height
        lines = lyrics.split('\n')
        # Load font - basic font
        try:
            # specific to system, fallback to default
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text image size
        # We need a dummy draw to measure
        dummy_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        
        max_width = 0
        total_height = 0
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            max_width = max(max_width, lw)
            # Add some padding
            lh += int(font_size * 0.2) 
            line_heights.append(lh)
            total_height += lh
            
        text_img_w = max_width + 20
        text_img_h = total_height + h # add screen height so it can scroll off
        
        text_img = Image.new('RGBA', (text_img_w, text_img_h), (0,0,0,0))
        draw_text = ImageDraw.Draw(text_img)
        
        y_cursor = h # Start text from bottom of screen
        for i, line in enumerate(lines):
            # Center text
            bbox = draw_text.textbbox((0, 0), line, font=font)
            lw = bbox[2] - bbox[0]
            x_pos = (text_img_w - lw) // 2
            draw_text.text((x_pos, y_cursor), line, font=font, fill=font_color)
            y_cursor += line_heights[i]
            
        text_arr = np.array(text_img)
        # video size
        param_w, param_h = w, h

        # Calculate speed
        # We want to scroll from y=0 (top of text_img at top of screen) to... 
        # Actually we started drawing at y=h (bottom of screen).
        # We want the End of the text (y = total_height + h) to reach the top of the screen (or somewhere appropriate).
        # Let's say we want to scroll the whole text_img up.
        
        scroll_distance = total_height + h 
        if auto_speed:
            speed = scroll_distance / duration
        else:
            speed = scroll_speed

        def make_frame_text(t):
            # Calculate y offset
            y_offset = int(speed * t)
            
            # Extract crop from the large text image
            # The viewport moves DOWN the text image? No, the text moves UP.
            # So we grab a slice.
            # crop area:
            # We want to emulate the text sliding UP.
            # At t=0, we see the top of the 'visual' area.
            # Since we drew text starting at y=h (bottom), initially the screen should be empty or just starting.
            # Let's adjust coordinate frame.
            # Image is (text_img_w, text_img_h).
            # We want to return a frame of size (w, h).
            # We slice text_arr. 
            
            # Let's simplify: 
            # We overlay the text_image onto a transparent frame at a changing y position.
            
            # Base frame (transparent)
            frame = Image.new('RGBA', (w, h), (0,0,0,0))
            
            # Position of text_img top-left corner relative to frame
            # Starts at 0? No, we drew text starting at 'h'.
            # At t=0, text should be just below screen or entering.
            # Let's say we offset the drawing by -y_offset.
            
            # Current y position of the text_img canvas:
            # We want it to move up.
            current_y = 0 - y_offset
            
            # But wait, we have `text_img` which is the whole strip.
            # We can use Pillow to paste it.
            # However, mapping to numpy for every frame is slow.
            # Optimization: slice the numpy array.
            
            # text_arr is [H_big, W_big, 4]
            # We want a slice [h, w, 4]
            # The slice corresponds to the window moving down the long image?
            # Creating scroll up effect means the window moves 'down' the text image strips.
            # Valid Y range in text_arr: 0 to text_img_h.
            
            # Let's refine the drawing: 
            # We define the text strip content. 
            # Content starts at 0.
            # We want Content[0] to be at Screen[h].
            # Then Content[0] moves to Screen[0] and then Screen[-N].
            
            # So we paste text_img at (UserX, ScreenY).
            # ScreenY = h - y_offset.
            
            paste_y = int(h - (speed * t))
            
            # Paste text_img into frame at (0, paste_y)
            # Since paste_y can be negative, we need to handle cropping.
            
            # Create a blank canvas of screen size
            canvas = Image.new('RGBA', (w, h), (0,0,0,0))
            
            # Calculate crop of text_img and paste position
            # This is standard simple overlap logic
            canvas.paste(text_img, ( (w - text_img_w)//2, paste_y), text_img)
            
            # Convert to numpy RGBA usually. MoviePy expects RGB or RGBA.
            return np.array(canvas)

        text_clip_duration = duration # Ensure variable is clear
        text_clip = VideoClip(make_frame_text, duration=text_clip_duration)
        
        # New MoviePy (2.0+) handles transparency automatically if make_frame returns 4 channels (RGBA)
        # We don't need explicit set_mask for RGBA clips anymore if compositing correctly.
        # But we must ensure it's treated as having a mask.
        
        # In 2.1.2 VideoClip with make_frame returning RGBA is usually enough.
        # Let's verify if we need to set the mask explicitly for older versions.
        
        # For MoviePy 1.x, we needed set_mask(ImageClip(..., ismask=True))
        # For MoviePy 2.x, we use with_mask() or just rely on RGBA.
        # However, VideoClip constructor doesn't automatically infer mask from function unless we use ImageClip or similar.
        
        # Actually, best way is to ensure the clip has a mask attribute.
        # If make_frame returns [W,H,4], MoviePy creates a mask from the alpha channel automatically during composition?
        # Not always for generic VideoClip.
        
        # Let's try the modern 2.x approach for masks if possible.
        try:
             # MoviePy 2.0+ approach
             # We create a mask clip from the alpha channel of the text frame?
             # That's slow to compute twice.
             # Actually, if we return RGBA, CompositeVideoClip should handle it.
             pass
        except:
             pass

        # Compatibility handling for set_mask / with_mask if we were to use it.
        # But wait, let's try just composing it. If it fails to show transparency, we fix it.
        # The error was AttributeError: 'VideoClip' object has no attribute 'set_mask'
        
        # In MoviePy 2.0, set_mask is removed/renamed.
        # If we just rely on RGBA output of make_frame, CompositeVideoClip usually works (it checks .shape of the frame).
        
        # So I will REMOVE the explicit set_mask call which is causing the crash.
        # And rely on the fact that make_frame_text returns RGBA.
        
        final_video = CompositeVideoClip([background_clip, text_clip])
        final_video.duration = duration
        
        # Save
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), 1, 1)
        output_file = f"{filename}_{counter:05}_.mp4"
        output_path = os.path.join(full_output_folder, output_file)
        
        final_video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')
        
        # Clean up temp
        try:
            os.remove(temp_audio_path)
        except:
            pass

        return io.NodeOutput()

