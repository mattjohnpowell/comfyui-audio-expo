import os
import uuid
import torch
import torchaudio
import numpy as np
import folder_paths
import wave
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# Try to import mutagen, handle if missing
try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TYER, TCON, USLT, TBPM, APIC, ID3NoHeaderError
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


# Try to import moviepy, handle if missing
try:
    # MoviePy 2.0+ uses direct imports, 1.x uses .editor
    try:
        from moviepy import VideoClip, AudioFileClip
    except ImportError:
        from moviepy.editor import VideoClip, AudioFileClip

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

    # Transpose to [Samples, Channels] for interleaving
    data = waveform.T

    # Scale float32 [-1, 1] to int16
    data = np.clip(data, -1.0, 1.0)
    data = (data * 32767).astype(np.int16)

    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


class SaveAudioWithTags:
    """Saves audio as MP3 and adds metadata tags like artist, title, lyrics, and cover art."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
                "artist": ("STRING", {"default": "Unknown Artist"}),
                "title": ("STRING", {"default": "Unknown Title"}),
                "album": ("STRING", {"default": "Unknown Album"}),
                "year": ("STRING", {"default": "2024"}),
                "genre": ("STRING", {"default": "Experimental"}),
                "bpm": ("STRING", {"default": ""}),
            },
            "optional": {
                "cover_image": ("IMAGE",),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "Audio/Expo"

    def save_audio(self, audio, filename_prefix, artist, title, album, year, genre, bpm, cover_image=None, lyrics=None):
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
            tags.add(USLT(encoding=3, lang='eng', desc='', text=lyrics))

        # Embed cover art if provided
        if cover_image is not None:
            img_tensor = cover_image[0]  # First image from batch
            img_np = (img_tensor.numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            buf = BytesIO()
            pil_img.save(buf, format='JPEG', quality=90)
            tags.add(APIC(
                encoding=3,
                mime='image/jpeg',
                type=3,  # Cover (front)
                desc='Cover',
                data=buf.getvalue()
            ))

        tags.save(mp3_file_path)

        return {}


class LyricsVideoGenerator:
    """Generates an MP4 video with scrolling lyrics over a static image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "lyrics": ("STRING", {"default": "Enter lyrics here...", "multiline": True}),
                "font_size": ("INT", {"default": 40, "min": 10, "max": 200}),
                "font_color": ("STRING", {"default": "white"}),
                "outline_size": ("INT", {"default": 2, "min": 0, "max": 20}),
                "outline_color": ("STRING", {"default": "black"}),
                "scroll_speed": ("INT", {"default": 50, "min": 1, "max": 500}),
                "auto_speed": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "video/LyricsVideo"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "generate_video"
    OUTPUT_NODE = True
    CATEGORY = "Audio/Expo"

    def generate_video(self, image, audio, lyrics, font_size, font_color, outline_size, outline_color, scroll_speed, auto_speed, filename_prefix, unique_id=None):
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy is required. Please install it: pip install moviepy")

        # 1. Prepare Audio
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.is_cuda:
            waveform = waveform.cpu()
        if waveform.dim() == 3:
            waveform = waveform[0]

        # Use unique temp filename to avoid race conditions
        temp_id = uuid.uuid4().hex[:8]
        temp_audio_path = os.path.join(folder_paths.get_temp_directory(), f"temp_lyrics_audio_{temp_id}.wav")
        save_wav_native(temp_audio_path, waveform, sample_rate)

        audio_clip = AudioFileClip(temp_audio_path)
        duration = audio_clip.duration

        # 2. Prepare Background Image as numpy array
        img_tensor = image[0]  # Take first image from batch
        bg_arr = (img_tensor.numpy() * 255).astype(np.uint8)  # [H, W, 3]
        h, w = bg_arr.shape[:2]

        # 3. Pre-render the full scrolling text strip ONCE
        if not lyrics or not lyrics.strip() or lyrics.strip() == "Enter lyrics here...":
            raise ValueError("Lyrics input is required. Please enter lyrics text.")

        lines = lyrics.split('\n')
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        # Measure all lines (accounting for outline if enabled)
        dummy_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        line_heights = []
        max_text_width = 0
        total_text_height = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=outline_size)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1] + int(font_size * 0.4)
            max_text_width = max(max_text_width, lw)
            line_heights.append(lh)
            total_text_height += lh

        text_strip_w = max_text_width + 40
        text_strip_h = total_text_height

        text_img = Image.new('RGBA', (text_strip_w, text_strip_h), (0, 0, 0, 0))
        draw_text = ImageDraw.Draw(text_img)

        # Draw text starting at y=0 — no dead padding
        y_cursor = 0
        for i, line in enumerate(lines):
            bbox = draw_text.textbbox((0, 0), line, font=font, stroke_width=outline_size)
            lw = bbox[2] - bbox[0]
            x_pos = (text_strip_w - lw) // 2
            draw_text.text(
                (x_pos, y_cursor), line, font=font, fill=font_color,
                stroke_width=outline_size, stroke_fill=outline_color
            )
            y_cursor += line_heights[i]

        # Convert text image to numpy ONCE — this is the key performance optimization.
        text_arr = np.array(text_img)  # [text_strip_h, text_strip_w, 4] RGBA
        x_offset = (w - text_strip_w) // 2

        # Debug: save text strip and log key values for troubleshooting
        debug_text_path = os.path.join(folder_paths.get_temp_directory(), f"debug_text_strip_{temp_id}.png")
        text_img.save(debug_text_path)
        alpha_max = int(text_arr[:, :, 3].max()) if text_strip_h > 0 else 0
        alpha_nonzero = int((text_arr[:, :, 3] > 0).sum()) if text_strip_h > 0 else 0
        print(f"[LyricsVideo] text_strip: {text_strip_w}x{text_strip_h}, viewport: {w}x{h}, lines: {len(lines)}")
        print(f"[LyricsVideo] total_text_height: {total_text_height}, alpha_max: {alpha_max}, non-zero alpha pixels: {alpha_nonzero}")
        print(f"[LyricsVideo] debug text strip saved to: {debug_text_path}")

        # 4. Calculate scroll speed
        # Text enters from the bottom (paste_y starts at h) and exits at the top
        # (paste_y ends at -total_text_height), total travel = total_text_height + h
        scroll_distance = total_text_height + h
        if auto_speed and duration > 0:
            effective_speed = scroll_distance / duration
        else:
            effective_speed = float(scroll_speed)

        print(f"[LyricsVideo] speed: {effective_speed:.2f} px/s, duration: {duration:.1f}s, scroll_distance: {scroll_distance}")

        # 5. Frame generator — pure numpy compositing, no PIL per frame
        def make_frame(t):
            frame = bg_arr.copy()
            paste_y = int(h - effective_speed * t)

            # Calculate overlap between text strip and the viewport
            src_y = max(0, -paste_y)
            dst_y = max(0, paste_y)
            src_x = max(0, -x_offset)
            dst_x = max(0, x_offset)

            vis_h = min(text_strip_h - src_y, h - dst_y)
            vis_w = min(text_strip_w - src_x, w - dst_x)

            if vis_h <= 0 or vis_w <= 0:
                return frame  # No text visible this frame

            # Alpha-blend text onto background using numpy vectorized ops
            text_slice = text_arr[src_y:src_y + vis_h, src_x:src_x + vis_w]
            alpha = text_slice[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
            text_rgb = text_slice[:, :, :3].astype(np.float32)
            bg_region = frame[dst_y:dst_y + vis_h, dst_x:dst_x + vis_w].astype(np.float32)

            blended = (text_rgb * alpha + bg_region * (1.0 - alpha)).astype(np.uint8)
            frame[dst_y:dst_y + vis_h, dst_x:dst_x + vis_w] = blended

            return frame

        # Debug: save a mid-point frame to verify compositing
        debug_frame = make_frame(duration / 2)
        debug_frame_img = Image.fromarray(debug_frame)
        debug_frame_path = os.path.join(folder_paths.get_temp_directory(), f"debug_frame_mid_{temp_id}.png")
        debug_frame_img.save(debug_frame_path)
        print(f"[LyricsVideo] debug mid-point frame saved to: {debug_frame_path}")

        # 6. Build video clip with audio attached
        video_clip = VideoClip(make_frame, duration=duration)

        try:
            video_clip = video_clip.with_audio(audio_clip)  # MoviePy 2.x
        except AttributeError:
            video_clip.audio = audio_clip  # MoviePy 1.x

        # 7. Save
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 1, 1
        )
        output_file = f"{filename}_{counter:05}_.mp4"
        output_path = os.path.join(full_output_folder, output_file)

        logger = ComfyMoviePyLogger(unique_id) if unique_id else None
        video_clip.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', logger=logger)

        # 8. Clean up
        audio_clip.close()
        try:
            os.remove(temp_audio_path)
        except OSError:
            pass

        return {}
