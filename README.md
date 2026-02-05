# ComfyUI Audio Expo

A collection of custom nodes for ComfyUI focused on Audio and Video generation post-processing.
This pack allows you to save audio with rich metadata (ID3 tags, cover art) and generate "karaoke-style" scrolling lyric videos.

## Nodes

### 1. Save Audio (MP3 w/ Tags)
Saves generated audio as an MP3 file with embedded ID3 metadata.
- **Inputs**:
  - `audio`: The audio waveform to save.
  - `filename_prefix`: Prefix for the saved file.
  - `artist`: Artist name tag.
  - `title`: Track title tag.
  - `album`: Album name tag.
  - `year`: Year tag.
  - `genre`: Genre tag.
  - `lyrics`: (Optional) Lyrics text to embed in the USLT frame.
  - `bpm`: (Optional) Beats Per Minute to embed.
  - `image`: (Optional) An image to embed as the MP3 Cover Art (Front Cover).

### 2. Lyrics Video Generator
Generates an `.mp4` video with a scrolling lyrics overlay on top of a background image.
- **Inputs**:
  - `image`: Background image.
  - `audio`: Audio track.
  - `lyrics`: Text to scroll.
  - `font_size`: Size of the text.
  - `font_color`: Hex code or color name for the text.
  - `scroll_speed`: Manual scroll speed (pixels per frame).
  - `auto_speed`: If True, calculates speed to ensure the text scrolls completely within the audio duration.

## Installation

### Option 1: Git Clone (Recommended)
1. Navigate to your ComfyUI `custom_nodes` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/mattjohnpowell/comfyui-audio-expo.git
   ```
3. Install the requirements:
   ```bash
   cd comfyui-audio-expo
   pip install -r requirements.txt
   ```
   *Note: Ensure you are using the python environment used by ComfyUI (e.g. portable versions use `python_embeded`).*

### Option 2: ComfyUI Manager
If using ComfyUI Manager, you can install via "Install via Git URL" using the repository URL.

## Requirements
- **FFmpeg**: Required for MoviePy to write video files and for audio conversion. Ensure it is installed and in your system PATH.
- **Python Packages**:
  - `moviepy`
  - `mutagen`
  - `proglog`
  - `numpy`, `torch`, `torchaudio` (Standard with ComfyUI)
