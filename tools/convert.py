import os
import subprocess

# Set your directory containing mp4 files (can be '.' for current folder)
input_dir = 'data/videos'

# Audio settings
audio_codec = 'aac'
bitrate = '192k'

# Loop through all .mp4 files
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.mp4'):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + '.m4a'
        output_path = os.path.join("recordings", output_filename)

        command = [
            'ffmpeg',
            '-i', input_path,
            '-vn',                    # No video
            '-acodec', audio_codec,
            '-b:a', bitrate,
            output_path
        ]

        print(f'Converting: {filename} → {output_filename}')
        subprocess.run(command, check=True)

print('✅ Conversion complete!')