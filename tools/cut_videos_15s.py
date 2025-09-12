import moviepy.editor as mp
import os
import glob
import math

OUTPUT_ROOT = os.path.join("data", "recordings", "by_15s_segment")

def cut_audio_fixed_segments(input_audio_path: str, segment_duration_seconds: int, output_base_dir: str) -> None:
    """Cut an audio file into consecutive, non-overlapping segments.

    Segments are exactly segment_duration_seconds except possibly the last one
    which can be shorter. All segments are saved inside
    output_base_dir/<audio_basename>/ as segment_000.m4a, etc.
    """
    print(f"processing audio: {input_audio_path}")
    audio = mp.AudioFileClip(input_audio_path)
    total_duration = audio.duration

    audio_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    # Output directory now prefixed with 'output_' as requested (e.g., by_15s_segment/output_v1/segment_000.m4a)
    output_dir = os.path.join(output_base_dir, f"output_{audio_name}")
    os.makedirs(output_dir, exist_ok=True)

    segment_count = int(math.ceil(total_duration / segment_duration_seconds))
    created = 0
    for segment_number in range(segment_count):
        start = segment_number * segment_duration_seconds
        end = min(start + segment_duration_seconds, total_duration)
        if end - start <= 0.3:  # skip extremely tiny remainders
            continue
        segment = audio.subclip(start, end)
        output_filename = os.path.join(output_dir, f"segment_{segment_number:03d}.m4a")
        duration_info = f"{start:.1f}s to {end:.1f}s (duration: {end - start:.1f}s)"
        print(f"creating segment {segment_number}: {duration_info}")
        segment.write_audiofile(output_filename, codec="aac", bitrate="192k", verbose=False, logger=None)
        created += 1

    audio.close()
    print(f"audio {input_audio_path} processed successfully! {created} segments created.\n")

def process_all_audios_in_folder(audio_folder_path: str, segment_duration_seconds: int) -> None:
    patterns = ['*.m4a']
    audio_files = []
    for pattern in patterns:
        audio_files.extend(glob.glob(os.path.join(audio_folder_path, pattern)))

    if not audio_files:
        print(f"no audio files found in folder: {audio_folder_path}")
        return

    print(f"found {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(f"  - {f}")

    for f in audio_files:
        try:
            cut_audio_fixed_segments(f, segment_duration_seconds, OUTPUT_ROOT)
        except Exception as e:
            print(f"error processing {f}: {e}")

if __name__ == "__main__":
    process_all_audios_in_folder("data/recordings/full", 15)