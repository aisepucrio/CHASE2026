import moviepy.editor as mp
import os
import glob

def cut_video_into_segments(input_video_path, segment_duration_seconds, output_directory, margin):
    print(f"processing audio: {input_video_path}")
    
    audio = mp.AudioFileClip(input_video_path)
    total_duration = audio.duration
    
    os.makedirs(output_directory, exist_ok=True)

    segment_number = 0
    segment_start = 0
    total_segment_duration = segment_duration_seconds + (2 * margin)

    while segment_start < total_duration:
        actual_start = max(0, segment_start - margin)
        segment_end = segment_start + segment_duration_seconds
        actual_end = min(segment_end + margin, total_duration)
        
        if actual_end - actual_start < 5:
            break
            
        segment = audio.subclip(actual_start, actual_end)
        output_filename = os.path.join(output_directory, f"segment_{segment_number:03d}.m4a")
        
        duration_info = f"{actual_start:.1f}s to {actual_end:.1f}s (duration: {actual_end - actual_start:.1f}s)"
        print(f"creating segment {segment_number}: {duration_info}")
        
        segment.write_audiofile(output_filename, codec="aac", bitrate="192k", verbose=False, logger=None)
        
        segment_start += segment_duration_seconds
        segment_number += 1

    audio.close()
    print(f"audio {input_video_path} processed successfully! {segment_number} segments created.\n")

def process_all_videos_in_folder(video_folder_path, segment_duration_seconds, margin):
    video_extensions = ['*.m4a']
    
    video_files = []
    for extension in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder_path, extension)))
    
    if not video_files:
        print(f"no audio files found in folder: {video_folder_path}")
        return
    
    print(f"found {len(video_files)} audio file(s):")
    for video_file in video_files:
        print(f"  - {video_file}")
    
    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_directory = f"data/recordings/by_19s_segment/output_{video_name}"
        
        try:
            cut_video_into_segments(video_file, segment_duration_seconds, output_directory, margin)
        except Exception as e:
            print(f"error processing {video_file}: {e}")

process_all_videos_in_folder("data/recordings/full", 15, 2) 