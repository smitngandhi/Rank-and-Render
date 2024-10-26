import streamlit as st
import os
import shutil
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

# Create a Streamlit app for video generation
st.title("🎬 Generate Video from Images and Audio")

# Markdown description
st.markdown("""
Welcome to the Video Generator! This app allows you to create a video by combining images with an audio track. 
You can specify how many images to include and upload an audio file in MP3 format. 
Simply follow the steps below to create your video!
""")

# Directory containing the sorted images
sorted_images_dir = "sorted_images"

# Check if the directory exists, if not, create it and show a message
if not os.path.exists(sorted_images_dir):
    os.makedirs(sorted_images_dir)
    st.warning(f"⚠️ The folder '{sorted_images_dir}' does not exist yet. Please add some images to the folder.")

# Get the list of sorted images
sorted_images = [img for img in os.listdir(sorted_images_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
sorted_images = sorted(sorted_images, key=lambda x: int(x.split('_')[0]))

# Determine minimum and maximum number of images
min_value = 1
max_value = len(sorted_images) if sorted_images else 0

# Let the user specify the number of images to include in the video
if max_value > 0:
    num_images = st.number_input("📷 Enter the number of images to include in the video:", min_value=min_value, max_value=max_value, step=1)

# Allow user to upload an audio file
audio_file = st.file_uploader("🎵 Upload an audio file for the video (mp3 format)", type=["mp3"])

# Create video only if conditions are met
if max_value > 0 and num_images > 0 and audio_file is not None:
    # Limit the images to the user-specified number
    selected_images = sorted_images[:num_images]
    print(selected_images)

    # Create video clips from selected images
    clips = []
    for image_name in selected_images:
        image_path = os.path.join(sorted_images_dir, image_name)
        clip = ImageClip(image_path).set_duration(3).fadein(0.5).fadeout(0.5).resize(height=720)  # Resize images to fit video
        clips.append(clip)

    # Concatenate the video clips
    video_clips = concatenate_videoclips(clips, method='compose')

    # Directory for saving audio files with numbering
    audio_dir = "audio_files"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # Get the highest numbered audio file and increment the number
    existing_audios = [f for f in os.listdir(audio_dir) if f.startswith('audio_') and f.endswith('.mp3')]
    if existing_audios:
        highest_audio_num = max([int(f.split('_')[1].split('.')[0]) for f in existing_audios])
        next_audio_num = highest_audio_num + 1
    else:
        next_audio_num = 1

    audio_temp_path = os.path.join(audio_dir, f"audio_{next_audio_num}.mp3")

    # Save the uploaded audio file temporarily with numbering
    with open(audio_temp_path, "wb") as f:
        f.write(audio_file.read())

    # Load the audio file
    audio_clip = AudioFileClip(audio_temp_path)

    # Trim the audio to match the video duration
    audio_duration = video_clips.duration
    audio_clip = audio_clip.subclip(0, audio_duration)  # Trim audio to the video duration

    # Set the audio to the video
    video_clips_with_audio = video_clips.set_audio(audio_clip)

    # Directory for saving video files with numbering
    video_dir = "videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Get the highest numbered video file and increment the number
    existing_videos = [f for f in os.listdir(video_dir) if f.startswith('video_') and f.endswith('.mp4')]
    if existing_videos:
        highest_video_num = max([int(f.split('_')[1].split('.')[0]) for f in existing_videos])
        next_video_num = highest_video_num + 1
    else:
        next_video_num = 1

    output_video_path = os.path.join(video_dir, f"video_{next_video_num}.mp4")

    # Write the video to a file
    video_clips_with_audio.write_videofile(output_video_path, fps=24, remove_temp=True, codec='libx264', audio_codec='aac')

    # Display success message and provide a link to download the video
    st.success("🎉 Video generated successfully!")

    # Display the video
    st.video(output_video_path)

    # Provide a download link for the video
    with open(output_video_path, "rb") as video_file:
        btn = st.download_button(
            label="⬇️ Download the video",
            data=video_file,
            file_name=f"video_{next_video_num}.mp4",
            mime="video/mp4"
        )



else:
    if max_value == 0:
        st.warning("⚠️ No images found in the 'sorted_images' folder. Please add images to generate a video.")
    else:
        st.warning("⚠️ Please select a number of images and upload an audio file to generate the video.")
