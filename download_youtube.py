import pdb
import cv2
from matplotlib import pyplot as plt
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os, json
from tqdm import tqdm
import re
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pytube.innertube import _default_clients
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]


# Function to extract the video ID from YouTube URL
def extract_video_id(youtube_url):
    # Use regex to find the video ID
    match = re.search(r"(?<=v=)[^&#]+", youtube_url)
    # Return the video ID if a match is found
    return match.group() if match else None


# Function to download and trim a YouTube video
def download_and_trim_video(youtube_url, start_time, end_time, id, output_path='downloaded_video.mp4'):
    # try:
    # Construct the YouTube URL and create the YouTube object
    # youtube_url = f'https://www.youtube.com/watch?v={video_id}'
    vid_id = extract_video_id(youtube_url)
    assert vid_id is not None

    # Download the video
    if not os.path.exists(f'youtube_videos/original/{vid_id}.mp4'):
        yt = YouTube(youtube_url)
        # Select the stream with the highest resolution
        video_stream = yt.streams.get_highest_resolution()
        if vid_id == "zkglsg0K1IY": # special case
            video_stream = yt.streams.get_by_itag(137)
        video_stream.download(filename=f'youtube_videos/original/{vid_id}.mp4')

    # trim video
    ffmpeg_extract_subclip(f'youtube_videos/original/{vid_id}.mp4', start_time, end_time, output_path)


if __name__ == "__main__":
    os.makedirs("./youtube_videos/original", exist_ok=True)

    data = json.load(open("i2v_prompts.json"))
    for d in tqdm(data):
        if 'video' in d:
            start_time, end_time = map(float, d['video'][1].split(","))
            save_dir = os.path.join("./youtube_videos/", f"{d['id']:05d}-00000.mp4")
            download_and_trim_video(d['video'][0], start_time, end_time,  d['id'], save_dir)
