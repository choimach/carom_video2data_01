import os
import subprocess
import json

def download_soop_video(url: str, output_dir: str = "data/videos"):
    """
    Downloads a SOOP VOD using yt-dlp and extracts metadata.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # yt-dlp command to download best video+audio and dump json metadata
    # The output format is set to download the video as mp4
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    command = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "-o", output_template,
        url
    ]
    
    print(f"Starting download for: {url}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Download completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        print(f"yt-dlp error output:\n{e.stderr}")
        return False

if __name__ == "__main__":
    # Example usage:
    # Replace the URL with a valid SOOP VOD url for testing
    sample_url = "https://vod.sooplive.co.kr/PLAYER/STATION/..."
    # download_soop_video(sample_url)
    pass
