import re
from youtube_transcript_api import YouTubeTranscriptApi

def get_video_id(url):
    """Extracts video ID from a YouTube URL."""
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(url):
    """Fetches YouTube transcript (prefers manual transcript if available)."""
    video_id = get_video_id(url)
    srt = YouTubeTranscriptApi.list_transcripts(video_id)

    transcript = ""
    for i in srt:
        transcript = i.fetch()
        if not i.is_generated:
            break  # Use manual transcript if available

    return transcript

def process_transcript(transcript):
    """Formats the transcript text."""
    return "\n".join([f"Text: {i['text']} Start: {i['start']}" for i in transcript])
