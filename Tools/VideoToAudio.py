from moviepy.editor import VideoFileClip

def video_to_wav(video_path, wav_path):
    video = VideoFileClip(video_path)

    video.audio.write_audiofile(wav_path, codec='pcm_s16le') 

    video.close()
    print(f"Audio saved to {wav_path}")

video_path = 'TestFiles\mp4test.mp4' 
wav_path = 'TestFiles\wavtest.wav'
video_to_wav(video_path, wav_path)