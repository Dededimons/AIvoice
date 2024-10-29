import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import os

file_path = r"TestFiles\test.wav" 
audio, sample_rate = librosa.load(file_path, sr=16000)


processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True,language="hr")

generated_tokens = model.generate(
    inputs["input_features"],
    num_beams=10,
    temperature=0.7,
)

with torch.no_grad():
    generated_tokens = model.generate(inputs["input_features"])

transcription = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print("Transcribed Text in Croatian:", transcription)

script_dir = os.path.dirname(os.path.abspath(__file__))
transcription_file_path = os.path.join(script_dir, "transcription.txt")

with open(transcription_file_path, "w", encoding="utf-8") as f:
    f.write(transcription)

print(f"Transcription saved to {transcription_file_path}")