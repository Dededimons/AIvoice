import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

file_path = r"TestFiles\test.wav" 
audio, sample_rate = librosa.load(file_path, sr=16000)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53", force_download=True)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53", force_download=True)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("Transcribed Text in Croatian:", transcription)
