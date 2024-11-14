import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import torchaudio

processor = AutoProcessor.from_pretrained("facebook/mms-tts")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mms-tts")

speaker_audio_path = "TestFiles\test.wav"
speaker_audio, speaker_sr = torchaudio.load(speaker_audio_path)

speaker_audio = torchaudio.transforms.Resample(speaker_sr, 16000)(speaker_audio)

text_input = "Ovo je primjer rečenice na hrvatskom jeziku."

inputs = processor(
    text=text_input, 
    speaker_audio=speaker_audio, 
    language="hr",  
    return_tensors="pt"
)

with torch.no_grad():
    generated_audio = model.generate(**inputs).cpu()

torchaudio.save("VoiceGeneration\GeneratedVoice\GeneratedGlas.wav", generated_audio, 16000)

print("Croatian TTS output saved as croatian_generated_voice.wav")
