import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torchaudio

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(dataset[0]["xvector"]).unsqueeze(0)

text_input = "Ovo je primjer rečenice na hrvatskom jeziku."

inputs = processor(text=text_input, return_tensors="pt")

with torch.no_grad():
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding)

torchaudio.save("croatian_speech_output.wav", speech, 16000)
print("Croatian TTS output saved as croatian_speech_output.wav")