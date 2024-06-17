import torch
import torch.nn as nn
import torchaudio
import librosa
from python_speech_features import mfcc as psf_mfcc
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tiktoken
import json
import os

class SpeechDataset(Dataset):
    def __init__(self, audio_paths, transcripts, target_num_frames=100, transform=None):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.target_num_frames = target_num_frames
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(transcripts)
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        transcript = self.transcripts[idx]
        audio_data, sample_rate = sf.read(audio_path)
        if self.transform:
            mfcc = self.transform(audio_path)
        else:
            mfcc = psf_mfcc(audio_data, samplerate=sample_rate, numcep=13)
            mfccs_normalized = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
        
        # Ensure equal number of frames
        if mfccs_normalized.shape[0] < self.target_num_frames:
            # Pad with zeros
            padding = np.zeros((self.target_num_frames - mfccs_normalized.shape[0], mfccs_normalized.shape[1]))
            mfccs_normalized = np.vstack((mfccs_normalized, padding))
        elif mfccs_normalized.shape[0] > self.target_num_frames:
            # Truncate
            mfccs_normalized = mfccs_normalized[:self.target_num_frames, :]
        
        # Adding a channel dimension for CNN input: (num_frames, num_mfcc_coeffs) -> (1, num_frames, num_mfcc_coeffs)
        mfccs_normalized = mfccs_normalized[np.newaxis, ...]
        
        label = self.label_encoder.transform([transcript])[0]
        
        return torch.tensor(mfccs_normalized, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def get_num_classes(self):
        return len(self.label_encoder.classes_)
    
    def decode(self, encoded_label):
        return self.label_encoder.inverse_transform([encoded_label])[0]
    
class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 32, num_frames//2, num_features//2)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # Output: (batch_size, 64, num_frames//4, num_features//4)
            nn.Flatten()
        )
        
        # Compute the flattened size after CNN layers
        num_frames = 1000
        num_features = 13
        cnn_output_size = 64 * (num_frames // 4) * (num_features // 4)
        
        self.bilstm = nn.LSTM(input_size=cnn_output_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.unsqueeze(1)  # Add time dimension for LSTM
        x, _ = self.bilstm(x)
        x = self.fc(x[:, -1, :])
        return x

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def main(train_corpus:list[dict], val_corpus:list[dict], test_corpus:list[dict]):
    
    print("Generating train dataset")
    
    train_audio_path, train_transcript = [],[]
    for files in train_corpus:
        train_audio_path.append(files["file"])
        train_transcript.append(files["transcript"])
    train_dataset = SpeechDataset(train_audio_path, train_transcript)
    
        
    num_classes = train_dataset.get_num_classes()
    
    print("Generating validation dataset")
    
    val_audio_path, val_transcript = [],[]
    for files in val_corpus:
        val_audio_path.append(files["file"])
        val_transcript.append(files["transcript"])
    val_dataset = SpeechDataset(val_audio_path, val_transcript)
    
    print("Generating test dataset")
    
    test_audio_path, test_transcript = [],[]
    for files in test_corpus:
        test_audio_path.append(files["file"])
        test_transcript.append(files["transcript"])
    test_dataset = SpeechDataset(test_audio_path, test_transcript)
    
    train_size = 10000
    train_sampler = SubsetRandomSampler(np.arange(train_size))
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    
    
    print("Model loading: ")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNBiLSTM(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("training start:")
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    test_loss = validate(model, test_dataloader, criterion, device)
    print("------------------------------------")
    print(f'Final Test Loss: {test_loss:.4f}')
    

    
def corpus(mode="train"):
    corpus_path = r"data\processed"
    curr_notebook_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(curr_notebook_dir,os.pardir))
    corpus_path_whole = os.path.join(parent_dir,corpus_path)
    
    mode_path = mode+"_corpus_new.json"
    
    corpus_path_mode = os.path.join(corpus_path_whole, mode_path)
    
    with open(corpus_path_mode, "r") as f:
        mode_corpus = json.load(f)
    
    return mode_corpus

if __name__ == "__main__":
    
    
    train_corpus = corpus("train")

    val_corpus = corpus("dev")

    test_corpus = corpus("test")
    
    main(train_corpus, val_corpus, test_corpus)
    
def get_predictions(test_audio, model, train_dataset):
    # Create a SpeechDataset instance for the single test audio
    test_audio_path = [test_audio["file"]]
    test_transcript = [test_audio["transcript"]]
    test_dataset = SpeechDataset(test_audio_path, test_transcript)
    
    # Use the __getitem__ method to get the processed input and label
    inputs, label = test_dataset[0]
    inputs = inputs.unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        inputs = inputs.to(next(model.parameters()).device)
        outputs = model(inputs)
        predicted = torch.argmax(outputs, 1).item()
    
    predicted_transcript = train_dataset.decode(predicted)
    
    print(f"Original transcript: {test_transcript[0]}")
    print(f"Predicted transcript: {predicted_transcript}")