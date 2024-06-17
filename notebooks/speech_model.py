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
import logging




logging.basicConfig(level=logging.INFO, filename="model_test.log", filemode="w")

class SpeechDataset(Dataset):
    """
    A custom Dataset class for loading and processing speech audio data and corresponding transcripts.

    Args:
        audio_paths (list of str): List of file paths to the audio files.
        transcripts (list of str): List of transcripts corresponding to the audio files.
        target_num_frames (int, optional): Target number of frames for the MFCC features. Defaults to 100.
        transform (callable, optional): Optional transform to be applied on an audio file. Defaults to None.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Loads and returns the MFCC features and encoded transcript for the given index.
        get_num_classes(): Returns the number of unique classes (transcripts) in the dataset.
        decode(encoded_label): Decodes an encoded label back to its original transcript.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The MFCC features of shape (1, num_frames, num_mfcc_coeffs).
            - torch.Tensor: The encoded label as a long tensor.

    Example:
        >>> dataset = SpeechDataset(audio_paths, transcripts, target_num_frames=100)
        >>> mfcc_features, label = dataset[0]
    """
    
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
    """
    A neural network model that combines Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) 
    for processing and classifying sequences of MFCC features extracted from speech audio.

    Args:
        num_classes (int): Number of target classes for the classification.
        num_frames (int): Number of frames in the input MFCC features.

    Methods:
        forward(x): Defines the forward pass of the model.

    Example:
        >>> model = CNNBiLSTM(num_classes=10, num_frames=100)
        >>> inputs = torch.randn(32, 1, 100, 13)  # Example input tensor with batch size 32
        >>> outputs = model(inputs)  # Forward pass
    """
    
    def __init__(self, num_classes, num_frames):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 32, num_frames//2, num_features//2)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),   # Output: (batch_size, 64, num_frames//4, num_features//4)
            nn.Flatten()
        )
        
        # Compute the flattened size after CNN layers
        num_features = 13
        cnn_output_size = 64 * (num_frames // 4) * (num_features // 4)
        
        self.bilstm = nn.LSTM(input_size=cnn_output_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 2, num_classes)
        self.layer_norm = nn.LayerNorm(128 * 2)  # Apply LayerNorm after LSTM output
        
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, num_frames, num_mfcc_coeffs).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.cnn(x)
        x = x.unsqueeze(1)  # Add time dimension for LSTM
        x, _ = self.bilstm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])
        return x


def train(model, dataloader, criterion, optimizer, device):
    """
    Trains the given model for one epoch using the provided dataloader, loss function, and optimizer.

    Args:
        model (nn.Module): The neural network model to be trained.
        dataloader (DataLoader): DataLoader providing the training data.
        criterion (nn.Module): Loss function to be used for training.
        optimizer (torch.optim.Optimizer): Optimizer to be used for updating the model parameters.
        device (torch.device): Device on which the model and data should be loaded (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average training loss for the epoch.

    Example:
        >>> model = CNNBiLSTM(num_classes=10, num_frames=100).to(device)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        >>> dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> train_loss = train(model, dataloader, criterion, optimizer, device)
        >>> print(f"Train Loss: {train_loss:.4f}")
    """
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
    """
    Evaluates the given model on the validation dataset using the provided dataloader and loss function.

    Args:
        model (nn.Module): The neural network model to be evaluated.
        dataloader (DataLoader): DataLoader providing the validation data.
        criterion (nn.Module): Loss function to be used for evaluation.
        device (torch.device): Device on which the model and data should be loaded (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average validation loss for the epoch.

    Example:
        >>> model = CNNBiLSTM(num_classes=10, num_frames=100).to(device)
        >>> criterion = nn.CrossEntropyLoss()
        >>> dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> val_loss = validate(model, dataloader, criterion, device)
        >>> print(f"Validation Loss: {val_loss:.4f}")
    """
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


def main(train_corpus: list[dict], val_corpus: list[dict], test_corpus: list[dict], device, mode="training"):
    """
    Main function to train or load a CNN-BiLSTM model for speech recognition.

    Args:
        train_corpus (list of dict): List of dictionaries containing training data paths and transcripts.
        val_corpus (list of dict): List of dictionaries containing validation data paths and transcripts.
        test_corpus (list of dict): List of dictionaries containing test data paths and transcripts.
        device (torch.device): Device on which to run the model (e.g., 'cuda' or 'cpu').
        mode (str): Mode of operation. Can be either "training" or "inference". Default is "training".

    Returns:
        model (torch.nn.Module): Trained or loaded CNN-BiLSTM model.
        train_dataset (SpeechDataset): Training dataset instance.
    """
    
    num_frames = 1000
    
    train_audio_path, train_transcript = [], []
    for files in train_corpus:
        train_audio_path.append(files["file"])
        train_transcript.append(files["transcript"])
    train_dataset = SpeechDataset(train_audio_path, train_transcript, target_num_frames=num_frames)
    
    num_classes = train_dataset.get_num_classes()
    
        
    val_audio_path, val_transcript = [], []
    for files in val_corpus:
        val_audio_path.append(files["file"])
        val_transcript.append(files["transcript"])
    val_dataset = SpeechDataset(val_audio_path, val_transcript, target_num_frames=num_frames)
    
        
    test_audio_path, test_transcript = [], []
    for files in test_corpus:
        test_audio_path.append(files["file"])
        test_transcript.append(files["transcript"])
    test_dataset = SpeechDataset(test_audio_path, test_transcript, target_num_frames=num_frames)
    
    batch_size = 64
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        
    if mode == "training":
        model = CNNBiLSTM(num_classes=num_classes, num_frames=num_frames).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

               
        num_epochs = 10
        for epoch in range(num_epochs):
            train_loss = train(model, train_dataloader, criterion, optimizer, device)
            val_loss = validate(model, val_dataloader, criterion, device)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        test_loss = validate(model, test_dataloader, criterion, device)
        print("------------------------------------")
    
    elif mode == "inference":
        model_path = r"models"
        curr_notebook_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(curr_notebook_dir, os.pardir))
        model_path_whole = os.path.join(parent_dir, model_path)
        
        model = torch.load(os.path.join(model_path_whole, "speech_model.pt"))
        model = model.to(device)
        
        print("Model loaded!")
        print("------------------------------------")
    
    return model, train_dataset


    
def corpus(mode="train"):
    """
    Load and return the corpus data from a JSON file based on the specified mode.

    Args:
        mode (str): The mode of the corpus to load. Options are "train", "dev", or "test". 
                    Default is "train".

    Returns:
        list of dict: A list of dictionaries, each containing the following keys:
            - "name": The name of the file.
            - "file": The file path.
            - "transcript": The transcript of the audio file.

    Raises:
        FileNotFoundError: If the JSON file for the specified mode is not found.
        json.JSONDecodeError: If the JSON file is not properly formatted.
    """
    corpus_path = r"data\processed"
    curr_notebook_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(curr_notebook_dir, os.pardir))
    corpus_path_whole = os.path.join(parent_dir, corpus_path)
    
    mode_path = mode + "_corpus_new.json"
    
    corpus_path_mode = os.path.join(corpus_path_whole, mode_path)
    
    with open(corpus_path_mode, "r") as f:
        mode_corpus = json.load(f)
    
    return mode_corpus



def get_predictions(test_audio, model, train_dataset, device):
    """
    Get the predicted transcript for a given audio file using a trained model.

    Args:
        test_audio (dict): A dictionary containing the file path and transcript of the test audio.
            Example: {"file": "path/to/audio.wav", "transcript": "actual transcript"}
        model (torch.nn.Module): The trained CNN-BiLSTM model.
        train_dataset (SpeechDataset): The dataset instance used during training, required for decoding labels.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        None: This function prints the original and predicted transcripts.
    """
    
    # Create a SpeechDataset instance for the single test audio
    test_audio_path = [test_audio["file"]]
    test_transcript = [test_audio["transcript"]]
    test_dataset = SpeechDataset(test_audio_path, test_transcript, target_num_frames=1000)
    
    inputs, label = test_dataset[0]
    inputs = inputs.unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        predicted = torch.argmax(outputs, 1).item()
    
    predicted_transcript = train_dataset.decode(predicted)
    
    print(f"Original transcript: {test_transcript[0]}")
    print(f"Predicted transcript: {predicted_transcript}")

if __name__ == "__main__":
    
    
    train_corpus = corpus("train")

    val_corpus = corpus("dev")

    test_corpus = corpus("test")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, train_dataset = main(train_corpus, val_corpus, val_corpus, device, mode="inference") 
    
    
    model_path = r"models"
    curr_notebook_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(curr_notebook_dir,os.pardir))
    model_path_whole = os.path.join(parent_dir,model_path)
    
    torch.save(model,os.path.join(model_path_whole,"speech_model.pt"))
    
    
    randtestidx = np.random.randint(0, len(val_corpus))
    test_data = val_corpus[randtestidx]
    get_predictions(test_data, model, train_dataset, device)
    
