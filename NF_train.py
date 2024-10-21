import numpy as np
import pandas as pd
import torch
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation
from nflows.transforms import MaskedAffineAutoregressiveTransform
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Function to load landmarks from Excel file
def load_landmarks(excel_path):
    df = pd.read_excel(excel_path)
    landmarks = df.drop(columns=['timestamp', 'frame_number']).to_numpy()
    return landmarks

# Function to create Normalizing Flow model with more layers and larger flow size
def create_normalizing_flow(input_dim):
    base_distribution = StandardNormal([input_dim])
    transforms = []

    # Increase the number of layers and the size of hidden features
    for _ in range(8):  # Increased number of layers from 5 to 8
        transforms.append(RandomPermutation(features=input_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, hidden_features=256))  # Increased hidden features from 128 to 256
    
    transform = CompositeTransform(transforms)
    return Flow(transform, base_distribution)

# Function to train Normalizing Flow on participant data
def train_normalizing_flow(flow, data_tensor, epochs=800, learning_rate=1e-4):
    print(f"Shape of data_tensor: {data_tensor.shape}")
    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    
    # Store loss values for plotting
    loss_values = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = -flow.log_prob(data_tensor).mean()
        loss.backward()

        # Apply gradient clipping to avoid large updates
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        loss_values.append(loss.item())  # Store the loss value
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

    return loss_values  # Return loss values for plotting

# Function to save the model and the distribution samples
def save_model_and_distribution(flow, participant, output_folder):
    model_path = os.path.join(output_folder, f'{participant}_NF_model.pth')
    print(f"Model path: {model_path}")

    try:
        torch.save(flow.state_dict(), model_path)
        print(f"Model saved for {participant}")
    except Exception as e:
        print(f"Error saving model for {participant}: {e}")

# Function to plot and save loss values
def plot_loss(loss_values, participant, loss_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title(f'Loss over Epochs for {participant}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    loss_path = os.path.join(loss_folder, f'{participant}_loss_plot.png')
    plt.savefig(loss_path)
    plt.close()  # Close the plot to avoid overlap

# Main logic: Train models and save them
def train_and_save(participant_names, excel_folder, output_folder, loss_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(loss_folder, exist_ok=True)  # Create loss folder if it doesn't exist

    for participant in participant_names:
        print(f"Processing participant {participant}...")
        data = load_landmarks(f'{excel_folder}/{participant}_general_dataset_landmarks.xlsx')
        input_dim = data.shape[1]
        flow = create_normalizing_flow(input_dim)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        print(f"Training Normalizing Flow for {participant}...")
        loss_values = train_normalizing_flow(flow, data_tensor)
        print(f"Training completed for {participant}.")
        
        print(f"Saving model and distribution for {participant}...")
        save_model_and_distribution(flow, participant, output_folder)
        
        print(f"Plotting loss for {participant}...")
        plot_loss(loss_values, participant, loss_folder)  # Plot and save loss values
        print(f"Saved model and loss plot for {participant}.")

# List of participant names
participant_names = ['81651276_65_right_control_interesting_3','81651286_65_right_control_interesting_3','81661296_66_right_control_interesting_1','81661306_66_right_control_interesting_3'] 
#participant_names = ['81651286_65_right_standup_4','81661306_66_right_standup_1','81661296_66_right_standup_2']
# Folders for data and model saving
excel_folder = f'C:/Users/user/OneDrive/שולחן העבודה/שנה ג/סמסטר ב/brain research work/Results_NF'
output_folder = f'C:/Users/user/OneDrive/שולחן העבודה/שנה ג/סמסטר ב/brain research work/NF_Models'
loss_folder = r"C:\Users\user\OneDrive\שולחן העבודה\שנה ג\סמסטר ב\brain research work\loss _func"
# Run the training and saving process
train_and_save(participant_names, excel_folder, output_folder, loss_folder)

