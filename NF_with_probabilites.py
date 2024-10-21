

import numpy as np
import pandas as pd
import torch
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation, MaskedAffineAutoregressiveTransform
import os
import matplotlib.pyplot as plt

# Function to load landmarks from Excel file (Only x, y coordinates)
def load_landmarks(excel_path):
    df = pd.read_excel(excel_path)
    # Drop 'z' coordinates if present, along with timestamp and frame_number
    landmarks = df.filter(like='_x').join(df.filter(like='_y')).to_numpy()  # Keep only x, y columns
    return landmarks

# Function to create Normalizing Flow model
def create_normalizing_flow(input_dim):
    base_distribution = StandardNormal([input_dim])  # Standard normal distribution
    transforms = []

    for _ in range(5):  # Number of layers in the model
        transforms.append(RandomPermutation(features=input_dim))  # Random permutation layer
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, hidden_features=256))  # Affine autoregressive transform
    
    transform = CompositeTransform(transforms)  # Composite of all transformations
    return Flow(transform, base_distribution)  # Create the flow model

# Function to load the trained model
def load_model(model_path, input_dim):
    flow = create_normalizing_flow(input_dim)  # Create a new flow model
    flow.load_state_dict(torch.load(model_path),strict=False)  # Load the trained model state
    flow.eval()  # Set the model to evaluation mode
    return flow

# Function to compute log probabilities for participant data
def compute_probabilities(flow, data_tensor):
    with torch.no_grad():  # Disable gradient calculation
        log_probabilities = flow.log_prob(data_tensor)  # Compute log probabilities
    return log_probabilities

# Function to compute the mean of the distribution learned for Participant A
def compute_mean_of_distribution(flow_a, data_tensor):
    with torch.no_grad():
        # Use the flow to sample from the base distribution and pass through the flow
        mean_a = torch.mean(flow_a.log_prob(data_tensor))  # Calculate mean of the log probabilities
    return mean_a

# Function to compute norms based on the mean of Participant A's distribution
def compute_norms_relative_to_mean(flow_a, data_tensor, mean_a):
    with torch.no_grad():
        log_probabilities = flow_a.log_prob(data_tensor)
        # Compute norms as the difference between log probabilities and the mean
        norms = torch.abs(log_probabilities - mean_a)
    return norms

# Function to save probabilities and norms to Excel
def save_results_to_excel(probabilities, norms, participant_name_a, participant_name_b, output_folder):
    results_df = pd.DataFrame({
        "Log Probability": probabilities.numpy(),
        "Norm": norms.numpy()
    })  # Create a DataFrame
    output_path = os.path.join(output_folder, f"{participant_name_a}_vs_{participant_name_b}_results.xlsx")  # Define output path
    results_df.to_excel(output_path, index=False)  # Save to Excel
    print(f"Results saved for {participant_name_a} vs {participant_name_b} at {output_path}")

# Function to calculate mean norms
def calculate_mean_norms(norms):
    return np.mean(norms)

# Function to plot norms and save graph
def plot_norms(norms_a, norms_b, norms_c, participant_name_a, participant_name_b, participant_name_c, graphs_folder):
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.hist(norms_a.numpy(), bins=30, alpha=0.5, color='blue', edgecolor='black', label=f'{participant_name_a} Norms')  # Plot histogram for A
    plt.hist(norms_b.numpy(), bins=30, alpha=0.5, color='red', edgecolor='black', label=f'{participant_name_b} Norms')  # Plot histogram for B
    plt.hist(norms_c.numpy(), bins=30, alpha=0.5, color='green', edgecolor='black', label=f'{participant_name_c} Norms')  # Plot histogram for C
    plt.title(f'Norms Distribution: {participant_name_a} ')  # Title
    plt.xlabel('Norms')  # X-axis label
    plt.ylabel('Count')  # Y-axis label
    plt.legend()  # Add legend
    plt.grid(axis='y', alpha=0.75)  # Grid
    plt_path = os.path.join(graphs_folder, f"{participant_name_a}_vs_{participant_name_b}_vs_{participant_name_c}_norms_distribution.png")  # Define graph path
    plt.savefig(plt_path)  # Save graph
    plt.close()  # Close the figure
    print(f"Graph saved for {participant_name_a} vs {participant_name_b} vs {participant_name_c} at {plt_path}")

# Main logic: Load models and compute probabilities and norms
def compute_and_save_results(participant_a_name, participant_b_name, participant_c_name, excel_folder, model_folder, output_folder, graphs_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    os.makedirs(graphs_folder, exist_ok=True)  # Create graphs folder if it doesn't exist
    
    # Load Participant A data
    data_a = load_landmarks(f'{excel_folder}/{participant_a_name}_general_dataset_landmarks.xlsx')  # Load data for A
    input_dim = data_a.shape[1]  # Get input dimensions
    data_tensor_a = torch.tensor(data_a, dtype=torch.float32)  # Convert to tensor
    
    # Load Participant B data
    data_b = load_landmarks(f'{excel_folder}/{participant_b_name}_general_dataset_landmarks.xlsx')  # Load data
    data_tensor_b = torch.tensor(data_b, dtype=torch.float32)  # Convert to tensor
    
    # Load Participant C data
    data_c = load_landmarks(f'{excel_folder}/{participant_c_name}_general_dataset_landmarks.xlsx')  # Load data
    data_tensor_c = torch.tensor(data_c, dtype=torch.float32)  # Convert to tensor
    
    # Load the trained NF model for Participant A
    model_path = os.path.join(model_folder, f'{participant_a_name}_NF_model.pth')  # Path to the model
    flow_a = load_model(model_path, input_dim)  # Load model for Participant A

    # Compute the mean of the distribution learned for Participant A
    mean_a = compute_mean_of_distribution(flow_a, data_tensor_a)

    # Compute probabilities for Participant B based on the model trained on Participant A
    probabilities_b = compute_probabilities(flow_a, data_tensor_b)  # Compute probabilities

    # Compute probabilities for Participant C based on the model trained on Participant A
    probabilities_c = compute_probabilities(flow_a, data_tensor_c)  # Compute probabilities

    # Compute norms relative to mean for Participant B based on the distribution of Participant A
    norms_b = compute_norms_relative_to_mean(flow_a, data_tensor_b, mean_a)  # Compute norms for B

    # Compute norms relative to mean for Participant C based on the distribution of Participant A
    norms_c = compute_norms_relative_to_mean(flow_a, data_tensor_c, mean_a)  # Compute norms for C

    # Compute norms for Participant A based on the distribution of Participant A
    norms_a = compute_norms_relative_to_mean(flow_a, data_tensor_a, mean_a)  # Compute norms for A

    # Save results to Excel
    save_results_to_excel(probabilities_b, norms_b, participant_a_name, participant_b_name, output_folder)  # Save results for B
    save_results_to_excel(probabilities_c, norms_c, participant_a_name, participant_c_name, output_folder)  # Save results for C
    
    # Plot norms and save graph
    plot_norms(norms_a, norms_b, norms_c, participant_a_name, participant_b_name, participant_c_name, graphs_folder)  # Save graph

# Participant names and folders
participant_a_name = '81771516_77_right_standup_1'  # Participant A's name (the one the model is trained on)
participant_b_name = '81771526_77_right_standup_4'  # Participant B's name (the one we want to compute probabilities for)
participant_c_name = '81791556_79_right_standup_3'  # Participant C's name (the one we want to compute norms for) - זר
excel_folder = 'C:/Users/user/OneDrive/שולחן העבודה/שנה ג/סמסטר ב/brain research work/Results_NF'
model_folder = 'C:/Users/user/OneDrive/שולחן העבודה/שנה ג/סמסטר ב/brain research work/NF_Models'
output_folder = 'C:/Users/user/OneDrive/שולחן העבודה/שנה ג/סמסטר ב/brain research work/Probabilities_Results'
graphs_folder = 'C:/Users/user/OneDrive/שולחן העבודה/שנה ג/סמסטר ב/brain research work/graphs'  # Graphs folder path


