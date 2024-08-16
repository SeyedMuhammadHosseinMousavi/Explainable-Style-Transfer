import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
tf.get_logger().setLevel(logging.ERROR)

# Function to parse BVH files
def parse_bvh(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header, motion_data = [], []
    capture_data = False
    for line in lines:
        if "MOTION" in line:
            capture_data = True
        elif capture_data:
            if line.strip().startswith("Frames") or line.strip().startswith("Frame Time"):
                continue
            motion_data.append(np.fromstring(line, sep=' '))
        else:
            header.append(line)
    return header, np.array(motion_data)

# Function to save BVH files
def save_bvh(header, motion_data, file_path):
    with open(file_path, 'w') as file:
        file.writelines(header)
        file.write("MOTION\n")
        file.write(f"Frames: {len(motion_data)}\n")
        file.write("Frame Time: 0.008333\n")
        for frame in motion_data:
            line = ' '.join(format(value, '.6f') for value in frame)
            file.write(line + '\n')

# Function to normalize motion data
def normalize_data(data):
    scaler = MinMaxScaler()
    data_shape = data.shape
    data_flattened = data.reshape(-1, data_shape[-1])
    data_normalized = scaler.fit_transform(data_flattened).reshape(data_shape)
    return data_normalized, scaler

# Build the modified style transfer model
def build_style_transfer_model(input_shape, style_shape, latent_dim, amplification_factor):
    initializer = tf.keras.initializers.GlorotUniform()

    # Content Encoder
    content_inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(content_inputs)
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
    content_encoded = layers.Dense(latent_dim, activation='relu', kernel_initializer=initializer)(x)
    content_encoder = Model(content_inputs, content_encoded, name='content_encoder')

    # Style Encoder
    style_inputs = layers.Input(shape=style_shape)
    y = layers.Flatten()(style_inputs)
    y = layers.Dense(128, activation='relu', kernel_initializer=initializer)(y)
    style_encoded = layers.Dense(latent_dim, activation='relu', kernel_initializer=initializer)(y)
    style_encoder = Model(style_inputs, style_encoded, name='style_encoder')

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    style_latent_inputs = layers.Input(shape=(latent_dim,))

    # Amplify style features
    amplified_style_features = layers.Lambda(lambda x: x * amplification_factor)(style_latent_inputs)

    combined_inputs = layers.Concatenate()([latent_inputs, amplified_style_features])
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(combined_inputs)
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer=initializer)(x)
    decoder_outputs = layers.Reshape(input_shape)(x)
    decoder = Model([latent_inputs, style_latent_inputs], decoder_outputs, name='decoder')

    # Style Transfer Model
    content_encoded = content_encoder(content_inputs)
    style_encoded = style_encoder(style_inputs)
    model_outputs = decoder([content_encoded, style_encoded])
    model = Model([content_inputs, style_inputs], model_outputs, name='style_transfer_model')

    # Compile the model with mean squared error loss and Adam optimizer
    learning_rate = 0.0001
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    return model, content_encoder, style_encoder, decoder

# Function to load data from a folder
def load_data_from_folder(folder_path):
    all_data = []
    max_length = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.bvh'):
            file_path = os.path.join(folder_path, file_name)
            _, motion_data = parse_bvh(file_path)
            all_data.append(motion_data)
            if motion_data.shape[0] > max_length:
                max_length = motion_data.shape[0]
    # Pad all sequences to the max_length
    padded_data = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant') for data in all_data]
    return np.array(padded_data)

# Function to smooth motion data using Gaussian filter
def smooth_motion_data_gaussian(motion_data, sigma=1.0):
    smoothed_data = gaussian_filter(motion_data, sigma=(sigma, 0))
    return smoothed_data

# Explainability: Function to plot joint trajectories with subplots for each emotion
def plot_joint_trajectories(emotion, original_data, generated_samples, joint_indices):
    num_samples = len(generated_samples)
    num_joints = len(joint_indices)
    fig, axs = plt.subplots(min(3, num_samples), 3, figsize=(15, 5 * min(3, num_samples)))
    fig.suptitle(f"Joint Trajectories for {emotion.capitalize()} Samples", fontsize=16)

    for i in range(min(3, num_samples)):
        for j in range(3):
            ax = axs[i, j] if num_samples > 1 else axs[j]
            ax.plot(original_data[:, joint_indices[j]], label='Original', linestyle='--', color='blue')
            ax.plot(generated_samples[i][:, joint_indices[j]], label='Generated', linestyle='-', color='red')
            ax.set_title(f"Sample {i + 1} - Joint {joint_indices[j]}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Position")
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Explainability: Function to calculate feature differences
def calculate_feature_differences(original_data, generated_data):
    differences = generated_data - original_data
    return differences

# Explainability: Print detailed feature differences with explanations
def print_feature_differences(differences, joint_indices):
    for joint_index in joint_indices:
        diff_mean = np.mean(differences[:, joint_index])
        diff_std = np.std(differences[:, joint_index])
        min_diff = np.min(differences[:, joint_index])
        max_diff = np.max(differences[:, joint_index])
        print(f"Joint {joint_index}:")
        print(f"  - Mean Difference: {diff_mean:.4f} (indicating average movement change)")
        print(f"  - Standard Deviation: {diff_std:.4f} (indicating variability of movement change)")
        print(f"  - Min Difference: {min_diff:.4f}, Max Difference: {max_diff:.4f}")
        if diff_mean > 0:
            print(f"  - Interpretation: On average, this joint moved forward or upwards in the generated sample.")
        else:
            print(f"  - Interpretation: On average, this joint moved backward or downwards in the generated sample.")
        print(f"  - The variability (std) suggests {'consistent' if diff_std < 1 else 'inconsistent'} changes in movement.\n")

# Explainability: Descriptive Statistics
def print_descriptive_statistics(differences, joint_indices):
    for joint_index in joint_indices:
        range_diff = np.ptp(differences[:, joint_index])  # Range (Max - Min)
        variance = np.var(differences[:, joint_index])    # Variance
        iqr = np.percentile(differences[:, joint_index], 75) - np.percentile(differences[:, joint_index], 25)  # IQR
        print(f"Joint {joint_index} Statistics:")
        print(f"  - Range: {range_diff:.4f}")
        print(f"  - Variance: {variance:.4f}")
        print(f"  - Interquartile Range (IQR): {iqr:.4f}\n")

# Explainability: Pairwise Comparison Visualization
def plot_pairwise_comparison(original_data, generated_data, joint_indices, emotion):
    fig, axs = plt.subplots(1, len(joint_indices), figsize=(15, 5))
    fig.suptitle(f"Pairwise Comparison for {emotion.capitalize()}", fontsize=16)

    for idx, joint_index in enumerate(joint_indices):
        ax = axs[idx]
        ax.scatter(original_data[:, joint_index], generated_data[:, joint_index], alpha=0.5)
        ax.set_xlabel('Original')
        ax.set_ylabel('Generated')
        ax.set_title(f"Joint {joint_index} Comparison")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Explainability: Simple Heatmaps
def plot_heatmap(differences, joint_indices, emotion):
    num_joints = len(joint_indices)
    num_samples = differences.shape[0]

    # Create a figure for the heatmaps
    fig, axs = plt.subplots(1, num_joints, figsize=(15, 5))
    fig.suptitle(f"Heatmaps of Feature Differences for {emotion.capitalize()}", fontsize=16)

    for idx, joint_index in enumerate(joint_indices):
        ax = axs[idx]
        sns.heatmap(differences[:, joint_index].reshape(-1, 1), ax=ax, cmap='coolwarm', cbar=True, annot=False)
        ax.set_title(f"Joint {joint_index}")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Sample Index")
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Function to generate samples with style transfer
def generate_samples_with_style(decoder, latent_dim, style_encoded, num_samples, original_data, content_scaler, original_shape):
    generated_samples = []
    zero_mask = (original_data == 0.0) | (original_data == -0.0)
    negative_zero_mask = np.signbit(original_data) & (original_data == 0.0)

    style_encoded_repeated = np.repeat(style_encoded[:1], num_samples, axis=0)

    for i in range(num_samples):
        latent_sample = np.random.normal(size=(1, latent_dim))
        generated_sample = decoder.predict([latent_sample, style_encoded_repeated[i:i+1]])
        generated_sample = content_scaler.inverse_transform(generated_sample.reshape(-1, generated_sample.shape[-1]))
        for j in range(generated_sample.shape[0]):
            generated_sample[j] = np.where(zero_mask[j], 0.0, generated_sample[j])
            generated_sample[j][negative_zero_mask[j]] = -0.0
        generated_samples.append(generated_sample.reshape(original_shape[1:]))
    return generated_samples

# Main process
base_folder = 'BFA Emotion'  # Path to your folder containing subfolders  
style_folder = 'styleemotion'  # Folder containing style data
output_folder = 'vae_output'
latent_dim = 10
num_new_samples = 1
epochs = 10
batch_size = 32
sigma = 2.0  # Standard deviation for Gaussian smoothing
amplification_factor = 6.0  # Factor to increase the influence of style

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

class_folders = ['angry', 'depressed', 'happy']

for class_folder in class_folders:
    print(f"Processing class: {class_folder}")
    class_path = os.path.join(base_folder, class_folder)
    class_output_folder = os.path.join(output_folder, class_folder)
    if not os.path.exists(class_output_folder):
        os.makedirs(class_output_folder)

    # Load and normalize content data for the current class
    content_data = load_data_from_folder(class_path)
    original_shape = content_data.shape
    content_data = content_data.reshape((content_data.shape[0], -1))  # Flatten data
    normalized_content_data, content_scaler = normalize_data(content_data)

    # Load and normalize style data
    style_data = load_data_from_folder(os.path.join(style_folder, class_folder))
    style_data = style_data.reshape((style_data.shape[0], -1))  # Flatten data
    normalized_style_data, style_scaler = normalize_data(style_data)

    # Ensure the number of style samples matches the number of content samples
    num_content_samples = normalized_content_data.shape[0]
    num_style_samples = normalized_style_data.shape[0]

    if num_style_samples < num_content_samples:
        # Repeat the style data to match the number of content samples
        repeats = (num_content_samples + num_style_samples - 1) // num_style_samples
        normalized_style_data = np.tile(normalized_style_data, (repeats, 1))[:num_content_samples]

    # Build and train the style transfer model for the current class
    input_shape = normalized_content_data.shape[1:]  # Correct input shape
    style_shape = normalized_style_data.shape[1:]  # Correct style input shape
    model, content_encoder, style_encoder, decoder = build_style_transfer_model(input_shape, style_shape, latent_dim, amplification_factor)
    model.fit([normalized_content_data, normalized_style_data], normalized_content_data, epochs=epochs, batch_size=batch_size)

    # Encode style data
    style_encoded = style_encoder.predict(normalized_style_data)

    # Generate new samples for the current class with style transfer
    generated_samples = generate_samples_with_style(decoder, latent_dim, style_encoded, num_new_samples, content_data, content_scaler, original_shape)

    # Plot all samples for each emotion in fewer figures with subplots
    for i, class_folder in enumerate(class_folders):
        class_path = os.path.join(base_folder, class_folder)
        print(f"Generated and smoothed samples for class {class_folder}")
        
        # Define joint indices to plot
        joint_indices_to_plot = [0, 1, 2]  # Adjust based on which joints you want to compare

        # Plot joint trajectories with subplots
        plot_joint_trajectories(class_folder, content_data.reshape(original_shape)[0], generated_samples, joint_indices_to_plot)
        
        # Calculate and print feature differences
        for j, generated_sample in enumerate(generated_samples):
            differences = calculate_feature_differences(content_data.reshape(original_shape)[0], generated_sample)
            print(f"Feature differences for sample {j + 1} in class {class_folder}:")
            print_feature_differences(differences, joint_indices_to_plot)
            print_descriptive_statistics(differences, joint_indices_to_plot)
            
            # Plot pairwise comparison
            plot_pairwise_comparison(content_data.reshape(original_shape)[0], generated_sample, joint_indices_to_plot, class_folder)
            
            # Plot heatmaps
            plot_heatmap(differences, joint_indices_to_plot, class_folder)

print("Generation completed.")
