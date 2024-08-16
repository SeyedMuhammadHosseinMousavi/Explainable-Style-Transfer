Motion Style Transfer with Explainable AI (XAI)
This repository contains a project that combines motion style transfer with various Explainable AI (XAI) techniques to enhance the interpretability of the generated motion data. The project demonstrates how different emotions can be applied to motion data using deep learning, and provides multiple levels of explanations for the generated results.

Features
Motion Style Transfer: Transfer style from one set of motion data (e.g., happy) to another (e.g., neutral) using a deep learning model.
Explainability: Understand the model's decisions with multiple XAI techniques, including:
Feature Difference Analysis
Descriptive Statistics
Pairwise Comparison Visualizations
Heatmaps
Project Structure
parse_bvh: Function to parse motion data from BVH files.
save_bvh: Function to save generated motion data back into BVH files.
normalize_data: Function to normalize the motion data for better performance during training.
build_style_transfer_model: Builds the deep learning model for style transfer.
generate_samples_with_style: Generates new motion samples by applying the desired style.
plot_joint_trajectories: Visualizes joint trajectories in subplots.
calculate_feature_differences: Computes differences between original and generated motion data.
print_feature_differences: Prints detailed explanations of these differences.
print_descriptive_statistics: Provides additional statistical explanations for the differences.
plot_pairwise_comparison: Visualizes pairwise comparisons between original and generated data.
plot_heatmap: Generates heatmaps to visually highlight differences across joints.
Explainable AI (XAI) Techniques
This project employs several XAI techniques to help understand the motion style transfer process. Below is a detailed explanation of each technique used:

1. Feature Difference Analysis
Purpose: To understand how each joint's movement changes as a result of the style transfer.
Method: Calculates the mean, standard deviation, and range of differences between original and generated motion data.
Output:
Mean Difference: Indicates the average change in joint movement.
Standard Deviation: Highlights the variability in joint movement.
Range: Shows the span between the minimum and maximum differences.
2. Descriptive Statistics
Purpose: To provide a deeper statistical analysis of the motion data.
Method: Calculates additional statistics, such as variance and interquartile range (IQR).
Output:
Variance: Measures the spread of the differences from the mean.
Interquartile Range (IQR): Indicates the middle 50% of the data spread.
3. Pairwise Comparison Visualization
Purpose: To visually compare the original and generated motion data for each joint.
Method: Plots scatter plots comparing the original vs. generated data for selected joints.
Output:
Scatter Plots: Each plot provides a direct visual comparison for each joint.
4. Heatmaps
Purpose: To visually represent the magnitude of differences across joints and samples.
Method: Generates heatmaps that show differences between the original and generated data across all samples.
Output:
Heatmaps: Use color gradients to indicate the intensity of differences, making it easy to spot significant changes.
5. Joint Trajectory Visualization
Purpose: To visualize the motion of each joint across time.
Method: Plots the trajectory of each joint over time, comparing the original and generated motions.
Output:
Line Plots: Show how each joint moves over time, with side-by-side comparisons for original and generated motions.
