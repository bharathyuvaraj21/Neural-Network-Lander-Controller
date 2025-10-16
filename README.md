# 🚀 Artificial Rocket Lander Control using Backpropagation Neural Network

## 📘 Project Overview
This project centers on developing an Artificial Neural Network (ANN) to autonomously control the landing trajectory of a simulated rocket lander. A Multilayer Perceptron (MLP), trained using the Backpropagation algorithm, is implemented to learn the optimal control strategy. The model's primary objective is to accurately predict the necessary X-Velocity and Y-Velocity adjustments based on the lander's current X and Y position to ensure a safe, precise landing on the target.

---

## 🗃️ Data Sources
- Simulated Trajectory Data (`Data.csv`): A supervised dataset collected from multiple successful, safe landings of the rocket lander.
- Data Features:  
  - Inputs: X-axis position, Y-axis position  
  - Outputs/Targets: X-velocity, Y-velocity

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Data Cleaning: Removed duplicates and NaN values.  
- Normalization: Applied Min-Max Scaling to all input and output features for uniformity and stable training.  
- Data Partitioning: Split dataset into Training (70%), Validation (15%), and Test (15%) sets to support model training and unbiased evaluation.

### 2. Model Development and Training
- Model Type: Custom-built Multilayer Perceptron (MLP) implemented from scratch in Python.  
- Architecture: Single hidden layer with 10 neurons; Sigmoid activation used for all layers.  
- Learning Algorithm: Backpropagation with momentum term (α=0.9) and L2 regularization (λ=0.6) to optimize convergence and reduce overfitting.

### 3. Hyperparameter Tuning
- Learning Rate (η): 0.1  
- Momentum Rate (α): 0.9  
- Regularization Rate (λ): 0.6  

### 4. Model Training and Evaluation
- Iterative training with performance monitored via RMSE on training and validation datasets.  
- Final optimized weights saved to `NeuralNetHolder.py` for real-time control integration.

---

## 📈 Key Results
- Achieved minimal RMSE on validation data indicating accurate velocity predictions.  
- ANN demonstrated robust and precise control over rocket landing trajectory.  
- Successfully integrated optimized weights into the rocket simulation for real-time autonomous control.

---

## 💡 Practical Implications
- Demonstrates how ANNs can learn complex control policies from supervised data.  
- Provides a foundational approach for control systems in robotics, drone navigation, and autonomous vehicles.  

---

## 🚀 Future Work
- Transition to Deep Reinforcement Learning (DRL) approaches such as DQN and A2C to learn policies via environment interaction.  
- Enhance environment simulation to include realistic physical dynamics (e.g., wind, fuel variability).  
- Explore transfer learning using current model weights for more complex landing scenarios.

---

This README template ensures clear and professional communication of your rocket lander control project with all structural elements for easy understanding and collaboration.
