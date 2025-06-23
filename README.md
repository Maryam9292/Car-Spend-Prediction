# Car-Spend-Prediction

## Overview
This project predicts car sales amounts based on socio-economic factors such as **Gender**, **Age**, **Annual Salary**, **Credit Card Debt**, and **Net Worth**. 

The prediction model can be utilized by car dealerships and automotive brands to estimate a customer's spending capacity on a car. By understanding a customer's financial profile, businesses can:
- Suggest cars within their budget range.
- Optimize their sales strategies by tailoring offers.
- Enhance customer experience by showing relevant options.

This approach uses a **neural network model** built with TensorFlow/Keras to provide accurate predictions, enabling data-driven decisions for both businesses and customers.

## Key Features
1. **Dataset**:
   - Input Variables: Gender, Age, Annual Salary, Credit Card Debt, Net Worth.
   - Target Variable: Car Sales Amount.

2. **Model**:
   - Neural Network with:
     - Two hidden layers (10 neurons each, ReLU activation).
     - One output layer (linear activation).
   - Optimizer: **Adam**
   - Loss Function: **Mean Squared Error**

3. **Data Preprocessing**:
   - Scaling of features and target variable using **MinMaxScaler**.
   - Train-test split (75% training, 25% testing).

---
## Setup Instructions
1. Download the `Sales_Forecast.ipynb` file and the dataset file
2. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/) by uploading it 
   or linking the GitHub repository.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
## Link 
[Open the notebook in Google Colab](https://colab.research.google.com/drive/1ZkYNjSx0s2X8S1aeVla1DMBK9tHH3hII?usp=sharing)
