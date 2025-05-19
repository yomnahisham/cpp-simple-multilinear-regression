# Simple Multilinear Regression in C++

This C++ program implements a multilinear regression model from scratch to predict heart disease risk based on several features: Age, Cholesterol, BMI, and Blood Pressure. The code constructs the regression model by performing all necessary matrix operations manually, including multiplication, transposition, and inversion using Gauss-Jordan elimination. It calculates the regression coefficients using the normal equation, then uses these coefficients to make predictions and compute residuals and the sum of squared errors. 

The program demonstrates the full workflow of multilinear regression: preparing the data matrix, solving for the coefficients, making predictions, and evaluating model performance. By working directly with matrices and not relying on external libraries, the code provides a transparent view of the mathematical steps involved in regression analysis.

