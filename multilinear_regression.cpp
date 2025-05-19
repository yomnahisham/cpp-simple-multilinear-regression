#include <iostream> 
#include <vector> 
#include <iomanip> 
#include <cmath> 

using namespace std;

// Function to print a matrix 
void printMatrix(const vector<vector<double>>& mat) { // prints a 2d matrix
    for (const auto& row : mat) { // loop through each row
        for (double val : row) { // loop through each value in row
            cout << setw(10) << val << " "; // print value with width 10
        }
        cout << endl; // new line after each row
    }
}

// Matrix multiplication
vector<vector<double>> multiply(const vector<vector<double>>& A, const vector<vector<double>>& B) { // multiplies two matrices
    int n = A.size(); // number of rows in A
    int m = B[0].size(); // number of columns in B
    int p = B.size(); // number of rows in B (columns in A)
    vector<vector<double>> C(n, vector<double>(m, 0.0)); // result matrix initialized to zero
    for (int i = 0; i < n; ++i) { // loop through rows of A
        for (int j = 0; j < m; ++j) { // loop through columns of B
            for (int k = 0; k < p; ++k) { // loop for dot product
                C[i][j] += A[i][k] * B[k][j]; // multiply and add
            }
        }
    }
    return C; // return result
}

// Matrix transpose
vector<vector<double>> transpose(const vector<vector<double>>& A) { // transposes a matrix
    int n = A.size(); // number of rows
    int m = A[0].size(); // number of columns
    vector<vector<double>> T(m, vector<double>(n, 0.0)); // transposed matrix
    for (int i = 0; i < n; ++i) { // loop through rows
        for (int j = 0; j < m; ++j) { // loop through columns
            T[j][i] = A[i][j]; // swap row and column
        }
    }
    return T; // return transposed matrix
}

// General matrix inverse using Gauss-Jordan elimination
vector<vector<double>> inverse(const vector<vector<double>>& A) { // finds inverse of any square matrix
    int n = A.size(); // size of matrix
    if (n != (int)A[0].size()) { // check if not square
        cerr << "Matrix must be square for inversion." << endl;
        exit(1);
    }
    // Create augmented matrix [A | I]
    vector<vector<double>> aug(n, vector<double>(2 * n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i][j] = A[i][j]; // left side is A
            aug[i][j + n] = (i == j) ? 1.0 : 0.0; // right side is identity
        }
    }
    // Forward elimination
    for (int i = 0; i < n; ++i) {
        // Find pivot
        double pivot = aug[i][i];
        if (fabs(pivot) < 1e-12) {
            // Try to swap with a lower row
            int swap_row = -1;
            for (int k = i + 1; k < n; ++k) {
                if (fabs(aug[k][i]) > 1e-12) {
                    swap_row = k;
                    break;
                }
            }
            if (swap_row == -1) {
                cerr << "Matrix is singular or nearly singular." << endl;
                exit(1);
            }
            swap(aug[i], aug[swap_row]);
            pivot = aug[i][i];
        }
        // Normalize pivot row
        for (int j = 0; j < 2 * n; ++j) {
            aug[i][j] /= pivot;
        }
        // Eliminate other rows
        for (int k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = aug[k][i];
            for (int j = 0; j < 2 * n; ++j) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }
    // Extract inverse from augmented matrix
    vector<vector<double>> inv(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inv[i][j] = aug[i][j + n];
        }
    }
    return inv;
}

// Multilinear regression: computes beta = (X^T X)^-1 X^T Y
vector<double> linearRegression(const vector<vector<double>>& X, const vector<double>& Y) { // computes regression coefficients
    int n = X.size(); // number of samples
    int p = X[0].size(); // number of features
    // Convert Y to a column vector
    vector<vector<double>> Y_col(n, vector<double>(1)); // Y as column vector
    for (int i = 0; i < n; ++i) Y_col[i][0] = Y[i]; // fill Y_col
    // X^T
    vector<vector<double>> Xt = transpose(X); // transpose of X
    // X^T X
    vector<vector<double>> XtX = multiply(Xt, X); // Xt times X
    // (X^T X)^-1
    vector<vector<double>> XtX_inv = inverse(XtX); // inverse of XtX
    // X^T Y
    vector<vector<double>> XtY = multiply(Xt, Y_col); // Xt times Y_col
    // beta = (X^T X)^-1 X^T Y
    vector<vector<double>> beta_col = multiply(XtX_inv, XtY); // regression coefficients as column
    // Convert beta to 1D vector
    vector<double> beta(p); // 1d vector for beta
    for (int i = 0; i < p; ++i) beta[i] = beta_col[i][0]; // fill beta
    return beta; // return coefficients
}

// Predict Y given X and beta
double predict(const vector<double>& x, const vector<double>& beta) { // predicts y value
    double y_pred = 0.0; // initialize prediction
    for (size_t i = 0; i < x.size(); ++i) { // loop through features
        y_pred += x[i] * beta[i]; // sum feature times coefficient
    }
    return y_pred; // return prediction
}

// Function to compute residuals and sum of squared errors
vector<double> computeResiduals(const vector<vector<double>>& X, const vector<double>& Y, const vector<double>& beta, double& sum_sq_error) { // computes residuals and error
    int n = X.size(); // number of samples
    vector<double> residuals(n); // residuals vector
    sum_sq_error = 0.0; // initialize error
    for (int i = 0; i < n; ++i) { // loop through samples
        double y_pred = predict(X[i], beta); // predicted value
        residuals[i] = Y[i] - y_pred; // actual minus predicted
        sum_sq_error += residuals[i] * residuals[i]; // add squared residual
    }
    return residuals; // return residuals
}

int main() {
    // Predicting heart disease risk from Age, Cholesterol, BMI, and Blood Pressure
    // Each row: {1, Age, Cholesterol, BMI, Blood Pressure} (1 for intercept)
    vector<vector<double>> X = {
        {1, 52, 200, 25, 120},
        {1, 53, 180, 27, 130}, 
        {1, 49, 210, 24, 110},
        {1, 60, 190, 29, 140}, 
        {1, 45, 170, 22, 115}, 
        {1, 55, 205, 28, 135}, 
        {1, 50, 195, 26, 125}, 
        {1, 48, 185, 23, 118} 
    };
    // Y values (e.g., risk score)
    vector<double> Y = {0.3, 0.25, 0.28, 0.35, 0.22, 0.32, 0.27, 0.24}; // target values

    cout << fixed << setprecision(4); // set output precision
    cout << "Multilinear Regression Example (Heart Disease Risk)\n";
    cout << "-----------------------------------------------\n"; // print separator
    cout << "X matrix (with intercept, Age, Cholesterol, BMI, Blood Pressure):\n"; // print X label
    printMatrix(X); // print X matrix
    cout << "Y vector (risk score):\n"; // print Y label
    for (double y : Y) cout << y << " "; // print Y values
    cout << endl << endl; // new lines

    // Compute regression coefficients
    vector<double> beta = linearRegression(X, Y); // get regression coefficients
    cout << "Regression coefficients (beta):\n";
    for (size_t i = 0; i < beta.size(); ++i)
        cout << "beta[" << i << "] = " << beta[i] << endl; // print each coefficient
    cout << endl; // new line

    double sum_sq_error; // variable for sum of squared errors
    vector<double> residuals = computeResiduals(X, Y, beta, sum_sq_error); // get residuals and error
    cout << "Predictions and residuals:\n"; 
    for (size_t i = 0; i < X.size(); ++i) {
        double y_pred = predict(X[i], beta); // get prediction
        cout << "Actual: " << Y[i] << ", Predicted: " << y_pred << ", Residual: " << residuals[i] << endl; // print actual, predicted, residual
    }
    cout << "\nSum of squared errors: " << sum_sq_error << endl; // print error

    return 0; // end program
}
