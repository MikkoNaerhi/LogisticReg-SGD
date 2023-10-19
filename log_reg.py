import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

def sgd_classifier(
    X:np.ndarray,
    y:np.ndarray,
    eta:float,
    iterations:int
) -> np.ndarray:
  """
  Stochastic Gradient Descent (SGD) implementation of a Logistic Regression Classifier.
  Parameters:
  -----------
  X: Feature matrix
  y: Target vector
  eta: Learning rate
  iterations: Number of iterations
  
  Returns:
  --------
  w: Learned weights
  """
  m, n = X.shape
  # Initialize weight vector
  w = np.zeros(n)

  for _ in range(iterations):
      for idx in range(m):  # Process examples in the order they appear
          xi, yi = X[idx], y[idx]

          # Compute gradient for the data point
          gradient = -yi * xi / (1 + np.exp(yi * np.dot(w, xi)))

          # Update weight vector
          w = w - eta * gradient

  return w

def load_data() -> (np.ndarray, np.ndarray):
  """ Load and return the breast cancer dataset
  """
  return load_breast_cancer(return_X_y=True)

def evaluate_model(
  y_true:np.ndarray,
  y_pred:np.ndarray
) -> None:
  """ Evaluate the model using ROC AUC Score and Confusion Matrix
  """
  score = roc_auc_score(y_true, y_pred)
  print("ROC AUC Score")
  print(score)

  print("Confusion Matrix:")
  print(confusion_matrix(y_true=y_true, y_pred=y_pred))

def main():
  # Parameters
  eta, iterations = 0.1, 1000

  X,y = load_data()
  mdata, ndim = X.shape

  # Convert the {0,1} output into {-1,+1}
  y = 2*y - 1
  # Scale the features by maximum absolute value
  X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
  weights = sgd_classifier(X=X_train,y=y_train, eta=eta, iterations=iterations)
  pred = np.sign(np.dot(X_test, weights))

  evaluate_model(
    y_true=y_test,
    y_pred=pred
  )

if __name__ == "__main__":
  main()

 
