from sklearn.base import RegressorMixin
import numpy as np


class SGDLinearRegressor(RegressorMixin):
    def __init__(self,
                 lr=0.01, regularization=1., delta_converged=1e-3, max_steps=1000,
                 batch_size=64):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None
        self.b = None

    def fit(self, X, Y):
        w_current = np.zeros(shape=X.shape[1])
        b_current = 0.0
        current_iter = 1

        while current_iter <= self.max_steps:
            w_old = w_current.copy()  # Копируем текущие веса
            b_old = b_current

            # Случайно выбираем индексы
            idx = np.random.choice(np.arange(len(X)), self.batch_size, replace=False)
            x = X[idx]  # x имеет форму (batch_size, 34)
            y = Y[idx]  # y имеет форму (batch_size,)

            # Вычисляем предсказания для выбранного батча
            predictions = np.dot(x, w_old) + b_old
            errors = y - predictions  # Ошибки

            # Вычисляем градиенты
            w_temp = -2 * np.dot(x.T, errors) / self.batch_size  # Градиент w
            b_temp = -2 * np.sum(errors) / self.batch_size  # Градиент b

            # Добавляем регуляризацию
            w_temp += self.regularization * w_old

            # Обновляем значения w и b
            w_current = w_old - self.lr * w_temp
            b_current = b_old - self.lr * b_temp

            # Проверка на сходимость
            if np.linalg.norm(w_old - w_current) < self.delta_converged:
                break

            current_iter += 1

        self.W = w_current
        self.b = b_current

        return self

    def predict(self, X):
        y_pred = []

        for i in range(len(X)):
            # Здесь мы вычисляем предсказание
            y = np.dot(self.W, X[i]) + self.b
            y_pred.append(y)

        return np.array(y_pred)



# from sklearn.base import RegressorMixin
# import numpy as np
#
# class SGDLinearRegressor(RegressorMixin):
#     def __init__(self,
#                  lr=0.01, regularization=1., delta_converged=1e-3, max_steps=1000,
#                  batch_size=64):
#         self.lr = lr  # Learning rate
#         self.regularization = regularization  # L2 regularization strength
#         self.max_steps = max_steps  # Maximum number of steps
#         self.delta_converged = delta_converged  # Convergence threshold
#         self.batch_size = batch_size  # Batch size for SGD
#         self.W = None  # Weights
#         self.b = None  # Bias
#
#     def fit(self, X, Y):
#         # Initialize weights and bias
#         n_features = X.shape[1]
#         self.W = np.zeros(n_features)
#         self.b = 0
#
#         n_samples = X.shape[0]
#         steps = 0
#         prev_loss = float('inf')
#
#         while steps < self.max_steps:
#             # Shuffle data
#             indices = np.random.permutation(n_samples)
#             X_shuffled = X[indices]
#             Y_shuffled = Y[indices]
#
#             for i in range(0, n_samples, self.batch_size):
#                 # Get mini-batch
#                 X_batch = X_shuffled[i:i + self.batch_size]
#                 Y_batch = Y_shuffled[i:i + self.batch_size]
#
#                 # Predictions
#                 Y_pred = np.dot(X_batch, self.W) + self.b
#
#                 # Compute gradients
#                 error = Y_pred - Y_batch
#                 grad_W = (np.dot(X_batch.T, error) + self.regularization * self.W) / self.batch_size
#                 grad_b = np.mean(error)
#
#                 # Update weights and bias
#                 self.W -= self.lr * grad_W
#                 self.b -= self.lr * grad_b
#
#             # Compute loss for convergence check
#             Y_pred_all = np.dot(X, self.W) + self.b
#             loss = np.mean((Y_pred_all - Y) ** 2) + 0.5 * self.regularization * np.sum(self.W ** 2)
#
#             # Check for convergence
#             if abs(prev_loss - loss) < self.delta_converged:
#                 break
#             prev_loss = loss
#
#             steps += 1
#
#     def predict(self, X):
#         if self.W is None or self.b is None:
#             raise ValueError("Model is not fitted yet.")
#         return np.dot(X, self.W) + self.b