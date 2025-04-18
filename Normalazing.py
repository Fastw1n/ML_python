from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
import arff
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from statsmodels.nonparametric.smoothers_lowess import lowess


with open('minifigs_data__preprocessed.arff', 'r', encoding='utf-8') as f:
    dataset = arff.load(f)

columns = [attr[0] for attr in dataset['attributes']]
data = dataset['data']
df = pd.DataFrame(data, columns=columns)

df = df.drop(columns=['ID', 'Name'])

categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, dtype=int)  

X = df.drop(columns=[col for col in df.columns if "Subtheme_" in col])  # Все признаки, кроме целевых
y = df[[col for col in df.columns if "Subtheme_" in col]] 

print("Данные:")
print(X.head())
print("--------------------")
print("Целевой признак")
print(y.head())
print("--------------------")




def normalize(d):
    return (d-d.min())/(d.max()-d.min())

X['Sets'] = normalize(X['Sets'])
X['Value, €'] = normalize(X['Value, €'])
X['Growth, %'] = normalize(X['Growth, %'])
X['Year'] = normalize(X['Year'])

X_train, X_pred, y_train, y_pred = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_pred, y_pred, test_size=0.5, random_state=42)

# print("Тренировочные данные:")
# print(X_train.head)
# print("--------------------")
# print("Тестовые данные:")
# print(X_test.head())
# print("--------------------")
# print("Валид данные:")
# print(X_val.head())

#----------------------------------------МЕТОД_БЛИЖ_СОСЕДОВ----------------------------------------------------------------------------------
class KNearestNeighbors:
    def __init__(self, metric='euclidean', kernel='uniform', fixed_window=True, k=5, h=1.0):
        self.metric = metric
        self.kernel = kernel
        self.fixed_window = fixed_window
        self.k = k
        self.h = h
        self.X_train = None
        self.y_train = None
        self.weights = None

    def fit(self, X, y, weights=None):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.weights = np.ones(len(self.X_train)) if weights is None else np.array(weights)

    def predict(self, X):
        X = np.array(X)
        distances = cdist(X, self.X_train, metric=self.metric)
        predictions = []
        for i, dist in enumerate(distances):
            predictions.append(self._predict_single(dist))
        return np.array(predictions)

    def _predict_single(self, distances):
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_weights = self.weights[sorted_indices]
        sorted_y = self.y_train[sorted_indices]

        if self.fixed_window:
            mask = sorted_distances <= self.h
        else:
            mask = np.zeros_like(sorted_distances, dtype=bool)
            mask[:self.k] = True

        relevant_distances = sorted_distances[mask]
        relevant_y = sorted_y[mask]
        relevant_weights = sorted_weights[mask]

        if relevant_y.shape[0] == 0:  
            return np.ones(self.y_train.shape[1]) / self.y_train.shape[1] 

        kernel_values = self._apply_kernel(relevant_distances / self.h)

        weighted_votes = relevant_weights * kernel_values
        probabilities = np.dot(relevant_y.T, weighted_votes)

        if probabilities.sum() > 0:
            probabilities /= probabilities.sum()

        return probabilities

    def _apply_kernel(self, u):
        if self.kernel == 'uniform':
            return np.where(np.abs(u) <= 1, 1, 0)
        elif self.kernel == 'gaussian':
            return np.exp(-u**2 / 2)
        elif self.kernel == 'triangular':
            return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)
        elif self.kernel == 'polynomial':
            a, b = 2, 2  
            return np.where(np.abs(u) <= 1, (1 - np.abs(u)**a)**b, 0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")


def normalize_predictions(predictions):
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    
    sums = predictions.sum(axis=1, keepdims=True)
    
    sums[sums == 0] = 1  
    return predictions / sums 

hyperparameter_results = []
metrics = ['euclidean', 'cosine', 'minkowski']
kernels = ['uniform', 'gaussian', 'triangular', 'polynomial']
k_values = range(1, 21)  
h_values = np.linspace(0.1, 5.0, 20)  

best_params = None
best_loss = float('inf')

# Перебор всех комбинаций
for metric in metrics:
    for kernel in kernels:
        for fixed_window in [True, False]:
            for k in k_values:  
                for h in h_values: 
                    knn = KNearestNeighbors(
                        metric=metric,
                        kernel=kernel,
                        fixed_window=fixed_window,
                        k=k,
                        h=h if fixed_window else 1.0
                    )
                    knn.fit(X_train, y_train)
                    predictions = knn.predict(X_val)
                    normalized_predictions = normalize_predictions(predictions)
                    loss = log_loss(Y_val, normalized_predictions)

                    if loss < best_loss:
                        best_loss = loss
                        best_params = {
                            'metric': metric,
                            'kernel': kernel,
                            'fixed_window': fixed_window,
                            'k': k,
                            'h': h
                        }
                    # Сохранение гиперпараметров и результата
                    hyperparameter_results.append({
                        'metric': metric,
                        'kernel': kernel,
                        'fixed_window': fixed_window,
                        'k': k,
                        'h': h,
                        'log_loss': loss
                    })

# Вывод лучших параметров 
#Val Best Params: {'metric': 'euclidean', 'kernel': 'polynomial', 'fixed_window': True, 'k': 1, 'h': np.float64(1.6473684210526318)}
#Val Best log_loss: 2.6793892814445357
print("Val Best Params:", best_params)
print("Val Best log_loss:", best_loss)
results_df = pd.DataFrame(hyperparameter_results)
results_df.to_csv('hyperparameter_results.csv', index=False)

# Построение графика
fixed_window = best_params['fixed_window']
param_range = h_values if fixed_window else k_values
train_losses = []
val_losses = []

for param in param_range:
    knn = KNearestNeighbors(
        metric='euclidean',
        kernel='polynomial',
        fixed_window=True,
        k=param if not fixed_window else 1,
        h=param if fixed_window else np.float64(1.6473684210526318)
    )
    knn.fit(X_train, y_train)
    train_predictions = knn.predict(X_train)
    val_predictions = knn.predict(X_val)
    train_losses.append(log_loss(y_train, normalize_predictions(train_predictions)))
    val_losses.append(log_loss(Y_val, normalize_predictions(val_predictions)))

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_losses, label='Train Loss', marker='o')
plt.plot(param_range, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Number of Neighbors (k)' if not fixed_window else 'Window Width (h)')
plt.ylabel('Log Loss')
plt.title('Error Dependency on Hyperparameter')
plt.legend()
plt.grid()
plt.show()


#--------------------------------------------------------------------ИНИЦИАЛИЗАЦИЯ----------------------------------------------------------        
knn = KNearestNeighbors(
    metric=best_params['metric'],  # Метрика расстояния: 'euclidean', 'cosine', 'minkowski'
    kernel=best_params['kernel'],   # Ядро: 'uniform', 'gaussian', 'triangular', 'polynomial'
    fixed_window=True,
    k=1,
    h=np.float64(1.6473684210526318)             # Радиус окна (для фиксированного окна)
)

knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print("--------------------")
print("Предсказания:", predictions)
print("--------------------")
normalized_predictions = normalize_predictions(predictions)

# Вычисление log_loss
loss = log_loss(Y_test, normalized_predictions)
print("Test Log Loss:", loss)

# Вычисление точности (accuracy)
y_true = np.argmax(Y_test.values, axis=1)
y_pred_labels = np.argmax(normalized_predictions, axis=1)
acc = accuracy_score(y_true, y_pred_labels)
print("Test Accuracy:", acc)

#--------------------------------------------------------------------LOWLESS----------------------------------------------------------        
# def apply_lowess(X, frac=0.2):
#     smoothed_data = {}
#     for column in X.columns:
#         smoothed = lowess(X[column], np.arange(len(X[column])), frac=frac)[:, 1]
#         residuals = X[column] - smoothed
#         smoothed_data[column] = residuals  # Остатки как аномалии
#     return pd.DataFrame(smoothed_data)

# X_lowess = apply_lowess(X_train)
# def calculate_weights(X_lowess, threshold=2.0):
#     # Вычисляем аномалии: объекты с большими остатками считаем более весомыми
#     weights = (np.abs(X_lowess) > threshold).sum(axis=1) + 1  # Минимальный вес = 1
#     return weights

# weights = calculate_weights(X_lowess)
# knn.fit(X_train, y_train, weights=weights)
# # Без взвешивания
# knn.fit(X_train, y_train)
# predictions_no_weights = knn.predict(X_test)
# normalized_predictions_no_weights = normalize_predictions(predictions_no_weights)
# loss_no_weights = log_loss(Y_test, normalized_predictions_no_weights)
# print("Test Log Loss без взвешивания:", loss_no_weights)

# # С взвешиванием
# knn.fit(X_train, y_train, weights=weights)
# predictions_with_weights = knn.predict(X_test)
# normalized_predictions_with_weights = normalize_predictions(predictions_with_weights)
# loss_with_weights = log_loss(Y_test, normalized_predictions_with_weights)
# print("Test Log Loss со взвешиванием:", loss_with_weights)
# print(f"Разница в Log Loss: {loss_no_weights - loss_with_weights}")


