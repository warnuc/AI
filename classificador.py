from sklearn.linear_model import LogisticRegression
import numpy as np

# Dados de exemplo: números e suas "respostas" (1 = positivo, 0 = negativo)
X = np.array([[1], [2], [3], [-1], [-2], [-3]])  # Números
y = np.array([1, 1, 1, 0, 0, 0])  # 1 pra positivo, 0 pra negativo

# Criando o modelo (como se fosse uma mini-IA)
modelo = LogisticRegression()

# Treinando o modelo com os dados
modelo.fit(X, y)

# Testando com novos números
teste = np.array([[4], [-5], [0]])
predicoes = modelo.predict(teste)

# Mostrando os resultados
for num, pred in zip(teste, predicoes):
    print(f"Número {num[0]} é {'positivo' if pred == 1 else 'negativo'}")
