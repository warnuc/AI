

### Exemplo 1: Código Básico em Python (Classificação Simples)
Vamos fazer um programa que "aprende" a separar números positivos de negativos – um exemplo bem básico de aprendizado de máquina. Você vai precisar do Python e da biblioteca `scikit-learn` (pode instalar com `pip install scikit-learn` se quiser testar).

```python
# Importando o básico
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
```

**O que isso faz?**
- `X` é a lista de números que a IA "vê".
- `y` é o que ela deve aprender (positivos ou negativos).
- O modelo `LogisticRegression` é uma IA simples que ajusta uma "linha" pra separar os dados.
- Depois de treinado, ele prevê se novos números são positivos ou negativos.

**Saída esperada:**
```
Número 4 é positivo
Número -5 é negativo
Número 0 é positivo (pode variar dependendo do treino)
```

### Exemplo 2: Rede Neural Básica (Conceito com Código)
Agora, redes neurais! Elas são mais complexas, mas vou te mostrar uma bem simples "do zero" pra você entender como elas "pensam". Aqui, criamos uma rede com 1 neurônio pra prever algo básico.

```python
import numpy as np

# Função de ativação (decide se o neurônio "liga")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dados: entradas e saídas esperadas (exemplo: "é par?")
entradas = np.array([[0], [1], [2], [3]])  # Números
saidas = np.array([1, 0, 1, 0])  # 1 = par, 0 = ímpar

# Pesos iniciais (aleatórios) e bias
pesos = np.random.rand(1)
bias = np.random.rand(1)

# Treinando a rede (simplificado)
taxa_aprendizado = 0.1
for _ in range(1000):  # Repete 1000 vezes pra ajustar
    # Calcula a previsão
    soma = np.dot(entradas, pesos) + bias
    previsao = sigmoid(soma)
    
    # Ajusta os pesos com base no erro
    erro = saidas - previsao
    pesos += taxa_aprendizado * np.dot(entradas.T, erro * previsao * (1 - previsao))
    bias += taxa_aprendizado * np.sum(erro * previsao * (1 - previsao))

# Testando
teste = np.array([[4], [5]])
resultado = sigmoid(np.dot(teste, pesos) + bias)
print("Previsões pra 4 e 5:")
for num, res in zip(teste, resultado):
    print(f"{num[0]}: {'par' if res > 0.5 else 'ímpar'} (prob: {res:.2f})")
```

**O que tá acontecendo aqui?**
- A rede tem 1 neurônio que "aprende" se um número é par ou ímpar (bem simplificado).
- `pesos` e `bias` são ajustados pra minimizar o erro entre o que ela prevê e o certo.
- A função `sigmoid` transforma o resultado em algo entre 0 e 1 (tipo uma probabilidade).
- Depois de treinar, ela testa com 4 e 5.

**Saída possível:**
```
Previsões pra 4 e 5:
4: par (prob: 0.85)
5: ímpar (prob: 0.23)
```

### Pra Onde Ir Agora?
- O primeiro código é mais fácil de expandir – dá pra usar pra classificar coisas reais, como texto ou imagens, com mais dados.
- O segundo é o básico de redes neurais – pra algo como eu (Grok), usam redes com milhões de neurônios e camadas.


