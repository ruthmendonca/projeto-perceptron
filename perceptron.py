import numpy as np

#Função de ativação (degrau bipolar: -1 ou +1)
def step_function(x):
    return 1 if x >= 0 else -1
#ler dataset
data_path = 'spambase/spambase.data'
parsed = []
with open(data_path, 'r', encoding='utf-8') as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        parts = line.split(',')
        #aceitar linhas com pelo menos 58 campos e truncar caso tenha mais
        if len(parts) >= 58:
            vals = list(map(float, parts[:58]))
            parsed.append(vals)
        else:
            #ignorar linhas mal formatadas
            continue

if len(parsed) == 0:
    raise RuntimeError(f'Nenhuma linha válida encontrada em {data_path}.')

data_arr = np.array(parsed, dtype=float)
X_all = data_arr[:, :-1]
y_all = data_arr[:, -1].astype(int)

#mapear labels 0 -> -1, 1 -> +1
y_all = np.where(y_all == 0, -1, 1)

#embaralhar os dados para misturar as classes
np.random.seed(42) 
indices = np.random.permutation(len(X_all))
X_all = X_all[indices]
y_all = y_all[indices]

#normalizar os dados (StandardScaler manual)
mean_X = np.mean(X_all, axis=0)
std_X = np.std(X_all, axis=0)
#evitar divisão por zero
std_X = np.where(std_X == 0, 1, std_X)
X_all = (X_all - mean_X) / std_X

X = X_all[:200]  #200 exemplos para treino
y = y_all[:200]

X_teste = X_all[200:300]  #100 exemplos para teste
Y_teste = y_all[200:300]

# Hiperparâmetros otimizados
lr = 0.1  #taxa de aprendizagem
n_epochs = 20  #épocas

#inicialização pesos com valores aleatórios pequenos 
np.random.seed(42)
#inicialização normal com desvio padrão um pouco maior
pesos = np.random.normal(0, 0.1, X.shape[1])
bias = 0.5  #Bias inicial maior

print(f"Treino: {len(X)} exemplos, Teste: {len(X_teste)} exemplos")
print("Distribuição de classes no treino:", np.bincount(y + 1))
print("Distribuição de classes no teste:", np.bincount(Y_teste + 1))
print("Pesos iniciais (Normal std=0.1):", pesos[:5], "... Bias inicial:", bias)

#treinamento 
for epoca in range(n_epochs):
    print(f"\nÉpoca {epoca+1}")
    for x_i, y_i in zip(X, y):
        soma = np.dot(pesos, x_i) + bias
        y_pred = step_function(soma)
        erro = y_i - y_pred
        pesos += lr * erro * x_i
        bias += lr * erro

        print(
            f"Entrada: {x_i[:5]}... (mostrando 5 primeiros atributos), "
            f"Esperado: {y_i}, Previsto: {y_pred}, Erro: {erro}"
        )
        #valor de cada peso atualizado (vetor completo)
        print(
            "Pesos atualizados:",
            np.array2string(pesos, precision=6, floatmode='fixed')
        )
        print("Bias atualizado:", bias)
    print("Novos pesos:", pesos, "Novo bias:", bias)

print("\nPesos finais:", pesos, "Bias final:", bias)

print("\n--- Predições (conjunto de teste) ---")
acertou = 0
exemplo_teste = 0

for x_i, y_i in zip(X_teste, Y_teste):
    soma = np.dot(pesos, x_i) + bias
    y_pred = step_function(soma)
    print(f"Entrada: {x_i} -> Saída prevista: {y_pred}")
    erro = y_i - y_pred
    if erro == 0:
        acertou += 1
    exemplo_teste += 1

acuracia = acertou / exemplo_teste
print(f"Acuracia: {acuracia:.4f} ({acertou}/{exemplo_teste})")
