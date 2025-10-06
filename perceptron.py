import numpy as np

# Função de ativação (degrau bipolar: -1 ou +1)
def step_function(x):
    return 1 if x >= 0 else -1
# Ler dataset Spambase a partir do arquivo local
data_path = 'spambase/spambase.data'
parsed = []
with open(data_path, 'r', encoding='utf-8') as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        parts = line.split(',')
        # aceitar linhas com pelo menos 58 campos e truncar caso tenha mais
        if len(parts) >= 58:
            vals = list(map(float, parts[:58]))
            parsed.append(vals)
        else:
            # ignorar linhas mal formatadas
            continue

if len(parsed) == 0:
    raise RuntimeError(f'Nenhuma linha válida encontrada em {data_path}.')

data_arr = np.array(parsed, dtype=float)
X_all = data_arr[:, :-1]
y_all = data_arr[:, -1].astype(int)

# Mapear labels 0 -> -1, 1 -> +1
y_all = np.where(y_all == 0, -1, 1)

# Usar os primeiros 40 exemplos para treino e os 30 seguintes para teste
X = X_all[:40]
y = y_all[:40]

X_teste = X_all[40:40+30]
Y_teste = y_all[40:40+30]

# Hiperparâmetros (mantidos próximos do original)
lr = 0.4
n_epochs = 10

# Inicializar pesos e bias com valores não-zero determinísticos
# (mantemos valores pequenos e variados para não usar zeros)
init_template = [0.4, -0.6, 0.6, -0.2, 0.1]
repeats = (X.shape[1] + len(init_template) - 1) // len(init_template)
pesos = np.array((init_template * repeats)[: X.shape[1]], dtype=float)
bias = 0.5

print("Treino: 25 exemplos, Teste: 15 exemplos")
print("Pesos iniciais:", pesos, "Bias inicial:", bias)

# Treinamento (regra do Perceptron; preserva prints do fluxo original)
for epoca in range(n_epochs):
    print(f"\nÉpoca {epoca+1}")
    for x_i, y_i in zip(X, y):
        soma = np.dot(pesos, x_i) + bias
        y_pred = step_function(soma)
        erro = y_i - y_pred

        # Atualização dos pesos e bias
        pesos += lr * erro * x_i
        bias += lr * erro

        print(
            f"Entrada: {x_i[:5]}... (mostrando 5 primeiros atributos), "
            f"Esperado: {y_i}, Previsto: {y_pred}, Erro: {erro}"
        )
        # Mostrar valor de cada peso atualizado (vetor completo)
        print(
            "Pesos atualizados:",
            np.array2string(pesos, precision=6, floatmode='fixed')
        )
        print("Bias atualizado:", bias)
    print("Novos pesos:", pesos, "Novo bias:", bias)

print("\nPesos finais:", pesos, "Bias final:", bias)

# Avaliar em conjunto de teste
print("\n--- Predições (conjunto de teste) ---")
acertou = 0
exemplo_teste = 0
# usar zip exatamente como na estrutura da professora
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
