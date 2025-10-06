# Projeto Perceptron - Classificação com a base Spambase

## Introdução
Este projeto implementa um Perceptron simples em Python usando apenas NumPy para classificar e-mails como spam ou não-spam a partir da base de dados Spambase. O objetivo é reproduzir a lógica de treinamento e avaliação apresentada pela professora Thissy, para que possamos compreender um modelo simples de rede neural artificial.

## Descrição do modelo
- Arquitetura: Perceptron de camada única (linear). O modelo calcula uma soma ponderada das características e aplica uma função de ativação degrau bipolar.
- Função de ativação: step_function(soma) -> +1 se soma >= 0, caso contrário -1.
- Regra de atualização: para cada amostra (x_i, y_i):
  - y_pred = step_function(pesos · x_i + bias)
  - erro = y_i - y_pred
  - pesos <- pesos + lr * erro * x_i
  - bias <- bias + lr * erro
- Taxa de aprendizado: lr = 0.4
- Épocas de treinamento (n_epochs): 10
- Inicialização: pesos inicializados de forma determinística e não-nula (template repetido e truncado para a dimensão correta), bias inicial = 0.5. Os rótulos originais (0/1) são convertidos para -1/+1.

## Base de dados utilizada
- Arquivo: `spambase/spambase.data` (formato CSV; 57 atributos numéricos + 1 rótulo por linha).
- Linhas válidas encontradas: 4601. Valores de rótulo: 0 e 1.
- Conversão de rótulos: 0 (não-spam) -> -1, 1 (spam) -> +1.

Exemplo (primeiras 5 linhas do arquivo):

| # | primeiros 5 atributos (amostra) | label (orig) | label (mapeado) |
|---:|---|---:|---:|
| 1 | 0, 0.64, 0.64, 0, 0.32 | 1 | +1 |
| 2 | 0.21, 0.28, 0.5, 0, 0.14 | 1 | +1 |
| 3 | 0.06, 0, 0.71, 0, 1.23 | 1 | +1 |
| 4 | 0, 0, 0, 0, 0.63 | 1 | +1 |
| 5 | 0, 0, 0, 0, 0.63 | 1 | +1 |

> Observação: a tabela acima mostra apenas os 5 primeiros atributos para facilitar a visualização. Cada linha do dataset possui 57 atributos numéricos seguidos do rótulo.

## Processo de treinamento e resultados
- Divisão usada (para reproduzir o selecionado): os primeiros 40 exemplos válidos são usados para treino; os próximos 30 para teste.
- Hiperparâmetros: lr = 0.4, n_epochs = 10.
- Inicialização determinística dos pesos para reprodutibilidade.

Treinamento executado localmente (rodando `python3 perceptron.py`) produziu:
- Treino: 40 exemplos, Teste: 30 exemplos
- O código imprime o vetor de pesos completo após cada atualização e o bias atualizado.
- Resultado observado na última execução:
  - Pesos finais e bias impressos ao final do treino.
  - Acurácia no conjunto de teste: 1.0000 (30/30)

## Discussão crítica
- Vantagens:
  - Implementação simples.
  - Uso apenas de NumPy mantém dependências mínimas.
  - Inicialização determinística garante reprodutibilidade.
- Limitações:
  - Perceptron linear não separável pode não convergir para conjuntos mais complexos; modelo limitado para problemas não-lineares.
  - Split fixo (primeiros 25/15) não valida generalização de forma robusta — idealmente usar shuffle + cross-validation.
  - Atualização usa `erro = y_i - y_pred`, gerando fatores ±2 em atualizações (atualizações relativamente grandes); isso é intencional na implementação atual, mas difere de variações que usam atualizações condicionais com passo lr * y_i * x_i.
  - Parser atual ignora linhas com menos de 58 campos; um parser mais tolerante (ex.: np.genfromtxt) pode ser usado, porém com impacto de desempenho.

Possíveis melhorias:
- Embaralhar os dados com uma semente fixa antes do split (np.random.RandomState(seed).shuffle) para obter uma avaliação mais representativa.
- Trocar a regra de atualização por uma versão clássica (ex.: se y_pred != y_i: pesos += lr * y_i * x_i) para controlar magnitude do passo e compará-las.
- Avaliar com k-fold cross-validation e métricas além de acurácia (precision, recall, F1) — especialmente importante em datasets desbalanceados.
- Substituir parsing manual por `np.loadtxt`/`np.genfromtxt` com tratamento de exceções para simplificar o código.

## Conclusão
O projeto demonstra a implementação de um Perceptron simples aplicado à base Spambase, preservando a lógica para compreensao pedagógica incial. A execução local mostrou que, com os primeiros 40 exemplos para treino e os 30 seguintes para teste, o classificador alcançou acurácia de 100% no conjunto de teste usado, resultado que pode ser enganoso devido ao split fixo e pequeno tamanho de treino/teste. Para avaliar, recomenda-se embaralhar e usar validação cruzada e testar variações da regra de atualização.