# Projeto Perceptron - Classificação de Spam com a base Spambase

## Introdução
Este projeto implementa um Perceptron simples em Python usando apenas NumPy para classificar e-mails como spam ou não-spam a partir da base de dados Spambase. O objetivo é compreender e aplicar um modelo simples de rede neural artificial, demonstrando os fundamentos do aprendizado de máquina supervisionado. O Perceptron, criado por Frank Rosenblatt em 1957, é um dos primeiros modelos de neurônio artificial e serve como base para redes neurais mais complexas. Neste projeto, exploramos sua capacidade de classificação binária em um problema real de detecção de spam.

## Descrição do modelo
- **Arquitetura**: Perceptron de camada única (linear). O modelo calcula uma soma ponderada das características de entrada e aplica uma função de ativação degrau bipolar para produzir a classificação final.
- **Função de ativação**: step_function(soma) → +1 se soma ≥ 0, caso contrário -1 (função degrau bipolar).
- **Regra de atualização**: Para cada amostra (x_i, y_i):
  - y_pred = step_function(pesos · x_i + bias)
  - erro = y_i - y_pred
  - pesos ← pesos + lr × erro × x_i
  - bias ← bias + lr × erro
- **Taxa de aprendizado**: lr = 0.1 (otimizada para melhor convergência)
- **Épocas de treinamento**: n_epochs = 20 (aumentadas para melhor aprendizado)
- **Inicialização**: 
  - Pesos: distribuição normal com média 0 e desvio padrão 0.1
  - Bias inicial: 0.5
  - Seed fixo (42) para reprodutibilidade
- **Pré-processamento**: Normalização dos dados (StandardScaler) para média 0 e desvio padrão 1
- **Conversão de rótulos**: 0 (não-spam) → -1, 1 (spam) → +1

## Base de dados utilizada
- **Arquivo**: `spambase/spambase.data` (formato CSV; 57 atributos numéricos + 1 rótulo por linha)
- **Total de linhas válidas**: 4601 exemplos
- **Classes**: 0 (não-spam) e 1 (spam)
- **Conversão de rótulos**: 0 (não-spam) → -1, 1 (spam) → +1
- **Pré-processamento aplicado**: 
  - Embaralhamento dos dados com seed=42
  - Normalização (StandardScaler): média=0, desvio padrão=1
  - Divisão: 200 exemplos treino, 100 exemplos teste

Exemplo (primeiras 5 linhas após normalização):

| # | Atributos normalizados (5 primeiros) | Label original | Label mapeado |
|---:|---|---:|---:|
| 1 | [-0.342, -0.165, -0.557, -0.047, -0.464] | 1 | +1 |
| 2 | [0.804, -0.165, 0.852, -0.047, -0.464] | 0 | -1 |
| 3 | [-0.342, -0.165, -0.557, -0.047, 1.067] | 1 | +1 |
| 4 | [-0.342, -0.026, -0.199, -0.047, 0.636] | 1 | +1 |
| 5 | [2.048, 0.114, 1.606, -0.047, -0.464] | 1 | +1 |

> **Observação**: A tabela mostra apenas os 5 primeiros atributos normalizados para facilitar a visualização. Cada linha possui 57 atributos numéricos seguidos do rótulo.

## Processo de treinamento e resultados obtidos
- **Divisão dos dados**: 200 exemplos para treino, 100 exemplos para teste (após embaralhamento)
- **Hiperparâmetros finais**: lr = 0.1, n_epochs = 20
- **Inicialização**: Pesos com distribuição normal (μ=0, σ=0.1), bias = 0.5
- **Distribuição de classes**:
  - Treino: [95 não-spam, 105 spam] - balanceada
  - Teste: [60 não-spam, 40 spam] - relativamente balanceada

**Resultados obtidos:**
- **Acurácia final**: **87% (87/100)**
- **Pré-processamento aplicado**: 
  - Normalização StandardScaler (essencial para convergência)
  - Embaralhamento com seed=42 para reprodutibilidade
- **Convergência**: O modelo treinou por todas as 20 épocas, atualizando pesos a cada exemplo
- **Saída do treinamento**: O código imprime detalhadamente:
  - Entrada normalizada (5 primeiros atributos)
  - Valor esperado vs predito
  - Erro calculado  
  - Pesos e bias atualizados após cada exemplo

## Discussão crítica

### Vantagens:
- **Implementação simples**: Código claro e educacional usando apenas NumPy
- **Reprodutibilidade**: Seed fixo (42) garante resultados consistentes
- **Pré-processamento adequado**: Normalização StandardScaler melhora significativamente a convergência
- **Dataset balanceado**: Distribuição equilibrada de classes facilita o aprendizado
- **Alta acurácia**: 87% é um resultado excelente para um modelo linear simples

### Limitações:
- **Modelo linear**: Perceptron não consegue resolver problemas não-linearmente separáveis
- **Sensibilidade à escala**: Sem normalização, o modelo não converge adequadamente
- **Função de ativação simples**: Degrau bipolar não permite gradientes suaves
- **Sem regularização**: Susceptível a overfitting em datasets pequenos
- **Split fixo**: Divisão treino/teste não rotativa pode mascarar problemas de generalização

### Possíveis melhorias:
- **Validação cruzada**: Implementar k-fold para avaliação mais robusta
- **Métricas adicionais**: Precision, Recall, F1-score para análise completa
- **Perceptron multicamadas**: MLPs para capturar relações não-lineares
- **Outros algoritmos**: SVM, Random Forest para comparação de performance
- **Análise de erro**: Visualização dos casos mal classificados
- **Otimização de hiperparâmetros**: Grid search para lr e n_epochs
- **Early stopping**: Parada antecipada baseada em validação

## Conclusão
O projeto demonstra com sucesso a implementação de um Perceptron simples aplicado à classificação de spam na base Spambase, alcançando uma acurácia de **87%** no conjunto de teste. A aplicação de técnicas de pré-processamento, especialmente a normalização dos dados e o embaralhamento com seed fixo, foram fundamentais para o sucesso do modelo. 

Os resultados obtidos validam a eficácia do Perceptron como classificador linear para problemas binários, mesmo mantendo a simplicidade da implementação usando apenas NumPy. A alta acurácia demonstra que, com os ajustes adequados nos hiperparâmetros (taxa de aprendizado de 0.1 e 20 épocas) e pré-processamento correto, modelos simples podem produzir resultados satisfatórios em problemas reais.

Este projeto serve como uma excelente introdução aos conceitos fundamentais de redes neurais e aprendizado de máquina supervisionado, fornecendo uma base sólida para compreender algoritmos mais complexos. A experiência hands-on com implementação, otimização e avaliação de um classificador contribui significativamente para o entendimento prático dos desafios e soluções em machine learning.