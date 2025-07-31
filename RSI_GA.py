# -----------------------------------------------------------------------------
# 1. Instalação e Importação de Bibliotecas
# -----------------------------------------------------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
import datetime
import math # Importado para checagem de números finitos


# -----------------------------------------------------------------------------
# 2. Definições de Parâmetros e Coleta de Dados
# -----------------------------------------------------------------------------

# Parâmetros do ativo e período de backtesting
TICKER = "ITSA4.SA"
START_DATE = "2020-01-01"
END_DATE = "2025-06-30"

# Função para obter dados históricos do Yahoo Finance
def get_historical_data(ticker, start, end):
    """
    Coleta dados históricos de preço de um ticker usando yfinance.
    """
    try:
        data = yf.download(ticker, start=start, end=end, progress=False) # progress=False para output mais limpo
        if data.empty:
            raise ValueError(f"Não foram encontrados dados para {ticker} no período especificado.")
        print(f"Dados históricos para {ticker} coletados de {start} a {end}.")
        return data
    except Exception as e:
        print(f"Erro ao coletar dados para {ticker}: {e}")
        return pd.DataFrame()

df = get_historical_data(TICKER, START_DATE, END_DATE)

if df.empty:
    print("Não foi possível prosseguir sem dados. Encerrando o script.")
    # No VS Code, podemos usar sys.exit() para uma saída controlada.
    import sys
    sys.exit(1)

# VERIFICAÇÃO CRÍTICA: Garante que a coluna 'Close' existe após o download
if 'Close' not in df.columns:
    print("Erro: A coluna 'Close' não foi encontrada nos dados baixados do Yahoo Finance. Verifique o ticker ou o período.")
    import sys
    sys.exit(1)

# Definindo intervalos para os parâmetros do RSI para o Algoritmo Genético
# Estes intervalos definem o espaço de busca para o otimizador.
N_MIN, N_MAX = 5, 30          # Período 'n' do RSI (geralmente entre 5 e 30 dias)
V_RSI_MIN, V_RSI_MAX = 60, 90 # Nível de sobrecompra 'v_rsi' (geralmente entre 60 e 90)
C_RSI_MIN, C_RSI_MAX = 10, 40 # Nível de sobrevenda 'c_rsi' (geralmente entre 10 e 40)

print(f"Parâmetros de busca definidos: n [{N_MIN}-{N_MAX}], v_rsi [{V_RSI_MIN}-{V_RSI_MAX}], c_rsi [{C_RSI_MIN}-{C_RSI_MAX}].\n")


# -----------------------------------------------------------------------------
# 3. Função de Aptidão (Fitness Function) - Estratégia de Backtesting
# -----------------------------------------------------------------------------

# Cache para armazenar os resultados do RSI e evitar recálculos desnecessários
rsi_cache = {}

def calculate_rsi(data_close_series, window):
    """
    Calcula o RSI manualmente usando pandas.ewm para suavização de Wilder,
    com a inicialização da média móvel.

    Args:
        data_close_series (pd.Series): Série de preços de fechamento.
        window (int): Período para o cálculo do RSI.

    Returns:
        pd.Series: Série com os valores de RSI calculados.
    """
    close_prices = data_close_series.squeeze()

    if not isinstance(close_prices, pd.Series) or close_prices.empty:
        return pd.Series(dtype=float)

    # Se não houver dados suficientes para a janela, retorna NaN para todos os pontos.
    if len(close_prices) < window:
        return pd.Series(np.nan, index=close_prices.index)

    # Calcula as diferenças diárias
    delta = close_prices.diff(1)

    # Separa ganhos e perdas
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0) # Perdas são valores positivos

    # Calcula a média móvel exponencial (RMA) para o alisamento de Wilder.
    # adjust=False é crucial para replicar o RSI padrão de Wilder.
    # min_periods=window garante que a média só comece após 'window' períodos, gerando NaNs antes.
    avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()

    # Calcula Relative Strength (RS)
    # np.seterr para gerenciar avisos/erros de divisão por zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss

    # Calcula RSI inicial (pode conter NaNs, inf)
    rsi_calculated = 100 - (100 / (1 + rs))

    # Tratamento explícito de casos especiais para garantir a robustez e valores corretos:
    # 1. Se AvgLoss é 0 (e há algum ganho), RSI deve ser 100
    rsi_calculated = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi_calculated)
    # 2. Se AvgGain é 0 (e há alguma perda), RSI deve ser 0
    rsi_calculated = np.where((avg_gain == 0) & (avg_loss > 0), 0.0, rsi_calculated)
    # 3. Se ambos são 0 (sem movimento), RSI é 50 por convenção (ou NaN, dependendo da sua preferência)
    rsi_calculated = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi_calculated)

    # Converte o resultado para uma Series Pandas com o índice original
    return pd.Series(rsi_calculated, index=close_prices.index)


def evaluate_strategy(individual):
    """
    Função de aptidão: Calcula o retorno da estratégia de trading para um dado indivíduo (parâmetros RSI).
    Individual: (n, v_rsi, c_rsi)
    """
    n, v_rsi, c_rsi = int(individual[0]), int(individual[1]), int(individual[2])

    # Garante que os parâmetros estejam dentro dos limites esperados
    n = max(N_MIN, min(n, N_MAX))
    v_rsi = max(V_RSI_MIN, min(v_rsi, V_RSI_MAX))
    c_rsi = max(C_RSI_MIN, min(c_rsi, C_RSI_MAX))

    # Cria um DataFrame para a avaliação que terá os dados de preço e o RSI
    # Começa com uma cópia completa do DataFrame original
    df_eval = df.copy()

    # Utiliza o cache para cálculo do RSI
    if n not in rsi_cache:
        # Passa apenas a Series 'Close' para calculate_rsi
        rsi_values = calculate_rsi(df_eval['Close'], n)
        rsi_cache[n] = rsi_values # Armazena no cache
    else:
        rsi_values = rsi_cache[n]

    # Atribui o RSI calculado à coluna 'RSI' no df_eval.
    # Garante que a coluna 'RSI' exista e seja populada corretamente.
    # O .loc garante o alinhamento de índices e atribuição correta.
    df_eval.loc[rsi_values.index, 'RSI'] = rsi_values

    # --- INÍCIO DOS PRINTS DE DEPURAR PARA INSPEÇÃO ---
    # Descomente as linhas abaixo se o erro persistir e você estiver usando o debugger do VS Code
    # print(f"\nDEBUG (evaluate_strategy): n={n}, v_rsi={v_rsi}, c_rsi={c_rsi}")
    # print(f"DEBUG (evaluate_strategy): Colunas de df_eval antes de dropna: {df_eval.columns.tolist()}")
    # print(f"DEBUG (evaluate_strategy): df_eval.head() antes de dropna:\n{df_eval[['Close', 'RSI']].head()}")
    # print(f"DEBUG (evaluate_strategy): df_eval.tail() antes de dropna:\n{df_eval[['Close', 'RSI']].tail()}")
    # print(f"DEBUG (evaluate_strategy): df_eval.info() antes de dropna:")
    # df_eval.info()
    # --- FIM DOS PRINTS DE DEPURAR ---

    # Remove NaNs gerados pelo cálculo do RSI e dados iniciais
    # REMOVIDO inplace=True para maior robustez, reatribuindo o resultado
    df_eval = df_eval.dropna(subset=['RSI', 'Close'])


    if df_eval.empty:
        # Se não houver dados válidos após o cálculo do RSI e dropna, o retorno é zero ou muito baixo.
        return 0.0,

    position = 0          # 0: sem posição, 1: comprado
    total_return = 0.0
    buy_price = 0.0

    # Itera sobre os dados para simular a estratégia
    for i in range(len(df_eval)):
        current_rsi = df_eval['RSI'].iloc[i]
        current_close = df_eval['Close'].iloc[i]

        # Verificação explícita de NaN para os valores de RSI e preço
        if pd.isna(current_rsi) or pd.isna(current_close):
            continue # Pula o dia se o valor do RSI ou do preço for inválido

        # Sinal de Compra
        if current_rsi <= c_rsi and position == 0:
            position = 1
            buy_price = current_close

        # Sinal de Venda
        elif current_rsi >= v_rsi and position == 1:
            sell_price = current_close
            profit = sell_price - buy_price
            total_return += profit
            position = 0

    # Robustness check: Garante que o valor de aptidão (fitness) seja um número finito.
    # Caso total_return seja NaN ou Infinito, atribui um valor muito baixo.
    if not math.isfinite(total_return):
        total_return = -1e9 # Um valor penalidade muito baixo para resultados inválidos

    # Garante que o retorno seja um float primitivo do Python.
    # Isso é crucial para evitar o erro "truth value of a Series is ambiguous".
    total_return_final = float(total_return)

    return total_return_final, # A DEAP espera uma tupla de fitness


# -----------------------------------------------------------------------------
# 4. Configuração do Algoritmo Genético com DEAP
# -----------------------------------------------------------------------------

# Criação dos tipos de fitness e indivíduo
# FitnessMax indica que queremos maximizar o valor de fitness (o retorno)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Um Indivíduo é uma lista de parâmetros (n, v_rsi, c_rsi) com uma Fitness associada
creator.create("Individual", list, fitness=creator.FitnessMax)

# Configuração da Toolbox
toolbox = base.Toolbox()

# Atributos (genes) para cada parâmetro do indivíduo
# n: inteiro no intervalo [N_MIN, N_MAX]
toolbox.register("attr_n", random.randint, N_MIN, N_MAX)
# v_rsi: inteiro no intervalo [V_RSI_MIN, V_RSI_MAX]
toolbox.register("attr_v_rsi", random.randint, V_RSI_MIN, V_RSI_MAX)
# c_rsi: inteiro no intervalo [C_RSI_MIN, C_RSI_MAX]
toolbox.register("attr_c_rsi", random.randint, C_RSI_MIN, C_RSI_MAX)

# Indivíduo: Usa initIterate e uma função lambda para gerar a lista completa de atributos
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [toolbox.attr_n(), toolbox.attr_v_rsi(), toolbox.attr_c_rsi()])

# População: Uma lista de indivíduos
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores Genéticos

# Função de Avaliação (Fitness Function)
toolbox.register("evaluate", evaluate_strategy)

# Seleção: Torneio (tools.selTournament)
# 'tournsize' define o número de indivíduos que participam de cada torneio.
toolbox.register("select", tools.selTournament, tournsize=3)

# Cruzamento (Crossover): Uniform Crossover
# Este método é apropriado para cruzamento de atributos numéricos.
toolbox.register("mate", tools.cxUniform, indpb=0.5) # indpb é a probabilidade de troca de cada atributo

# Mutação: Gaussiana
# Muta cada atributo com uma probabilidade de indpb=0.5 (sua taxa de mutação).
# mu=0: centraliza a mutação em torno do valor atual do gene.
# sigma=1: desvio padrão da mutação (baixa intensidade para pequenas alterações).
# Garante que os valores permaneçam inteiros e dentro dos limites após a mutação.
def mutate_individual(individual, mu, sigma, indpb, min_n, max_n, min_v, max_v, min_c, max_c):
    """
    Função de mutação personalizada para garantir que os parâmetros permaneçam inteiros e dentro dos limites.
    """
    for i, gene in enumerate(individual):
        if random.random() < indpb:
            if i == 0: # n
                mutated_gene = gene + random.gauss(mu, sigma)
                individual[i] = max(min_n, min(max_n, int(round(mutated_gene))))
            elif i == 1: # v_rsi
                mutated_gene = gene + random.gauss(mu, sigma)
                individual[i] = max(min_v, min(max_v, int(round(mutated_gene))))
            elif i == 2: # c_rsi
                mutated_gene = gene + random.gauss(mu, sigma)
                individual[i] = max(min_c, min(max_c, int(round(mutated_gene))))
    return individual,

toolbox.register("mutate", mutate_individual, mu=0, sigma=1, indpb=0.5,
                 min_n=N_MIN, max_n=N_MAX, min_v=V_RSI_MIN, max_v=V_RSI_MAX, min_c=C_RSI_MIN, max_c=C_RSI_MAX)


# -----------------------------------------------------------------------------
# 5. Execução do Algoritmo Genético
# -----------------------------------------------------------------------------


# Parâmetros do AG
N_POP = 50 # Tamanho da população (número de indivíduos em cada geração)
N_GEN = 100 # Número máximo de gerações
CXPB = 0.7  # Probabilidade de cruzamento (crossover)
MUTPB = 0.3 # Probabilidade de mutação (aplicada após o cruzamento)

# Registra estatísticas
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Hall of Fame: Armazena o melhor indivíduo encontrado em todas as gerações
hof = tools.HallOfFame(1) # Armazena apenas o melhor de todos

# --- VERIFICAÇÃO CRÍTICA DO DATAFRAME GLOBAL 'df' ANTES DA EXECUÇÃO ---
if df.empty or 'Close' not in df.columns:
    print("FATAL ERROR: O DataFrame global 'df' está vazio ou faltando a coluna 'Close'.")
    print("Isso indica que a Parte 2 (coleta de dados) não foi executada com sucesso ou o DataFrame foi modificado/perdido.")
    print(f"Colunas atuais de 'df': {df.columns.tolist()}")
    print(f"Cabeçalho de 'df':\n{df.head()}")
    import sys
    sys.exit(1)
# --- FIM DA VERIFICAÇÃO CRÍTICA ---

# Cria a população inicial
population = toolbox.population(n=N_POP)

# Hotstart: Injeta a solução inicial (14, 70, 30) na população
hotstart_individual = creator.Individual([14, 70, 30])
# É crucial avaliar o hotstart individual *antes* de adicioná-lo ao Hall of Fame,
# ou de ser usado em operações que esperam um fitness válido.
# Isso garante que ele já tenha um 'fitness.values' quando as operações de DEAP começarem.
hotstart_individual.fitness.values = toolbox.evaluate(hotstart_individual)
population[0] = hotstart_individual # Substitui o primeiro indivíduo aleatório pelo hotstart

# Avalia a população inicial (apenas indivíduos sem fitness válido ainda)
print("Avaliando população inicial...")
invalid_ind = [ind for ind in population if not ind.fitness.valid]
fits = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fits):
    ind.fitness.values = fit

# Variáveis para a condição de término (20 gerações sem mudança no fitness)
best_fitness_history = []
stagnation_counter = 0
MAX_STAGNATION_GENERATIONS = 20

# Loop principal do Algoritmo Genético
logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + stats.fields

print("Iniciando loop de gerações...")
for gen in range(1, N_GEN + 1):
    # Seleciona os próximos indivíduos para a próxima geração
    # Utilizando selTournament para seleção dos pais
    offspring = toolbox.select(population, len(population))

    # Clona os indivíduos selecionados para evitar modificação direta
    offspring = list(map(toolbox.clone, offspring))

    # Aplica o cruzamento (crossover) nos indivíduos da prole
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Aplica a mutação nos indivíduos da prole
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Avalia os indivíduos com fitness inválido (após cruzamento/mutação)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fits = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fits):
        ind.fitness.values = fit

    # Substitui a população antiga pela nova prole
    population[:] = offspring

    # Atualiza o Hall of Fame com o melhor indivíduo
    hof.update(population)

    # Registra estatísticas
    record = stats.compile(population)
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    print(f"Geração {gen:3d} | Média: {record['avg']:.2f} | Max: {record['max']:.2f} | Melhor até agora: {hof[0].fitness.values[0]:.2f} (Params: {hof[0]})")

    # Condição de Término por Convergência
    current_best_fitness = hof[0].fitness.values[0]
    best_fitness_history.append(current_best_fitness)

    if gen > MAX_STAGNATION_GENERATIONS:
        # Compara o melhor fitness atual com o melhor de MAX_STAGNATION_GENERATIONS atrás
        # Adicione uma pequena tolerância para evitar sensibilidade a pequenas flutuações
        tolerance = 0.001 # 0.1% de tolerância
        # Verifica se o melhor fitness atual é 'igual' ou pior do que o de MAX_STAGNATION_GENERATIONS atrás
        if current_best_fitness <= best_fitness_history[gen - MAX_STAGNATION_GENERATIONS - 1] * (1 + tolerance):
             stagnation_counter += 1
        else:
            stagnation_counter = 0 # Reset se houver melhora

        if stagnation_counter >= MAX_STAGNATION_GENERATIONS:
            print(f"\nCondição de término: Fitness não mudou por {MAX_STAGNATION_GENERATIONS} gerações. Encerrando.")
            break
    else: # Reset no contador para o início se o hof melhorar
        if len(best_fitness_history) > 1 and current_best_fitness > best_fitness_history[-2]:
            stagnation_counter = 0
