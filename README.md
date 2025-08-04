# Tech-Challenge-2
# 📈 Otimização de Estratégias de Negociação Financeira com Algoritmos Genéticos

## 🚀 Descrição do Projeto

A crescente complexidade do mercado financeiro e a disponibilidade massiva de dados impulsionaram o desenvolvimento de abordagens quantitativas para a análise de ativos financeiros. Nesse contexto, a análise técnica se consolidou como um dos principais métodos utilizados para identificar padrões e tendências nos preços das ações, permitindo aos investidores tomarem decisões baseadas no comportamento passado dos ativos.

Este projeto implementa um Algoritmo Genético (AG) para otimizar os parâmetros de uma estratégia de negociação baseada no cruzamento de Médias Móveis (MM) Simples. O objetivo principal é **maximizar o lucro total** em um backtest para um ativo financeiro específico, utilizando dados históricos. O script abrange desde a aquisição de dados e a simulação da estratégia (backtesting) até a otimização via AG e a visualização detalhada dos resultados, incluindo o progresso do algoritmo genético.

A estratégia de cruzamento das médias móveis é relativamente simples. Essa técnica envolve calcular duas médias móveis: uma de curto prazo e outra de longo prazo. O cruzamento da média curta com a longa sinaliza tendências de alta ou baixa, mas os parâmetros ideais para otimizar retornos variam conforme diversos fatores, como setor econômico e volatilidade do ativo.

O desenvolvimento tecnológico e a ciência de dados permitem testes contínuos para identificar parâmetros que otimizem retornos. Diante desse cenário, este estudo tem como objetivo otimizar os parâmetros das duas médias móveis por meio da aplicação de algoritmo genético, buscando maximizar o retorno de investimento em ações da Itaú Sa no curto e médio prazo (swing trade). A metodologia proposta envolve a utilização de AG para identificar combinações de parâmetros que apresentem melhor desempenho, comparando os resultados obtidos com o método da busca exaustiva.

Este é um projeto ideal para estudos em ciência de dados aplicada ao mercado financeiro, combinando técnicas de otimização e análise técnica.

## 🤯 Definição do problema
### Descrição do problema
A estratégia operacional de compra e venda de ativos utilizando as médias móveis pressupõe a definição dois parâmetros, m e n:
SMA_Slow = representa a média móvel lenta de m períodos;
SMA_Fast = representa a média móvel rápida de n períodos;
Sujeito à restrição de m > n.
O problema consiste em definir qual o valor ótimo das duas médias móveis que otimizem o retorno (ganho de capital) sobre as ações da Itaú Sa (ITSA4) para operações de swing trade entre 01/01/2020 e 31/07/2025.

### Objetivos
Definir o valor ótimo das duas médias móveis que otimizem o retorno de operações de swing trade sobre o ativo ITSA4.

### Critérios de sucesso
Espera-se que os resultados obtidos a partir dos parâmetros otimizados sejam superiores aos resultados obtidos pelo método de otimização chamado “busca exaustiva” para o mesmo ativo ao longo do mesmo período que resultaram em ganhos da ordem de R$ 107.710,00.
Para conhecer mais sobre o método suas aplicações consulte o **Boletim de Mercado de Capitais da Unifor** no link https://unifor.br/nupe/boletim-mercado-de-capitais.

### Implementação
O código foi implementado em sete etapas, conforme descrito abaixo:
1.	Etapa 1: Instalação e importação das principais bibliotecas necessárias para o desenvolvimento do algoritmo;
2.	Etapa 2: Definições de parâmetros e obtenção dos dados. Os dados foram obtidos no Yahoo Finance. Foram definidos limite máximos e mínimos para cada uma das médias móveis rápida e lenda, sob a restrição de m>n, conforme a descrito no próprio código, uma vez que valores fora desses limites não fazem muito sentido no âmbito do mercado financeiro;
3.	Etapa 3: Definição da função de aptidão, que representa o retorno esperado do somatório do ganho de capital de todas as operações de compra e venda resultantes da aplicação da estratégia. O ganho de capital é calculado pela diferença entre o preço de fechamento na data da venda e o preço de fechamento na data da compra. O capital inicial investido foi de R$ 100.000,00.
4.	Etapa 4: Configuração e implementação do algoritmo genético. 
    a.	Foi definido um tamanho de população em cada geração igual a 100
    b.	Cada indivíduo representa uma solução do problema com 2 parâmetros: fast_p, slow_p. 
    c.	A população inicial foi criada aleatoriamente
    d.	A seleção de indivíduos para cruzamento utilizada foi o torneio
    e.	Para a função de cruzamento foi utilizado o crossover aritmético com taxa de 80%
    f.	Para a mutação foi aplicada uma probabilidade de 50% para ambos os genes, com intensidade de 20%
    g.	Foi estabelecido um limite máximo de 100 gerações

5.	Etapa 5: Visualização dos resultados. Nesta etapa foram desenvolvidos gráficos de candle mostrando as médias móveis e os pontos de compra e venda. Além disso, são apresentados os gráficos de curva de capital e de evolução do fitness do longo das gerações.

## ✨ Funcionalidades

*   **Aquisição de Dados Históricos**: Utiliza a robusta biblioteca `yfinance` para baixar dados de OHLCV (Abertura, Máxima, Mínima, Fechamento, Volume) de ativos da B3 (bolsa brasileira) ou outras bolsas.
*   **Resolve problema de colunas MultiIndex**: É muito comum no yfinance para múltiplos tickers ou para dados de ações. Caso não haja ao ajuste, o algoritmo não reconhece novas colunas geradas.
*   **Estratégia de Cruzamento de Médias Móveis**:
    *   **Sinal de Compra**: Gerado quando a Média Móvel Rápida cruza a Média Móvel Lenta de baixo para cima.
    *   **Sinal de Venda**: Gerado quando a Média Móvel Rápida cruza a Média Móvel Lenta de cima para baixo (sinaliza o fechamento da posição).
*   **Backtesting Robusto**: Simula o desempenho da estratégia de negociação com um capital inicial definido e custos de transação (ambos configuráveis), fornecendo um lucro total claro.
*   **Algoritmo Genético (AG)**:
    *   Otimiza os períodos das Médias Móveis (MM Rápida e MM Lenta) para encontrar a combinação mais lucrativa.
    *   A **função de aptidão (fitness function)** é baseada diretamente no lucro total obtido no backtest, buscando maximizá-lo.
    *   Implementa operadores genéticos fundamentais: seleção (por torneio), cruzamento (crossover aritmético) e mutação, para explorar eficientemente o espaço de soluções.
    *   Inclui **elitismo** para garantir que as melhores soluções encontradas em cada geração sejam preservadas na próxima.
*   **Visualização de Resultados**:
    *   Um **gráfico de candlestick interativo** (via `mplfinance`) exibe o histórico de preços do ativo, as Médias Móveis otimizadas e, de forma clara, os pontos de sinais de compra e venda.
    *   Um **gráfico de progresso do Algoritmo Genético** demonstra a evolução do "Melhor Lucro (Fitness)" por geração, permitindo visualizar a convergência e o desempenho da otimização ao longo do tempo.
*   **Resumo de Performance**: Ao final da execução, o script exibe o lucro total otimizado e o **capital final total** da simulação, fornecendo uma métrica clara do sucesso da estratégia.

## �� Como Usar

### Pré-requisitos

O script é escrito em Python e é ideal para ser executado em ambientes como Google Colab ou Jupyter Notebook, mas também pode ser rodado localmente.

As seguintes bibliotecas Python são necessárias:
*   `yfinance`
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `mplfinance`

Você pode instalá-las executando a primeira célula do script (o comando `!pip install` já está incluso no código):

```bash
!pip install yfinance pandas numpy matplotlib mplfinance -q
