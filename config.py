"""
config.py - Parâmetros centrais do projeto.

Tudo que é regra do cenário fica aqui (e não espalhado pelo código), para ser
fácil de documentar no relatório e de ajustar.
"""

# Semente fixa: rodar o gerador sempre produz exatamente a mesma base.
SEMENTE = 42

# Quantidade de chamados resolvidos mantidos no histórico dinâmico.
N_HISTORICO = 200

# Quantidade de chamados abertos criados na primeira inicialização, para demonstração.
N_ABERTOS_INICIAIS = 25

# Prazos de resolução (SLA) hipotéticos do cenário simulado, em horas.
SLA_HORAS = {"Baixa": 72, "Média": 24, "Alta": 8, "Crítica": 4}

# Categorias ordenadas, usadas na geração e nas conversões numéricas.
CATEGORIAS = ["Rede", "Hardware", "Software", "Acesso", "Impressora", "Sistema"]
ORDEM_URGENCIA = ["Baixa", "Média", "Alta", "Crítica"]
ORDEM_COMPLEXIDADE = ["Baixa", "Média", "Alta"]


# ============================================================================
# Modelo de decisão (pontuação ponderada)
# ============================================================================

# Critérios, na ordem em que aparecem no sistema.
CRITERIOS = ["urgencia", "criticidade", "usuarios", "idade", "recorrencias", "risco"]

NOMES_CRITERIOS = {
    "urgencia": "Urgência",
    "criticidade": "Criticidade",
    "usuarios": "Usuários afetados",
    "idade": "Idade (vs. prazo do SLA)",
    "recorrencias": "Recorrências",
    "risco": "Risco de violação do SLA",
}

# Pesos iniciais (em %). O usuário pode alterá-los na tela; o sistema normaliza para somar 100%.
PESOS_PADRAO = {"urgencia": 25, "criticidade": 20, "usuarios": 15, "idade": 15, "recorrencias": 10, "risco": 15}

# Nota (0-100) de cada nível de urgência. A criticidade (1-5) vale nível x 20.
NOTA_URGENCIA = {"Baixa": 25, "Média": 50, "Alta": 75, "Crítica": 100}

# Normalização dos demais critérios:
PERCENTIL_USUARIOS = 95   # usuários afetados: nota = min(usuários / P95, 1) x 100
RECORRENCIAS_MAX = 5      # recorrências: nota = min(recorrências / 5, 1) x 100
# idade: nota = min(idade / prazo do SLA da urgência, 1) x 100
# risco: nota = probabilidade de violação do SLA x 100

# Faixas de risco (probabilidade de violar o SLA): baixo < 30%, médio < 60%, alto >= 60%.
LIMITES_RISCO = (0.30, 0.60)

# Faixas de prioridade (pontuação final): P1 >= 60, P2 >= 40, P3 abaixo disso.
LIMITES_PRIORIDADE = (60, 40)

# Quantidade de chamados que a equipe consegue atender por rodada (destacados no topo da fila).
CAPACIDADE_PADRAO = 8

# Tamanho do topo (top-N) usado na validação retrospectiva da fila.
TOPO_VALIDACAO = 20


# ============================================================================
# Funções analíticas
# ============================================================================

# Classificação (árvore de decisão): profundidade e tamanho mínimo de folha limitados
# para que as probabilidades não sejam apenas 0 ou 1.
ARVORE_PROFUNDIDADE = 4
ARVORE_MIN_FOLHA = 10
PROPORCAO_TESTE = 0.20     # 80% treino / 20% teste (divisão estratificada)
FOLDS_VALIDACAO_CRUZADA = 5

# Agrupamento (K-Means)
K_PADRAO = 3
