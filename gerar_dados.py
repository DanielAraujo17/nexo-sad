"""
gerar_dados.py - Gera o conjunto sintético inicial para popular o banco do Nexo.

Ideia central (importante para evitar vazamento de dados):
  1. Sorteamos as características do chamado conhecidas na hora da triagem
     (categoria, urgência, criticidade, usuários afetados, complexidade, recorrências).
  2. Sorteamos a IDADE do chamado: quanto ele já esperou no momento em que o
     coordenador olha a fila. A idade é sorteada ANTES e sem olhar o desfecho.
  3. Só então calculamos o TEMPO RESTANTE para resolver, que depende da
     complexidade, da categoria, das recorrências e de um ruído aleatório.
  4. tempo_resolucao = idade + tempo restante  (logo, nunca é menor que a idade,
     o que é coerente com um chamado que ainda estava aberto).
  5. violou_sla = 1 se tempo_resolucao ultrapassou o prazo da urgência.

A base inicial é sintética e usada para popular o banco na primeira execução.
A fila inicial de demonstração também é gerada por este módulo; a persistência real
é responsabilidade de database.py, que grava abertos e resolvidos na mesma tabela.

Uso:  python gerar_dados.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from config import (CATEGORIAS, N_ABERTOS_INICIAIS, N_HISTORICO, ORDEM_COMPLEXIDADE,
                    ORDEM_URGENCIA, SEMENTE, SLA_HORAS)

# ----------------------------------------------------------------------------
# Regras de geração (o que torna os dados "minimamente realistas")
# ----------------------------------------------------------------------------
P_CATEGORIA = [0.15, 0.20, 0.20, 0.20, 0.12, 0.13]  # mesma ordem de CATEGORIAS

# Probabilidade de cada urgência (Baixa, Média, Alta, Crítica) por categoria:
# rede e sistema tendem a ser mais urgentes; impressora, menos.
P_URGENCIA = {
    "Rede":       [0.10, 0.30, 0.35, 0.25],
    "Sistema":    [0.10, 0.30, 0.35, 0.25],
    "Hardware":   [0.20, 0.40, 0.30, 0.10],
    "Software":   [0.25, 0.40, 0.28, 0.07],
    "Acesso":     [0.20, 0.40, 0.30, 0.10],
    "Impressora": [0.50, 0.35, 0.12, 0.03],
}

# Criticidade média (1 a 5) esperada para cada urgência.
CRITICIDADE_MEDIA = {"Baixa": 1.8, "Média": 2.8, "Alta": 3.8, "Crítica": 4.5}

# Usuários afetados: problemas de rede/sistema atingem muita gente; impressora e hardware, poucos.
USUARIOS_BASE = {"Rede": 12, "Sistema": 10, "Acesso": 3, "Software": 3, "Hardware": 1.5, "Impressora": 2}

# Complexidade (Baixa, Média, Alta) por categoria.
P_COMPLEXIDADE = {
    "Rede":       [0.20, 0.45, 0.35],
    "Sistema":    [0.20, 0.40, 0.40],
    "Hardware":   [0.30, 0.45, 0.25],
    "Software":   [0.35, 0.45, 0.20],
    "Acesso":     [0.60, 0.30, 0.10],
    "Impressora": [0.55, 0.35, 0.10],
}

# Média de ocorrências anteriores do mesmo problema, por categoria.
RECORRENCIA_MEDIA = {"Impressora": 1.2, "Acesso": 1.5, "Rede": 1.0, "Software": 0.8, "Sistema": 0.7, "Hardware": 0.5}

# Horas típicas que ainda faltam para resolver, por complexidade.
TEMPO_RESTANTE_BASE = {"Baixa": 1.5, "Média": 5.0, "Alta": 14.0}

# A equipe atende primeiro o que é mais urgente, então urgentes demoram menos.
FATOR_URGENCIA = {"Crítica": 0.5, "Alta": 0.6, "Média": 1.0, "Baixa": 2.0}

# Títulos de exemplo (só para deixar a fila de chamados abertos legível): problemas "leves" para
# urgência Baixa ou Média e "graves" para Alta ou Crítica.
TITULOS = {
    "Rede": {"leve": ["Wi-Fi instável na sala de reuniões", "Lentidão na rede interna", "Ponto de rede sem sinal em uma mesa"],
             "grave": ["Sem acesso à internet no setor", "VPN fora do ar para toda a equipe", "Switch do andar sem energia"]},
    "Hardware": {"leve": ["Teclado e mouse sem resposta", "Disco cheio e computador lento", "Troca de monitor solicitada"],
                 "grave": ["Computador não liga", "Servidor de arquivos sem resposta", "Notebook superaquecendo e desligando"]},
    "Software": {"leve": ["Instalar software solicitado", "Programa fecha sozinho", "Atualização travada"],
                 "grave": ["Antivírus bloqueando aplicativo essencial", "Planilhas não abrem em vários computadores",
                           "Falha ao atualizar o sistema operacional"]},
    "Acesso": {"leve": ["Esqueci a senha do e-mail", "Cadastrar novo colaborador", "Liberar acesso à pasta do setor"],
               "grave": ["Conta de equipe bloqueada", "Autenticação em dois fatores falhando para o setor",
                         "Usuários sem acesso ao diretório"]},
    "Impressora": {"leve": ["Impressora sem toner", "Impressão com listras", "Papel atolado"],
                   "grave": ["Impressora do setor fora da rede", "Scanner não envia e-mail", "Impressora fiscal sem comunicação"]},
    "Sistema": {"leve": ["Relatório não carrega", "Lentidão no sistema de vendas", "Ajuste de permissão no sistema"],
                "grave": ["ERP fora do ar", "Erro ao emitir nota fiscal", "Sistema de ponto indisponível"]},
}


def _sortear_triagem(rng):
    """Características conhecidas na triagem (a ordem dos sorteios não deve mudar: garante a mesma base)."""
    categoria = rng.choice(CATEGORIAS, p=P_CATEGORIA)
    urgencia = rng.choice(ORDEM_URGENCIA, p=P_URGENCIA[categoria])
    criticidade = int(np.clip(round(rng.normal(CRITICIDADE_MEDIA[urgencia], 1.0)), 1, 5))
    usuarios = USUARIOS_BASE[categoria] * (0.5 + 0.3 * criticidade) * rng.lognormal(0, 0.6)
    usuarios_afetados = int(np.clip(round(usuarios), 1, 80))
    complexidade = rng.choice(ORDEM_COMPLEXIDADE, p=P_COMPLEXIDADE[categoria])
    recorrencias = int(min(rng.poisson(RECORRENCIA_MEDIA[categoria]), 8))
    return {"categoria": str(categoria), "urgencia": str(urgencia), "usuarios_afetados": usuarios_afetados,
            "criticidade": criticidade, "recorrencias": recorrencias, "complexidade": str(complexidade)}


def _sortear_idade(rng, urgencia):
    """Idade no momento da avaliação: proporcional ao prazo da urgência; pode passar do prazo."""
    return round(max(0.1, SLA_HORAS[urgencia] * rng.beta(1.3, 2.5) * 1.3), 1)


def gerar_chamados(n=N_HISTORICO, semente=SEMENTE):
    """Gera n chamados fictícios e devolve um DataFrame."""
    rng = np.random.default_rng(semente)
    linhas = []

    for i in range(1, n + 1):
        # 1) Características conhecidas na triagem
        t = _sortear_triagem(rng)
        urgencia, complexidade, recorrencias = t["urgencia"], t["complexidade"], t["recorrencias"]

        # 2) Idade no momento da avaliação (sorteada sem olhar o desfecho)
        prazo = SLA_HORAS[urgencia]
        idade_horas = _sortear_idade(rng, urgencia)

        # 3) Tempo que ainda falta para resolver (depende de complexidade, urgência,
        #    recorrências e ruído).
        restante = (TEMPO_RESTANTE_BASE[complexidade] * FATOR_URGENCIA[urgencia]
                    * (1 + 0.18 * recorrencias) * rng.lognormal(0, 0.35))

        # 4) e 5) Desfecho histórico (só existe depois que o chamado é encerrado)
        tempo_resolucao_horas = round(idade_horas + restante, 1)
        violou_sla = int(tempo_resolucao_horas > prazo)

        linhas.append({
            "id_chamado": i,
            "categoria": t["categoria"],
            "urgencia": urgencia,
            "usuarios_afetados": t["usuarios_afetados"],
            "criticidade": t["criticidade"],
            "idade_horas": idade_horas,
            "recorrencias": recorrencias,
            "complexidade": complexidade,
            "tempo_resolucao_horas": tempo_resolucao_horas,
            "violou_sla": violou_sla,
        })
    return pd.DataFrame(linhas)


def gerar_abertos(n=N_ABERTOS_INICIAIS, semente=SEMENTE + 1, id_inicial=N_HISTORICO + 1):
    """
    Gera a fila inicial de chamados ABERTOS (sem tempo de resolução nem resultado de SLA).
    Usa as mesmas regras da base histórica, com outra semente, e ids a partir de id_inicial.
    """
    rng = np.random.default_rng(semente)
    linhas = []
    for i in range(n):
        t = _sortear_triagem(rng)
        idade_horas = _sortear_idade(rng, t["urgencia"])
        gravidade = "grave" if t["urgencia"] in ("Alta", "Crítica") else "leve"
        titulo = str(rng.choice(TITULOS[t["categoria"]][gravidade]))
        linhas.append({"id_chamado": id_inicial + i, "titulo": titulo, "categoria": t["categoria"],
                       "urgencia": t["urgencia"], "usuarios_afetados": t["usuarios_afetados"],
                       "criticidade": t["criticidade"], "idade_horas": idade_horas,
                       "recorrencias": t["recorrencias"], "complexidade": t["complexidade"]})
    return pd.DataFrame(linhas)


def resumo(df):
    """Imprime verificações de qualidade da base gerada."""
    print(f"\nChamados: {len(df)} | colunas: {df.shape[1]} | nulos: {int(df.isna().sum().sum())}")
    print(f"IDs únicos: {df['id_chamado'].is_unique}")
    print(f"tempo_resolucao >= idade em todos: {bool((df['tempo_resolucao_horas'] >= df['idade_horas']).all())}")
    regra = (df["tempo_resolucao_horas"] > df["urgencia"].map(SLA_HORAS)).astype(int)
    print(f"violou_sla consistente com o SLA: {bool((regra == df['violou_sla']).all())}")
    print(f"\nTaxa geral de violação do SLA: {df['violou_sla'].mean():.1%}")
    for col in ["urgencia", "complexidade", "categoria"]:
        ordem = {"urgencia": ORDEM_URGENCIA, "complexidade": ORDEM_COMPLEXIDADE}.get(col)
        t = df.groupby(col)["violou_sla"].agg(["count", "mean"])
        t = t.reindex(ordem) if ordem else t.sort_values("mean")
        print(f"\nViolação por {col}:")
        print(t.rename(columns={"count": "chamados", "mean": "taxa_violacao"}).round(2).to_string())
    print("\nEstatísticas das variáveis numéricas:")
    print(df[["usuarios_afetados", "criticidade", "idade_horas", "recorrencias",
              "tempo_resolucao_horas"]].describe().round(1).to_string())


if __name__ == "__main__":
    pasta = Path(__file__).parent / "data"
    pasta.mkdir(exist_ok=True)

    base = gerar_chamados()
    base.to_csv(pasta / "chamados.csv", index=False, encoding="utf-8")
    print(f"Amostra sintética exportada para {pasta / 'chamados.csv'} (uso opcional para inspeção).")

    abertos = gerar_abertos()
    print(f"Fila inicial para popular o banco: {len(abertos)} chamados abertos.")

    resumo(base)
