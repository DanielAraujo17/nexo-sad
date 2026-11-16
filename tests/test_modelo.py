"""
Testes automatizados do modelo (rodar com:  python -m pytest -q  na pasta do projeto).

Cobrem: qualidade da base, ausência de vazamento de dados, pesos, notas, faixas de
prioridade, fila, agrupamento, chamado novo e comparação de políticas.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import config as cfg  # noqa: E402
import gerar_dados  # noqa: E402
import modelo as m  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    return m.preparar()


# ---------------------------------------------------------------- base de dados
def test_base_tem_estrutura_esperada(ctx):
    df = ctx.df
    assert len(df) == cfg.N_HISTORICO
    assert all(c in df.columns for c in m.COLUNAS_BASE)
    assert df[m.COLUNAS_BASE].isna().sum().sum() == 0
    assert df["id_chamado"].is_unique


def test_base_e_coerente(ctx):
    df = ctx.df
    assert (df["tempo_resolucao_horas"] >= df["idade_horas"]).all()
    regra = (df["tempo_resolucao_horas"] > df["urgencia"].map(cfg.SLA_HORAS)).astype(int)
    assert (regra == df["violou_sla"]).all()
    assert 0.2 <= df["violou_sla"].mean() <= 0.5          # classes razoavelmente balanceadas


def test_historico_e_fila_de_demo_sao_reproduziveis_com_a_semente():
    pd.testing.assert_frame_equal(
        gerar_dados.gerar_chamados(), m.carregar_dados()[m.COLUNAS_BASE], check_dtype=False
    )
    pd.testing.assert_frame_equal(gerar_dados.gerar_abertos(), m.carregar_abertos(), check_dtype=False)


def test_chamados_abertos_nao_tem_desfecho_e_ids_nao_colidem(ctx):
    abertos = m.carregar_abertos()
    assert len(abertos) == cfg.N_ABERTOS_INICIAIS
    assert not set(m.COLUNAS_DESFECHO) & set(abertos.columns)
    assert abertos["id_chamado"].is_unique
    assert not set(abertos["id_chamado"]) & set(ctx.df["id_chamado"])


# ------------------------------------------------------------ vazamento de dados
def test_classificador_nao_usa_desfecho(ctx):
    colunas = m.matriz_atributos(ctx.df).columns
    for proibida in m.COLUNAS_DESFECHO + ["id_chamado"]:
        assert proibida not in colunas


def test_fila_nao_expoe_desfecho(ctx):
    fila, _, _ = m.montar_fila(ctx, m.carregar_abertos(), cfg.PESOS_PADRAO, 8)
    assert not set(m.COLUNAS_DESFECHO) & set(fila.columns)


# ----------------------------------------------------------------------- pesos
def test_normalizar_pesos():
    w = m.normalizar_pesos(cfg.PESOS_PADRAO)
    assert pytest.approx(sum(w.values())) == 1
    w2 = m.normalizar_pesos({c: 2 * v for c, v in cfg.PESOS_PADRAO.items()})
    assert all(pytest.approx(w[c]) == w2[c] for c in cfg.CRITERIOS)
    negativo = m.normalizar_pesos({**cfg.PESOS_PADRAO, "urgencia": -10})
    assert negativo["urgencia"] == 0 and pytest.approx(sum(negativo.values())) == 1
    zerado = m.normalizar_pesos({c: 0 for c in cfg.CRITERIOS})
    assert all(pytest.approx(v) == 1 / len(cfg.CRITERIOS) for v in zerado.values())


def test_notas_e_pontuacao_ficam_entre_0_e_100(ctx):
    risco = ctx.classificador["risco_oof"]
    notas, contrib, pont = m.pontuar(ctx.df, risco, cfg.PESOS_PADRAO, ctx.resumo["p95_usuarios"])
    assert notas.min().min() >= 0 and notas.max().max() <= 100
    assert pont.min() >= 0 and pont.max() <= 100
    assert np.allclose(contrib.sum(axis=1), pont)


def test_escalar_todos_os_pesos_nao_muda_a_pontuacao(ctx):
    risco = ctx.classificador["risco_oof"]
    _, _, a = m.pontuar(ctx.df, risco, cfg.PESOS_PADRAO, ctx.resumo["p95_usuarios"])
    _, _, b = m.pontuar(ctx.df, risco, {c: 3 * v for c, v in cfg.PESOS_PADRAO.items()}, ctx.resumo["p95_usuarios"])
    assert np.allclose(a, b)


def test_peso_so_em_urgencia_ordena_pela_urgencia(ctx):
    pesos = {c: 0 for c in cfg.CRITERIOS}
    pesos["urgencia"] = 100
    fila, _, _ = m.montar_fila(ctx, m.carregar_abertos(), pesos, 10)
    ordem = fila["urgencia"].map({u: i for i, u in enumerate(cfg.ORDEM_URGENCIA)})
    assert ordem.is_monotonic_decreasing
    assert fila.iloc[0]["urgencia"] == "Crítica"


# ----------------------------------------------------------------------- faixas
def test_faixas_de_prioridade_e_risco():
    p1, p2 = cfg.LIMITES_PRIORIDADE
    assert m.classificar_prioridade(p1) == "P1" and m.classificar_prioridade(p1 - 0.1) == "P2"
    assert m.classificar_prioridade(p2) == "P2" and m.classificar_prioridade(p2 - 0.1) == "P3"
    b, md = cfg.LIMITES_RISCO
    assert m.classificar_risco(b - 0.01) == "Baixo"
    assert m.classificar_risco(b) == "Médio" and m.classificar_risco(md - 0.01) == "Médio"
    assert m.classificar_risco(md) == "Alto"


# ------------------------------------------------------------------------ fila
def test_fila_ordenada_e_com_capacidade(ctx):
    abertos = m.carregar_abertos()
    fila, _, _ = m.montar_fila(ctx, abertos, cfg.PESOS_PADRAO, 8)
    assert len(fila) == len(abertos)
    assert list(fila["posicao"]) == list(range(1, len(fila) + 1))
    assert fila["pontuacao"].is_monotonic_decreasing
    assert int(fila["atender_agora"].sum()) == 8
    assert set(fila["prioridade"]) <= {"P1", "P2", "P3"}
    assert fila["risco_pct"].between(0, 100).all()
    assert set(fila["prioridade"]) <= {"P1", "P2", "P3"}


# ------------------------------------------------------------------ classificação
def test_risco_fora_da_amostra_e_probabilidade(ctx):
    r = ctx.classificador["risco_oof"]
    assert r.between(0, 1).all() and r.index.equals(ctx.df.index)
    assert r.round(3).nunique() > 5                         # não é apenas 0 ou 1


def test_classificador_supera_a_classe_majoritaria(ctx):
    c = ctx.classificador
    assert c["metricas_teste"]["AUC"] > 0.7
    assert c["validacao_cruzada"].loc["Acurácia", "média"] > c["metricas_base"]["Acurácia"] - 0.02


# ---------------------------------------------------------------------- agrupamento
@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_agrupamento_para_varios_k(ctx, k):
    ag = m.agrupar(ctx.df, k)
    assert ag["rotulos"].nunique() == k
    assert set(ag["nomes"]) == set(ag["acoes"]) == set(range(k))
    assert len(set(ag["nomes"].values())) == k             # nomes únicos
    assert m.atribuir_grupo(ag, ctx.df.head(5)).isin(range(k)).all()


# ---------------------------------------------------------------- registro de chamado
def test_registrar_chamado_entra_na_fila(ctx):
    abertos = m.carregar_abertos()
    dados = dict(titulo="  ", categoria="Rede", urgencia="Crítica", usuarios_afetados=40, criticidade=5,
                 idade_horas=0, recorrencias=2, complexidade="Alta")
    novos = m.registrar_chamado(abertos, dados, 999)
    assert len(novos) == len(abertos) + 1 and len(abertos) == cfg.N_ABERTOS_INICIAIS      # original não é alterado
    assert novos.iloc[-1]["titulo"] == "Chamado sem título"
    fila, _, _ = m.montar_fila(ctx, novos, cfg.PESOS_PADRAO, 8)
    linha = fila[fila["id_chamado"] == 999].iloc[0]
    assert linha["prioridade"] == "P1" and linha["posicao"] <= 5            # crítico, grande impacto e complexo


def test_novo_chamado_sem_idade_informada_comeca_em_zero():
    base = dict(titulo="Novo chamado", categoria="Rede", urgencia="Média", usuarios_afetados=3,
                criticidade=2, recorrencias=0, complexidade="Baixa")
    novo = m.registrar_chamado(m.carregar_abertos().head(0), base, 501)
    assert novo.iloc[0]["idade_horas"] == 0.0


def test_registrar_chamado_rejeita_valores_invalidos():
    base = dict(titulo="x", categoria="Rede", urgencia="Alta", usuarios_afetados=1, criticidade=3,
                idade_horas=0, recorrencias=0, complexidade="Baixa")
    abertos = m.carregar_abertos()
    for campo, valor in [("categoria", "Telefonia"), ("urgencia", "Urgentíssima"), ("complexidade", "Enorme")]:
        with pytest.raises(ValueError):
            m.registrar_chamado(abertos, {**base, campo: valor}, 999)


def test_fila_aceita_chamado_com_idade_zero(ctx):
    novo = m.registrar_chamado(m.carregar_abertos().head(0), dict(
        titulo="", categoria="Impressora", urgencia="Baixa", usuarios_afetados=1, criticidade=1,
        idade_horas=0, recorrencias=0, complexidade="Baixa"), 500)
    fila, _, _ = m.montar_fila(ctx, novo, cfg.PESOS_PADRAO, 8)
    assert len(fila) == 1 and fila.iloc[0]["prioridade"] == "P3"


# ------------------------------------------------------------------------ validação
def test_comparacao_de_politicas(ctx):
    risco = ctx.classificador["risco_oof"]
    _, _, pont = m.pontuar(ctx.df, risco, cfg.PESOS_PADRAO, ctx.resumo["p95_usuarios"])
    t = m.comparar_politicas(ctx.df, pont, risco, cfg.TOPO_VALIDACAO)
    assert len(t) == 3 and t["Precisão"].between(0, 1).all()
    viol = t.set_index("Política")[f"Violações no top {cfg.TOPO_VALIDACAO}"]
    assert viol["Modelo ponderado (SAD)"] >= viol["FIFO (mais antigo primeiro)"]


def test_sensibilidade(ctx):
    s = m.analise_sensibilidade(ctx, cfg.PESOS_PADRAO, cfg.TOPO_VALIDACAO)
    assert len(s) == len(cfg.CRITERIOS)
    assert s.filter(like="Sobreposição").min().min() >= 0.5


def test_classificador_tem_fallback_quando_historico_tem_uma_classe():
    base = gerar_dados.gerar_chamados(n=80)
    base["violou_sla"] = 0
    ctx_uniclasse = m.preparar_de_dataframe(base)
    assert ctx_uniclasse.classificador["aviso"]
    assert ctx_uniclasse.classificador["metricas_teste"]["AUC"] == 0.5
    fila, _, _ = m.montar_fila(ctx_uniclasse, m.carregar_abertos().head(3), cfg.PESOS_PADRAO, 2)
    assert len(fila) == 3 and fila["risco_pct"].eq(0).all()
