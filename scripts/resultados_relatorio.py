"""
resultados_relatorio.py - Recalcula TODOS os números citados no relatório.

Por padrão usa a base sintética determinística (semente fixa) em um SQLite em memória:
o banco local NÃO é lido nem alterado. Assim, os valores não dependem de quantos chamados
você cadastrou ou encerrou durante os testes.

Uso (na pasta do projeto):
    python scripts/resultados_relatorio.py                       # imprime os resultados
    python scripts/resultados_relatorio.py --json docs/resultados_referencia.json   # salva
    python scripts/resultados_relatorio.py --conferir docs/resultados_referencia.json
    python scripts/resultados_relatorio.py --db sqlite:///instance/chamados.db      # lê um banco (somente leitura)

--conferir compara com um arquivo salvo antes e informa se algum valor mudou (por exemplo,
por diferença de versão do scikit-learn). Se mudar, use os valores da página "Avaliação".
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
import modelo as m  # noqa: E402
from database import RepositorioChamados  # noqa: E402

# Chamados HIPOTÉTICOS usados só para ilustrar o modelo (caso concreto do relatório).
# A idade é o tempo de espera no instante da avaliação (valor fixo, para ser reproduzível).
CASO_CONCRETO = [
    (9001, "Sem acesso à internet em todo o setor", "Rede", "Crítica", 25, 5, 1, "Média", 2.0),
    (9002, "Impressora sem toner", "Impressora", "Baixa", 1, 1, 3, "Baixa", 60.0),
    (9003, "ERP lento no financeiro", "Sistema", "Alta", 12, 4, 0, "Alta", 6.5),
    (9004, "Troca de monitor solicitada", "Hardware", "Média", 1, 2, 0, "Baixa", 20.0),
    (9005, "Conta de equipe bloqueada", "Acesso", "Alta", 8, 3, 2, "Baixa", 9.0),
    (9006, "Programa fecha sozinho", "Software", "Média", 3, 2, 1, "Média", 30.0),
]
COLUNAS_CASO = ["id_chamado", "titulo", "categoria", "urgencia", "usuarios_afetados",
                "criticidade", "recorrencias", "complexidade", "idade_horas"]


def _r(x, n=4):
    return round(float(x), n)


def calcular(db_url: str | None = None) -> dict:
    if db_url:
        repo = RepositorioChamados(db_url)  # somente leitura: não chamamos inicializar()
    else:
        repo = RepositorioChamados("sqlite://")
        repo.inicializar(popular_demo=True)
    historico = repo.listar_resolvidos_df()
    ctx = m.preparar_de_dataframe(historico)
    cl, ag, rs, df = ctx.classificador, ctx.agrupamento, ctx.resumo, ctx.df

    out: dict = {}
    out["ambiente"] = {
        "python": platform.python_version(), "scikit-learn": __import__("sklearn").__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    out["base"] = {
        "origem": db_url or "semente determinística (SQLite em memória)",
        "resolvidos": rs["total"], "violacoes": int(df["violou_sla"].sum()),
        "taxa_violacao_pct": _r(rs["taxa_violacao"] * 100, 2),
        "tempo_medio_h": _r(rs["tempo_medio_h"], 2),
        "p95_usuarios": _r(rs["p95_usuarios"], 1),
        "ja_acima_do_prazo": rs["ja_acima_do_prazo"],
        "taxa_violacao_se_ja_acima_do_prazo": _r(df.loc[df["idade_rel_sla"] > 1, "violou_sla"].mean()),
        "taxa_violacao_se_dentro_do_prazo": _r(df.loc[df["idade_rel_sla"] <= 1, "violou_sla"].mean()),
    }
    cm = cl["matriz_confusao"]
    out["classificacao"] = {
        "n_treino": cl["n_treino"], "n_teste": cl["n_teste"], "violacoes_teste": cl["violacoes_teste"],
        "teste_arvore": {k: _r(v) for k, v in cl["metricas_teste"].items()},
        "teste_referencia_majoritaria": {k: _r(v) for k, v in cl["metricas_base"].items()},
        "matriz_confusao": {"VN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "VP": int(cm[1, 1])},
        "validacao_cruzada_media": {i: _r(r["média"]) for i, r in cl["validacao_cruzada"].iterrows()},
        "validacao_cruzada_desvio": {i: _r(r["desvio-padrão"]) for i, r in cl["validacao_cruzada"].iterrows()},
        "importancias_top5": {k: _r(v, 3) for k, v in cl["importancias"].head(5).items()},
    }
    perfil = ag["perfil"]
    out["agrupamento"] = {
        "silhueta_k3": _r(ag["silhueta"]), "variancia_pca_2_componentes": _r(ag["variancia_pca"], 3),
        "silhueta_por_k": {int(r["k"]): _r(r["silhueta"]) for r in m.avaliar_k(df).to_dict("records")},
        "perfis": {
            ag["nomes"][g]: {
                "chamados": int(perfil.loc[g, "chamados"]),
                "usuarios_afetados_media": _r(perfil.loc[g, "usuarios_afetados"], 2),
                "criticidade_media": _r(perfil.loc[g, "criticidade"], 2),
                "idade_horas_media": _r(perfil.loc[g, "idade_horas"], 2),
                "recorrencias_media": _r(perfil.loc[g, "recorrencias"], 2),
            } for g in perfil.index
        },
    }
    pesos = cfg.PESOS_PADRAO
    _, _, pont = m.pontuar(df, cl["risco_oof"], pesos, rs["p95_usuarios"])
    comp = m.comparar_politicas(df, pont, cl["risco_oof"], cfg.TOPO_VALIDACAO)
    out["politicas_top20"] = {
        r["Política"]: {
            "violacoes_no_top": int(r[f"Violações no top {cfg.TOPO_VALIDACAO}"]),
            "pct_das_violacoes_capturadas": _r(r["% das violações capturadas"] * 100, 1),
            "precisao_pct": _r(r["Precisão"] * 100, 1),
            "pct_do_impacto_coberto": _r(r["% do impacto coberto"] * 100, 1),
        } for r in comp.to_dict("records")
    }
    sens = m.analise_sensibilidade(ctx, pesos, cfg.TOPO_VALIDACAO)
    col_ov = [c for c in sens.columns if "Sobreposição" in c]
    col_co = [c for c in sens.columns if "Correlação" in c]
    out["sensibilidade_top20"] = {
        "sobreposicao_min_pct": _r(sens[col_ov].min().min() * 100, 1),
        "sobreposicao_max_pct": _r(sens[col_ov].max().max() * 100, 1),
        "correlacao_spearman_min": _r(sens[col_co].min().min(), 3),
        "correlacao_spearman_max": _r(sens[col_co].max().max(), 3),
    }

    # ---- Caso concreto (chamados hipotéticos, modelo treinado na base acima) ----
    casos = pd.DataFrame(CASO_CONCRETO, columns=COLUNAS_CASO)
    fila, notas, contrib = m.montar_fila(ctx, casos, pesos, cfg.CAPACIDADE_PADRAO)
    fifo = casos.sort_values(["idade_horas", "id_chamado"], ascending=[False, True])["id_chamado"].tolist()
    urg_ordem = casos.assign(un=casos["urgencia"].map({u: i for i, u in enumerate(cfg.ORDEM_URGENCIA)})) \
        .sort_values(["un", "idade_horas"], ascending=False)["id_chamado"].tolist()
    pesos_norm = m.normalizar_pesos(pesos)
    linhas = []
    for idx, linha in fila.iterrows():
        cid = int(linha["id_chamado"])
        linhas.append({
            "posicao_sad": int(linha["posicao"]), "posicao_fifo": fifo.index(cid) + 1,
            "posicao_urgencia": urg_ordem.index(cid) + 1, "id_chamado": cid, "titulo": linha["titulo"],
            "urgencia": linha["urgencia"], "usuarios": int(linha["usuarios_afetados"]),
            "criticidade": int(linha["criticidade"]), "recorrencias": int(linha["recorrencias"]),
            "idade_horas": _r(linha["idade_horas"], 1), "risco_pct": _r(linha["risco_pct"], 1),
            "pontuacao": _r(linha["pontuacao"], 1), "prioridade": linha["prioridade"],
            "notas": {k: _r(v, 1) for k, v in notas.loc[idx].items()},
            "contribuicoes": {k: _r(v, 1) for k, v in contrib.loc[idx].items()},
        })
    linhas.sort(key=lambda r: r["posicao_sad"])
    out["caso_concreto"] = {"pesos_normalizados": {k: _r(v, 3) for k, v in pesos_norm.items()}, "chamados": linhas}
    return out


def _achatar(d, prefixo=""):
    for k, v in d.items():
        chave = f"{prefixo}{k}"
        if isinstance(v, dict):
            yield from _achatar(v, chave + ".")
        elif isinstance(v, list):
            for i, item in enumerate(v):
                yield from _achatar(item, f"{chave}[{i}].") if isinstance(item, dict) else [(f"{chave}[{i}]", item)]
        else:
            yield chave, v


def imprimir(res: dict) -> None:
    b, c, a = res["base"], res["classificacao"], res["agrupamento"]
    print("=== AMBIENTE ===")
    print(", ".join(f"{k} {v}" for k, v in res["ambiente"].items()))
    print(f"\n=== BASE ({b['origem']}) ===")
    print(f"Resolvidos: {b['resolvidos']} | violações: {b['violacoes']} ({b['taxa_violacao_pct']}%) | tempo médio: {b['tempo_medio_h']} h")
    print(f"P95 de usuários: {b['p95_usuarios']} | já acima do prazo no registro: {b['ja_acima_do_prazo']} "
          f"(taxa de violação entre eles: {b['taxa_violacao_se_ja_acima_do_prazo']:.0%}; demais: {b['taxa_violacao_se_dentro_do_prazo']:.1%})")
    print("\n=== CLASSIFICAÇÃO (árvore de decisão) ===")
    print(f"Treino {c['n_treino']} | teste {c['n_teste']} (violações no teste: {c['violacoes_teste']})")
    print("Teste - árvore:", c["teste_arvore"])
    print("Teste - referência (classe majoritária):", c["teste_referencia_majoritaria"])
    print("Matriz de confusão:", c["matriz_confusao"])
    print("Validação cruzada (média):", c["validacao_cruzada_media"])
    print("Validação cruzada (desvio):", c["validacao_cruzada_desvio"])
    print("Importâncias (top 5):", c["importancias_top5"])
    print("\n=== AGRUPAMENTO (K-Means, k=3) ===")
    print(f"Silhueta k=3: {a['silhueta_k3']} | por k: {a['silhueta_por_k']} | variância PCA: {a['variancia_pca_2_componentes']}")
    for nome, p in a["perfis"].items():
        print(f"  {nome}: {p}")
    print("\n=== POLÍTICAS DE FILA (top 20, validação retrospectiva) ===")
    for nome, p in res["politicas_top20"].items():
        print(f"  {nome}: {p}")
    print("\n=== SENSIBILIDADE (±50% em cada peso, top 20) ===")
    print(" ", res["sensibilidade_top20"])
    print("\n=== CASO CONCRETO (chamados hipotéticos; pesos padrão) ===")
    print("Pesos normalizados:", res["caso_concreto"]["pesos_normalizados"])
    print("pos.SAD | pos.FIFO | pos.urgência | id | urgência | usuários | crit. | rec. | idade(h) | risco% | pontuação | prioridade | título")
    for r in res["caso_concreto"]["chamados"]:
        print(f"{r['posicao_sad']:7d} | {r['posicao_fifo']:8d} | {r['posicao_urgencia']:12d} | {r['id_chamado']} | {r['urgencia']:8s} | "
              f"{r['usuarios']:8d} | {r['criticidade']:5d} | {r['recorrencias']:4d} | {r['idade_horas']:8.1f} | {r['risco_pct']:6.1f} | "
              f"{r['pontuacao']:9.1f} | {r['prioridade']:10s} | {r['titulo']}")
    top = res["caso_concreto"]["chamados"][0]
    print(f"\nDetalhe do 1º colocado (#{top['id_chamado']}): notas {top['notas']}")
    print(f"  contribuições (nota x peso): {top['contribuicoes']}  -> soma = {top['pontuacao']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="URL do banco a ler (somente leitura). Padrão: semente determinística em memória.")
    ap.add_argument("--json", help="Salva os resultados neste arquivo JSON.")
    ap.add_argument("--conferir", help="Compara com um JSON salvo anteriormente e informa diferenças.")
    args = ap.parse_args()

    res = calcular(args.db)
    imprimir(res)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nResultados salvos em {args.json}")
    if args.conferir:
        ref = json.loads(Path(args.conferir).read_text(encoding="utf-8"))
        a, b = dict(_achatar(res)), dict(_achatar(ref))
        difs = [(k, b.get(k), a.get(k)) for k in sorted(set(a) | set(b))
                if not k.startswith("ambiente.") and not k.startswith("base.origem")
                and (a.get(k) != b.get(k) if not isinstance(a.get(k), float) or not isinstance(b.get(k), float)
                     else abs(a[k] - b[k]) > 1e-4)]
        print("\n=== CONFERÊNCIA COM O ARQUIVO DE REFERÊNCIA ===")
        if not difs:
            print("OK: todos os valores coincidem com a referência.")
        else:
            print(f"ATENÇÃO: {len(difs)} valor(es) diferente(s) (referência -> atual):")
            for k, ref_v, atual in difs[:40]:
                print(f"  {k}: {ref_v} -> {atual}")
            print("Se o ambiente for diferente (versões de bibliotecas), use os valores da página Avaliação no relatório.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
