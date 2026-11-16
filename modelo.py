"""
modelo.py - Funções analíticas, modelo de decisão e validação do SAD de chamados de TI.

Organização do arquivo (na ordem em que o sistema usa):
  1. Dados          -> carregar_dados, adicionar_derivadas
  2. Sumarização    -> sumarizar
  3. Agrupamento    -> agrupar (K-Means), avaliar_k, nomear_grupos
  4. Classificação  -> treinar_classificador (árvore de decisão)
  5. Decisão        -> pontuar, montar_fila (pontuação ponderada)
  6. Validação      -> comparar_politicas, curva_politicas, analise_sensibilidade
  7. Chamados abertos -> carregar_abertos, registrar_chamado (helpers de análise offline)
  8. Contexto       -> preparar_de_dataframe (usa o histórico persistido no banco)

IMPORTANTE (vazamento de dados): as colunas COLUNAS_DESFECHO só existem depois que o
chamado é encerrado. Elas NUNCA entram como entrada do classificador; são usadas na
sumarização histórica e na validação retrospectiva.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import config as cfg
from gerar_dados import gerar_abertos as gerar_fila_exemplo

CAMINHO_DADOS = Path(__file__).parent / "data" / "chamados.csv"

COLUNAS_BASE = ["id_chamado", "categoria", "urgencia", "usuarios_afetados", "criticidade",
                "idade_horas", "recorrencias", "complexidade", "tempo_resolucao_horas", "violou_sla"]
COLUNAS_DESFECHO = ["tempo_resolucao_horas", "violou_sla"]
# Chamados abertos não têm desfecho: só o que se sabe na hora da triagem.
COLUNAS_ABERTOS = ["id_chamado", "titulo", "categoria", "urgencia", "usuarios_afetados", "criticidade",
                   "idade_horas", "recorrencias", "complexidade"]


# ============================================================================
# 1. Dados
# ============================================================================
def adicionar_derivadas(df):
    """Acrescenta colunas calculadas só com informações disponíveis na hora da avaliação."""
    df = df.copy()
    df["sla_horas"] = df["urgencia"].map(cfg.SLA_HORAS)
    df["idade_rel_sla"] = df["idade_horas"] / df["sla_horas"]          # idade em relação ao prazo
    df["urgencia_n"] = df["urgencia"].map({u: i for i, u in enumerate(cfg.ORDEM_URGENCIA)})
    df["complexidade_n"] = df["complexidade"].map({c: i + 1 for i, c in enumerate(cfg.ORDEM_COMPLEXIDADE)})
    return df


def carregar_dados(caminho=None):
    """Carrega um CSV de análise ou gera a base sintética padrão quando não há arquivo."""
    if caminho is not None:
        df = pd.read_csv(caminho)
    elif CAMINHO_DADOS.exists():
        df = pd.read_csv(CAMINHO_DADOS)
    else:
        from gerar_dados import gerar_chamados
        df = gerar_chamados()
    faltando = [c for c in COLUNAS_BASE if c not in df.columns]
    if faltando:
        raise ValueError(f"A base não possui as colunas: {faltando}")
    return adicionar_derivadas(df.reset_index(drop=True))


# ============================================================================
# 2. Sumarização
# ============================================================================
def _resumo_por(df, coluna, ordem=None):
    t = df.groupby(coluna).agg(chamados=("id_chamado", "count"),
                               taxa_violacao=("violou_sla", "mean"),
                               tempo_medio_h=("tempo_resolucao_horas", "mean"))
    if ordem:
        t = t.reindex(ordem)
    return t.reset_index()


def sumarizar(df):
    """Indicadores gerais do histórico de chamados (função de sumarização)."""
    return {
        "total": int(len(df)),
        "taxa_violacao": float(df["violou_sla"].mean()),
        "tempo_medio_h": float(df["tempo_resolucao_horas"].mean()),
        "ja_acima_do_prazo": int((df["idade_rel_sla"] > 1).sum()),
        "por_categoria": _resumo_por(df, "categoria").sort_values("chamados", ascending=False),
        "por_urgencia": _resumo_por(df, "urgencia", cfg.ORDEM_URGENCIA),
        "por_complexidade": _resumo_por(df, "complexidade", cfg.ORDEM_COMPLEXIDADE),
        # referência usada pelo modelo de decisão para normalizar "usuários afetados"
        "p95_usuarios": float(np.percentile(df["usuarios_afetados"], cfg.PERCENTIL_USUARIOS)),
    }


# ============================================================================
# 3. Agrupamento (K-Means)
# ============================================================================
def _matriz_agrupamento(df):
    """Variáveis do agrupamento. Usuários e idade são muito assimétricos, então usamos log."""
    return pd.DataFrame({
        "usuarios_afetados": np.log1p(df["usuarios_afetados"]),
        "criticidade": df["criticidade"],
        "idade_horas": np.log1p(df["idade_horas"]),
        "recorrencias": df["recorrencias"],
        "complexidade_n": df["complexidade_n"],
    }, index=df.index)


def avaliar_k(df, ks=range(2, 7)):
    """Coeficiente de silhueta para cada k (quanto maior, mais bem separados os grupos)."""
    Xs = StandardScaler().fit_transform(_matriz_agrupamento(df))
    linhas = []
    for k in ks:
        rotulos = KMeans(n_clusters=k, n_init=10, random_state=cfg.SEMENTE).fit_predict(Xs)
        linhas.append({"k": k, "silhueta": float(silhouette_score(Xs, rotulos))})
    return pd.DataFrame(linhas)


def nomear_grupos(perfil, df):
    """
    Dá nome e uma ação sugerida a cada grupo a partir do PERFIL encontrado (nada é presumido).
    Compara a média do grupo com a média geral, em desvios-padrão (z):
      - Alto impacto: usuários e criticidade bem acima da média
      - Recorrente:   recorrências bem acima da média
      - Complexo:     complexidade bem acima da média
      - nenhum dos três: Pontual, baixo impacto
    """
    cols = ["usuarios_afetados", "criticidade", "recorrencias", "complexidade_n"]
    medias, desvios = df[cols].mean(), df[cols].std()
    nomes, acoes = {}, {}
    for g in perfil.index:
        z = (perfil.loc[g, cols] - medias) / desvios
        marcas, acao = [], []
        if (z["usuarios_afetados"] + z["criticidade"]) / 2 > 0.5:
            marcas.append("Alto impacto")
            acao.append("atender com prioridade e comunicar os usuários afetados")
        if z["recorrencias"] > 0.8:
            marcas.append("Recorrente")
            acao.append("investigar a causa raiz e abrir registro de problema")
        if z["complexidade_n"] > 0.8:
            marcas.append("Complexo")
            acao.append("acionar especialista (escalar de nível)")
        if not marcas:
            marcas.append("Pontual, baixo impacto")
            acao.append("resolver na fila normal; avaliar autoatendimento ou resolução em lote")
        nomes[g] = " + ".join(marcas)
        texto = "; ".join(acao)
        acoes[g] = texto[0].upper() + texto[1:]
    # evita nomes repetidos quando k é grande
    vistos = {}
    for g in list(nomes):
        vistos[nomes[g]] = vistos.get(nomes[g], 0) + 1
        if vistos[nomes[g]] > 1:
            nomes[g] = f"{nomes[g]} ({vistos[nomes[g]]})"
    return nomes, acoes


def agrupar(df, k=cfg.K_PADRAO):
    """Aplica o K-Means e devolve rótulos, perfil, nomes, ações e coordenadas para o gráfico."""
    X = _matriz_agrupamento(df)
    escala = StandardScaler().fit(X)
    Xs = escala.transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.SEMENTE).fit(Xs)
    rotulos = pd.Series(km.labels_, index=df.index, name="grupo")

    perfil = (df.assign(grupo=rotulos).groupby("grupo")
              .agg(chamados=("id_chamado", "count"),
                   usuarios_afetados=("usuarios_afetados", "mean"),
                   criticidade=("criticidade", "mean"),
                   idade_horas=("idade_horas", "mean"),
                   recorrencias=("recorrencias", "mean"),
                   complexidade_n=("complexidade_n", "mean")))
    nomes, acoes = nomear_grupos(perfil, df)
    pca = PCA(n_components=2, random_state=cfg.SEMENTE).fit(Xs)
    coords = pd.DataFrame(pca.transform(Xs), columns=["PC1", "PC2"], index=df.index)
    return {
        "k": k, "modelo": km, "escala": escala, "rotulos": rotulos, "perfil": perfil,
        "nomes": nomes, "acoes": acoes, "coords": coords,
        "variancia_pca": float(pca.explained_variance_ratio_.sum()),
        "silhueta": float(silhouette_score(Xs, km.labels_)),
    }


def atribuir_grupo(agrupamento, df_novo):
    """Descobre o grupo de chamados novos usando o K-Means já treinado."""
    Xs = agrupamento["escala"].transform(_matriz_agrupamento(df_novo))
    return pd.Series(agrupamento["modelo"].predict(Xs), index=df_novo.index)


# ============================================================================
# 4. Classificação (árvore de decisão: prever violação do SLA)
# ============================================================================
def matriz_atributos(df):
    """
    Entradas da árvore: só o que se sabe na hora da avaliação.
    (id_chamado, tempo_resolucao_horas e violou_sla ficam de fora.)
    """
    X = pd.DataFrame({
        "urgencia_n": df["urgencia_n"],
        "complexidade_n": df["complexidade_n"],
        "criticidade": df["criticidade"],
        "usuarios_afetados": df["usuarios_afetados"],
        "idade_horas": df["idade_horas"],
        "idade_rel_sla": df["idade_rel_sla"],
        "recorrencias": df["recorrencias"],
    }, index=df.index)
    for c in cfg.CATEGORIAS:
        X[f"categoria_{c}"] = (df["categoria"] == c).astype(int)
    return X


def _nova_arvore():
    return DecisionTreeClassifier(max_depth=cfg.ARVORE_PROFUNDIDADE,
                                  min_samples_leaf=cfg.ARVORE_MIN_FOLHA,
                                  random_state=cfg.SEMENTE)


def _metricas(y, pred, proba):
    return {"Acurácia": accuracy_score(y, pred),
            "Precisão": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1-score": f1_score(y, pred, zero_division=0),
            "AUC": roc_auc_score(y, proba)}


def treinar_classificador(df):
    """Treina/avalia a árvore e calcula risco fora da amostra para o histórico.

    O número de folds é reduzido quando uma das classes fica pequena. Se o
    histórico retido tiver somente uma classe, usa um DummyClassifier como
    fallback transparente até existirem exemplos dos dois resultados.
    """
    X, y = matriz_atributos(df), df["violou_sla"].astype(int)
    classes = sorted(y.unique().tolist())
    contagens = y.value_counts()
    min_classe = int(contagens.min()) if len(contagens) else 0

    if len(classes) < 2 or min_classe < 2:
        fallback = DummyClassifier(strategy="most_frequent").fit(X, y)
        classe_modelo = int(fallback.classes_[0])
        risco_constante = 1.0 if classe_modelo == 1 else 0.0
        risco_oof = pd.Series(risco_constante, index=df.index, name="risco")
        pred = fallback.predict(X)
        metricas = {
            "Acurácia": float(accuracy_score(y, pred)),
            "Precisão": 0.0,
            "Recall": 0.0,
            "F1-score": 0.0,
            "AUC": 0.5,
        }
        matriz = confusion_matrix(y, pred, labels=[0, 1])
        nomes_cv = ["Acurácia", "Precisão", "Recall", "F1-score", "AUC"]
        validacao_cruzada = pd.DataFrame({"média": [metricas[n] for n in nomes_cv],
                                          "desvio-padrão": [0.0] * len(nomes_cv)}, index=nomes_cv)
        importancias = pd.Series(0.0, index=X.columns).sort_values(ascending=False)
        return {
            "metricas_teste": pd.Series(metricas),
            "metricas_base": pd.Series(metricas),
            "matriz_confusao": matriz,
            "validacao_cruzada": validacao_cruzada,
            "n_treino": int(len(y)), "n_teste": 0,
            "violacoes_teste": int(y.sum()),
            "importancias": importancias,
            "regras": "Histórico atual com apenas uma classe (ou menos de dois exemplos da classe minoritária). "
                      "O sistema usa uma previsão de referência até acumular registros dos dois resultados.",
            "risco_oof": risco_oof,
            "modelo_final": fallback,
            "aviso": "Ainda não há exemplos suficientes dos dois resultados de SLA para treinar uma árvore confiável.",
        }

    n_folds = min(cfg.FOLDS_VALIDACAO_CRUZADA, min_classe)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.PROPORCAO_TESTE, stratify=y, random_state=cfg.SEMENTE)

    arvore = _nova_arvore().fit(X_tr, y_tr)
    pred, proba = arvore.predict(X_te), arvore.predict_proba(X_te)[:, 1]

    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    pred_base = base.predict(X_te)
    metricas_base = _metricas(y_te, pred_base, np.full(len(y_te), 0.5))

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEMENTE)
    res = cross_validate(_nova_arvore(), X, y, cv=cv,
                         scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    nomes_cv = {"accuracy": "Acurácia", "precision": "Precisão", "recall": "Recall",
                "f1": "F1-score", "roc_auc": "AUC"}
    validacao_cruzada = pd.DataFrame({
        "média": {nomes_cv[k]: res[f"test_{k}"].mean() for k in nomes_cv},
        "desvio-padrão": {nomes_cv[k]: res[f"test_{k}"].std() for k in nomes_cv},
    })

    risco_oof = pd.Series(cross_val_predict(_nova_arvore(), X, y, cv=cv, method="predict_proba")[:, 1],
                          index=df.index, name="risco")
    modelo_final = _nova_arvore().fit(X, y)

    return {
        "metricas_teste": pd.Series(_metricas(y_te, pred, proba)),
        "metricas_base": pd.Series(metricas_base),
        "matriz_confusao": confusion_matrix(y_te, pred, labels=[0, 1]),
        "validacao_cruzada": validacao_cruzada,
        "n_treino": int(len(y_tr)), "n_teste": int(len(y_te)),
        "violacoes_teste": int(y_te.sum()),
        "importancias": pd.Series(modelo_final.feature_importances_, index=X.columns).sort_values(ascending=False),
        "regras": export_text(modelo_final, feature_names=list(X.columns)),
        "risco_oof": risco_oof,
        "modelo_final": modelo_final,
        "aviso": None,
    }


def classificar_risco(prob):
    """Transforma a probabilidade de violar o SLA em faixa: Baixo, Médio ou Alto."""
    baixo, medio = cfg.LIMITES_RISCO
    return "Baixo" if prob < baixo else ("Médio" if prob < medio else "Alto")


# ============================================================================
# 5. Modelo de decisão (pontuação ponderada)
# ============================================================================
def normalizar_pesos(pesos):
    """Pesos negativos viram zero; se a soma não for 100%, normaliza. Tudo zero -> pesos iguais."""
    limpos = {c: max(0.0, float(pesos.get(c, 0))) for c in cfg.CRITERIOS}
    total = sum(limpos.values())
    if total <= 0:
        return {c: 1 / len(cfg.CRITERIOS) for c in cfg.CRITERIOS}
    return {c: v / total for c, v in limpos.items()}


def notas_criterios(df, risco, p95_usuarios):
    """Converte cada critério para uma nota de 0 a 100."""
    n = pd.DataFrame(index=df.index)
    n["urgencia"] = df["urgencia"].map(cfg.NOTA_URGENCIA)
    n["criticidade"] = df["criticidade"] * 20
    n["usuarios"] = (df["usuarios_afetados"] / p95_usuarios).clip(upper=1) * 100
    n["idade"] = df["idade_rel_sla"].clip(upper=1) * 100
    n["recorrencias"] = (df["recorrencias"] / cfg.RECORRENCIAS_MAX).clip(upper=1) * 100
    n["risco"] = np.asarray(risco) * 100
    return n


def classificar_prioridade(pontuacao):
    """P1 (alta), P2 (intermediária) ou P3 (baixa), conforme a pontuação final."""
    p1, p2 = cfg.LIMITES_PRIORIDADE
    return "P1" if pontuacao >= p1 else ("P2" if pontuacao >= p2 else "P3")


def pontuar(df, risco, pesos, p95_usuarios):
    """
    Pontuação final = soma(peso normalizado x nota do critério).
    Devolve (notas, contribuições, pontuação). Contribuição = quanto cada critério
    somou à pontuação (permite explicar a posição do chamado).
    """
    w = normalizar_pesos(pesos)
    notas = notas_criterios(df, risco, p95_usuarios)
    contribuicoes = pd.DataFrame({c: notas[c] * w[c] for c in cfg.CRITERIOS}, index=df.index)
    return notas, contribuicoes, contribuicoes.sum(axis=1)


def fatores_principais(contribuicoes_linha, n=2):
    """Texto com os critérios que mais pesaram na pontuação do chamado."""
    top = contribuicoes_linha.sort_values(ascending=False).head(n)
    return " · ".join(f"{cfg.NOMES_CRITERIOS[c]} (+{v:.1f})".replace(".", ",") for c, v in top.items())


def montar_fila(ctx, chamados, pesos, capacidade):
    """
    Monta a fila de atendimento recomendada para os chamados ABERTOS.
    O risco de violar o SLA vem do classificador treinado com o histórico; como o chamado
    ainda está aberto, não existe desfecho (tempo de resolução ou SLA) para ele.
    """
    chamados = adicionar_derivadas(chamados.reset_index(drop=True))
    probas = ctx.classificador["modelo_final"].predict_proba(matriz_atributos(chamados))
    classes = list(ctx.classificador["modelo_final"].classes_)
    prob_violacao = probas[:, classes.index(1)] if 1 in classes else np.zeros(len(chamados))
    risco = pd.Series(prob_violacao, index=chamados.index)
    notas, contrib, pont = pontuar(chamados, risco, pesos, ctx.resumo["p95_usuarios"])
    grupo = atribuir_grupo(ctx.agrupamento, chamados)
    fila = pd.DataFrame({
        "id_chamado": chamados["id_chamado"],
        "titulo": chamados["titulo"],
        "categoria": chamados["categoria"],
        "urgencia": chamados["urgencia"],
        "usuarios_afetados": chamados["usuarios_afetados"],
        "criticidade": chamados["criticidade"],
        "idade_horas": chamados["idade_horas"],
        "recorrencias": chamados["recorrencias"],
        "complexidade": chamados["complexidade"],
        "risco_pct": risco * 100,
        "faixa_risco": risco.map(classificar_risco),
        "pontuacao": pont,
        "prioridade": pont.map(classificar_prioridade),
        "grupo": grupo.map(ctx.agrupamento["nomes"]),
        "acao_sugerida": grupo.map(ctx.agrupamento["acoes"]),
        "principais_fatores": contrib.apply(fatores_principais, axis=1),
    })
    # desempate: maior pontuação, depois maior risco, depois mais antigo
    fila = fila.sort_values(["pontuacao", "risco_pct", "idade_horas"], ascending=False)
    fila.insert(0, "posicao", np.arange(1, len(fila) + 1))
    fila["atender_agora"] = fila["posicao"] <= capacidade
    return fila, notas, contrib


# ============================================================================
# 6. Validação (comparação com outras formas de organizar a fila)
# ============================================================================
def _ordens(df, pontuacao, risco):
    """Três políticas de fila, cada uma devolvendo os índices do chamado mais ao menos prioritário."""
    aux = df.assign(pontuacao=pontuacao, risco=risco)
    return {
        "FIFO (mais antigo primeiro)": aux.sort_values(["idade_horas", "id_chamado"], ascending=[False, True]).index,
        "Só por urgência": aux.sort_values(["urgencia_n", "idade_horas"], ascending=False).index,
        "Modelo ponderado (SAD)": aux.sort_values(["pontuacao", "risco", "idade_horas"], ascending=False).index,
    }


def comparar_politicas(df, pontuacao, risco, capacidade):
    """
    Avaliação RETROSPECTIVA: usa o desfecho histórico apenas para medir as filas.
    Para as N primeiras posições de cada política, calcula:
      - violações capturadas e % do total de violações da base;
      - precisão (violações / N);
      - impacto coberto = % de (usuários afetados x criticidade) concentrado nas N primeiras.
    """
    impacto = df["usuarios_afetados"] * df["criticidade"]
    total_viol, total_imp = df["violou_sla"].sum(), impacto.sum()
    linhas = []
    for nome, ordem in _ordens(df, pontuacao, risco).items():
        topo = ordem[:capacidade]
        viol = int(df.loc[topo, "violou_sla"].sum())
        linhas.append({
            "Política": nome,
            f"Violações no top {capacidade}": viol,
            "% das violações capturadas": (viol / total_viol) if total_viol else 0.0,
            "Precisão": (viol / len(topo)) if len(topo) else 0.0,
            "% do impacto coberto": float(impacto.loc[topo].sum() / total_imp),
        })
    return pd.DataFrame(linhas)


def curva_politicas(df, pontuacao, risco, ns=range(5, 101, 5)):
    """% das violações capturadas por política para vários tamanhos de top-N."""
    total_viol = df["violou_sla"].sum()
    linhas = []
    for nome, ordem in _ordens(df, pontuacao, risco).items():
        for n in ns:
            linhas.append({"Política": nome, "N": n,
                           "% das violações capturadas": (df.loc[ordem[:n], "violou_sla"].sum() / total_viol) if total_viol else 0.0})
    return pd.DataFrame(linhas)


def analise_sensibilidade(ctx, pesos, capacidade, variacao=0.5):
    """
    Aumenta e diminui o peso de cada critério (por padrão em 50%) e mede quanto a fila muda:
      - sobreposição do top-N com a fila original;
      - correlação de posições (Spearman) entre a fila original e a nova.
    """
    risco, p95 = ctx.classificador["risco_oof"], ctx.resumo["p95_usuarios"]
    _, _, base = pontuar(ctx.df, risco, pesos, p95)
    topo_base = set(base.sort_values(ascending=False).index[:capacidade])
    linhas = []
    for crit in cfg.CRITERIOS:
        linha = {"Critério": cfg.NOMES_CRITERIOS[crit]}
        for rotulo, fator in [("+", 1 + variacao), ("-", 1 - variacao)]:
            novos = dict(pesos)
            novos[crit] = pesos[crit] * fator
            _, _, nova = pontuar(ctx.df, risco, novos, p95)
            topo = set(nova.sort_values(ascending=False).index[:capacidade])
            linha[f"Sobreposição do top {capacidade} ({rotulo}{int(variacao * 100)}%)"] = len(topo & topo_base) / capacidade
            linha[f"Correlação de posições ({rotulo}{int(variacao * 100)}%)"] = float(base.corr(nova, method="spearman"))
        linhas.append(linha)
    return pd.DataFrame(linhas)


# ============================================================================
# 7. Chamados abertos (fila e registro de chamados novos)
# ============================================================================
def carregar_abertos():
    """Gera a fila inicial de demonstração em memória, sem um segundo CSV.

    Os registros são reproduzíveis porque o gerador utiliza uma semente fixa.
    Assim, o único arquivo de dados persistido no projeto é data/chamados.csv.
    """
    abertos = gerar_fila_exemplo()
    faltando = [c for c in COLUNAS_ABERTOS if c not in abertos.columns]
    if faltando:
        raise ValueError(f"A fila de demonstração não possui as colunas: {faltando}")
    return abertos[COLUNAS_ABERTOS].reset_index(drop=True)


def registrar_chamado(abertos, dados, id_chamado):
    """Acrescenta um chamado novo à fila de abertos e devolve a fila atualizada (nada é gravado em arquivo)."""
    if dados["categoria"] not in cfg.CATEGORIAS:
        raise ValueError(f"Categoria inválida: {dados['categoria']}")
    if dados["urgencia"] not in cfg.ORDEM_URGENCIA:
        raise ValueError(f"Urgência inválida: {dados['urgencia']}")
    if dados["complexidade"] not in cfg.ORDEM_COMPLEXIDADE:
        raise ValueError(f"Complexidade inválida: {dados['complexidade']}")
    linha = {
        "id_chamado": int(id_chamado),
        "titulo": (dados.get("titulo") or "").strip() or "Chamado sem título",
        "categoria": dados["categoria"],
        "urgencia": dados["urgencia"],
        "usuarios_afetados": max(1, int(dados["usuarios_afetados"])),
        "criticidade": int(min(5, max(1, dados["criticidade"]))),
        "idade_horas": max(0.0, float(dados.get("idade_horas", 0.0))),
        "recorrencias": max(0, int(dados["recorrencias"])),
        "complexidade": dados["complexidade"],
    }
    return pd.concat([abertos[COLUNAS_ABERTOS], pd.DataFrame([linha])], ignore_index=True)


# ============================================================================
# 8. Contexto: executa as análises uma única vez
# ============================================================================
@dataclass
class Contexto:
    df: pd.DataFrame
    resumo: dict
    agrupamento: dict
    classificador: dict


def preparar_de_dataframe(df, k=cfg.K_PADRAO):
    """Prepara as três funções analíticas a partir dos N chamados resolvidos do banco."""
    faltando = [c for c in COLUNAS_BASE if c not in df.columns]
    if faltando:
        raise ValueError(f"O histórico não possui as colunas necessárias: {faltando}")
    base = adicionar_derivadas(df[COLUNAS_BASE].copy().reset_index(drop=True))
    if len(base) < 50:
        raise ValueError("São necessários pelo menos 50 chamados resolvidos para treinar e avaliar o modelo.")
    return Contexto(df=base, resumo=sumarizar(base), agrupamento=agrupar(base, k),
                    classificador=treinar_classificador(base))


def preparar(k=cfg.K_PADRAO, caminho=None):
    """Compatibilidade com os testes e análises offline a partir de CSV."""
    return preparar_de_dataframe(carregar_dados(caminho), k)
