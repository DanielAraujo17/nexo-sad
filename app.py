"""Aplicação web Flask do SAD para priorização de chamados de suporte de TI."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from flask import Flask, abort, flash, make_response, redirect, render_template, request, session, url_for
from flask_wtf.csrf import CSRFProtect, generate_csrf

import config as cfg
import modelo as m
from database import RepositorioChamados

load_dotenv()


def _env_flag(nome: str, padrao: bool = False) -> bool:
    """Interpreta variáveis booleanas sem tratar a string 'false' como True."""
    valor = os.getenv(nome)
    if valor is None:
        return padrao
    return valor.strip().lower() in {"1", "true", "yes", "on", "sim"}


def _ambiente_producao() -> bool:
    return os.getenv("APP_ENV", "").strip().lower() == "production" or _env_flag("RENDER")


PALETA = {
    "azul": "#244C7A",
    "azul_escuro": "#142B45",
    "verde": "#168477",
    "vermelho": "#C65B58",
    "cinza": "#718096",
}


def criar_app(repositorio: RepositorioChamados | None = None, inicializar_banco: bool = True) -> Flask:
    app = Flask(__name__)
    em_producao = _ambiente_producao()
    secret_key = os.getenv("FLASK_SECRET_KEY")
    if em_producao and not secret_key:
        raise RuntimeError("Defina FLASK_SECRET_KEY nas variáveis de ambiente antes de publicar a aplicação.")
    if em_producao and secret_key.strip().lower().startswith("troque-por"):
        raise RuntimeError("FLASK_SECRET_KEY ainda é o valor de exemplo do .env.example. Gere uma chave aleatória longa.")

    app.config.update(
        SECRET_KEY=secret_key or "chave-local-apenas-para-desenvolvimento",
        HISTORICO_LIMITE=int(os.getenv("HISTORICO_LIMITE", cfg.N_HISTORICO)),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=_env_flag("SESSION_COOKIE_SECURE", padrao=em_producao),
    )

    repo = repositorio or RepositorioChamados(historico_limite=app.config["HISTORICO_LIMITE"])
    if em_producao and repo.dialeto == "sqlite":
        app.logger.warning(
            "Ambiente de produção usando SQLite: em hospedagens com disco efêmero (ex.: Render gratuito) "
            "os dados são perdidos a cada reinício ou novo deploy. Configure DATABASE_URL (PostgreSQL/Supabase)."
        )
    if inicializar_banco:
        repo.inicializar(popular_demo=True)
    CSRFProtect(app)

    cache: dict[str, object] = {"contexto": None}
    app.extensions["repositorio_chamados"] = repo
    app.extensions["contexto_analitico"] = cache

    def contexto(force: bool = False):
        if force or cache["contexto"] is None:
            historico = repo.listar_resolvidos_df()
            if len(historico) < 50:
                raise ValueError("O histórico precisa ter pelo menos 50 chamados resolvidos para executar as análises.")
            cache["contexto"] = m.preparar_de_dataframe(historico)
        return cache["contexto"]

    def pesos_atuais() -> dict[str, float]:
        salvos = session.get("pesos", {})
        return {criterio: float(salvos.get(criterio, valor)) for criterio, valor in cfg.PESOS_PADRAO.items()}

    def capacidade_atual() -> int:
        return max(1, min(30, int(session.get("capacidade", cfg.CAPACIDADE_PADRAO))))

    def fig_html(fig, altura: int = 330) -> str:
        fig.update_layout(
            template="plotly_white", height=altura,
            margin=dict(l=18, r=18, t=55, b=24),
            font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#40516A"),
            title_font=dict(size=16, color="#20344D"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False, "responsive": True})

    @app.context_processor
    def inject_globals():
        return {
            "app_name": "Nexo | Suporte de TI",
            "historico_limite": repo.historico_limite,
            "nav_endpoint": request.endpoint,
            "csrf_token": generate_csrf,
        }

    @app.get("/health")
    def health():
        # 'banco' permite conferir, na versão publicada, se os dados estão no PostgreSQL
        # (persistente) ou em SQLite (efêmero em hospedagens gratuitas). Não expõe credenciais.
        return {"status": "ok", "banco": repo.dialeto}

    @app.get("/")
    def index():
        ctx = contexto()
        abertos = repo.listar_abertos()
        pesos = pesos_atuais()
        capacidade = capacidade_atual()
        fila, notas, contribuicoes = m.montar_fila(ctx, abertos, pesos, capacidade) if not abertos.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        prioridade = request.args.get("prioridade", "Todas")
        busca = request.args.get("q", "").strip()
        visivel = fila.copy()
        if prioridade in {"P1", "P2", "P3"} and not visivel.empty:
            visivel = visivel[visivel["prioridade"] == prioridade]
        if busca and not visivel.empty:
            termo = busca.lower().replace("#", "")
            mask = (
                visivel["id_chamado"].astype(str).str.contains(termo, na=False)
                | visivel["titulo"].astype(str).str.lower().str.contains(termo, na=False)
                | visivel["categoria"].astype(str).str.lower().str.contains(termo, na=False)
            )
            visivel = visivel[mask]

        counts = {p: int((fila["prioridade"] == p).sum()) for p in ["P1", "P2", "P3"]} if not fila.empty else {"P1": 0, "P2": 0, "P3": 0}
        metricas = {
            "abertos": len(fila),
            "p1": counts["P1"],
            "p2": counts["P2"],
            "risco_alto": int((fila["faixa_risco"] == "Alto").sum()) if not fila.empty else 0,
        }
        return render_template(
            "dashboard.html", fila=visivel.to_dict("records"), total_fila=len(fila), metricas=metricas,
            prioridade=prioridade, busca=busca, capacidade=capacidade, pesos=pesos,
            nomes_criterios=cfg.NOMES_CRITERIOS, criterios=cfg.CRITERIOS,
            limites_prioridade=cfg.LIMITES_PRIORIDADE,
        )

    @app.post("/preferencias")
    def preferencias():
        pesos = {}
        for criterio in cfg.CRITERIOS:
            try:
                pesos[criterio] = max(0, min(100, int(request.form.get(f"peso_{criterio}", 0))))
            except (TypeError, ValueError):
                pesos[criterio] = cfg.PESOS_PADRAO[criterio]
        try:
            capacidade = max(1, min(30, int(request.form.get("capacidade", cfg.CAPACIDADE_PADRAO))))
        except (TypeError, ValueError):
            capacidade = cfg.CAPACIDADE_PADRAO
        session["pesos"] = pesos
        session["capacidade"] = capacidade
        flash("Critérios de priorização atualizados. A fila foi recalculada.", "success")
        return redirect(url_for("index"))

    @app.post("/preferencias/restaurar")
    def restaurar_preferencias():
        session.pop("pesos", None)
        session.pop("capacidade", None)
        flash("Pesos e capacidade restaurados para os valores padrão.", "info")
        return redirect(url_for("index"))

    @app.post("/chamados")
    def criar_chamado():
        try:
            dados = {
                "titulo": request.form.get("titulo", ""),
                "categoria": request.form.get("categoria", ""),
                "urgencia": request.form.get("urgencia", ""),
                "complexidade": request.form.get("complexidade", ""),
                "usuarios_afetados": int(request.form.get("usuarios_afetados", 1)),
                "criticidade": int(request.form.get("criticidade", 3)),
                "recorrencias": int(request.form.get("recorrencias", 0)),
            }
            if not 1 <= dados["usuarios_afetados"] <= 200:
                raise ValueError("A quantidade de usuários deve estar entre 1 e 200.")
            if not 1 <= dados["criticidade"] <= 5:
                raise ValueError("A criticidade deve estar entre 1 e 5.")
            if not 0 <= dados["recorrencias"] <= 10:
                raise ValueError("As recorrências devem estar entre 0 e 10.")
            novo_id = repo.criar(dados)
            flash(f"Chamado #{novo_id} registrado. A prioridade já foi calculada.", "success")
            return redirect(url_for("detalhe_chamado", id_chamado=novo_id))
        except (ValueError, TypeError) as exc:
            flash(str(exc) or "Confira os dados informados.", "error")
            return redirect(url_for("index"))

    @app.get("/chamados/<int:id_chamado>")
    def detalhe_chamado(id_chamado: int):
        chamado = repo.obter(id_chamado)
        if chamado is None:
            abort(404)
        recomendacao = None
        fatores = []
        if chamado["status"] == "aberto":
            ctx = contexto()
            abertos = repo.listar_abertos()
            fila, notas, contribuicoes = m.montar_fila(ctx, abertos, pesos_atuais(), capacidade_atual())
            linha = fila[fila["id_chamado"] == id_chamado]
            if not linha.empty:
                recomendacao = linha.iloc[0].to_dict()
                contrib = contribuicoes.loc[linha.index[0]].sort_values(ascending=False)
                fatores = [{"nome": cfg.NOMES_CRITERIOS[k], "valor": float(v)} for k, v in contrib.items()]
        return render_template("ticket_detail.html", chamado=chamado, recomendacao=recomendacao, fatores=fatores,
                               sla_horas=cfg.SLA_HORAS.get(chamado["urgencia"], 0))

    @app.post("/chamados/<int:id_chamado>/resolver")
    def resolver_chamado(id_chamado: int):
        resultado = repo.resolver(id_chamado)
        if resultado is None:
            flash("O chamado não foi encontrado ou já está resolvido.", "warning")
            return redirect(url_for("index"))
        cache["contexto"] = None
        status = "ultrapassou" if resultado["violou_sla"] else "permaneceu dentro"
        flash(
            f"Chamado #{id_chamado} encerrado. Tempo registrado: {resultado['tempo_resolucao_horas']:.2f} h; "
            f"o atendimento {status} do SLA. O histórico foi atualizado.", "success"
        )
        return redirect(url_for("analises"))

    @app.get("/analises")
    def analises():
        ctx = contexto()
        resumo = ctx.resumo
        ag = ctx.agrupamento
        historico = ctx.df.copy()
        contagens = repo.contagens()

        situacao = historico.assign(situacao=historico["violou_sla"].map({1: "Violou o SLA", 0: "Dentro do SLA"}))
        dados_categoria = situacao.groupby(["categoria", "situacao"]).size().reset_index(name="chamados")
        fig_categoria = px.bar(
            dados_categoria, x="chamados", y="categoria", color="situacao", orientation="h",
            color_discrete_map={"Violou o SLA": PALETA["vermelho"], "Dentro do SLA": PALETA["azul"]},
            title="Chamados por categoria e resultado do SLA", labels={"categoria": "", "chamados": "Quantidade", "situacao": "Resultado"},
            barmode="stack",
        )
        chart_categoria = fig_html(fig_categoria, 360)

        por_urgencia = resumo["por_urgencia"].copy()
        por_urgencia["Prazo do SLA"] = por_urgencia["urgencia"].map(cfg.SLA_HORAS)
        fig_urgencia = px.bar(
            por_urgencia, x="urgencia", y=["tempo_medio_h", "Prazo do SLA"], barmode="group",
            category_orders={"urgencia": cfg.ORDEM_URGENCIA},
            color_discrete_sequence=[PALETA["vermelho"], "#AAB7C6"],
            title="Tempo médio de resolução e prazo do SLA",
            labels={"urgencia": "Urgência", "value": "Horas", "variable": "Medida", "tempo_medio_h": "Tempo médio"},
        )
        chart_urgencia = fig_html(fig_urgencia, 360)

        perfil = ag["perfil"].copy()
        perfil.insert(0, "Perfil", perfil.index.map(ag["nomes"]))
        violacoes_grupo = historico.assign(grupo=ag["rotulos"]).groupby("grupo")["violou_sla"].mean() * 100
        perfil["Violação histórica (%)"] = violacoes_grupo
        perfil["Ação sugerida"] = perfil.index.map(ag["acoes"])
        perfil = perfil.rename(columns={
            "chamados": "Chamados", "usuarios_afetados": "Usuários afetados (média)",
            "criticidade": "Criticidade média", "idade_horas": "Idade na avaliação (h)",
            "recorrencias": "Recorrências médias",
        })
        colunas_num = ["Usuários afetados (média)", "Criticidade média", "Idade na avaliação (h)", "Recorrências médias", "Violação histórica (%)"]
        perfil[colunas_num] = perfil[colunas_num].round(1)
        perfis = perfil[["Perfil", "Chamados", *colunas_num, "Ação sugerida"]].reset_index(drop=True).to_dict("records")

        chart_clusters = ""
        if not ag["coords"].empty:
            pontos = ag["coords"].assign(Perfil=ag["rotulos"].map(ag["nomes"]), Chamado=historico["id_chamado"])
            fig_clusters = px.scatter(pontos, x="PC1", y="PC2", color="Perfil", hover_data=["Chamado"],
                                      title="Perfis de chamados (projeção PCA)", labels={"PC1": "Componente 1", "PC2": "Componente 2"})
            chart_clusters = fig_html(fig_clusters, 390)

        recentes = repo.listar_historico_recente(repo.historico_limite)
        return render_template(
            "analytics.html", resumo=resumo, contagens=contagens,
            chart_categoria=chart_categoria, chart_urgencia=chart_urgencia,
            perfis=perfis, silhueta=ag["silhueta"], chart_clusters=chart_clusters,
            recentes=recentes,
        )

    @app.get("/avaliacao")
    def avaliacao():
        ctx = contexto()
        cl = ctx.classificador
        historico = ctx.df
        pesos = pesos_atuais()
        risco_oof = cl["risco_oof"]
        _, _, pontuacao_oof = m.pontuar(historico, risco_oof, pesos, ctx.resumo["p95_usuarios"])
        top_n = min(cfg.TOPO_VALIDACAO, len(historico))
        comparacao = m.comparar_politicas(historico, pontuacao_oof, risco_oof, top_n)
        comparacao_records = comparacao.replace({np.nan: None}).to_dict("records")
        curva = m.curva_politicas(historico, pontuacao_oof, risco_oof)
        fig_curva = px.line(
            curva, x="N", y="% das violações capturadas", color="Política", markers=True,
            title="Violações capturadas conforme o tamanho do topo",
            labels={"N": "Quantidade de chamados considerados"},
            color_discrete_sequence=[PALETA["cinza"], PALETA["verde"], PALETA["azul"]],
        )
        fig_curva.add_vline(x=top_n, line_dash="dash", line_color="#8A98A9")
        chart_curva = fig_html(fig_curva, 350)
        sensibilidade = m.analise_sensibilidade(ctx, pesos, min(20, len(historico)))
        sensibilidade_records = sensibilidade.replace({np.nan: None}).to_dict("records")

        cm = cl["matriz_confusao"]
        cm_html = "<table class='matrix'><thead><tr><th></th><th>Previsto: não</th><th>Previsto: sim</th></tr></thead>"
        cm_html += "<tbody>"
        cm_html += f"<tr><th>Real: não</th><td>{int(cm[0,0])}</td><td>{int(cm[0,1])}</td></tr>"
        cm_html += f"<tr><th>Real: sim</th><td>{int(cm[1,0])}</td><td>{int(cm[1,1])}</td></tr></tbody></table>"

        metricas = cl["metricas_teste"].to_dict()
        metricas_base = cl["metricas_base"].to_dict()
        validacao_cv = cl["validacao_cruzada"].reset_index(names="Métrica").to_dict("records")
        importancias = cl["importancias"].head(8).reset_index()
        importancias.columns = ["atributo", "importancia"]
        fig_imp = px.bar(importancias.sort_values("importancia"), x="importancia", y="atributo", orientation="h",
                         color_discrete_sequence=[PALETA["verde"]], title="Atributos mais utilizados pela árvore")
        chart_importancias = fig_html(fig_imp, 320)

        total_violacoes = int(historico["violou_sla"].sum())
        return render_template(
            "evaluation.html", metricas=metricas, metricas_base=metricas_base,
            n_treino=cl["n_treino"], n_teste=cl["n_teste"], violacoes_teste=cl["violacoes_teste"],
            matriz_confusao=cm_html, validacao_cv=validacao_cv,
            comparacao=comparacao_records, top_n=top_n, total_violacoes=total_violacoes,
            chart_curva=chart_curva,
            sensibilidade=sensibilidade_records, chart_importancias=chart_importancias,
            recall_baixo=metricas.get("Recall", 0) < 0.65,
            aviso_classificador=cl.get("aviso"),
            regras=cl["regras"],
        )

    @app.get("/exportar/fila.csv")
    def exportar_fila():
        ctx = contexto()
        abertos = repo.listar_abertos()
        if abertos.empty:
            df = pd.DataFrame(columns=["id_chamado", "titulo", "status"])
        else:
            fila, _, _ = m.montar_fila(ctx, abertos, pesos_atuais(), capacidade_atual())
            df = fila
        return _csv_response(df, "fila_recomendada.csv")

    @app.get("/exportar/historico.csv")
    def exportar_historico():
        df = repo.listar_resolvidos_df()
        return _csv_response(df, "historico_ultimos_resolvidos.csv")

    @app.errorhandler(404)
    def nao_encontrado(_erro):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def erro_interno(_erro):
        return render_template("500.html"), 500

    return app


def _csv_response(df: pd.DataFrame, filename: str):
    content = df.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig")
    response = make_response(content)
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


app = criar_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=(not _ambiente_producao() and _env_flag("FLASK_DEBUG", False)))
