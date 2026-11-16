"""Testes de integração das páginas e do fluxo web Flask."""
import os

# Evita que a aplicação WSGI global (app = criar_app()) toque no banco real durante a coleta.
# Atribuição direta (e não setdefault): mesmo que o terminal tenha DATABASE_URL apontando
# para o Supabase, os testes nunca usam esse banco.
os.environ["DATABASE_URL"] = "sqlite://"
os.environ.setdefault("FLASK_SECRET_KEY", "chave-de-teste-nao-usar-em-producao")
os.environ.pop("APP_ENV", None)
os.environ.pop("RENDER", None)

import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_wtf")

from app import criar_app  # noqa: E402
from database import RepositorioChamados  # noqa: E402


@pytest.fixture
def web_app():
    repo = RepositorioChamados("sqlite://", historico_limite=200)
    repo.inicializar(popular_demo=True)
    app = criar_app(repositorio=repo, inicializar_banco=False)
    app.config.update(TESTING=True, WTF_CSRF_ENABLED=False, SECRET_KEY="segredo-de-teste")
    yield app, repo
    repo.fechar()


def test_paginas_principais_e_health_renderizam(web_app):
    app, repo = web_app
    client = app.test_client()
    primeiro_aberto = int(repo.listar_abertos().iloc[0]["id_chamado"])

    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get(f"/chamados/{primeiro_aberto}").status_code == 200
    resposta_analises = client.get("/analises")
    assert resposta_analises.status_code == 200
    assert "IDADE NA AVALIAÇÃO" in resposta_analises.get_data(as_text=True)
    assert client.get("/avaliacao").status_code == 200
    assert client.get("/rota-inexistente").status_code == 404


def test_fluxo_web_cadastro_fila_detalhe_e_resolucao(web_app):
    app, repo = web_app
    client = app.test_client()

    resposta = client.post("/chamados", data={
        "titulo": "Equipe sem acesso à rede",
        "categoria": "Rede",
        "urgencia": "Alta",
        "complexidade": "Média",
        "usuarios_afetados": "12",
        "criticidade": "4",
        "recorrencias": "1",
    })
    assert resposta.status_code == 302
    caminho = resposta.headers["Location"]
    detalhe = client.get(caminho)
    assert detalhe.status_code == 200
    assert "RECOMENDAÇÃO DO SAD" in detalhe.get_data(as_text=True)
    chamado_id = int(caminho.rstrip("/").split("/")[-1])
    assert repo.obter(chamado_id)["status"] == "aberto"

    fila = client.get("/")  # gera e registra o snapshot de idade
    assert fila.status_code == 200
    assert f"#{chamado_id}" in fila.get_data(as_text=True)
    resolucao = client.post(f"/chamados/{chamado_id}/resolver")
    assert resolucao.status_code == 302
    encerrado = repo.obter(chamado_id)
    assert encerrado["status"] == "resolvido"
    assert encerrado["tempo_resolucao_horas"] is not None
    assert encerrado["violou_sla"] in (0, 1)
    resposta_analises = client.get("/analises")
    assert resposta_analises.status_code == 200
    assert "IDADE NA AVALIAÇÃO" in resposta_analises.get_data(as_text=True)
    assert client.get("/avaliacao").status_code == 200


def test_preferencias_e_exports_funcionam(web_app):
    app, _repo = web_app
    client = app.test_client()
    resposta = client.post("/preferencias", data={
        "peso_urgencia": "40",
        "peso_criticidade": "20",
        "peso_usuarios": "10",
        "peso_idade": "10",
        "peso_recorrencias": "10",
        "peso_risco": "10",
        "capacidade": "6",
    })
    assert resposta.status_code == 302
    assert client.post("/preferencias/restaurar").status_code == 302
    assert client.get("/exportar/fila.csv").status_code == 200
    assert client.get("/exportar/historico.csv").status_code == 200


def test_health_informa_tipo_de_banco_sem_expor_credenciais(web_app):
    app, _repo = web_app
    dados = app.test_client().get("/health").get_json()
    assert dados == {"status": "ok", "banco": "sqlite"}


def test_resolucao_mantem_historico_no_limite(web_app):
    app, repo = web_app
    client = app.test_client()
    antes = repo.contagens()
    assert antes["resolvidos"] == 200
    primeiro_aberto = int(repo.listar_abertos().iloc[0]["id_chamado"])
    assert client.post(f"/chamados/{primeiro_aberto}/resolver").status_code == 302
    depois = repo.contagens()
    assert depois["resolvidos"] == 200  # entrou 1 e saiu o mais antigo
    assert depois["abertos"] == antes["abertos"] - 1
    assert repo.obter(1) is None  # o resolvido mais antigo foi descartado pela retenção


def test_fluxo_com_csrf_habilitado_como_no_navegador():
    """Os demais testes desligam o CSRF; aqui o token é lido da página, como faz o navegador."""
    repo = RepositorioChamados("sqlite://", historico_limite=200)
    repo.inicializar(popular_demo=True)
    app = criar_app(repositorio=repo, inicializar_banco=False)
    app.config.update(TESTING=True, WTF_CSRF_ENABLED=True, SECRET_KEY="segredo-de-teste")
    client = app.test_client()
    try:
        pagina = client.get("/").get_data(as_text=True)
        token = re.search(r'name="csrf_token" value="([^"]+)"', pagina).group(1)
        dados = {"titulo": "Teste com CSRF", "categoria": "Rede", "urgencia": "Alta", "complexidade": "Média",
                 "usuarios_afetados": "5", "criticidade": "4", "recorrencias": "0"}
        assert client.post("/chamados", data=dados).status_code == 400  # sem token: recusado
        resposta = client.post("/chamados", data={**dados, "csrf_token": token})
        assert resposta.status_code == 302
        novo_id = int(resposta.headers["Location"].rstrip("/").split("/")[-1])
        assert repo.obter(novo_id)["status"] == "aberto"
        assert client.post(f"/chamados/{novo_id}/resolver", data={"csrf_token": token}).status_code == 302
        assert repo.obter(novo_id)["status"] == "resolvido"
    finally:
        repo.fechar()


def test_preferencias_invalidas_nao_quebram_a_pagina(web_app):
    app, _repo = web_app
    client = app.test_client()
    resposta = client.post("/preferencias", data={"peso_urgencia": "abc", "peso_criticidade": "-5",
                                                  "peso_usuarios": "999", "capacidade": "xyz"})
    assert resposta.status_code == 302
    assert client.get("/").status_code == 200


def test_producao_recusa_chave_de_exemplo_e_chave_ausente(monkeypatch):
    repo = RepositorioChamados("sqlite://", historico_limite=200)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("FLASK_SECRET_KEY", "troque-por-uma-chave-aleatoria-longa")
    with pytest.raises(RuntimeError):
        criar_app(repositorio=repo, inicializar_banco=False)
    monkeypatch.delenv("FLASK_SECRET_KEY")
    with pytest.raises(RuntimeError):
        criar_app(repositorio=repo, inicializar_banco=False)
    monkeypatch.setenv("FLASK_SECRET_KEY", "x" * 40)
    assert criar_app(repositorio=repo, inicializar_banco=False) is not None
    repo.fechar()
