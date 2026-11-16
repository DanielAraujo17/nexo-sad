"""Testes dos scripts auxiliares (reinício seguro da demonstração e números do relatório)."""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

import config as cfg  # noqa: E402
import reiniciar_demo  # noqa: E402
import resultados_relatorio  # noqa: E402
from database import RepositorioChamados  # noqa: E402


def _payload(titulo):
    return {"titulo": titulo, "categoria": "Rede", "urgencia": "Média", "complexidade": "Baixa",
            "usuarios_afetados": 4, "criticidade": 3, "recorrencias": 0}


def test_reiniciar_demo_simulacao_nao_altera_nada(tmp_path):
    url = f"sqlite:///{(tmp_path / 'demo.db').as_posix()}"
    repo = RepositorioChamados(url)
    repo.inicializar()
    repo.criar(_payload("Chamado do usuário"))
    antes = repo.contagens()
    res = reiniciar_demo.reiniciar(repo, tmp_path / "backups", confirmar=False)
    assert res["executado"] is False
    assert repo.contagens() == antes
    assert not (tmp_path / "backups").exists()
    repo.fechar()


def test_reiniciar_demo_faz_backup_antes_de_apagar_e_restaura_base_inicial(tmp_path):
    url = f"sqlite:///{(tmp_path / 'demo.db').as_posix()}"
    repo = RepositorioChamados(url)
    repo.inicializar()
    repo.criar(_payload("Chamado do usuário"))
    res = reiniciar_demo.reiniciar(repo, tmp_path / "backups", confirmar=True)
    assert res["executado"] is True
    backup = pd.read_csv(res["backup"])
    assert len(backup) == cfg.N_HISTORICO + cfg.N_ABERTOS_INICIAIS + 1
    assert "Chamado do usuário" in set(backup["titulo"])  # o registro do usuário está no backup
    assert list((tmp_path / "backups").glob("chamados_*.db"))  # cópia do arquivo SQLite
    assert repo.contagens()["resolvidos"] == cfg.N_HISTORICO
    assert repo.contagens()["abertos"] == cfg.N_ABERTOS_INICIAIS
    repo.fechar()


def test_resultados_relatorio_reproduz_a_base_sintetica():
    res = resultados_relatorio.calcular()
    assert res["base"]["resolvidos"] == 200
    assert res["base"]["violacoes"] == 65
    assert res["base"]["taxa_violacao_pct"] == 32.5
    assert res["classificacao"]["n_treino"] == 160 and res["classificacao"]["n_teste"] == 40
    assert sum(res["classificacao"]["matriz_confusao"].values()) == 40
    assert len(res["caso_concreto"]["chamados"]) == len(resultados_relatorio.CASO_CONCRETO)
    pontos = [c["pontuacao"] for c in res["caso_concreto"]["chamados"]]
    assert pontos == sorted(pontos, reverse=True)
