"""
reiniciar_demo.py - Reinicia a demonstração com a base sintética inicial (200 resolvidos + 25 abertos).

Por que existe: a idade dos chamados abertos é calculada em tempo real (agora - abertura).
Depois de alguns dias, os 25 chamados de demonstração ficam muito mais velhos que qualquer
chamado do histórico; o classificador passa a atribuir 100% de risco a todos e a fila perde
poder de discriminação. Reiniciar a demonstração antes de gravar o vídeo ou de avaliar o
sistema restaura o cenário inicial.

SEGURANÇA
  * Sem --confirmar, nada é alterado: o script só mostra o que faria (simulação).
  * Com --confirmar, ANTES de apagar, exporta todos os registros para backups/chamados_<data>.csv
    (e, em SQLite, copia também o arquivo .db).
  * Em PostgreSQL/Supabase é exigida, além disso, a digitação de REINICIAR.
  * Apaga SOMENTE os registros da tabela 'chamados' desta aplicação.

Uso (na pasta do projeto):
    python scripts/reiniciar_demo.py                 # simulação
    python scripts/reiniciar_demo.py --confirmar     # executa, com backup
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAIZ / ".env")

from database import Chamado, RepositorioChamados  # noqa: E402


def exportar_backup(repo: RepositorioChamados, pasta: Path) -> Path:
    """Grava todos os registros da tabela em CSV (UTF-8) e, em SQLite, copia o arquivo do banco."""
    pasta.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = pasta / f"chamados_{carimbo}.csv"
    df = pd.read_sql(select(Chamado), repo.engine)
    df.to_csv(destino, index=False, encoding="utf-8")
    if repo.dialeto == "sqlite":
        origem = repo.engine.url.database
        if origem and origem != ":memory:" and Path(origem).exists():
            shutil.copy2(origem, pasta / f"chamados_{carimbo}.db")
    return destino


def reiniciar(repo: RepositorioChamados, pasta_backup: Path, confirmar: bool, digitado: str | None = None) -> dict:
    """Executa (ou simula) o reinício. ``digitado`` é a confirmação textual exigida fora do SQLite."""
    repo.inicializar(popular_demo=False)  # garante que a tabela exista
    antes = repo.contagens()
    resultado = {"dialeto": repo.dialeto, "antes": antes, "executado": False, "backup": None}
    if not confirmar:
        return resultado
    if repo.dialeto != "sqlite" and (digitado or "").strip() != "REINICIAR":
        raise PermissionError("Confirmação textual ausente: digite REINICIAR para alterar um banco PostgreSQL.")
    resultado["backup"] = exportar_backup(repo, pasta_backup)
    with repo.Session() as session:
        session.execute(delete(Chamado))
        session.commit()
    repo.inicializar(popular_demo=True)
    resultado["depois"] = repo.contagens()
    resultado["executado"] = True
    return resultado


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmar", action="store_true", help="Executa de fato (com backup). Sem isto, apenas simula.")
    ap.add_argument("--backup-dir", default=str(RAIZ / "backups"), help="Pasta dos backups (padrão: backups/).")
    args = ap.parse_args()

    repo = RepositorioChamados()  # usa DATABASE_URL ou instance/chamados.db
    print(f"Banco em uso: {repo.dialeto}")
    digitado = None
    if args.confirmar and repo.dialeto != "sqlite":
        print("ATENÇÃO: este é um banco PostgreSQL (possivelmente o Supabase da versão publicada).")
        digitado = input("Para continuar, digite REINICIAR: ")
    try:
        res = reiniciar(repo, Path(args.backup_dir), args.confirmar, digitado)
    except PermissionError as exc:
        print(str(exc))
        return 2
    a = res["antes"]
    print(f"Antes: {a['resolvidos']} resolvidos e {a['abertos']} abertos.")
    if not res["executado"]:
        print("SIMULAÇÃO: nada foi alterado. Use --confirmar para reiniciar (um backup é criado antes).")
        return 0
    print(f"Backup salvo em: {res['backup']}")
    d = res["depois"]
    print(f"Depois: {d['resolvidos']} resolvidos e {d['abertos']} abertos (base sintética inicial).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
