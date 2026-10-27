"""Engine and session factory."""

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

CONNECT_TIMEOUT_S = 5


def make_engine(url: str) -> Engine:
    # pool_pre_ping discards connections the database closed while idle, so a
    # Postgres restart does not surface as a 500 on the next request.
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_S},
    )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def ping(engine: Engine) -> bool:
    with engine.connect() as connection:
        return connection.execute(text("SELECT 1")).scalar_one() == 1
