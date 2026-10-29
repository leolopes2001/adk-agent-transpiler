from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

CONNECT_TIMEOUT_S = 5


def make_engine(url: str) -> Engine:
    return create_engine(
        url,
        # Discards connections Postgres closed while idle, so a restart is not a 500.
        pool_pre_ping=True,
        # Keeps request data out of exception messages, which end up in the 500 log.
        hide_parameters=True,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_S},
    )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def ping(engine: Engine) -> bool:
    with engine.connect() as connection:
        return connection.execute(text("SELECT 1")).scalar_one() == 1
