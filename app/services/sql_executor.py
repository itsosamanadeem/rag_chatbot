from sqlalchemy import create_engine, text


def execute_sql(db_url: str, sql: str):

    engine = create_engine(db_url)

    with engine.connect() as conn:

        result = conn.execute(text(sql))

        rows = result.fetchall()

        columns = result.keys()

    return {
        "columns": list(columns),
        "rows": [list(r) for r in rows]
    }