from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.documents import Document


def extract_schema_documents(db_url: str):

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

    docs = []

    with engine.connect() as conn:
        # GET TABLES
        tables = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)).fetchall()

        for (table_name,) in tables:

            
            try:
                # FOREIGN KEYS
                foreign_keys = conn.execute(text("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_name = :table_name
                """), {"table_name": table_name}).fetchall()
                
                fk_text = "\n".join([
                    f"{fk[0]} -> {fk[1]}.{fk[2]}"
                    for fk in foreign_keys
                ])
                # -----------------------------
                # GET COLUMNS
                # -----------------------------
                columns = conn.execute(
                    text("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = :table_name
                    """),
                    {"table_name": table_name}
                ).fetchall()

                column_text = ", ".join(
                    [f"{c[0]} ({c[1]})" for c in columns]
                )

                content = f"""
                Table: {table_name}

                Columns:
                {column_text}

                Foreign Keys:
                {fk_text}
                """

                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "table": table_name,
                            "type": "schema"
                        }
                    )
                )

                # -----------------------------
                # SAMPLE ROWS
                # -----------------------------
                try:

                    sample_query = text(
                        f'SELECT * FROM "{table_name}" LIMIT 1'
                    )

                    rows = conn.execute(sample_query).fetchall()

                    for row in rows:

                        docs.append(
                            Document(
                                page_content=f"""
                                    Table: {table_name}

                                    Sample Row:
                                    {str(row)}
                                    """,
                                metadata={
                                    "table": table_name,
                                    "type": "sample_row"
                                }
                            )
                        )

                except Exception as row_error:

                    print(
                        f"[WARNING] Sample rows failed for table {table_name}: {row_error}"
                    )

                    # IMPORTANT
                    conn.rollback()

            except SQLAlchemyError as e:

                print(
                    f"[ERROR] Failed processing table {table_name}: {e}"
                )

                # VERY IMPORTANT
                conn.rollback()

                continue

    return docs