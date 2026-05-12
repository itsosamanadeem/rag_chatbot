from app.services.vector_store import get_vector_store
from app.services.sql_generator import generate_sql
from app.services.sql_executor import execute_sql
from app.services.response_generator import generate_human_response


def query_database(
    db_id: str,
    db_url: str,
    query: str,
):

    vector_store = get_vector_store(db_id)

    # --------------------------------
    # VECTOR SEARCH
    # --------------------------------
    docs = vector_store.max_marginal_relevance_search(
        query,
        k=2,
        fetch_k=10,
    )

    context = "\n\n".join([
        d.page_content
        for d in docs
    ])

    # --------------------------------
    # GENERATE SQL
    # --------------------------------
    sql = generate_sql(query, context)

    # --------------------------------
    # EXECUTE SQL
    # --------------------------------
    result = execute_sql(db_url, sql)

    # --------------------------------
    # HUMAN RESPONSE
    # --------------------------------
    answer = generate_human_response(
        question=query,
        sql=sql,
        result=result,
    )

    return {
        "answer": answer,
        "sql": sql,
        "data": result,
    }