from cubyc import query

statement = """
                SELECT config.batch_size, metadata.runtime
                FROM config
                INNER JOIN metadata ON config.id = metadata.id
                WHERE config.batch_size is not NULL
                ORDER BY metadata.runtime ASC
            """

print(query(statement=statement))
