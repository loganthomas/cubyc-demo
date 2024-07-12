from cubyc import query

statement = """
                SELECT config.batch_size, metadata.runtime, logs.acc
                FROM config
                INNER JOIN metadata ON config.id = metadata.id
                INNER JOIN logs ON config.id = logs.id
                WHERE config.batch_size is not NULL
                AND logs.name = 'acc'
                ORDER BY metadata.runtime ASC
            """

print(query(statement=statement))
