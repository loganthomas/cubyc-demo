from cubyc import query

# statement = """
#                 SELECT config.batch_size, metadata.runtime, logs.acc
#                 FROM config
#                 INNER JOIN metadata ON config.id = metadata.id
#                 INNER JOIN logs ON config.id = logs.id
#                 WHERE config.batch_size is not NULL
#                 AND logs.name = 'acc'
#                 ORDER BY metadata.runtime ASC
#             """

# statement = """
#             SELECT config.id, config.batch_size, logs.name, logs.value
#             FROM config
#             INNER JOIN logs ON config.id = logs.id
#             WHERE logs.name = 'test_loss' OR logs.name = 'acc'
#             GROUP BY config.id
#             """

statement = """
            SELECT config.batch_size
            FROM config
            """

print(query(statement=statement))
