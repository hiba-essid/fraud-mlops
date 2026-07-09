from kafka import KafkaConsumer
import psycopg2, json

consumer = KafkaConsumer('transactions',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m))

conn = psycopg2.connect("dbname=fraud password=secret")
cur  = conn.cursor()

for msg in consumer:
    tx = msg.value
    cur.execute(
        "INSERT INTO transactions VALUES (%s,%s,%s,%s)",
        (tx['id'], tx['amount'], tx['merchant'], tx['label'])
    )
    conn.commit()