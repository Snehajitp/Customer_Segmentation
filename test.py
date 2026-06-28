from pymongo import MongoClient
import certifi

client = MongoClient(
    "your_mongo_url",
    tls=True,
    tlsCAFile=certifi.where()
)

print(client.list_database_names())