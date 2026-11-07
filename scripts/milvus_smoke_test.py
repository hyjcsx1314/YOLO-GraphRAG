from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="http://127.0.0.1:19530",
    token="root:Milvus"
)

name = "milvus_smoke_v1"

print("Connected to Milvus")
print("Collections:", client.list_collections())

if not client.has_collection(collection_name=name):
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False
    )

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True
    )
    schema.add_field(
        field_name="paper_id",
        datatype=DataType.VARCHAR,
        max_length=64
    )
    schema.add_field(
        field_name="page",
        datatype=DataType.INT64
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=2048
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=4
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params
    )

    result = client.insert(
        collection_name=name,
        data=[
            {
                "id": 1,
                "paper_id": "YOLOX",
                "page": 1,
                "text": "YOLOX uses an anchor-free design.",
                "vector": [1.0, 0.0, 0.0, 0.0]
            },
            {
                "id": 2,
                "paper_id": "YOLOv10",
                "page": 1,
                "text": "YOLOv10 proposes NMS-free detection.",
                "vector": [0.0, 1.0, 0.0, 0.0]
            }
        ]
    )
    print("Insert:", result)

results = client.search(
    collection_name=name,
    data=[[1.0, 0.0, 0.0, 0.0]],
    anns_field="vector",
    search_params={"metric_type": "COSINE"},
    limit=2,
    output_fields=["paper_id", "page", "text"],
    consistency_level="Strong"
)

for hit in results[0]:
    print(
        "id:", hit["id"],
        "score:", round(hit["distance"], 4),
        "data:", hit["entity"]
    )

client.close()
print("Milvus smoke test passed")
