import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/datasets/all_chunks_v3.jsonl"
RESULT_DIR = ROOT / "results/milvus"

COLLECTION = "yolo_text_e5s_v1"
MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384
WINDOW = 450
OVERLAP = 45


def connect():
    return MilvusClient(
        uri="http://127.0.0.1:19530",
        token="root:Milvus"
    )


def load_model():
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )
    assert model.get_sentence_embedding_dimension() == DIM
    return model


def load_chunks():
    chunks = []

    with SOURCE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            obj = json.loads(line)

            for key in ("chunk_id", "paper_id", "page", "text"):
                if key not in obj:
                    raise ValueError(
                        f"Line {line_no}: missing {key}"
                    )

            if not isinstance(obj["text"], str):
                raise ValueError(
                    f"Line {line_no}: text must be string"
                )

            text = obj["text"].strip()
            if not text:
                raise ValueError(
                    f"Line {line_no}: empty text"
                )

            chunk = {
                "chunk_id": str(obj["chunk_id"]),
                "paper_id": str(obj["paper_id"]),
                "page": int(obj["page"]),
                "text": text
            }

            chunks.append(chunk)

    if len(chunks) != 78:
        raise ValueError(
            f"Expected 78 source chunks, got {len(chunks)}"
        )

    return chunks


def split_passages(text, tokenizer):
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    step = WINDOW - OVERLAP
    parts = []

    for start in range(0, len(tokens), step):
        piece = tokenizer.decode(
            tokens[start:start + WINDOW],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        if piece:
            parts.append(piece)

        if start + WINDOW >= len(tokens):
            break

    return parts


def create_collection(client):
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False
    )

    schema.add_field(
        "id", DataType.INT64,
        is_primary=True
    )
    schema.add_field(
        "chunk_id", DataType.VARCHAR,
        max_length=256
    )
    schema.add_field(
        "paper_id", DataType.VARCHAR,
        max_length=128
    )
    schema.add_field(
        "page", DataType.INT64
    )
    schema.add_field(
        "piece_index", DataType.INT64
    )
    schema.add_field(
        "text", DataType.VARCHAR,
        max_length=16384
    )
    schema.add_field(
        "vector", DataType.FLOAT_VECTOR,
        dim=DIM
    )

    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=indexes,
        consistency_level="Strong"
    )


def ingest():
    chunks = load_chunks()

    print("Source chunks:", len(chunks))
    print(
        "Papers:",
        dict(Counter(c["paper_id"] for c in chunks))
    )

    model = load_model()
    tokenizer = model.tokenizer

    rows = []

    for chunk in chunks:
        parts = split_passages(
            chunk["text"], tokenizer
        )

        for piece_index, piece in enumerate(parts):
            if len(piece.encode("utf-8")) > 16384:
                raise ValueError("Text piece exceeds Milvus limit")

            if len(chunk["chunk_id"].encode("utf-8")) > 256:
                raise ValueError("chunk_id too long")

            if len(chunk["paper_id"].encode("utf-8")) > 128:
                raise ValueError("paper_id too long")

            length = len(
                tokenizer.encode(
                    "passage: " + piece,
                    add_special_tokens=True
                )
            )

            if length > model.max_seq_length:
                raise ValueError(
                    f"Piece too long: {length} tokens"
                )

            rows.append({
                "id": len(rows) + 1,
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "page": chunk["page"],
                "piece_index": piece_index,
                "text": piece
            })

    print("Vector records:", len(rows))

    vectors = model.encode(
        ["passage: " + r["text"] for r in rows],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    for row, vector in zip(rows, vectors):
        row["vector"] = vector.tolist()

    client = connect()

    if not client.has_collection(
        collection_name=COLLECTION
    ):
        create_collection(client)
        print("Collection created:", COLLECTION)
    else:
        print("Collection exists, updating records")

    client.load_collection(
        collection_name=COLLECTION
    )

    for i in range(0, len(rows), 32):
        batch = rows[i:i + 32]
        client.upsert(
            collection_name=COLLECTION,
            data=batch
        )

    result = client.query(
        collection_name=COLLECTION,
        filter="",
        output_fields=["count(*)"],
        consistency_level="Strong"
    )

    count = int(result[0]["count(*)"])
    print("Milvus count:", count)

    if count != len(rows):
        raise RuntimeError(
            f"Count mismatch: {count} != {len(rows)}"
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "collection": COLLECTION,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": DIM,
        "metric": "COSINE",
        "device": "cpu",
        "source_chunks": len(chunks),
        "stored_vectors": count,
        "window_tokens": WINDOW,
        "overlap_tokens": OVERLAP,
        "source_sha256": hashlib.sha256(
            SOURCE.read_bytes()
        ).hexdigest()
    }

    (RESULT_DIR / "text_index_v1.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    client.close()
    print("TEXT INDEX BUILD PASSED")


def search(question, topk):
    model = load_model()
    client = connect()

    if not client.has_collection(
        collection_name=COLLECTION
    ):
        raise RuntimeError(
            "Collection not found. Run ingest first."
        )

    client.load_collection(
        collection_name=COLLECTION
    )

    vector = model.encode(
        ["query: " + question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].tolist()

    results = client.search(
        collection_name=COLLECTION,
        data=[vector],
        anns_field="vector",
        search_params={
            "metric_type": "COSINE"
        },
        limit=topk,
        output_fields=[
            "chunk_id",
            "paper_id",
            "page",
            "piece_index",
            "text"
        ],
        consistency_level="Strong"
    )

    print("\nQuestion:", question)
    print("Collection:", COLLECTION)

    for rank, hit in enumerate(results[0], 1):
        entity = hit["entity"]

        print(f"\nTop {rank}")
        print("Paper:", entity["paper_id"])
        print("PDF page:", entity["page"])
        print("Chunk:", entity["chunk_id"])
        print("Part:", entity["piece_index"])
        print("Cosine:", round(hit["distance"], 4))
        print(
            "Text:",
            entity["text"][:450].replace("\n", " ")
        )

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["ingest", "search"]
    )
    parser.add_argument(
        "--question",
        default="How does YOLOX use SimOTA?"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if args.action == "ingest":
        ingest()
    else:
        search(args.question, args.topk)
