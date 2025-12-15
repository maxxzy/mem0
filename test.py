import os
from mem0 import Memory

os.environ["OPENAI_API_KEY"] = "sk-0mCxucY3RJ6nS8619rJITPRY6tARrIueT3s2HFLbsQgxHyI5"
os.environ["OPENAI_BASE_URL"] = "https://yansd666.top/v1"

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-5.1",
            "temperature": 0,
            "max_tokens": 512,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0_topic_test",
            "host": "localhost",
            "port": 6333,
        }
    }
}

def pretty_results(label, res):
    print(f"\n=== {label} ===")
    for r in res["results"]:
        rid = r.get("id")
        mem = r.get("memory")
        print(f"id={rid} memory={mem}")


def test_topic():
    user_id = "u_demo"
    memory = Memory.from_config(config)

    mem_a = memory.add(
        messages=[
            {"role": "user", "content": "I love playing basketball on weekends."},
            {"role": "assistant", "content": "Got it."},
        ],
        user_id=user_id,
        metadata={"topics": ["sports", "basketball"]},
    )

    mem_b = memory.add(
        messages=[
            {"role": "user", "content": "I enjoy reading science fiction novels."},
            {"role": "assistant", "content": "Nice!"},
        ],
        user_id=user_id,
        metadata={"topics": ["reading", "sci-fi"]},
    )

    mem_c = memory.add(
        messages=[
            {"role": "user", "content": "Soccer practice is every Friday evening."},
            {"role": "assistant", "content": "Noted."},
        ],
        user_id=user_id,
        metadata={"topics": ["sports", "soccer"]},
    )

    mem_a_id = mem_a["results"][0]["id"] if mem_a.get("results") else None
    mem_b_id = mem_b["results"][0]["id"] if mem_b.get("results") else None
    mem_c_id = mem_c["results"][0]["id"] if mem_c.get("results") else None
    print(f"Created memories: A={mem_a_id} B={mem_b_id} C={mem_c_id}")

    res_sports = memory.search(query="What sports do I like?", user_id=user_id, topics=["sports"], limit=5)
    pretty_results("Search with topics=['sports']", res_sports)

    res_all = memory.search(query="What do I like?", user_id=user_id, limit=5)
    pretty_results("Search without topics", res_all)

    if mem_a_id:
        memory.delete(mem_a_id)
    mem_a2 = memory.add(
        messages=[
            {"role": "user", "content": "Actually, I now prefer reading tech blogs."},
            {"role": "assistant", "content": "Updated."},
        ],
        user_id=user_id,
        metadata={"topics": ["reading", "technology"]},
    )

    res_sports_after = memory.search(query="Sports schedule?", user_id=user_id, topics=["sports"], limit=5)
    pretty_results("Search sports after update", res_sports_after)

    res_reading = memory.search(query="Reading interests?", user_id=user_id, topics=["reading"], limit=5)
    pretty_results("Search with topics=['reading']", res_reading)

    mem_c = memory.add(
        messages=[
            {"role": "user", "content": "Soccer practice is every Monday morning."},
            {"role": "assistant", "content": "Noted."},
        ],
        user_id=user_id,
    )

    res_sports_after = memory.search(query="Sports schedule?", user_id=user_id, topics=["sports"], limit=5)
    pretty_results("Search sports after two updates", res_sports_after)


if __name__ == "__main__":
    test_topic()