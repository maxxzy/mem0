from openai import OpenAI
import os
from mem0 import Memory

os.environ["OPENAI_API_KEY"] = "sk-XqFpYZIJj6Z3mCVGoCbu2k4SJYxRQrXFcuIsQvdLkdmxQWzJ"
os.environ["OPENAI_BASE_URL"] = "https://yansd666.top/v1"

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-nano-2025-04-14",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
        }
    }
}

# 初始化 Memory
memory = Memory.from_config(config)

# 初始化 OpenAI 客户端
openai_client = OpenAI(api_key="sk-XqFpYZIJj6Z3mCVGoCbu2k4SJYxRQrXFcuIsQvdLkdmxQWzJ",
                       base_url="https://yansd666.top/v1")

last_topics = []

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # Retrieve relevant memories
    global last_topics
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3,topics=last_topics)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])

    # Generate Assistant response
    system_prompt = (
        f"You are a helpful AI. Answer the question based on query and memories.\n"
        f"User Memories:\n{memories_str}\n\n"
        f"IMPORTANT: At the very end of your response, please summarize the relevant topics (at least one) of this conversation in the following format:\n"
        f"TOPICS: topic1, topic2, topic3"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    response = openai_client.chat.completions.create(model="gpt-4.1-nano-2025-04-14", messages=messages)

    full_response = response.choices[0].message.content

    # 解析回答和 Topics
    assistant_response = full_response
    topics = []

    if "TOPICS:" in full_response:
        parts = full_response.split("TOPICS:")
        assistant_response = parts[0].strip()
        topics_str = parts[1].strip()
        # 分割并清理 topic 列表
        topics = [t.strip() for t in topics_str.split(",") if t.strip()]

    # Create new memories from the conversation
    # 将解析出的 topics 传入 metadata
    messages.append({"role": "assistant", "content": assistant_response})

    if topics:
        memory.add(messages, user_id=user_id, metadata={"topics": topics})
    else:
        memory.add(messages, user_id=user_id)

    last_topics = topics

    return full_response

def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")

if __name__ == "__main__":
    main()