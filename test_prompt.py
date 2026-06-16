from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Hello {name}"),
])

try:
    print(prompt.invoke({"name": "David", "extra": "data"}))
except Exception as e:
    print(f"ERROR: {e}")
