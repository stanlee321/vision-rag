from smart_llm_loader import SmartLLMLoader
import json

# # Using Gemini Flash model
# os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"
# model = "gemini/gemini-1.5-flash"

# Using openai model
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
model = "openai/gpt-4o-mini"

# # Using anthropic model
# os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"
# model = "anthropic/claude-3-5-sonnet"


# api_base="http://localhost:11434"
# Initialize the document loader
loader = SmartLLMLoader(
    file_path="./data/2502.06472v1.pdf",
    chunk_strategy="contextual",
    model=model,
    # api_base=api_base,
)
# Load and split the document into chunks
documents = loader.load_and_split()

print(documents)

for i, doc in enumerate(documents):
    print(doc.metadata)
    
    if doc.metadata['page'] == 16:
        with open(f"documents_{i}.json", "w") as f:
            json.dump(doc.page_content, f)
