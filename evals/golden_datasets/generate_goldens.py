import os
import json
import re
from pathlib import Path

from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.models import OllamaModel

from deepeval.synthesizer.config import EvolutionConfig, StylingConfig
from deepeval.dataset import EvaluationDataset
from src.data_ingesations.load_docs import load_web_page, URLS
from src.data_ingesations.chunking import create_chunks

# --------------------------------------------------
# 1. Environment configuration
# --------------------------------------------------

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1000"

client = OllamaModel(
    model="qwen2.5:1.5b", base_url="http://localhost:11434", temperature=0
)


# --------------------------------------------------
# 2. Generate question evolution configuration
# --------------------------------------------------

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.CONCRETIZING: 0.40,
        Evolution.CONSTRAINED: 0.30,
        Evolution.COMPARATIVE: 0.30,
    },
    num_evolutions=1,
)


# --------------------------------------------------
# 3. Question and answer styling
# --------------------------------------------------

styling_config = StylingConfig(
    input_format=(
        "Generate realistic, self-contained questions "
        "about the supplied technical documentation. "
        "Include factual, conceptual and practical "
        "application questions. Avoid vague questions."
    ),
    expected_output_format=(
        "Generate a correct and concise answer using "
        "only the supplied context. Do not introduce "
        "unsupported facts or assumptions."
    ),
    task=(
        "Generate high-quality evaluation questions "
        "and reference answers for a RAG chatbot."
    ),
    scenario=(
        "Users ask technical questions about information " "stored in a knowledge base."
    ),
)


# --------------------------------------------------
# 4. Load documents and create contexts
# --------------------------------------------------

contexts = []
source_files = []

for url in URLS:

    print(f"Loading: {url}")

    text = load_web_page(url)

    chunks = create_chunks(text)

    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:

        if not chunk or len(chunk.split()) < 100:
            continue

        contexts.append([chunk])
        source_files.append(url)


print("Total contexts:", len(contexts))


# --------------------------------------------------
# 5. Initialize Synthesizer
# --------------------------------------------------

synthesizer = Synthesizer(
    model=client,
    async_mode=False,
    max_concurrent=1,
    evolution_config=evolution_config,
    styling_config=styling_config,
)


# --------------------------------------------------
# 6. Generate goldens
# --------------------------------------------------

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    source_files=source_files,
    include_expected_output=True,
    max_goldens_per_context=1,
)

dataset = EvaluationDataset(goldens=goldens)

dataset.save_as(
    file_type="json", directory="data", file_name="retriever_golden_dataset"
)


# --------------------------------------------------
# 7. Validate and remove duplicate questions
# --------------------------------------------------

# dataset = []
# seen_questions = set()

# for idx, golden in enumerate(goldens):

#     question = (golden.input or "").strip()
#     answer = (golden.expected_output or "").strip()

#     if not question or not answer:
#         continue

#     normalized = re.sub(r"\s+", " ", question.lower())

#     if normalized in seen_questions:
#         continue

#     if len(question.split()) < 5:
#         continue

#     seen_questions.add(normalized)

#     dataset.append(
#         {
#             "id": idx,
#             "input": question,
#             "expected_output": answer,
#             "context": golden.context,
#         }
#     )


# # --------------------------------------------------
# # 8. Save dataset
# # --------------------------------------------------

# output_path = Path("data/retriever_golden_dataset.json")

# output_path.parent.mkdir(parents=True, exist_ok=True)

# with open(output_path, "w", encoding="utf-8") as file:

#     json.dump(dataset, file, indent=4, ensure_ascii=False)


# print("\n" + "=" * 60)
# print("GOLDEN GENERATION COMPLETED")
# print("=" * 60)

# print("Total generated:", len(goldens))
# print("Total valid goldens:", len(dataset))
# print("Total duplicates removed:", len(goldens) - len(dataset))
# print("Saved:", output_path)
