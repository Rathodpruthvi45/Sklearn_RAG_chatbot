import json
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.models import OpenAIModel
from src.data_ingesations.load_docs import load_web_page, URLS
from src.data_ingesations.chunking import create_chunks

client = OpenAIModel(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
    model="meta/muse-glimmer-30b",
)

context = []

for url in URLS:

    print(f"Loading {url}")

    text = load_web_page(url)
    chunks = create_chunks(text)

    print(f"Chunks created : {len(chunks)}")

    for chunk in chunks:
        if len(chunk.strip()) > 100:
            context.append([chunk])


print(f"Total context available : {len(context)}")

# Step 2: Initialize DeepEval Synthesize

synthesizer = Synthesizer(model=client, async_mode=False, max_concurrent=1)

# Step 3: Generate synthetic goldens

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=context[:3], include_expected_output=True, max_goldens_per_context=1
)


# Step 4: Display the generated goldens
for i, golden in enumerate(goldens, start=1):
    print("\n" + "=" * 80)
    print(f"GOLDEN {i}")
    print("=" * 80)

    print("QUESTION:", golden.input)
    print("EXPECTED ANSWER:", golden.expected_output)
    print("CONTEXT:", golden.context)


# Step 5: Save the dataset as JSON

dataset = []

for golden in goldens:

    dataset.append(
        {
            "input": golden.input,
            "expected_output": golden.expected_output,
            "context": golden.context,
        }
    )


output_path = Path("data/generated_goldens.json")
output_path.parent.mkdir(parents=True, exist_ok=True)


with open(output_path, "w", encoding="utf-8") as file:
    json.dump(dataset, file, indent=4, ensure_ascii=False)

print(f"\nSaved {len(dataset)} goldens to {output_path}")
