"""Measure retrieval quality: does the search surface the moment that answers the question?

    python -m evals.retrieval_eval

Compares several chunk sizes on hand-labelled questions. A chunk counts as a hit
when it overlaps the labelled answer by at least MIN_OVERLAP seconds. Needs
internet access for the captions but no API key.
"""

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from video_chatbot.chunking import chunk_segments
from video_chatbot.vector_store import VectorStore
from video_chatbot.youtube import load_video

CONFIGS = {  # target, max and overlap words
    "default": (50, 75, 15),
    "no overlap": (50, 75, 0),
    "medium": (110, 150, 30),
    "large": (250, 300, 60),
}
KS = (1, 3, 6, 8)
MIN_OVERLAP = 5.0


def first_hit(results, answers) -> int | None:
    """1-based rank of the first chunk covering one of the labelled answers."""
    for rank, result in enumerate(results, 1):
        for start, end in answers:
            if min(result.chunk.end, end) - max(result.chunk.start, start) >= MIN_OVERLAP:
                return rank
    return None


def main() -> None:
    data = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
    info, segments = load_video(data["video"])
    questions = data["questions"]
    print(f"{info.title}: {len(questions)} questions\n")
    print(f"{'config':<12}{'chunks':>7}{f'words@{max(KS)}':>9}" + "".join(f"{f'hit@{k}':>8}" for k in KS) + f"{'MRR':>7}")

    # Chroma keeps its files open on Windows, so let the OS clean up if we can't.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        store = VectorStore(Path(folder))
        for i, (name, sizes) in enumerate(CONFIGS.items()):
            video = replace(info, video_id=f"eval-{i}")
            chunks = chunk_segments(segments, *sizes)
            store.add(video, chunks)
            ranks, words, misses = [], 0, []
            for q in questions:
                results = store.search(video.video_id, q["question"], max(KS))
                words += sum(len(r.chunk.text.split()) for r in results)
                ranks.append(first_hit(results, q["answers"]))
                if ranks[-1] is None:
                    misses.append(q["question"])
            hits = [sum(r is not None and r <= k for r in ranks) / len(ranks) for k in KS]
            mrr = sum(1 / r for r in ranks if r) / len(ranks)
            print(
                f"{name:<12}{len(chunks):>7}{words // len(questions):>9}"
                + "".join(f"{h:>8.0%}" for h in hits)
                + f"{mrr:>7.2f}"
            )
            for question in misses:
                print(f"    missed: {question}")


if __name__ == "__main__":
    main()
