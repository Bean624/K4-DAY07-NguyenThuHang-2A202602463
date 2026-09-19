from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)

        if not results:
            return "Không tìm thấy thông tin liên quan."

        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[{i}] {r['content']}")
        context = "\n".join(context_parts)

        prompt = (
            "Bạn là trợ lý tra cứu thông tin. Chỉ trả lời dựa trên ngữ cảnh được cung cấp bên dưới. "
            "Nếu thông tin không có trong ngữ cảnh, hãy nói 'Không tìm thấy thông tin liên quan'.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Câu trả lời:"
        )

        return self.llm_fn(prompt)
