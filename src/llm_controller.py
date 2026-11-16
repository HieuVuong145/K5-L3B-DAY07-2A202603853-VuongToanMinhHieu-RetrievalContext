"""LLM Controller using OpenRouter chat completions for agentic decision making."""
import json
import os
import re
import urllib.request
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SYSTEM_PROMPT = """Bạn là bộ điều phối (Controller) cho một hệ thống Trả lời câu hỏi có căn cứ (Grounded QA Agent) về Quy chế đào tạo Đại học Bách khoa Hà Nội.

Nhiệm vụ của bạn: Dựa vào câu hỏi của người dùng và ngữ cảnh (context) các đoạn trích tài liệu đã truy xuất được, đưa ra quyết định hành động tiếp theo.

Các hành động bạn có thể chọn (chỉ trả về ĐÚNG 1 JSON object):

1. NẾU ngữ cảnh hiện tại ĐÃ CÓ ĐỦ BẰNG CHỨNG để trả lời chính xác, đầy đủ câu hỏi:
Trả về JSON:
{
  "action": "answer",
  "answer": "<câu trả lời súc tích, chính xác bằng tiếng Việt, bắt buộc phải trích dẫn mã chunk dưới dạng [chunk_id] ngay sau thông tin được trích xuất>",
  "citations": ["<chunk_id_1>", ...]
}
* RÀNG BUỘC CỰC KỲ QUAN TRỌNG VỀ CITATION:
- Mọi khẳng định trong câu trả lời bắt buộc phải có bằng chứng từ các chunk trong context.
- Mã chunk trong citations và [chunk_id] trong văn bản PHẢI thuộc danh sách context_ids đã cung cấp.
- Không trích dẫn chunk không liên quan hoặc không chứa bằng chứng trực tiếp.

2. NẾU ngữ cảnh hiện tại CHƯA ĐỦ hoặc THIẾU BẰNG CHỨNG, nhưng có thể tìm kiếm thêm bằng từ khóa khác hoặc khía cạnh bổ sung:
Trả về JSON:
{
  "action": "search",
  "query": "<câu truy vấn tìm kiếm tiếng Việt mới, tập trung vào từ khóa cần tìm>"
}

3. NẾU câu hỏi KHÔNG THỂ TRẢ LỜI ĐƯỢC từ tài liệu (ví dụ: câu hỏi về mức học phí cụ thể, ký túc xá, bảo hiểm y tế, bơi lội, giờ mở cổng... những nội dung ngoài phạm vi quy chế), hoặc sau khi đã tìm kiếm mà tài liệu hoàn toàn không có thông tin:
TỪ CHỐI TRẢ LỜI NGAY, TUYỆT ĐỐI KHÔNG ẢO GIÁC HOẶC TỰ BỊA ĐẶT ĐÁP ÁN:
Trả về JSON:
{
  "action": "answer",
  "answer": "Không đủ bằng chứng để trả lời.",
  "citations": []
}

CHỈ TRẢ VỀ DUY NHẤT MỘT KHỐI JSON HỢP LỆ, KHÔNG KÈM GIẢI THÍCH NGOÀI JSON.
"""


class OpenRouterController:
    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.temperature = temperature

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            }).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ConnectionError(f"OpenRouter chat completion error: {e}") from e

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide next action: search or answer with citations."""
        # Initial turn: if no search has been performed yet
        if not history:
            user_content = f"""Câu hỏi: {question}

Hiện tại chưa có dữ liệu ngữ cảnh nào được truy xuất.
Hãy tạo câu truy vấn tìm kiếm đầu tiên tốt nhất để tìm kiếm bằng chứng trong tài liệu quy chế."""
        else:
            last = history[0]
            context = last.get("context", "")
            context_ids = last.get("context_ids", [])
            past_queries = last.get("queries", [])

            user_content = f"""Câu hỏi: {question}

Các truy vấn tìm kiếm đã thực hiện trước đó: {past_queries}
Danh sách context_ids hiện có trong ngữ cảnh: {context_ids}

Ngữ cảnh hiện tại:
\"\"\"
{context}
\"\"\"

Dựa vào ngữ cảnh trên, hãy quyết định hành động tiếp theo:
- Nếu đã có đủ bằng chứng: Trả lời kèm trích dẫn [chunk_id].
- Nếu chưa đủ nhưng cần tìm kiếm khía cạnh khác: Đưa ra truy vấn mới.
- Nếu không có thông tin hoặc không đủ bằng chứng: Từ chối với answer: 'Không đủ bằng chứng để trả lời.' và citations: []."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw_content = self._call_llm(messages)

        # Parse JSON output
        cleaned = raw_content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback search if decode fails
            return {"action": "search", "query": question}

        action = parsed.get("action")
        if action == "search":
            query = parsed.get("query", "").strip() or question
            return {"action": "search", "query": query}
        elif action == "answer":
            answer = parsed.get("answer", "")
            citations = parsed.get("citations", [])
            if not isinstance(citations, list):
                citations = []
            return {
                "action": "answer",
                "answer": answer,
                "citations": [str(c) for c in citations],
            }
        elif action == "abstain":
            return {
                "action": "answer",
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
            }
        else:
            return {"action": "search", "query": question}
