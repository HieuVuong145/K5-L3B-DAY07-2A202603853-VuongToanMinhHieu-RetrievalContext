# Hướng dẫn Mã nguồn & Tái lập (Source Code & Reproduction Guide)

Thư mục `src/` chứa toàn bộ mã nguồn hiện thực hệ thống **Agentic RAG: Retrieval & Context Engineering** cho bài lab Day 7, kế thừa dữ liệu và vector store từ Day 6 (Quy chế Đào tạo ĐHBK Hà Nội 2025).

---

## 1. Cấu trúc mã nguồn

| File | Thành phần | Mô tả chức năng |
|---|---|---|
| `dense_retriever.py` | `PgVectorDenseRetriever` | Truy xuất vector dày đặc kết nối PostgreSQL 16 + `pgvector` cục bộ (cổng 55433). Sử dụng mô hình `openai/text-embedding-3-small` (1536 chiều) qua OpenRouter API. Tích hợp bộ nhớ đệm `data/embedding_cache.json` giúp chạy offline/tái lập không tốn phí API. |
| `lexical_retriever.py` | `BM25LexicalRetriever` | Truy xuất từ vựng thuần túy BM25 offline ($k_1=1.5, b=0.75$). Tách từ tiếng Việt đơn giản, tính tần suất thuật ngữ và trọng số IDF. Rất mạnh với các mã học phần, điểm số (`điểm F`, `CPA 3.6`, `16 tín chỉ`). |
| `hybrid_retriever.py` | `HybridRRFRetriever` | Hợp nhất đa phương thức Reciprocal Rank Fusion (RRF với hằng số $k=60$) kết hợp Dense và BM25. Cân bằng giữa sự chính xác từ khóa và hiểu biết ngữ nghĩa sâu. |
| `llm_controller.py` | `OpenRouterController` | Bộ điều phối Agentic RAG gọi mô hình `openai/gpt-4o-mini` qua OpenRouter API (nhiệt độ 0.0, format JSON mode). Quyết định `search` hoặc `answer` kèm trích dẫn `citations`. Tự động từ chối (`abstain`) khi context không đủ bằng chứng. |
| `run_pipeline.py` | Agentic Pipeline | Điều phối vòng lặp Agentic RAG tự động cho toàn bộ tập câu hỏi benchmark. Hỗ trợ cấu hình retriever (`dense`, `lexical`, `hybrid`), tùy biến context budget, ghi nhận toàn bộ `traces.json` và metrics chi phí/thời gian. |
| `benchmark_retrieval.py` | Retrieval Benchmark | Công cụ đo đạc định lượng các chỉ số tìm kiếm thuần túy (Hit@1, Hit@3, Context Evidence Rate, thời gian, chi phí) trên tập Dev và Test cho 5 cấu hình khác nhau. |

---

## 2. Thiết lập môi trường

### Yêu cầu hệ thống
- Python $\ge$ 3.11
- Cơ sở dữ liệu PostgreSQL 16 + extension `pgvector` đang hoạt động (kế thừa từ Day 6 trên cổng `127.0.0.1:55433`).

### Cài đặt thư viện phụ thuộc
```bash
pip install psycopg requests python-dotenv
```

### Cấu hình biến môi trường
Tạo file `.env` tại thư mục gốc của repository (tham khảo `.env.example`):
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
AGENT_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
DB_HOST=127.0.0.1
DB_PORT=55433
DB_NAME=day6_lab
DB_USER=day6
DB_PASS=day6_local_only
```

---

## 3. Hướng dẫn chạy kiểm thử tự động (Unit Tests)

Chạy toàn bộ 25 bài unit test (bao gồm kiểm thử schema dữ liệu, giao thức Controller/Retriever, tính hợp nhất Hybrid RRF và cơ chế abstention):

```bash
# Chạy toàn bộ test suite
python3 -m unittest discover -s tests -v
```

Chạy riêng từng nhóm bài test:
```bash
# Kiểm tra tính toàn vẹn của dữ liệu Day 6 và 100% khớp bằng chứng nguyên văn
python3 -m unittest tests/test_dataset_day6.py

# Kiểm tra giao thức Checkpoint 2 (Controller, Dense Retriever, Abstention & Citations)
python3 -m unittest tests/test_checkpoint2.py

# Kiểm tra giao thức Checkpoint 3 (Lexical Retriever, thuật toán Hybrid RRF)
python3 -m unittest tests/test_checkpoint3.py
```

---

## 4. Hướng dẫn chạy đo đạc Retrieval Benchmark

Đo đạc Hit@1, Hit@3, Context Evidence Rate trên tập Dev ($N=17$) và Test ($N=7$):

```bash
python3 src/benchmark_retrieval.py
```

Kết quả chi tiết được tự động xuất ra:
- `starter/results/retrieval_benchmark/benchmark_dev.json`
- `starter/results/retrieval_benchmark/benchmark_test.json`

---

## 5. Hướng dẫn chạy Agentic RAG Pipeline

### 5.1. Chạy Cấu hình Tối ưu (Chosen: Hybrid RRF + Context Budget 2000)
Cấu hình đạt chất lượng cao nhất trong nghiên cứu:
- Trên tập **Test** ($N=10$):
  ```bash
  python3 src/run_pipeline.py --config hybrid --budget 2000 --split test --output starter/results/chosen_hybrid_test
  ```
- Trên tập **Dev** ($N=20$):
  ```bash
  python3 src/run_pipeline.py --config hybrid --budget 2000 --split dev --output starter/results/checkpoint3_hybrid_b2000_dev
  ```

### 5.2. Chạy Cấu hình Cơ sở (Baseline: Dense Retrieval + Context Budget 800)
- Trên tập **Test**:
  ```bash
  python3 src/run_pipeline.py --config dense --budget 800 --split test --output starter/results/baseline_dense_test
  ```
- Trên tập **Dev**:
  ```bash
  python3 src/run_pipeline.py --config dense --budget 800 --split dev --output starter/results/checkpoint2_dense_dev
  ```

---

## 6. Hướng dẫn tổng hợp số liệu đánh giá (Agent Metrics)

Sau khi pipeline sinh ra file `traces.json` và `reviews_draft.json`, chạy script của starter để đối chiếu nhãn và tính toán các chỉ số `Grounded Mean`, `False Abstention`, `Citation Support`:

```bash
python3 starter/agent_metrics.py \
  --cases data/converted/cases.json \
  --traces starter/results/chosen_hybrid_test/traces.json \
  --reviews starter/results/chosen_hybrid_test/reviews_draft.json \
  --split test \
  --output starter/results/chosen_hybrid_test/summary.json
```

---

## 7. Xác thực kết nối Day 6

Để kiểm tra nhanh trạng thái kết nối tới PostgreSQL và đếm số lượng chunks:
```bash
python3 starter/verify_day6_connection.py
```
