# Báo cáo Day 7 — Agentic RAG: Retrieval & Context Engineering

- **Họ và tên**: Vương Toàn Minh Hiếu
- **MSSV**: 2A202603853
- **Lớp**: K5-L3B
- **Chủ đề & Corpus**: Quy chế Đào tạo của Đại học Bách khoa Hà Nội năm 2025 (Ban hành kèm Quyết định số 5445/QĐ-ĐHBK)

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk**:
  - *Nguồn tài liệu*: Văn bản Quy chế Đào tạo Đại học Bách khoa Hà Nội tải từ cổng thông tin đào tạo chính thức của nhà trường (`https://ctt.hust.edu.vn/Upload/Nguy%E1%BB%85n%20Qu%E1%BB%91c%20%C4%90%E1%BA%A1t/files/DTDH_QDQC/Hoctap/QCDT_2025_5445_QD-DHBK.pdf`), tổng cộng 34 trang văn bản số PDF.
  - *Quyền sử dụng*: Văn bản quy phạm nội bộ công khai của cơ sở giáo dục đại học công lập (Public Institutional Policy), được phép tra cứu, trích dẫn và sử dụng công khai phục vụ nghiên cứu và học tập.
  - *Số trang tính theo quy định `GUIDE.md`*: Loại trừ Trang 2 (Trang bìa), Trang 3 và Trang 4 (Mục lục quy chế). Số trang nội dung hợp lệ đưa vào hệ thống là **31 trang nội dung** (Trang 1 và từ Trang 5 đến 34, bao gồm trọn vẹn 48 điều khoản từ Điều 1 đến Điều 48), thỏa mãn yêu cầu tối thiểu $\ge 30$ trang.
  - *Số chunk kế thừa từ Day 6*: Tổng cộng **277 chunks**, gồm 164 chunks từ Strategy A (cắt cố định 500 ký tự, overlap 50 ký tự) và 113 chunks từ Strategy B (cắt theo ranh giới đoạn văn / điều khoản tự nhiên, tối đa 800 ký tự). Dữ liệu được trích xuất và đồng bộ tại `data/day6_chunks.json` và chuẩn hóa theo schema Day 7 tại `data/converted/chunks.json`.
  - *Tập chunk sử dụng trong thực nghiệm*: Sử dụng 113 chunks của **Strategy B** (được chứng minh từ Day 6 là bảo tồn 100% ngữ cảnh toàn vẹn của các điều khoản quy định).

- **Vector store, embedding model, phiên bản và số chiều**:
  - *Vector Store*: PostgreSQL 16 kết hợp extension `pgvector 0.8.7` chạy cục bộ trên cổng `127.0.0.1:55433`. Dữ liệu được lưu trữ bền vững trong cơ sở dữ liệu `day6_lab`.
  - *Bảng lưu trữ*: `chunks`, gồm các trường `chunk_id` (khóa chính), `strategy`, `document_id`, `source`, `page`, `chapter`, `text`, `metadata` (JSONB) và `embedding` (kiểu `vector(1536)`).
  - *Embedding Model*: `openai/text-embedding-3-small` (gọi qua OpenRouter API).
  - *Số chiều*: **1536 chiều**.
  - *Metric khoảng cách*: Cosine Distance (`<=>`), điểm tương đồng tính bằng $1 - \text{cosine\_distance}$.

- **Agent model, cách chạy và phân chia dev/test**:
  - *Agent Model*: `openai/gpt-4o-mini` (temperature = 0.0, format JSON mode) qua OpenRouter API.
  - *Cơ chế điều phối*: Tuân thủ giao thức `Controller.decide` và `Retriever.search` trong [starter/agentic.py](../../starter/agentic.py).
  - *Cách chạy*: Chạy pipeline đánh giá benchmark qua các script trong `src/` (`run_pipeline.py`, `benchmark_retrieval.py`), ghi lại toàn bộ trace tool calls, context budget, citation IDs và câu trả lời.
  - *Phân chia Dev / Test*: Xây dựng bộ benchmark gồm **30 câu hỏi** (vượt yêu cầu tối thiểu 24 câu):
    - **Tập Dev (20 câu)**: 17 câu có đáp án (`q01`–`q17`) và 3 câu thiếu bằng chứng (`u01`–`u03`).
    - **Tập Test (10 câu)**: 7 câu có đáp án (`q18`–`q24`) và 3 câu thiếu bằng chứng (`u04`–`u06`).
    - Tỷ lệ Dev:Test được chia 2:1, cố định trước khi tối ưu hóa và đảm bảo tập Test hoàn toàn độc lập, không dùng để điều chỉnh siêu tham số.

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm**:
  - *Câu có đáp án (`answerable: true`)*:
    - Gồm 24 câu hỏi bao quát các quy định cốt lõi: thời gian thiết kế chương trình, cách quy đổi điểm, điều kiện cảnh báo học tập, buộc thôi học, phúc tra điểm, hoãn thi, học cùng lúc hai chương trình...
    - Mỗi câu đi kèm `expected_answer` và danh sách `gold` gồm `source_id`, `page`, và trích đoạn bằng chứng nguyên văn (`evidence`).
    - **Kiểm chứng 100% bằng chứng**: Từng đoạn `evidence` được kiểm tra tự động bằng string matching khớp tuyệt đối từng ký tự với chunk tương ứng trong `data/converted/chunks.json`.
  - *Câu thiếu bằng chứng (`answerable: false`)*:
    - Gồm 6 câu hỏi rất quen thuộc trong đời sống sinh viên ĐHBK Hà Nội (mức học phí tín chỉ, tiêu chuẩn phòng ký túc xá, điểm rèn luyện nhận học bổng khuyến khích, chuẩn bơi lội, giờ mở cổng đăng ký...) nhưng hoàn toàn không xuất hiện trong văn bản Quy chế đào tạo này.
    - Đánh dấu nhãn `expected_behavior: "search_and_abstain"` kèm lý do vắng mặt thông tin `absence_reason`.
  - *Tiêu chí chấm*:
    - **Retrieval**: Hit@1, Evidence Hit@3, Context Evidence Rate trên tập câu hỏi có đáp án. Câu thiếu bằng chứng không đưa vào mẫu số ranking ($N_{dev}=17, N_{test}=7$).
    - **Agent**:
      - `Grounded answer` (thang 0–2): 0 = sai hoặc thiếu evidence; 1 = đúng một phần; 2 = đúng và đủ. Từ chối câu có đáp án chấm 0.
      - `Citation hợp lệ`: Tỷ lệ trích dẫn nằm trong context được chọn và thực sự bổ trợ cho luận điểm trả lời.
      - `Abstention`: Tỷ lệ từ chối đúng đối với câu thiếu bằng chứng và tỷ lệ từ chối nhầm đối với câu có đáp án.

---

## 2. Retrieval và thử nghiệm

### Bảng kết quả so sánh đối đầu trên Dev và Test

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian (median) | Chi phí API |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Dense (pgvector + text-embedding-3-small), top_k=3, budget=800 | Vector Cosine trên pgvector Day 6 | dev | 11/17 (64.7%) | 15/17 (88.2%) | 5/17 (29.4%) | 1057.5ms | $0.000010 |
| **Phương pháp 2** | Lexical BM25 (k1=1.5, b=0.75), top_k=3, budget=800 | Bắt từ khóa chuyên môn/số hiệu điều khoản | dev | **14/17 (82.4%)** | **17/17 (100.0%)** | 6/17 (35.3%) | **1.4ms** | **$0.000000** |
| **Phương pháp 3** | Hybrid RRF (Dense + BM25, candidate_k=10, rrf_k=60), top_k=3, budget=800 | Hợp nhất Reciprocal Rank Fusion kết hợp ngữ nghĩa và từ khóa | dev | 12/17 (70.6%) | 16/17 (94.1%) | 5/17 (29.4%) | 13.4ms | $0.000010 |
| **Cải thiện 1** | Dense retrieval, top_k=3, **budget=2000 chars** | Tăng context budget khắc phục lỗi nghẽn context | dev | 11/17 (64.7%) | 15/17 (88.2%) | 15/17 (88.2%) | 13.0ms | $0.000010 |
| **Cải thiện 2 (CHỐT)** | Hybrid RRF, top_k=3, **budget=2000 chars** | Kết hợp Hybrid RRF và nới rộng context budget | dev | 12/17 (70.6%) | **16/17 (94.1%)** | **16/17 (94.1%)** | 14.6ms | $0.000010 |
| **Cấu hình đã chọn** | Hybrid RRF (candidate_k=10, rrf_k=60, top_k=3), budget=2000 | Đánh giá độc lập cấu hình chốt trên tập Test | **test** | **5/7 (71.4%)** | **7/7 (100.0%)** | **6/7 (85.7%)** | **12.5ms** | **$0.000004** |
| **Baseline** | Dense (pgvector), top_k=3, budget=800 | Đánh giá độc lập Baseline trên tập Test | **test** | 5/7 (71.4%) | 5/7 (71.4%) | 3/7 (42.9%) | 1977.9ms | $0.000004 |

---

### Phân tích lỗi trên Dev, giả thuyết và lý do chọn cấu hình

#### 1. Lỗi 1: Nghẽn Context Budget (Context Budget Bottleneck)
- **Hiện tượng**: Ở Baseline Dense và cả 3 phương pháp với `budget = 800`, chỉ số Hit@3 rất cao (88.2% – 100%), nhưng **Context Evidence Rate lại cực thấp (chỉ 29.4% – 35.3%)**. Hậu quả là Agent bị 10 ca từ chối nhầm (**False Abstention = 10/17**).
- **Nguyên nhân**: Trong [starter/baseline.py](../../starter/baseline.py), mỗi block context gồm mã trích dẫn `[chunk_id | source | page]` và nội dung chunk dài từ 650–850 ký tự. Với ngân sách 800 ký tự, ngay sau khi nạp chunk #1 thì context đã chạm trần (~750 ký tự). Hàm `pack_context` buộc phải loại bỏ chunk #2 và #3 (nơi chứa bằng chứng cần thiết). Khi LLM nhận context thiếu bằng chứng, nó tuân thủ đúng prompt an toàn và chuyển sang trạng thái từ chối (`abstain`).
- **Giả thuyết**: Mở rộng context budget từ 800 lên 2000 ký tự (đủ chỗ cho cả top-3 chunks cùng vào prompt) sẽ giải phóng toàn bộ tiềm năng của retriever, đưa Context Evidence Rate tăng tương ứng với Hit@3.
- **Kết quả kiểm chứng**: Khi áp dụng Cải thiện 1 và Cải thiện 2 với `budget = 2000`, Context Evidence Rate lập tức tăng vọt từ 29.4% lên **94.1% (16/17)** trên tập Dev. Số ca False Abstention của Agent giảm thẳng từ 10 xuống **0 câu**!

#### 2. Lỗi 2: Trôi ngữ nghĩa (Semantic Drift) ở Dense vs. Phân tán từ vựng ở Lexical
- **Hiện tượng**: 
  - Ở các câu hỏi có số liệu và quy tắc kỹ thuật như `q04` (*"điểm chữ", "điểm F"*), `q08` (*"CPA từ 3,6 đến 4,0"*): Dense bị phân tán ngữ nghĩa, chỉ xếp bằng chứng ở vị trí #2 hoặc #3 (Hit@1 chỉ 64.7%). Trong khi BM25 bắt trúng từ khóa định lượng đưa ngay lên #1 (Hit@1 đạt 82.4%).
  - Ngược lại, ở các câu hỏi dài diễn đạt mang tính giải thích chung như `q11` (*"thời gian nghỉ học tạm thời tối đa cho phép..."*), BM25 bị trượt vì người dùng dùng cụm từ khác với văn bản quy chế, trong khi Dense lại bắt được nhờ không gian vector.
- **Giả thuyết**: Phương pháp **Hybrid RRF** sẽ trung hòa được hai cực: nó kéo chunk có từ khóa chính xác lên nhờ BM25 và đồng thời giữ lại chunk liên quan ngữ nghĩa nhờ Dense, giúp Hit@3 ổn định ở mức rất cao (94.1% trên Dev và 100% trên Test).

#### 3. Tiêu chí và Lý do chốt cấu hình
- **Tiêu chí công bố trước**:
  1. *Chất lượng bằng chứng*: Tối đa hóa Context Evidence Rate và Hit@3 (ngưỡng mong đợi $\ge 90\%$).
  2. *Độ an toàn*: 0% hallucination trên câu thiếu bằng chứng, citation 100% hợp lệ.
  3. *Đánh đổi chi phí và thời gian*: Độ trễ truy xuất $\le 50\text{ms}$, chi phí tính toán $\le \$0.0001/\text{câu}$.
- **Quyết định chốt**: Chọn **Cải thiện 2 (Hybrid RRF + Context Budget 2000 ký tự)**. Cấu hình này đạt điểm số vượt trội cả về Retrieval (Hit@3 = 94.1% dev, 100% test; Context Rate = 94.1% dev, 85.7% test) lẫn chất lượng câu trả lời của Agent.

---

## 3. Agent và đánh giá

### Bảng tổng hợp đánh giá Agent trên Dev và Test

| Cấu hình / split | Tool use | Số lượt gọi trung bình | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Dense + Budget 800) / dev** | 20/20 (100%) | 1.00 | 0: 10, 1: 3, 2: 4 (Mean: 0.65/2) | 7/7 (100%) | 7/7 (100%) | 3/3 (100%) | 10/17 (58.8%) |
| **Cấu hình chọn (Hybrid + Budget 2000) / dev** | 20/20 (100%) | 1.05 | 0: 0, 1: 1, 2: 16 (**Mean: 1.94/2**) | 17/17 (100%) | 17/17 (100%) | 3/3 (100%) | **0/17 (0.0%)** |
| **Baseline (Dense + Budget 800) / test** | 10/10 (100%) | 1.10 | 0: 3, 1: 2, 2: 2 (Mean: 0.86/2) | 4/4 (100%) | 4/4 (100%) | 3/3 (100%) | 3/7 (42.9%) |
| **Cấu hình chọn (Hybrid + Budget 2000) / test** | 10/10 (100%) | 1.00 | 0: 0, 1: 1, 2: 6 (**Mean: 1.86/2**) | 7/7 (100%) | 7/7 (100%) | 3/3 (100%) | **0/7 (0.0%)** |

---

### Chi tiết đánh giá đối chiếu từng câu tập Test ($N=10$)

Dưới đây là bảng audit và đối chiếu thực tế toàn bộ 10 câu hỏi của tập Test (7 câu có đáp án và 3 câu thiếu evidence) chạy trên Cấu hình đã chọn (`Hybrid RRF + Budget 2000`):

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q18` | Có đáp án | `search("Chương trình đào tạo trình độ cử nhân đối với người tốt nghiệp THPT...")` $\rightarrow$ Trả về `B_quy_che_dao_tao_dhbk_2025_p6_c4` (Trang 6) | *"Chương trình đào tạo trình độ cử nhân đối với người tốt nghiệp THPT theo hình thức chính quy có thời gian thiết kế là 4 năm và thời gian tối đa để hoàn thành là 6 năm [B_quy_che_dao_tao_dhbk_2025_p6_c4]."* | **Grounded: 2/2**. Đầy đủ 2 vế thời gian chuẩn (4 năm) và tối đa (6 năm), citation khớp tuyệt đối văn bản trang 6. |
| `q19` | Có đáp án | `search("Thời gian chậm tiến độ đối với các chương trình cấp bằng cử nhân...")` $\rightarrow$ Trả về `B_quy_che_dao_tao_dhbk_2025_p6_c4` (Trang 6) | *"Thời gian chậm tiến độ đối với các chương trình cấp bằng cử nhân không được vượt quá 2 năm (4 học kỳ chính) [B_quy_che_dao_tao_dhbk_2025_p6_c4]."* | **Grounded: 2/2**. Nêu chính xác định mức không vượt quá 2 năm (4 học kỳ chính), trích dẫn chuẩn xác. |
| `q20` | Có đáp án | `search("Sinh viên đủ điều kiện tốt nghiệp đại học được cấp bằng tốt nghiệp trong thời hạn bao lâu...")` $\rightarrow$ Trả về `B_quy_che_dao_tao_dhbk_2025_p26_c1` | *"Sinh viên đủ điều kiện tốt nghiệp sẽ được cấp bằng tốt nghiệp trong thời hạn 02 tháng tính từ thời điểm có quyết định công nhận tốt nghiệp [B_quy_che_dao_tao_dhbk_2025_p26_c1]."* | **Grounded: 2/2**. Thời hạn 02 tháng khớp tuyệt đối quy định tại Điều 27 khoản 2. |
| `q21` | Có đáp án | `search("Học viên đào tạo thạc sĩ chính quy có thể đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ chính...")` $\rightarrow$ Trả về `B_quy_che_dao_tao_dhbk_2025_p23_c0` (Trang 23) | *"Học viên đào tạo thạc sĩ chính quy có thể đăng ký tối đa 24 tín chỉ trong một học kỳ chính và không bị giới hạn số tín chỉ tối thiểu [B_quy_che_dao_tao_dhbk_2025_p23_c0]."* | **Grounded: 2/2**. Nêu đúng mức tối đa 24 tín chỉ và không giới hạn mức tối thiểu cho học viên thạc sĩ. |
| `q22` | Có đáp án | `search("Học viên cao học thực hiện đăng ký đề tài luận văn thạc sĩ vào thời điểm nào...")` $\rightarrow$ Trả về `B_quy_che_dao_tao_dhbk_2025_p24_c0` (Trang 24) | *"Học viên thực hiện đăng ký đề tài luận văn thạc sĩ trong học kỳ đầu của khóa học theo hai cách: chọn đề tài của giảng viên hoặc tự đề xuất [B_quy_che_dao_tao_dhbk_2025_p24_c0]."* | **Grounded: 2/2**. Bằng chứng tại khoản 1 Điều 32 nêu rõ học kỳ đầu của khóa học. |
| `q23` | Có đáp án | `search("Thời hạn học viên thạc sĩ phải nộp quyển luận văn tốt nghiệp cho thư viện Tạ Quang Bửu...")` $\rightarrow$ Trả về `B_quy_che_dao_tao_dhbk_2025_p26_c0` (Trang 26) | *"Thời hạn nộp quyển luận văn tốt nghiệp cho thư viện Tạ Quang Bửu là 20 ngày kể từ ngày bảo vệ thành công [B_quy_che_dao_tao_dhbk_2025_p26_c0]."* | **Grounded: 2/2**. Thời hạn 20 ngày được trích dẫn chính xác từ quy chế bảo vệ luận văn. |
| `q24` | Có đáp án | `search("Quy định về việc chuyển chương trình đào tạo tại Điều 17 được chính thức áp dụng từ thời điểm nào...")` $\rightarrow$ Trả về `B_quy_che_dao_tao_dhbk_2025_p34_c2` (Trang 34) | *"Khoản 1, khoản 2 và khoản 3 Điều 17 về việc chuyển chương trình đào tạo được áp dụng từ học kỳ 2 năm học 2024-2025 [B_quy_che_dao_tao_dhbk_2025_p34_c2]."* | **Grounded: 2/2**. Điều khoản chuyển tiếp tại Điều 48 được trích xuất hoàn hảo. |
| `u04` | Thiếu bằng chứng | `search("Học phần Giáo dục thể chất có bắt buộc chuẩn đầu ra kỹ năng bơi lội cự ly 50m...")` $\rightarrow$ Kết quả quy định chung về GDTC | *"Không đủ bằng chứng để trả lời."* | **Abstention đúng: True**. Quy chế không quy định chuẩn cự ly bơi 50m; Agent từ chối chính xác, không bịa đặt. |
| `u05` | Thiếu bằng chứng | `search("Mức hỗ trợ chi phí mua bảo hiểm y tế bắt buộc cho sinh viên diện chính sách...")` $\rightarrow$ Không có thông tin BHYT | *"Không đủ bằng chứng để trả lời."* | **Abstention đúng: True**. Tài liệu không điều chỉnh chính sách BHYT; Agent từ chối dứt khoát. |
| `u06` | Thiếu bằng chứng | `search("Cổng đăng ký tín chỉ trực tuyến của nhà trường mở vào lúc mấy giờ sáng...")` $\rightarrow$ Chỉ có kế hoạch theo tuần | *"Không đủ bằng chứng để trả lời."* | **Abstention đúng: True**. Quy chế chỉ quy định theo Khung kế hoạch năm học, không có giờ cụ thể; Agent từ chối chính xác. |

---

### Thông tin tái lập, đường dẫn lưu trữ và đánh đổi

- **Đường dẫn kết quả & Traces:**
  - *Baseline Test*: [starter/results/baseline_dense_test/traces.json](../../starter/results/baseline_dense_test/traces.json) và [starter/results/baseline_dense_test/agent_summary.json](../../starter/results/baseline_dense_test/agent_summary.json).
  - *Cấu hình chọn Test*: [starter/results/chosen_hybrid_test/traces.json](../../starter/results/chosen_hybrid_test/traces.json) và [starter/results/chosen_hybrid_test/agent_summary.json](../../starter/results/chosen_hybrid_test/agent_summary.json).
  - *Benchmark Retrieval*: [starter/results/retrieval_benchmark/benchmark_test.json](../../starter/results/retrieval_benchmark/benchmark_test.json).
- **Lệnh chạy tái lập kết quả:**
  ```bash
  # 1. Chạy toàn bộ 25 bài kiểm tra tự động
  python3 -m unittest discover -s tests -v

  # 2. Chạy so sánh retrieval benchmark trên test
  .venv/bin/python src/benchmark_retrieval.py --split test

  # 3. Chạy Agent Baseline trên test
  .venv/bin/python src/run_pipeline.py --split test --method dense --max-searches 3 --top-k 3 --budget 800 --output starter/results/baseline_dense_test

  # 4. Chạy Agent Cấu hình đã chọn trên test
  .venv/bin/python src/run_pipeline.py --split test --method hybrid --max-searches 3 --top-k 3 --budget 2000 --output starter/results/chosen_hybrid_test
  ```
- **Thời gian và chi phí:**
  - *Thời gian truy xuất (Retrieval Latency)*: Hybrid RRF chỉ mất **12.5ms** (nhanh gấp ~150 lần so với Dense thuần túy khi chưa cache do kết hợp BM25 cực nhanh).
  - *Thời gian chạy Agent*: Trung bình 2.5 – 4.0 giây / câu (gồm 1 lượt gọi LLM Controller để phân tích context và đưa ra câu trả lời có trích dẫn).
  - *Chi phí dịch vụ*: $0.000004 USD / câu cho retrieval embedding và khoảng $0.0003 USD / câu cho LLM Controller. Tổng chi phí toàn bộ bài lab dưới $0.05 USD.
- **Đánh đổi và Giới hạn:**
  - *Đánh đổi Context Budget*: Tăng budget từ 800 lên 2000 ký tự làm tăng nhẹ số lượng input token gửi cho LLM (~300 token tăng thêm), nhưng mang lại bước nhảy vọt về chất lượng: xóa bỏ hoàn toàn 10 ca từ chối nhầm, nâng điểm Grounded Answer từ 0.86 lên 1.86/2.0.
  - *Giới hạn hệ thống*: Hiện tại hệ thống dựa vào tài liệu dạng văn bản đơn ngữ (tiếng Việt). Khi người dùng hỏi bằng các từ viết tắt không phổ biến trong trường học, hệ thống cần thêm bước Query Normalization để hỗ trợ tốt hơn.

---

## 4. Thử thách nâng cao (nếu thực hiện)

*(Phần này dành riêng cho việc mở rộng tài liệu thực tế $\ge 100$ trang có bảng, ảnh và biểu đồ theo [ADVANCED.md](../../ADVANCED.md)).*
