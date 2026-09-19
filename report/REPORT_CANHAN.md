# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thu Hằng
**Nhóm:** Aura
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai đoạn văn bản có độ tương tự cosine cao nghĩa là chúng nằm gần nhau về **hướng** trong không gian vector embedding — tức là chúng truyền đạt ý nghĩa tương đồng, dù độ dài có thể khác nhau. Giá trị cosine gần 1.0 cho thấy hai văn bản gần như "nói về cùng một điều", còn giá trị gần 0 nghĩa là chúng hoàn toàn không liên quan.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên cần đóng học phí trước ngày 15 mỗi tháng."
- Câu B: "Hạn nộp học phí cho sinh viên là ngày 15 hằng tháng."
- Tại sao tương đồng: Hai câu diễn đạt cùng một thông tin (hạn đóng học phí là ngày 15), chỉ khác cách dùng từ — model embedding sẽ ánh xạ chúng vào các vector gần nhau trong không gian ngữ nghĩa.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên cần đóng học phí trước ngày 15 mỗi tháng."
- Câu B: "Thư viện mở cửa từ 7 giờ sáng đến 9 giờ tối các ngày trong tuần."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (tài chính vs. dịch vụ thư viện), không chia sẻ ngữ nghĩa chung nên vector embedding sẽ trỏ về hai hướng khác biệt.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị ảnh hưởng bởi **độ lớn (magnitude)** của vector — văn bản dài tự nhiên tạo ra vector có norm lớn hơn, khiến khoảng cách Euclid giữa hai văn bản dài luôn lớn hơn dù nội dung tương đồng. Cosine similarity chỉ đo **góc** giữa hai vector (bỏ qua độ lớn), nên công bằng hơn khi so sánh văn bản có độ dài khác nhau — đây là đặc điểm phổ biến trong thực tế.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Áp dụng công thức: `số chunk = làm_tròn_lên((độ_dài - overlap) / (chunk_size - overlap))`
>
> `= làm_tròn_lên((10.000 - 50) / (500 - 50))`
> `= làm_tròn_lên(9.950 / 450)`
> `= làm_tròn_lên(22,11)`
> `= **23 chunks**`

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> `= làm_tròn_lên((10.000 - 100) / (500 - 100))`
> `= làm_tròn_lên(9.900 / 400)`
> `= làm_tròn_lên(24,75)`
> `= **25 chunks**` (tăng thêm 2 so với overlap=50)
>
> Tăng overlap làm **tăng số chunks** vì mỗi bước trượt (step) ngắn hơn. Ta muốn overlap lớn hơn khi thông tin quan trọng nằm ở **ranh giới giữa hai chunk** — overlap giúp chunk sau "nhớ" được ngữ cảnh của chunk trước, giảm nguy cơ câu bị cắt đứt mất nghĩa khi retrieval.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`FixedSizeChunker.chunk`** — đã được implement sẵn, tôi đọc kỹ để hiểu cấu trúc:
> `FixedSizeChunker` chia văn bản thành các đoạn có độ dài cố định (`chunk_size` ký tự) với độ chồng chéo `overlap`. Mỗi bước trượt `step = chunk_size - overlap` ký tự, lấy đoạn `text[start : start + chunk_size]`. Trường hợp đặc biệt: nếu văn bản ngắn hơn `chunk_size` thì trả về `[text]` nguyên vẹn, và nếu `text` rỗng thì trả về `[]`.

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng `re.split(r'(?<=[.!?])\s+', text)` để tách văn bản thành danh sách câu — pattern này nhìn ra phía sau (lookbehind) để giữ lại dấu câu ở cuối câu trước. Sau đó nhóm các câu thành từng chunk `max_sentences_per_chunk` câu bằng vòng lặp bước nhảy (step loop), nối bằng khoảng trắng và `.strip()` để loại bỏ khoảng trắng thừa. Edge case xử lý: văn bản rỗng trả về `[]`, câu cuối cùng không đủ nhóm vẫn được gộp thành một chunk riêng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> `chunk()` là điểm vào, gọi `_split(text, self.separators)`. `_split()` thử lần lượt từng separator trong danh sách ưu tiên `["\n\n", "\n", ". ", " ", ""]`: nếu tách được thành các phần ≤ `chunk_size` thì trả về, nếu phần nào vẫn quá dài thì đệ quy tiếp với các separator còn lại. Base case: khi `remaining_separators` rỗng (dùng `""`) thì cắt cứng theo ký tự. Cách này ưu tiên tách ở ranh giới đoạn văn trước, rồi mới xuống dòng, câu, từ — giúp giữ tính mạch lạc ngữ nghĩa tốt hơn `FixedSizeChunker`.

**`compute_similarity`** — hướng tiếp cận:
> Áp dụng công thức cosine: `dot(a, b) / (||a|| × ||b||)`. Tính `||a||` và `||b||` bằng `math.sqrt(_dot(vec, vec))`. Bảo vệ chia cho 0: nếu bất kỳ norm nào bằng 0 (vector zero) thì trả về `0.0` ngay lập tức, tránh `ZeroDivisionError`.

**`ChunkingStrategyComparator.compare`** — hướng tiếp cận:
> Gọi cả ba chunker (`FixedSizeChunker`, `SentenceChunker`, `RecursiveChunker`) với cùng `chunk_size`. Thu thập danh sách chunks từ mỗi chiến lược, rồi tính 3 chỉ số: `chunk_count` (số lượng chunk), `avg_length` (độ dài trung bình), `min_length`/`max_length`. Trả về dict `{"fixed": {...}, "sentence": {...}, "recursive": {...}}` để dễ so sánh.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `add_documents()` duyệt từng `Document`, gọi `self._embedding_fn(doc.content)` để lấy vector, rồi lưu vào `self._store` dưới dạng dict `{"id": ..., "embedding": [...], "content": ..., "metadata": {...}}` (metadata chứa toàn bộ `doc.metadata` bao gồm `doc_id`, `audience`, ...). `search()` embed câu truy vấn, tính dot product với từng embedding đã lưu bằng hàm `_dot()`, sắp xếp giảm dần theo score và trả về `top_k` kết quả tốt nhất kèm `score`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter()` **lọc trước, tìm kiếm sau**: đầu tiên lọc `self._store` lấy các record có metadata khớp với `metadata_filter` (ví dụ `{"audience": "student"}`), sau đó chỉ chạy similarity search trên tập đã lọc — cách này tránh việc chunk không phù hợp lọt vào top-k. `delete_document()` duyệt `self._store`, loại bỏ tất cả record có `metadata["doc_id"] == doc_id` bằng list comprehension, trả về `True` nếu số lượng thay đổi.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `__init__()` lưu `self.store` và `self.llm_fn` vào instance. `answer()` gọi `self.store.search(question, top_k)` để lấy danh sách các chunk liên quan, sau đó xây dựng prompt theo cấu trúc RAG chuẩn: phần **System** hướng dẫn agent chỉ trả lời dựa trên context, phần **Context** liệt kê nội dung từng chunk (đánh số thứ tự), phần **Question** đặt câu hỏi. Cuối cùng gọi `self.llm_fn(prompt)` và trả về kết quả chuỗi. Nếu không có chunk nào được tìm thấy, trả về thông báo "Không tìm thấy thông tin liên quan."

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI-Lab\K4-DAY07-NguyenThuHang-2A202602463

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.10s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Mức học phí sinh viên đại học chính quy năm học 2026-2027" | "Học phí hệ đại học năm học 2026-2027 là bao nhiêu?" | CAO | 0.2262 | ✗ |
| 2 | "Sinh viên học lại phải đóng học phí theo tín chỉ" | "Thư viện mở cửa từ 7 giờ sáng đến 9 giờ tối" | THẤP | 0.1148 | ✓ |
| 3 | "Mức thu học phí 530.000đ/tín chỉ áp dụng cho hệ nào?" | "530.000 đồng mỗi tín chỉ áp dụng cho sinh viên khóa QH-2023 trở về trước" | CAO | -0.2315 | ✗ |
| 4 | "Học phí chương trình đào tạo chất lượng cao" | "Học bổng dành cho sinh viên xuất sắc học kỳ" | THẤP | -0.1699 | ✓ |
| 5 | "Nghị định 238/2025/NĐ-CP quy định mức học phí" | "Theo Nghị định 238 năm 2025, học phí được tính như thế nào?" | CAO | 0.0985 | ✗ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là Cặp 3 và Cặp 5 — hai câu cùng nói về học phí 530.000đ và Nghị định 238 nhưng điểm cosine lại âm hoặc rất thấp. Nguyên nhân là lab dùng **MockEmbedder** (hash MD5 → vector ngẫu nhiên), không phải model ngôn ngữ thật, nên vector không mang ngữ nghĩa thực sự — hai câu gần nghĩa vẫn có thể cho cosine thấp hoặc âm. Điều này cho thấy chất lượng embedding phụ thuộc hoàn toàn vào model: với model ngôn ngữ thật (sentence-transformers, Gemini...), Cặp 1, 3, 5 sẽ cho score cao hơn nhiều vì chúng dùng chung từ khóa và ngữ nghĩa.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân với **FixedSizeChunker** (`chunk_size=500, overlap=50`), 94 chunks từ 6 tài liệu học phí.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên ĐH chính quy USSH đóng học phí bao nhiêu/tháng 2026-2027? | "...khóa QH-2026-X còn lại thu theo Nghị định 238/2025..." | 0.307 | ✓ Có (đề cập mức thu theo tháng) | Mức thu 1.910.000đ/tháng cho các ngành theo NĐ 238 |
| 2 | Học phí học lại tại HVTC 2025-2026 theo tín chỉ hay tháng? | "...QĐ-HV của Giám đốc HVTC ngày 28/12/2022..." | 0.226 | ✗ Không trực tiếp (sau filter audience=student) | Thu theo tín chỉ đăng ký học |
| 3 | Mức 1.910.000đ/tháng áp dụng cho sinh viên khóa nào? | "...USSH Hanoi Forum 2026 — duy trì trạng thái đăng nhập..." | 0.254 | ✗ Không (lấy nhầm footer website) | Không tìm thấy thông tin liên quan |
| 4 | Học phí sinh viên quốc tế tại ĐH KHXH&NV là bao nhiêu/năm? | "...QĐ-HV của Giám đốc HVTC ngày 28/12/2022..." | 0.297 | ✗ Không (sai trường) | Không tìm thấy thông tin liên quan |
| 5 | Báo cáo lộ trình HP 2026-2027 do phòng nào USSH phát hành? | "...Các đơn vị chức năng — Các đơn vị đào tạo..." | 0.253 | ✗ Không (lấy nhầm menu) | Không tìm thấy thông tin liên quan |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 2 / 5

**Phân tích lỗi (Failure Analysis):**
> Retrieval đạt 2/5 do **MockEmbedder không mang ngữ nghĩa thật** — vector hash ngẫu nhiên nên không phân biệt được chunk nội dung với chunk footer/menu. Nếu dùng model thật (sentence-transformers/paraphrase-multilingual), top-1 sẽ khớp nội dung học phí thay vì lấy nhầm nav bar. Ngoài ra, cần **làm sạch triệt để** nội dung md (bỏ menu, footer) trước khi index để tăng chất lượng.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua demo nhóm, tôi nhận ra rằng **RecursiveChunker** giúp giữ nguyên cấu trúc đoạn văn tốt hơn FixedSizeChunker khi tài liệu có định dạng rõ ràng (bảng mục, tiêu đề). Tôi cũng học được rằng việc làm sạch nội dung (loại bỏ footer, menu) trước khi chunk ảnh hưởng rất lớn đến retrieval quality — đây là bước quan trọng không kém gì việc chọn chiến lược chunking.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 4 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10 |
| **Tổng phần cá nhân** | **55 / 60** |
