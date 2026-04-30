Hệ thống được triển khai dưới dạng một ứng dụng web tương tác, trong đó người học đóng vai trò trung tâm và AI Agent đóng vai trò là một người hướng dẫn thông minh có khả năng hỗ trợ học tập thông qua việc điều phối các thành phần xử lý khác nhau. Quy trình tương tác bắt đầu khi người dùng gửi yêu cầu thông qua giao diện chat, sau đó dữ liệu được truyền đến hệ thống backend để xử lý.
Tại lớp xử lý trung tâm, hệ thống thực hiện phân tích đầu vào nhằm xác định mục đích và ngữ cảnh của người dùng thông qua cơ chế phân loại ý định (intent detection). Đồng thời, hệ thống truy xuất thông tin từ bộ nhớ (Memory) để nắm bắt lịch sử hội thoại và tiến trình học tập của người học, từ đó đảm bảo tính liên tục và cá nhân hóa trong quá trình phản hồi.
Dựa trên thông tin đầu vào và trạng thái hiện tại, hệ thống sử dụng cơ chế định tuyến (routing) kết hợp giữa các quy tắc xác định trước và mô hình ngôn ngữ để lựa chọn chiến lược xử lý phù hợp. Thay vì phụ thuộc hoàn toàn vào khả năng suy luận của mô hình ngôn ngữ, các luồng xử lý chính (learning strategies) được thiết kế sẵn nhằm đảm bảo tính nhất quán và định hướng sư phạm trong quá trình học tập.
Trong giai đoạn thực thi, hệ thống kích hoạt các công cụ (Tools) chuyên biệt tương ứng với từng loại yêu cầu. Cụ thể, Grammar Checker được sử dụng để phân tích và phát hiện lỗi ngữ pháp, Translation Tool hỗ trợ dịch thuật, trong khi Exercise Generator tạo bài tập luyện tập dựa trên lỗi sai của người học. Đối với các yêu cầu cần bổ sung kiến thức hoặc giải thích khái niệm, hệ thống áp dụng cơ chế Retrieval-Augmented Generation (RAG) để truy xuất thông tin liên quan từ cơ sở dữ liệu vector thông qua thành phần Retriever, nhằm cung cấp ngữ cảnh chính xác cho quá trình sinh phản hồi.
Mô hình ngôn ngữ lớn (LLM) chủ yếu đảm nhiệm vai trò xử lý ngôn ngữ tự nhiên, bao gồm diễn đạt lại kết quả, cung cấp giải thích và hỗ trợ tạo nội dung học tập ở dạng dễ hiểu. Việc phân tách rõ vai trò giữa backend logic và mô hình ngôn ngữ giúp hệ thống duy trì tính ổn định và giảm thiểu các sai lệch không mong muốn trong quá trình phản hồi.
Trước khi phản hồi được gửi đến người dùng, kết quả được đưa qua lớp kiểm định (Validator/Guardrails) nhằm đánh giá mức độ phù hợp về nội dung, tính nhất quán với dữ liệu truy xuất và tuân thủ các nguyên tắc sư phạm đã được thiết kế. Trong trường hợp kết quả chưa đạt yêu cầu, hệ thống có thể thực hiện điều chỉnh hoặc tái xử lý nhằm cải thiện chất lượng đầu ra.
Cuối cùng, phản hồi được gửi đến người dùng thông qua giao diện chat, đồng thời hệ thống cập nhật lại bộ nhớ bao gồm lịch sử hội thoại, lỗi sai và tiến trình học tập của người học. Điều này giúp đảm bảo khả năng tích lũy tri thức dài hạn và hỗ trợ cá nhân hóa trong các lần tương tác tiếp theo.







2.2 Tổng quan chương trình
Để hiện thực hóa hệ thống AI Agent hỗ trợ học ngoại ngữ với khả năng phản hồi chính xác, cá nhân hóa cao và đảm bảo bảo mật dữ liệu, toàn bộ hệ thống được thiết kế vận hành hoàn toàn trên máy chủ nội bộ (Local Server) thông qua nền tảng Ollama.
Hệ thống được xây dựng dựa trên kiến trúc hiện đại, dễ mở rộng với các công nghệ chính sau: Backend sử dụng FastAPI để xử lý các API một cách nhanh chóng và bất đồng bộ; Frontend được triển khai bằng Streamlit nhằm tạo giao diện chat tương tác trực quan, phù hợp cho việc phát triển prototype và demo. Framework quản lý Agent là LangChain kết hợp LangGraph, giúp xây dựng quy trình xử lý có trạng thái, lập kế hoạch và điều phối công cụ một cách linh hoạt. Vector Database sử dụng FAISS để lưu trữ và truy xuất embedding nhanh chóng trên môi trường local. Ngoài ra, PostgreSQL được sử dụng làm User Database để quản lý thông tin người dùng, lịch sử học tập và tiến trình cá nhân hóa.
Hệ thống AI Agent bao gồm hai thành phần cốt lõi chính:

2.2.1 Mô hình điều phối và tương tác chính (Main Agent Model)
Hệ thống sử dụng mô hình Qwen 2.5 3B chạy trên nền tảng Ollama làm mô hình điều phối trung tâm. Với kích thước nhỏ gọn nhưng vẫn đảm bảo tốc độ phản hồi thời gian thực và khả năng hiểu ngôn ngữ tự nhiên tốt, Qwen 2.5 3B đóng vai trò là LLM Agent chính, chịu trách nhiệm tiếp nhận đầu vào từ giao diện chat, truy xuất bộ nhớ, lập kế hoạch xử lý và điều phối các công cụ chuyên biệt để tạo ra phản hồi phù hợp cho người học.
Hệ thống sử dụng mô hình Qwen 2.5 3B chạy trên nền tảng Ollama làm mô hình điều phối trung tâm. Với kích thước nhỏ gọn nhưng vẫn đảm bảo tốc độ phản hồi thời gian thực và khả năng hiểu ngôn ngữ tự nhiên tốt, Qwen 2.5 3B đóng vai trò là LLM Agent chính, chịu trách nhiệm tiếp nhận đầu vào từ giao diện chat, truy xuất bộ nhớ, lập kế hoạch xử lý và điều phối các công cụ chuyên biệt để tạo ra phản hồi phù hợp cho người học.

2.2.2 Mô hình nhúng và truy xuất tri thức (Embedding & Retrieval)
Để hỗ trợ truy xuất tri thức chính xác, hệ thống sử dụng mô hình BGE-M3 làm mô hình nhúng. Nhờ cơ chế hybrid retrieval (kết hợp Dense, Sparse và Multi-vector), BGE-M3 cho phép hệ thống thực hiện Retrieval-Augmented Generation (RAG) hiệu quả trên cơ sở dữ liệu vector FAISS. Điều này giúp AI Agent truy xuất nhanh chóng và chính xác các tài liệu liên quan về ngữ pháp, từ vựng và mẫu câu thực tế, từ đó nâng cao độ tin cậy và tính sư phạm của phản hồi.

1. PROJECT ROOT (phiên bản “build được”, không over-engineer)
project_root/
│
├── app/
│   ├── api/                    # FastAPI endpoints
│   │   ├── routes_chat.py
│   │   └── routes_user.py
│   │
│   ├── core/                   # Logic điều phối chính (quan trọng nhất)
│   │   ├── pipeline.py         # MAIN FLOW (1 request đi qua đây)
│   │   ├── router.py           # định tuyến: dùng tool / rag / llm
│   │   ├── strategy.py         # learning strategy (cá nhân hóa)
│   │   └── validator.py        # kiểm tra output
│   │
│   ├── memory/                 # bộ nhớ học tập
│   │   ├── short_term.py       # history chat
│   │   ├── long_term.py        # user profile (level, lỗi sai)
│   │   └── memory_service.py
│   │
│   ├── tools/                  # tool thật (không phải chỉ prompt)
│   │   ├── grammar_checker.py
│   │   ├── translator.py
│   │   ├── exercise_generator.py
│   │   └── tool_registry.py
│   │
│   ├── rag/                    # chỉ làm khi đã có core
│   │   ├── retriever.py
│   │   ├── embedder.py
│   │   └── vector_store.py
│   │
│   ├── llm/                    # wrapper model
│   │   ├── llm_client.py
│   │   └── prompts.py
│   │
│   ├── db/
│   │   ├── models.py
│   │   ├── repository.py
│   │   └── session.py
│   │
│   └── services/               # business logic (optional nhưng nên có)
│       └── learning_service.py
│
├── data/
│   ├── raw/                    # tài liệu học
│   └── processed/              # embedding
│
├── tests/
│   └── test_pipeline.py
│
├── main.py                     # entry FastAPI
├── requirements.txt
└── README.md
2. TRÁI TIM HỆ THỐNG (pipeline.py)

Đây là thứ quan trọng nhất — nếu cái này sai, project fail.

def run_pipeline(user_input, user_id):
    # 1. load memory
    memory = memory_service.load(user_id)

    # 2. detect intent (simple trước)
    intent = detect_intent(user_input)

    # 3. chọn strategy
    plan = strategy.decide(intent, memory)

    # 4. route
    if plan.type == "tool":
        result = tool_registry.run(plan.tool, user_input)

    elif plan.type == "rag":
        docs = retriever.search(user_input)
        result = llm.generate_with_context(user_input, docs)

    else:
        result = llm.generate(user_input)

    # 5. validate
    checked = validator.check(result)

    # 6. retry nếu cần
    if not checked.ok:
        result = retry_logic(...)

    # 7. update memory
    memory_service.update(user_id, user_input, result)

    return result

👉 Đây chính là:

AI Engineer = thiết kế flow này

3. ROADMAP 8 TUẦN (chuẩn để build thật)
🔥 TUẦN 1: Setup + Skeleton
FastAPI chạy được
tạo project structure
viết route /chat

👉 Output:

API nhận input → trả string (fake)
🔥 TUẦN 2: CORE PIPELINE (MVP)
viết pipeline.py
intent detection (rule-based đơn giản)
gọi LLM basic

👉 Output:

Chat được (chưa thông minh)

🔥 TUẦN 3: TOOL SYSTEM (quan trọng)
grammar_checker (không chỉ prompt, có parsing)
translator
tool_registry

👉 Output:

system biết:

khi nào sửa lỗi
khi nào dịch
🔥 TUẦN 4: MEMORY
lưu history (short-term)
lưu:
lỗi sai
level user

👉 Output:

system nhớ người dùng

🔥 TUẦN 5: PERSONALIZATION (điểm ăn tiền)
strategy.py:
dễ → khó
lỗi nhiều → luyện lại

👉 Output:

system “dạy học” thật sự

🔥 TUẦN 6: RAG
build vector DB (FAISS)
retriever
integrate pipeline

👉 Output:

giải thích ngữ pháp chuẩn hơn

🔥 TUẦN 7: VALIDATION + LOOP
validator:
check format
check hallucination đơn giản
retry logic

👉 Output:

response ổn định hơn

🔥 TUẦN 8: POLISH
Streamlit UI
logging
test
viết report
4. NGUYÊN TẮC VÀNG (bắt buộc nhớ)
⚠️ 1. Không build tất cả cùng lúc

build theo pipeline

⚠️ 2. Tool phải là logic thật

Không phải:

"fix grammar" → prompt
⚠️ 3. Memory > RAG
memory = core learning
RAG = phụ trợ
⚠️ 4. LLM chỉ là “component”

Không phải “bộ não duy nhất”

5. Mức độ project sau 8 tuần

Nếu làm đúng:

👉 bạn đạt:

portfolio AI Engineer thật
không phải demo
6. Câu chốt

Đừng cố làm AI “thông minh hơn”
hãy làm hệ thống “kiểm soát AI tốt hơn”





1. Context window – vấn đề thật là gì?
🧠 Context window = “trí nhớ ngắn hạn của model”

Ví dụ:

model 4k tokens → chỉ “nhìn thấy” ~3000–3500 tokens input
phần còn lại bị cắt mất
❗ Vấn đề bạn sẽ gặp

Trong system của bạn:

history chat dài
có memory
có RAG
có prompt system

👉 tất cả cộng lại = vỡ context ngay

2. Cách xử lý CHUẨN (AI Engineer dùng)
✅ (1) Không bao giờ nhét full history

Sai phổ biến:

history = full chat từ đầu đến giờ ❌
✔️ Đúng:
history = last N messages (3–5)

👉 chỉ giữ:

phần gần nhất
đủ để hiểu context
✅ (2) Summarize history (cực quan trọng)

Sau mỗi vài turn:

"User hay sai thì hiện tại, đang học thì quá khứ đơn"

👉 lưu thành:

{
  "grammar_weakness": ["past tense"],
  "vocab_level": "A2"
}

👉 Lúc inference:

KHÔNG gửi full chat
chỉ gửi summary + recent messages
✅ (3) Memory tách khỏi context

👉 Đây là insight quan trọng:

❌ Memory không = prompt
✅ Memory = database

Khi cần mới inject:
System:
User is weak at past tense.

👉 thay vì nhét cả history dài

✅ (4) RAG cũng phải giới hạn

Sai:

top_k = 10 ❌

Đúng:

top_k = 2–3 ✔

👉 vì:

mỗi doc = tokens
nhiều doc = vỡ context
✅ (5) Prompt phải “lean”

Đừng viết:

You are a highly intelligent AI system...

👉 lãng phí tokens

Thay bằng:
Fix grammar and explain briefly.
3. Nếu dùng model LOCAL (Ollama)

👉 Tin tốt:

💰 = KHÔNG tốn tiền tokens
bạn chạy local
không trả phí theo token
❗ Nhưng vẫn có “cost khác”
⚠️ (1) Chậm hơn
context dài → inference lâu
⚠️ (2) Tốn RAM / VRAM
context càng dài → càng nặng
⚠️ (3) Model nhỏ dễ “ngu hơn”
khi bị nhồi nhiều context

👉 nên:

dù free → vẫn phải tối ưu context

4. Nếu sau này dùng API (OpenAI, v.v.)

👉 thì:

💸 Tokens = tiền thật

Ví dụ:

1 request = 2000 tokens
1000 users → rất tốn

👉 nên design từ đầu:

token-efficient system

5. Cách bạn áp dụng vào project này (rất cụ thể)
Trong pipeline:
def build_prompt(user_input, memory):
    recent_history = get_last_messages(3)

    summary = memory.summary  # ngắn gọn

    return f"""
    User level: {summary.level}
    Weakness: {summary.weakness}

    Conversation:
    {recent_history}

    Task:
    {user_input}
    """
Memory lưu kiểu này:
{
  "level": "A2",
  "common_errors": ["past tense", "articles"],
  "last_topics": ["travel", "food"]
}
6. Rule vàng (rất quan trọng)

❌ Đừng cố “nhét mọi thứ vào context”
✅ Hãy “chọn lọc thông tin cần thiết”

7. Câu chốt

AI không cần biết mọi thứ
nó chỉ cần biết đúng thứ tại đúng thời điểm