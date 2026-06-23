import os
import glob
import json
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import networkx as nx
import matplotlib.pyplot as plt

# Khởi tạo cấu hình API
dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

llm = ChatOpenAI(temperature=0, model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL)

prompt_template = """
Trích xuất thông tin từ đoạn văn bản sau thành các bộ ba (triples) để xây dựng Knowledge Graph.
Lưu ý quan trọng: Các mốc thời gian, số liệu (năm, doanh thu, tỷ lệ...) KHÔNG được làm Object độc lập mà phải được lưu dưới dạng metadata (thuộc tính) của quan hệ.
Trả về kết quả dưới dạng danh sách các JSON object. Mỗi object có cấu trúc chính xác như sau: {{"subject": "...", "relation": "...", "object": "...", "metadata": {{"key": "value"}}}}.

Chỉ trả về chuỗi JSON hợp lệ, không giải thích gì thêm.

Văn bản: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = prompt | llm

dataset_dir = "./dataset"
all_triples = []

# Load 5 file đầu tiên từ dataset thật
file_paths = glob.glob(f"{dataset_dir}/*.txt")[:5]

print("Đang đọc dữ liệu thật từ dataset và trích xuất Triples...")
for file_path in file_paths:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()[:1500] # Giới hạn token
            
            try:
                response = chain.invoke({"text": text})
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]
                
                triples_data = json.loads(content)
                for item in triples_data:
                    if "subject" in item and "relation" in item and "object" in item:
                        all_triples.append((item["subject"], item["relation"], item["object"], item.get("metadata", {})))
            except Exception as e:
                print(f"Lỗi khi xử lý {file_path}: {e}")

print(f"Trích xuất thành công {len(all_triples)} triples. Đang vẽ đồ thị...")

G = nx.DiGraph()

# Thêm các edges và metadata từ dữ liệu thật
for subject, relation, obj, metadata in all_triples:
    G.add_edge(subject, obj, label=relation, **metadata)

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Vẽ đồ thị
nx.draw(G, pos, with_labels=True, node_color='#87CEEB', 
        node_size=2000, font_size=8, font_weight='bold', arrows=True,
        arrowsize=15, edge_color='#666666', alpha=0.9)

# Thêm nhãn cho các cạnh
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='#FF4500', font_size=7)

plt.title("Knowledge Graph from Real Dataset", fontsize=18, fontweight='bold', pad=20)

# Lưu hình ảnh đồ thị thật
plt.savefig("graph.png", dpi=300, bbox_inches="tight")
print("Đã lưu graph.png từ dữ liệu thật thành công.")
