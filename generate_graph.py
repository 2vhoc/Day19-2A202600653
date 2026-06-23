import networkx as nx
import matplotlib.pyplot as plt

# Sử dụng một số triples tiêu biểu để vẽ hình minh họa cho báo cáo
triples = [
    ("VinFast Auto Ltd.", "TICKER", "VFS"),
    ("VinFast Auto Ltd.", "INDUSTRY", "Electric Vehicle Manufacturer"),
    ("VinFast Auto Ltd.", "LISTED_ON", "Nasdaq"),
    ("Madam Thuy Le", "POSITION", "Chairwoman"),
    ("Madam Thuy Le", "AFFILIATION", "VinFast Auto Ltd."),
    ("OpenAI", "FOUNDED_BY", "Sam Altman"),
    ("OpenAI", "FOUNDED_BY", "Elon Musk"),
    ("OpenAI", "FOUNDED_IN", "2015"),
    ("US Energy Information Administration", "PUBLISHES", "Total Energy Monthly Data"),
    ("Total Energy Monthly Data", "COVERS", "Transportation sector"),
    ("Total Energy Monthly Data", "COVERS", "Electric power sector"),
]

G = nx.DiGraph()

for subject, relation, obj in triples:
    G.add_edge(subject, obj, label=relation)

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.8)

# Vẽ nodes và edges
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=2500, font_size=9, font_weight='bold', arrows=True,
        arrowsize=20, edge_color='gray')

# Thêm nhãn cho các cạnh
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

plt.title("Knowledge Graph from Tech Company Corpus", fontsize=16)

# Lưu hình ảnh
plt.savefig("graph.png", dpi=300, bbox_inches="tight")
print("Saved graph.png successfully.")
