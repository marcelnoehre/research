import requests
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

edge_weights = defaultdict(int)   # (voter, recipient) -> raw points
voter_totals = defaultdict(int)   # voter -> total points ever given out

for year in range(1975, 2026):
    if year == 2020:
        continue

    try:
        print(year)
        url = f'https://eurovisionapi.runasp.net/api/senior/contests/{year}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        final_round = next(
            (r for r in data['rounds'] if r.get('name') == 'final'), None
        )
        if not final_round:
            continue

        contestants = {c['id']: c['country'] for c in data.get('contestants', [])}

        for performance in final_round['performances']:
            recipient = contestants.get(performance.get('contestantId', -1), '')
            if not recipient:
                continue

            if year >= 2016:
                score = next(
                    (s for s in performance['scores'] if s.get('name') == 'public'), None
                )
            else:
                score = next(
                    (s for s in performance['scores'] if s.get('name') == 'total'), None
                )

            for voter, pts in (score['votes'] if score else {}).items():
                if pts and pts > 0:
                    edge_weights[(voter, recipient)] += pts
                    voter_totals[voter] += pts

    except requests.exceptions.RequestException as e:
        print(f"  Request error for {year}: {e}")

# Normalize: weight(A,B) = votes(A->B)/total(A) + votes(B->A)/total(B)
# This removes the bias from countries that simply participate more years.
G = nx.Graph()
all_pairs = set((min(a, b), max(a, b)) for (a, b) in edge_weights)
for a, b in all_pairs:
    w = 0
    if voter_totals[a]:
        w += edge_weights[(a, b)] / voter_totals[a]
    if voter_totals[b]:
        w += edge_weights[(b, a)] / voter_totals[b]
    if w > 0:
        G.add_edge(a, b, weight=w)

# Community detection on the full graph
communities = nx.community.greedy_modularity_communities(G, weight='weight')
node_community = {n: i for i, com in enumerate(communities) for n in com}
cmap = plt.cm.get_cmap('tab20', len(communities))

# Layout uses ALL edges so communities attract each other spatially
pos = nx.spring_layout(G, weight='weight', seed=42, k=2.5)

# Draw only the top 15% strongest edges to avoid a hairball
all_weights = sorted(d['weight'] for _, _, d in G.edges(data=True))
threshold = all_weights[int(len(all_weights) * 0.85)]
G_draw = nx.Graph(
    (u, v, d) for u, v, d in G.edges(data=True) if d['weight'] >= threshold
)

node_colors = [cmap(node_community[n]) for n in G.nodes()]
edge_w = [d['weight'] for _, _, d in G_draw.edges(data=True)]
max_w = max(edge_w)

fig, ax = plt.subplots(figsize=(20, 16))

nx.draw_networkx_edges(
    G_draw, pos, ax=ax,
    width=[0.5 + 4.5 * w / max_w for w in edge_w],
    alpha=0.4,
    edge_color='#888888',
)
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_size=300,
    node_color=node_colors,
    alpha=0.9,
)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color='white', font_weight='bold')

ax.set_title('Eurovision Song Contest — Nation Voting Affinity 1975–2025\n'
             '(top 10% edges by normalised mutual loyalty, colours = detected communities)', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig('esc_force.png', dpi=150, bbox_inches='tight')
plt.show()
