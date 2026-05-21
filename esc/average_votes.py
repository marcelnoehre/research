import requests
import pandas as pd
from collections import defaultdict

voting_history = defaultdict(lambda: defaultdict(list))

for year in range(1957, 2026):
    if year == 2020:
        continue
    try:
        print(year)
        url = f'https://eurovisionapi.runasp.net/api/senior/contests/{year}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        final_round = next((item for item in data['rounds'] if item.get('name') == 'final'), None)
        for performance in final_round['performances']:
            contestant = next((c for c in data['contestants'] if c.get('id') == performance['contestantId']), None)['country']
            votes = next((s for s in performance['scores'] if s.get('name') == 'total'), None)['votes']

            for voter, points in votes.items():
                voting_history[voter][contestant].append(points)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

for giver, receivers in voting_history.items():
    for receiver, scores in receivers.items():
        avg = sum(scores) / len(scores)
        print(f"{giver} gives an average of {avg:.2f} to {receiver}")

binary_matrix = {}

all_voters = sorted(voting_history.keys())

all_receivers = sorted({
    receiver
    for receivers in voting_history.values()
    for receiver in receivers.keys()
})

for voter in all_voters:
    binary_matrix[voter] = {}

    for receiver in all_receivers:

        scores = voting_history[voter].get(receiver, [])

        if scores:
            avg_score = sum(scores) / len(scores)
            binary_matrix[voter][receiver] = avg_score >= 8
        else:
            binary_matrix[voter][receiver] = False

df = pd.DataFrame.from_dict(binary_matrix, orient="index")

df = df.sort_index(axis=0)
df = df.sort_index(axis=1)

print(df)

df.to_csv("eurovision_binary_context.csv")