import pandas as pd

df = pd.read_csv("eurovision_binary_context.csv", index_col=0)

df = df.astype(bool)

rows = list(df.index)
cols = list(df.columns)

with open("eurovision_binary_context.cxt", "w", encoding="utf-8") as f:
    f.write("B\n")
    f.write("\n")

    f.write(f"{len(rows)}\n")
    f.write(f"{len(cols)}\n")

    for row in rows:
        f.write(f"{row}\n")

    for col in cols:
        f.write(f"{col}\n")

    for row in rows:
        line = "".join("x" if value else "." for value in df.loc[row])
        f.write(f"{line}\n")

print("Burmeister .cxt file created: eurovision_binary_context.cxt")