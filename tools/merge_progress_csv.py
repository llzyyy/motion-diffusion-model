import argparse
import os
import pandas as pd


def read_csv_checked(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Empty file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows in file: {path}")
    if "step" not in df.columns:
        raise ValueError(f"'step' column not found in: {path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv1", type=str, required=True, help="First progress.csv")
    parser.add_argument("--csv2", type=str, required=True, help="Second progress.csv")
    parser.add_argument("--out", type=str, required=True, help="Output merged csv path")
    args = parser.parse_args()

    df1 = read_csv_checked(args.csv1)
    df2 = read_csv_checked(args.csv2)

    merged = pd.concat([df1, df2], ignore_index=True)

    # 按 step 排序
    merged = merged.sort_values("step").reset_index(drop=True)

    # 如果有重复 step，保留后出现的那条
    merged = merged.drop_duplicates(subset=["step"], keep="last").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)

    print("=" * 70)
    print("MERGE FINISHED")
    print("=" * 70)
    print(f"csv1 rows : {len(df1)}")
    print(f"csv2 rows : {len(df2)}")
    print(f"merged rows : {len(merged)}")
    print(f"step range : {merged['step'].min()} ~ {merged['step'].max()}")
    print(f"saved to   : {args.out}")


if __name__ == "__main__":
    main()