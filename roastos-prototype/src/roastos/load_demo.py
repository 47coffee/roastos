from pathlib import Path
from roastos.data_loader import load_full_dataset


def main():
    dataset = load_full_dataset(Path("data/mock"))

    print("Dataset loaded successfully.")
    print(f"Sessions:   {len(dataset.sessions)}")
    print(f"Timeseries: {len(dataset.timeseries)}")
    print(f"Features:   {len(dataset.features)}")
    print(f"Outcomes:   {len(dataset.outcomes)}")
    print(f"CoffeeLots: {len(dataset.coffee_lots)}")

    if dataset.sessions:
        print("\nFirst session:")
        print(dataset.sessions[0])
    else:
        print("\nNo sessions loaded.")

    if dataset.timeseries:
        print("\nFirst timeseries row:")
        print(dataset.timeseries[0])
    else:
        print("\nNo timeseries rows loaded.")


if __name__ == "__main__":
    main()