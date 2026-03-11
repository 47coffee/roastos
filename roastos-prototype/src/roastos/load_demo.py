from pathlib import Path
from roastos.data_loader import load_full_dataset

"""This script serves as a simple demonstration of loading the full dataset using the load_full_dataset function from the data_loader module. 
It prints out the number of sessions, timeseries rows, features, outcomes, 
and coffee lots loaded, as well as a sample session and timeseries row to verify 
that the data has been loaded correctly. This can be used as a quick check to ensure 
that the data loading process is working as expected before proceeding with further analysis or model training."""
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