#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read the CSV file
    # Make sure "results.csv" is in the same directory, or provide a full path:
    df = pd.read_csv("results.csv")

    # Create a figure with 2 subplots side by side
    plt.figure(figsize=(10, 5))

    # --- Plot 1: checkSymSeq vs. checkSymMPI ---
    plt.subplot(1, 2, 1)
    plt.plot(df['n'], df['checkSymSeq'], marker='o', label='Sequential')
    plt.plot(df['n'], df['checkSymMPI'], marker='o', label='MPI (4 ranks)')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Check Symmetry Performance')
    plt.legend()
    plt.grid(True)
    # Use a log scale for both axes (optional, but often helpful for performance data)
    plt.xscale('log', base=2)
    plt.yscale('log')

    # --- Plot 2: matTransposeSeq vs. matTransposeMPI ---
    plt.subplot(1, 2, 2)
    plt.plot(df['n'], df['matTransposeSeq'], marker='o', label='Sequential')
    plt.plot(df['n'], df['matTransposeMPI'], marker='o', label='MPI (4 ranks)')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Transpose Performance')
    plt.legend()
    plt.grid(True)
    # Log scale again (optional)
    plt.xscale('log', base=2)
    plt.yscale('log')

    # Adjust layout so plots don’t overlap
    plt.tight_layout()

    # Show the plots on screen
    plt.show()

    # If you want to save the figure to a file instead, uncomment the line below:
    # plt.savefig("performance_comparison.png")

if __name__ == "__main__":
    main()
