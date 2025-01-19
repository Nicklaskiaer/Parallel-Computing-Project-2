#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read the CSV file
    # Make sure "results.csv" is in the same directory, or provide a full path:
    df = pd.read_csv("src/results-8-ranks.csv")

    # Create a figure with 2 subplots side by side
    plt.figure(figsize=(12, 5))

    # --- Plot 1: checkSymSeq vs. checkSymMPI vs. checkSymOMP ---
    plt.subplot(1, 2, 1)
    plt.plot(df['n'], df['checkSymSeq'], marker='o', label='Sequential')
    plt.plot(df['n'], df['checkSymMPI'], marker='o', label='MPI (8 ranks)')
    plt.plot(df['n'], df['checkSymOMP'], marker='o', label='OpenMP')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Check Symmetry Performance')
    plt.legend()
    plt.grid(True)
    # Use a log scale for both axes (optional)
    plt.xscale('log', base=2)
    plt.yscale('log')

    # --- Plot 2: matTransposeSeq vs. matTransposeMPI vs. matTransposeOMP ---
    plt.subplot(1, 2, 2)
    plt.plot(df['n'], df['matTransposeSeq'], marker='o', label='Sequential')
    plt.plot(df['n'], df['matTransposeMPI'], marker='o', label='MPI (8 ranks)')
    plt.plot(df['n'], df['matTransposeOMP'], marker='o', label='OpenMP')
    plt.xlabel('Matrix size (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Transpose Performance')
    plt.legend()
    plt.grid(True)
    # Use a log scale for both axes (optional)
    plt.xscale('log', base=2)
    plt.yscale('log')

    # Adjust layout so plots don’t overlap
    plt.tight_layout()

    # Display the plots
    plt.show()

    # If you'd like to save the figure, uncomment the following line:
    # plt.savefig("performance_comparison.png")

if __name__ == "__main__":
    main()
