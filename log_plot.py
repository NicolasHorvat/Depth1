import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def simple_bar_plot(csv_file, last_n=10, save_name="training_log.png"):
    # Load CSV
    df = pd.read_csv(csv_file, header=None, names=["DateTime", "ModelName", "TrainTime", "TestLoss"])

    # Take last n rows
    df = df.tail(last_n)

    x = np.arange(len(df))
    width = 0.25

    train_time = df['TrainTime'] / 60
    test_loss = df['TestLoss']

    fig, ax1 = plt.subplots(figsize=(12,9))
    plt.subplots_adjust(bottom=0.5)

    bars1 = ax1.bar(x - width/2, train_time, width, color='b', label='Train Time (min)')
    ax1.set_ylabel("Training Time (min)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, test_loss, width, color='r', label='Test Loss')
    ax2.set_ylabel("Test Loss", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_xticks(x)
    ax1.set_xticklabels(df['ModelName'], rotation=90, fontsize=8)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')


    # Draw values on top
    for rect, value in zip(bars1, train_time):
        ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{value:.1f}", ha='center', va='bottom', fontsize=8, color='b')
    for rect, value in zip(bars2, test_loss):
        ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{value:.2f}", ha='center', va='bottom', fontsize=8, color='r')

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Log Plot saved to {save_path}")

simple_bar_plot(r"H:\Depth1\training_runs_log.csv", last_n=10)