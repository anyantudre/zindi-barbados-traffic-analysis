# Plotting and visualization utilities
"""
This module contains functions for data visualization and plotting.
"""

import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Video
from sklearn.metrics import confusion_matrix
from typing import List, Optional


class Visualizer:
    """Class for creating visualizations."""
    
    def __init__(self, style: str = "whitegrid"):
        """
        Initialize the Visualizer.
        
        Args:
            style: Seaborn style to use.
        """
        sns.set_theme(style=style)
    
    @staticmethod
    def show_video(
        video_path: str, 
        trim: bool = False, 
        duration: int = 10, 
        width: int = 800
    ) -> None:
        """
        Convert (and optionally trim) a video, suppress ffmpeg output, and display it inline.
        
        Args:
            video_path: Path to the input video file.
            trim: Whether to trim the video to a short preview.
            duration: Duration in seconds if trimming.
            width: Display width in pixels.
        """
        output_path = "preview.mp4"
        
        # Build ffmpeg command
        cmd = ["ffmpeg", "-i", video_path]
        if trim:
            cmd += ["-t", str(duration)]
        cmd += [output_path, "-y"]  # overwrite existing
        
        # Run ffmpeg silently (no stdout/stderr)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Display the video inline
        display(Video(output_path, width=width, embed=True))
    
    @staticmethod
    def plot_label_distribution(
        df: pd.DataFrame, 
        figsize: tuple = (12, 5)
    ) -> None:
        """
        Plot distribution of congestion labels (enter and exit).
        
        Args:
            df: DataFrame with congestion rating columns.
            figsize: Figure size.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        total = len(df)
        
        # Left plot: congestion_enter_rating
        ax1 = sns.countplot(x="congestion_enter_rating", data=df, palette="viridis", ax=axes[0])
        ax1.set_title("Congestion Enter Rating")
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Count")
        
        # Add percentage labels
        for p in ax1.patches:
            height = p.get_height()
            percent = 100 * height / total
            ax1.annotate(f'{percent:.1f}%', 
                         (p.get_x() + p.get_width()/2., height), 
                         ha='center', va='bottom', fontsize=9)
        
        # Right plot: congestion_exit_rating
        ax2 = sns.countplot(x="congestion_exit_rating", data=df, palette="magma", ax=axes[1])
        ax2.set_title("Congestion Exit Rating")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("")
        
        # Add percentage labels
        for p in ax2.patches:
            height = p.get_height()
            percent = 100 * height / total
            ax2.annotate(f'{percent:.1f}%', 
                         (p.get_x() + p.get_width()/2., height), 
                         ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_hourly_distribution(
        df: pd.DataFrame, 
        figsize: tuple = (14, 6)
    ) -> None:
        """
        Plot hourly distribution of congestion ratings.
        
        Args:
            df: DataFrame with hour and congestion rating columns.
            figsize: Figure size.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # Left plot: congestion_enter_rating by hour
        ax1 = sns.countplot(
            x="hour", hue="congestion_enter_rating", data=df,
            palette="viridis", ax=axes[0]
        )
        ax1.set_title("Hourly Distribution of Congestion Enter Rating", fontsize=13)
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Count")
        ax1.legend(title="Enter Rating", loc="upper right")
        
        # Right plot: congestion_exit_rating by hour
        ax2 = sns.countplot(
            x="hour", hue="congestion_exit_rating", data=df,
            palette="magma", ax=axes[1]
        )
        ax2.set_title("Hourly Distribution of Congestion Exit Rating", fontsize=13)
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("")
        ax2.legend(title="Exit Rating", loc="upper right")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_signaling_distribution(df: pd.DataFrame) -> None:
        """
        Plot distribution of signaling values.
        
        Args:
            df: DataFrame with signaling column.
        """
        ax = sns.countplot(x="signaling", data=df, palette="viridis")
        total = len(df)
        
        # Add percentage labels
        for p in ax.patches:
            height = p.get_height()
            percent = 100 * height / total
            ax.annotate(f'{percent:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10, color='black')
        
        plt.title("Distribution of Signaling Values", fontsize=14)
        plt.xlabel("Signaling Category")
        plt.ylabel("Count")
        plt.show()
    
    @staticmethod
    def _add_percentages(ax, data: pd.DataFrame, x_col: str, hue_col: str) -> None:
        """
        Add percentage annotations to a countplot.
        
        Args:
            ax: Matplotlib axis.
            data: DataFrame with the data.
            x_col: Column name for x-axis.
            hue_col: Column name for hue.
        """
        contingency_table = pd.crosstab(data[x_col], data[hue_col], normalize='index') * 100
        
        bars = ax.patches
        n_hues = data[hue_col].nunique()
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            
            hue_index = i % n_hues
            x_index = i // n_hues
            
            x_label = ax.get_xticklabels()[x_index].get_text()
            hue_value = list(contingency_table.columns)[hue_index]
            
            percent = contingency_table.loc[x_label, hue_value]
            
            if height > 0:
                ax.annotate(f"{percent:.1f}%",
                            (bar.get_x() + bar.get_width()/2., height),
                            ha="center", va="bottom", fontsize=9)
    
    @staticmethod
    def plot_signaling_vs_congestion(
        df: pd.DataFrame, 
        figsize: tuple = (18, 6)
    ) -> None:
        """
        Plot signaling vs congestion ratings.
        
        Args:
            df: DataFrame with signaling and congestion columns.
            figsize: Figure size.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # Left plot
        ax1 = sns.countplot(
            x="signaling", hue="congestion_enter_rating",
            data=df, palette="viridis", ax=axes[0]
        )
        ax1.set_title("Signaling vs Congestion Enter Rating", fontsize=13)
        Visualizer._add_percentages(ax1, df, "signaling", "congestion_enter_rating")
        
        # Right plot
        ax2 = sns.countplot(
            x="signaling", hue="congestion_exit_rating",
            data=df, palette="magma", ax=axes[1]
        )
        ax2.set_title("Signaling vs Congestion Exit Rating", fontsize=13)
        Visualizer._add_percentages(ax2, df, "signaling", "congestion_exit_rating")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        class_names: List[str],
        figsize: tuple = (8, 6)
    ) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            class_names: List of class names.
            figsize: Figure size.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Validation Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
    
    @staticmethod
    def plot_feature_importance(
        importances: np.ndarray, 
        feature_names: List[str],
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot feature importance bar chart.
        
        Args:
            importances: Array of feature importances.
            feature_names: List of feature names.
            figsize: Figure size.
        """
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=figsize)
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
        plt.title("Feature Importance in RandomForest")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()

