import matplotlib.pyplot as plt

# Data
phages = ['Phage_17', 'Phage_22']
probabilities = [99.99, 99.85]

# Plot
plt.figure(figsize=(6, 6))
bars = plt.bar(phages, probabilities, color=['green', 'orange'])
plt.ylim(99.7, 100)
plt.ylabel('Predicted Match Probability (%)')
plt.title('Figure 2: Predicted Match Scores for Top Phage Candidates')

# Annotate bars
for bar, prob in zip(bars, probabilities):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02,
             f'{prob:.2f}%', ha='center', va='top', fontsize=12, color='white')

plt.tight_layout()
plt.savefig("figure2_phage_scores.png", dpi=300)
plt.show()
