import numpy as np

class EloRanking:
    def __init__(self, initial_rating=100, k_base=40, n_base=3):
        """
        Initialize the EloRanking system.

        Parameters:
        - initial_rating (int): Default initial rating for each stimulus.
        - k_base (int): Base K-factor for Elo updates.
        - n_base (int): Number of matches at which the K-factor is not scaled.
        """
        self.ratings = {}
        self.initial_rating = initial_rating
        self.k_base = k_base
        self.n_base = n_base
        self.match_counts = {}
        self.scores = {}

    def _dynamic_k_factor(self, stimulus):
        """
        Calculate the dynamic K-factor for a given stimulus based on matches played.
        """
        matches_played = self.match_counts.get(stimulus, 0)
        scaling_factor = np.sqrt(self.n_base / max(1, matches_played))
        return self.k_base * scaling_factor

    def add_stimulus(self, stimulus):
        """
        Add a new stimulus to the rating system.
        """
        if stimulus not in self.ratings:
            self.ratings[stimulus] = self.initial_rating
            self.match_counts[stimulus] = 0

    def update_ratings(self, winner, loser):
        """
        Update Elo ratings after a match.

        Parameters:
        - winner (str): Identifier for the winning stimulus.
        - loser (str): Identifier for the losing stimulus.
        """
        self.add_stimulus(winner)
        self.add_stimulus(loser)

        # Dynamic K-factors
        k_winner = self._dynamic_k_factor(winner)
        k_loser = self._dynamic_k_factor(loser)

        # Current ratings
        rating_winner = self.ratings[winner]
        rating_loser = self.ratings[loser]

        # Probability of winning
        expected_winner = 1 / (1 + 10 ** ((rating_loser - rating_winner) / 400))
        expected_loser = 1 - expected_winner

        # Update ratings
        self.ratings[winner] += k_winner * (1 - expected_winner)
        self.ratings[loser] += k_loser * (0 - expected_loser)

        # Increment match counts
        self.match_counts[winner] += 1
        self.match_counts[loser] += 1

    def get_rankings(self):
        """
        Get the current rankings sorted by Elo score.
        """
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
    
    def get_scores(self):
        """
        Get the current scores sorted by Elo score.
        """
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

    def post_hoc_adjustment(self):
        """
        Apply a post-hoc adjustment to penalize stimuli with fewer comparisons.
        """
        adjusted_ratings = {}
        for stimulus, rating in self.ratings.items():
            matches_played = self.match_counts.get(stimulus, 0)
            penalty = 25 / (1 + matches_played)  # Adjust penalty as needed
            adjusted_ratings[stimulus] = rating - penalty
        self.ratings = adjusted_ratings

    def standardize_rankings(self):
        """
        Standardize the adjusted Elo ratings to a 1-10 scale and update `scores` attribute.
        """
        if not self.ratings:
            return {}

        min_rating = min(self.ratings.values())
        max_rating = max(self.ratings.values())

        self.scores = {
            stimulus: 1 + 9 * (rating - min_rating) / (max_rating - min_rating)
            for stimulus, rating in self.ratings.items()
        }

        return self.scores
