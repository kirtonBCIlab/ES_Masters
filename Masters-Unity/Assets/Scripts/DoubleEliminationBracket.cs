using System.Collections;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.Linq;

public class DoubleEliminationBracket
{
    private Dictionary<int, string> stimuliDict; // Mapping indices to stimulus names
    private Queue<Match> winnerBracket;
    private Queue<Match> loserBracket;
    private Queue<int> loserList;
    private Queue<int> winnerList;
    private Queue<int> secondRoundLoserList;
    private Match currentMatch;
    private List<int> eliminated;
    private int? winner;

    private bool winnerBracketComplete = false; // Tracks if the winner bracket is complete
    private bool loserBracketComplete = false;  // Tracks if the loser bracket is complete
    private int currentRound = 1; // Initialize the round counter

    public DoubleEliminationBracket(Dictionary<int, string> stimuli)
    {
        stimuliDict = stimuli;
        winnerBracket = new Queue<Match>();
        loserBracket = new Queue<Match>();
        loserList = new Queue<int>();
        winnerList = new Queue<int>();
        secondRoundLoserList = new Queue<int>();
        eliminated = new List<int>();
        winner = null;

        InitializeBracket();
    }

    private void InitializeBracket()
    {
        var indices = stimuliDict.Keys.ToList();

        // Create the first round of winner bracket matches
        for (int i = 0; i < indices.Count; i += 2)
        {
            if (i + 1 < indices.Count)
            {
                winnerBracket.Enqueue(new Match(indices[i], indices[i + 1]));
            }
            else
            {
                winnerBracket.Enqueue(new Match(indices[i], null)); // Bye for the last stimulus
            }
        }

        SetNextMatch();
    }

    public Match GetCurrentMatch()
    {
        return currentMatch;
    }

    public void RecordMatchResult(int winningIndex)
    {
        if (currentMatch == null) return;

        // Get the losing index (the other stimulus in the match)
        int losingIndex = currentMatch.GetOtherStimulus(winningIndex);

        if (currentRound < 10 || currentRound > 20)
        {
            if (currentRound == 7 || currentRound == 8 || currentRound == 9)
            {
                secondRoundLoserList.Enqueue(losingIndex);
                Debug.Log($"secondRoundLoserList contents: {secondRoundLoserList.Count}");
            }
            else
            {
                // Move to the loser bracket if not eliminated
                loserList.Enqueue(losingIndex);
                Debug.Log($"LoserList contents: {loserList.Count}");
            }
        }
        else
        {
            //Eliminate if already in the loser bracket
            eliminated.Add(losingIndex);
            Debug.Log($"eliminated contents: {eliminated.Count}");
        }

        winnerList.Enqueue(winningIndex);

        // Log the winner and loser
        Debug.Log($"Round {currentRound} result: Stimulus {winningIndex} wins, Stimulus {losingIndex} {(currentRound >= 10 && currentRound <= 20 ? "(not recorded)" : "loses")}");

        // Check for completion or set up the next match
        if (winnerBracket.Count == 0 && loserBracket.Count == 0 && loserList.Count == 1 && winnerList.Count == 1)
        {
            winner = winnerList.Dequeue(); // Final match reached
            Debug.Log($"Final match reached. Winner is Stimulus {winner}");
            currentMatch = null; // No more matches after the final
        }
        else
        {
            SetNextMatch();
            currentRound++; // Increment the round after processing the match
        }
    }


    private void SetNextMatch()
    {
        if (!winnerBracketComplete)
        {
            ProcessWinnerBracket();
        }
        else
        {
            // Process loser's bracket and inject winner bracket losers if needed
            ProcessLoserBracket();

            if (loserBracket.Count == 0 && loserList.Count > 1 && winnerBracketComplete)
            {
                InjectWinnerBracketLoser();
            }
        }

        // Handle final match logic
        if (winnerBracketComplete && loserBracketComplete && winnerList.Count == 1 && loserList.Count == 1)
        {
            // Final match between the last winner and loser
            winnerBracket.Enqueue(new Match(winnerList.Dequeue(), loserList.Dequeue()));
            currentMatch = winnerBracket.Dequeue();
        }
    }


    private void ProcessWinnerBracket()
    {
        // Ensure there are enough players to create a winner bracket match
        if (winnerList.Count >= 2)
        {
            winnerBracket.Enqueue(new Match(winnerList.Dequeue(), winnerList.Dequeue()));
        }

        // Handle bye situation for the winner bracket
        if (winnerBracket.Count == 0 && winnerList.Count == 1)
        {
            int winnerFromBye = winnerList.Dequeue();
            winnerBracket.Enqueue(new Match(winnerFromBye, null));
        }

        if (winnerBracket.Count > 0)
        {
            currentMatch = winnerBracket.Dequeue();

            // Mark the winner bracket as complete after the final match in the winner's bracket
            if (winnerBracket.Count == 0 && winnerList.Count == 0)
            {
                winnerBracketComplete = true;
            }
        }
    }

    private void ProcessLoserBracket()
    {
        // Process the first three matches in the loser's bracket
        if (!loserBracketComplete && loserBracket.Count == 0 && loserList.Count >= 2)
        {
            int processedMatches = 0;
            while (processedMatches < 3 && loserList.Count >= 2)
            {
                loserBracket.Enqueue(new Match(loserList.Dequeue(), loserList.Dequeue()));
                processedMatches++;
            }

            // After processing first three matches, stop to wait for second-round losers from the winner's bracket
            if (processedMatches == 3 && loserList.Count > 0)
            {
                Debug.Log("First three matches in the loser's bracket processed. Waiting for second-round losers.");
                return; // Exit to wait for more players from the winner's bracket
            }
        }

        // Handle bye situation for the losers bracket
        if (loserBracket.Count == 0 && loserList.Count == 1)
        {
            int winnerFromBye = loserList.Dequeue();
            loserBracket.Enqueue(new Match(winnerFromBye, null));
        }

        // Process matches in the loser's bracket
        if (loserBracket.Count > 0)
        {
            currentMatch = loserBracket.Dequeue();
        }

        // Mark the loser's bracket as complete if conditions are met
        if (loserBracket.Count == 0 && loserList.Count <= 1)
        {
            loserBracketComplete = true;
        }
    }

    private void InjectWinnerBracketLoser()
    {
        // Inject losers from the second round of the winner's bracket into the loser's bracket
        if (winnerBracketComplete && secondRoundLoserList.Count >= 1 && winnerList.Count > 0)
        {
            int loserFromWinnerBracket = secondRoundLoserList.Dequeue();
            int winnerBracketLoser = winnerList.Dequeue();

            // Add a match to the loser's bracket with the current loser's bracket winner
            loserBracket.Enqueue(new Match(loserFromWinnerBracket, winnerBracketLoser));
            Debug.Log($"Injected loser from winner's bracket into loser's bracket: {loserFromWinnerBracket} vs. {winnerBracketLoser}");
        }
    }

    public bool IsComplete()
    {
        return winner.HasValue;
    }

    public string GetWinner()
    {
        return winner.HasValue ? stimuliDict[winner.Value] : null;
    }

    public List<string> GetEliminated()
    {
        return eliminated.Select(index => stimuliDict[index]).ToList();
    }

    public class Match
    {
        public int Stimulus1 { get; }
        public int? Stimulus2 { get; }
        public bool IsFinal { get; set; }

        public Match(int stimulus1, int? stimulus2)
        {
            Stimulus1 = stimulus1;
            Stimulus2 = stimulus2;
        }

        public int GetOtherStimulus(int winningStimulus)
        {
            if (winningStimulus == Stimulus1) return Stimulus2 ?? -1;
            if (winningStimulus == Stimulus2) return Stimulus1;

            throw new ArgumentException("Winning stimulus not in match");
        }
    }
}
