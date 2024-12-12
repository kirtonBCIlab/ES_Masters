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
    private Match currentMatch;
    private List<int> eliminated;
    private int? winner;

    private bool winnerBracketComplete = false; // Tracks if the winner bracket is complete
    private bool loserBracketComplete = false;  // Tracks if the loser bracket is complete


    public DoubleEliminationBracket(Dictionary<int, string> stimuli)
    {
        stimuliDict = stimuli;
        winnerBracket = new Queue<Match>();
        loserBracket = new Queue<Match>();
        loserList = new Queue<int>();
        winnerList = new Queue<int>();
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

        if (loserList.Contains(losingIndex))
        {
            // Eliminate if already in the loser bracket
            eliminated.Add(losingIndex);
        }
        else
        {
            // Move to the loser bracket if not eliminated
            loserList.Enqueue(losingIndex);
        }

        winnerList.Enqueue(winningIndex);


        // Log the winner and loser
        Debug.Log($"Match result: Stimulus {winningIndex} wins, Stimulus {losingIndex} loses");

        if (winnerBracket.Count == 0 && loserBracket.Count == 0 && loserList.Count == 1 && winnerList.Count == 1)
        {
            winner = winnerList.Dequeue(); // Final match reached
            Debug.Log($"Final match reached. Winner is Stimulus {winner}");
            currentMatch = null; // No more matches after the final
        }
        else
        {
            SetNextMatch();
        }
    }

    private void SetNextMatch()
    {
        if (winnerList.Count == 3 && !winnerBracketComplete)
        {
            // Create a match for the first two winners
            int participant1 = winnerList.Dequeue();
            int participant2 = winnerList.Dequeue();

            winnerBracket.Enqueue(new Match(participant1, participant2));
            Debug.Log($"Match created: Stimulus {participant1} vs. Stimulus {participant2}");

            // Move the third participant directly to the next round (bye)
            int byeParticipant = winnerList.Dequeue();
            winnerList.Enqueue(byeParticipant);
            Debug.Log($"Stimulus {byeParticipant} gets a bye to the next round.");
        }
        else if (loserList.Count == 3 && !loserBracketComplete)
        {
            // Create a match for the first two winners
            int participant1 = loserList.Dequeue();
            int participant2 = loserList.Dequeue();

            loserBracket.Enqueue(new Match(participant1, participant2));
            Debug.Log($"Match created: Stimulus {participant1} vs. Stimulus {participant2}");

            // Move the third participant directly to the next round (bye)
            int byeParticipant = loserList.Dequeue();
            loserList.Enqueue(byeParticipant);
            Debug.Log($"Stimulus {byeParticipant} gets a bye to the next round.");
        }
        // Check if both winner and loser lists have enough players to create a match
        else if (winnerList.Count >= 2 && !winnerBracketComplete)
        {
            // Create the next winner bracket match from the winner list
            winnerBracket.Enqueue(new Match(winnerList.Dequeue(), winnerList.Dequeue()));
        }

        if (loserList.Count >= 2 && !loserBracketComplete)
        {
            // Create the next loser bracket match from the loser list
            loserBracket.Enqueue(new Match(loserList.Dequeue(), loserList.Dequeue()));
        }

        if (winnerBracket.Count > 0 && !winnerBracketComplete)
        {
            currentMatch = winnerBracket.Dequeue();

            if (winnerBracket.Count == 0 && winnerList.Count == 1)
                {
                    winnerBracketComplete = true;
                }
        }
        else if (loserBracket.Count > 0 && !loserBracketComplete)
        {
            currentMatch = loserBracket.Dequeue();

            if(loserBracket.Count == 0 && loserList.Count == 1)
            {
                loserBracketComplete = true;
            }
        }
    
        else if (winnerBracketComplete && loserBracketComplete && winnerList.Count == 1 && loserList.Count == 1)
        {
            // Create final match between winner and loser
            Debug.Log("Preparing final match...");
            winnerBracket.Enqueue(new Match(winnerList.Dequeue(), loserList.Dequeue()));
            currentMatch = winnerBracket.Dequeue();
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
