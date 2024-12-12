using System.Collections.Generic;
using System.Linq;
using System;

public class DoubleEliminationBracket
{
    private Dictionary<int, string> stimuliDict; // Mapping indices to stimulus names
    private Queue<Match> winnerBracket;
    private Queue<Match> loserBracket;
    private Match currentMatch;
    private List<int> eliminated;
    private int? winner;

    public DoubleEliminationBracket(Dictionary<int, string> stimuli)
    {
        stimuliDict = stimuli;
        winnerBracket = new Queue<Match>();
        loserBracket = new Queue<Match>();
        eliminated = new List<int>();
        winner = null;

        InitializeBracket();
    }

    private void InitializeBracket()
    {
        var indices = stimuliDict.Keys.ToList();
        for (int i = 0; i < indices.Count; i += 2)
        {
            if (i + 1 < indices.Count)
            {
                winnerBracket.Enqueue(new Match(indices[i], indices[i + 1]));
            }
            else
            {
                winnerBracket.Enqueue(new Match(indices[i], null));
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

        int losingIndex = currentMatch.GetOtherStimulus(winningIndex);

        if (currentMatch.IsFinal)
        {
            winner = winningIndex;
            currentMatch = null;
        }
        else
        {
            loserBracket.Enqueue(new Match(losingIndex, null));
        }

        winnerBracket.Enqueue(new Match(winningIndex, loserBracket.Dequeue().Stimulus1));
        eliminated.Add(losingIndex);

        SetNextMatch();
    }

    private void SetNextMatch()
    {
        if (winnerBracket.Count > 0)
        {
            currentMatch = winnerBracket.Dequeue();
        }
        else if (loserBracket.Count > 0)
        {
            currentMatch = loserBracket.Dequeue();
        }
        else
        {
            currentMatch = null;
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
        public bool IsFinal => Stimulus2 == null;

        public Match(int stimulus1, int? stimulus2)
        {
            Stimulus1 = stimulus1;
            Stimulus2 = stimulus2;
        }

        public int GetOtherStimulus(int winningStimulus)
        {
            if (winningStimulus == Stimulus1) return Stimulus2 ?? -1;
            if (winningStimulus == Stimulus2) return Stimulus1;

            // Handle invalid input with a default return value or throw an exception
            throw new ArgumentException("Winning stimulus not in match");
        }

    }
}