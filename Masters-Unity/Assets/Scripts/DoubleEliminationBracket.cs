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

    private bool winnerBracketComplete = false; 
    private bool loserBracketComplete = false; 
    private bool loserBracketRound1Complete = false;
    private bool loserBracketRound2Complete = false;
    private int winnerWinner;
    private int loserWinner;
    private int currentRound = 1;

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
        winnerWinner = -1;
        loserWinner = -1;

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

        //handle losers
        if (currentRound < 10)
        {
            if (currentRound == 7 || currentRound == 8 || currentRound == 9)
            {
                //if the loser lost in the 2nd round of the winner bracket, save it in a special list
                secondRoundLoserList.Enqueue(losingIndex);
            }
            else
            {
                //if the loser lost in the 1st round of the winner bracket, save it in the normal loser list
                loserList.Enqueue(losingIndex);
            }
        }
        else
        {
            //if the loser loses after round 10, they are eliminated
            eliminated.Add(losingIndex);
        }

        //handle losers
        if(currentRound == 11)
        {
            // Save winner of winners bracket
            winnerWinner = winningIndex;
        }
        else if (currentRound == 19)
        {
            // save winner of losers bracket
            loserWinner = winningIndex;
        }
        else if (currentRound == 20)
        {
            //save winner of whole bracket
            winner = winningIndex;
            currentMatch = null;
        }
        else
        {
            //add the winner of the match to the winner list if it has no special circumstances
            winnerList.Enqueue(winningIndex);
        }

        //Handle last match
        if(loserBracketComplete)
        {
            SetLastMatch();
            currentRound++;
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
            //process full winner's bracket
            ProcessWinnerBracket();
        }
        else if (!loserBracketRound1Complete)
        {
            //process first round of losers bracket
            ProcessLoserBracket();
        }
        else if (!loserBracketRound2Complete)
        {     
            //process 2nd round of the losers bracket   
            InjectWinnerBracketLoser();
        }
        else
        {
            //process the end of the lsoers bracket
            Last2OfLosers();
        }
    }

    private void SetLastMatch()
    {
        //winner of losers bracket vs winner of winners bracket
        winnerBracket.Enqueue(new Match(winnerWinner, loserWinner));
        currentMatch = winnerBracket.Dequeue();
    }


    private void ProcessWinnerBracket()
    {
        // Ensure there are enough players to create a winner bracket match
        if (winnerList.Count >= 2)
        {
            winnerBracket.Enqueue(new Match(winnerList.Dequeue(), winnerList.Dequeue()));
        }

        //get next match from winner bracket
        if (winnerBracket.Count > 0 && !winnerBracketComplete)
        {
            currentMatch = winnerBracket.Dequeue();

            // Mark the winner bracket as complete only after Match 11
            if (winnerBracket.Count == 0 && winnerList.Count == 0)
            {
                winnerBracketComplete = true;
            }
        }
    }

    private void ProcessLoserBracket()
    {
        // Process the first three matches in the loser's bracket
        if (!loserBracketComplete && loserBracket.Count == 0)
        {
            int processedMatches = 0;
            while (processedMatches < 3 && loserList.Count >= 2)
            {
                loserBracket.Enqueue(new Match(loserList.Dequeue(), loserList.Dequeue()));
                processedMatches++;
            }
        }

        // Process matches in the loser's bracket
        if (loserBracket.Count > 0)
        {
            currentMatch = loserBracket.Dequeue();

            if (loserBracket.Count == 0)
            {
                loserBracketRound1Complete = true;
            }
        }
    }

    private void InjectWinnerBracketLoser()
    {
        // Inject losers from the second round of the winner's bracket into the loser's bracket
        if (secondRoundLoserList.Count > 0 && winnerList.Count > 0)
        {
            int addedMatches = 0;
            while (addedMatches < 3 )
            {
                int loserFromWinnerBracket = secondRoundLoserList.Dequeue();
                int winnerBracketLoser = winnerList.Dequeue();

                // Add a match to the loser's bracket with the current loser's bracket winner
                loserBracket.Enqueue(new Match(loserFromWinnerBracket, winnerBracketLoser));
                addedMatches++;
            }
        }

        if (loserBracket.Count > 0 && !loserBracketRound2Complete)
        {
            currentMatch = loserBracket.Dequeue();

            // Mark the winner bracket as complete only after Match 11
            if (loserBracket.Count == 0)
            {
                loserBracketRound2Complete = true;
            }
        }
    }

    private void Last2OfLosers()
    {
        if (winnerList.Count >= 2)
        {
            winnerBracket.Enqueue(new Match(winnerList.Dequeue(), winnerList.Dequeue()));
        }

        if (winnerBracket.Count > 0 && !loserBracketComplete)
        {
            currentMatch = winnerBracket.Dequeue();

            // Mark the winner bracket as complete only after Match 19
            if (winnerBracket.Count == 0 && winnerList.Count == 0)
            {
                loserBracketComplete = true;
            }
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
