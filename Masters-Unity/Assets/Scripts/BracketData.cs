using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace BCIEssentials.Utilities
{
    public class BracketData
    {
        // Dictionary to hold stimulus indices and their corresponding names
        public Dictionary<int, string> StimulusIndex { get; private set; }

        // List of pairs for comparisons, with a pair number as a string
        public List<(string pairNumber, string item1, string item2)> ComparisonPairs { get; private set; }

        // List to hold winners
        public List<int> Winners { get; private set; }

        // Constructor to initialize the data structure
        public BracketData()
        {
            StimulusIndex = new Dictionary<int, string>();
            ComparisonPairs = new List<(string, string, string)>();
            Winners = new List<int>();
        }

        // Method to set the entire StimulusIndex dictionary
        public void SetStimulusIndex(Dictionary<int, string> stimulusDictionary)
        {
            StimulusIndex = new Dictionary<int, string>(stimulusDictionary);
        }

        // Method to add a comparison pair with a pair number
        public void AddComparisonPair(int pairNumber, string item1, string item2)
        {
            string pairNumberString = "Pair Number " + pairNumber.ToString(); // Convert pair number to string
            ComparisonPairs.Add((pairNumberString, item1, item2));
        }

        // Method to add a winner
        public void AddWinner(int winner)
        {
            Winners.Add(winner);
        }
        public void ExportToCsv(string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                // Write headers
                writer.WriteLine("Pair Number,Item 1,Item 2,Winner");

                // Iterate through ComparisonPairs and Winners
                for (int i = 0; i < ComparisonPairs.Count; i++)
                {
                    string pairNumber = ComparisonPairs[i].pairNumber;
                    string item1 = ComparisonPairs[i].item1;
                    string item2 = ComparisonPairs[i].item2;

                    // Get the corresponding winner (stimulus name) if it exists
                    string winner = "N/A"; // Default value in case winner is not found
                    if (i < Winners.Count)
                    {
                        int winnerIndex = Winners[i];
                        // Get the stimulus name from the StimulusIndex dictionary using the winner index
                        if (StimulusIndex.ContainsKey(winnerIndex))
                        {
                            winner = StimulusIndex[winnerIndex];
                        }
                    }

                    // Write a row to the CSV
                    writer.WriteLine($"{pairNumber},{item1},{item2},{winner}");
                }
            }
        }
    }
}