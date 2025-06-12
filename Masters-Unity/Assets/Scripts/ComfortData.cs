using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace BCIEssentials.Utilities
{
    public class ComfortData
    {
        public Dictionary<string, List<int>> StimulusRatings { get; private set; }

        public ComfortData()
        {
            StimulusRatings = new Dictionary<string, List<int>>();
        }

        public void SetStimulusNames(Dictionary<int, string> stimulusDictionary)
        {
            StimulusRatings.Clear(); // Clear existing ratings

            foreach (var kvp in stimulusDictionary)
            {
                if (!StimulusRatings.ContainsKey(kvp.Value))
                {
                    StimulusRatings[kvp.Value] = new List<int>();
                }
            }
        }

        public void AddScore(string stimulusName, int score)
        {
            if (!StimulusRatings.ContainsKey(stimulusName))
            {
                Debug.LogWarning($"Stimulus name '{stimulusName}' not found in StimulusRatings.");
                return;
            }

            StimulusRatings[stimulusName].Add(score);
        }

        public double GetMeanComfort(string stimulusId)
        {
            if (StimulusRatings.ContainsKey(stimulusId))
            {
                var scores = StimulusRatings[stimulusId];
                return scores.Count > 0 ? scores.Average() : 0.0;
            }
            else
            {
                Debug.LogWarning($"Stimulus ID '{stimulusId}' not found in StimulusRatings.");
                return -1000000;
            }
        }

        // Export only mean comfort values (2-row format: headers and means)
        public List<List<string>> GetComfortDataForExport()
        {
            List<string> headerRow = new List<string>();
            List<string> meanRow = new List<string>();

            foreach (var stimulus in StimulusRatings.Keys)
            {
                headerRow.Add(stimulus);

                double mean = GetMeanComfort(stimulus);
                meanRow.Add(mean.ToString("F4"));
            }

            return new List<List<string>> { headerRow, meanRow };
        }

        // Write means to CSV
        public void ExportMeansToCsv(string filePath)
        {
            var exportData = GetComfortDataForExport();
            using (var writer = new StreamWriter(filePath))
            {
                foreach (var row in exportData)
                {
                    writer.WriteLine(string.Join(",", row));
                }
            }
        }

        public void ExportScoresInLongFormat(string filePath)
        {
            using (var writer = new StreamWriter(filePath))
            {
                writer.WriteLine("Contrast,Size,Epoch,Comfort_Score");

                foreach (var kvp in StimulusRatings)
                {
                    string stimulus = kvp.Key;
                    List<int> scores = kvp.Value;

                    // Parse Contrast and Size from the stimulus string
                    var contrastMatch = System.Text.RegularExpressions.Regex.Match(stimulus, @"Contrast(\d+)");
                    var sizeMatch = System.Text.RegularExpressions.Regex.Match(stimulus, @"Size(\d+)");

                    if (!contrastMatch.Success || !sizeMatch.Success)
                    {
                        Debug.LogWarning($"Invalid stimulus format: {stimulus}");
                        continue;
                    }

                    int contrast = int.Parse(contrastMatch.Groups[1].Value);
                    int size = int.Parse(sizeMatch.Groups[1].Value);

                    for (int epoch = 0; epoch < scores.Count; epoch++)
                    {
                        int score = scores[epoch];
                        writer.WriteLine($"{contrast},{size},{epoch + 1},{score}");
                    }
                }
            }
        }
    }
}