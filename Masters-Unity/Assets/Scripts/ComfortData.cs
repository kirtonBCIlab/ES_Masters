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

        // Export all individual scores per stimulus with mean (1 row per stimulus)
        public List<List<string>> GetFullComfortDataForExport()
        {
            List<List<string>> rows = new List<List<string>>();

            int maxEpochs = StimulusRatings.Values.Max(list => list.Count);
            List<string> header = new List<string> { "Stimulus" };
            for (int i = 0; i < maxEpochs; i++)
                header.Add($"Epoch {i + 1}");
            rows.Add(header);

            foreach (var kvp in StimulusRatings)
            {
                List<string> row = new List<string> { kvp.Key };
                var scores = kvp.Value;

                for (int i = 0; i < maxEpochs; i++)
                {
                    if (i < scores.Count)
                        row.Add(scores[i].ToString());
                    else
                        row.Add("");
                }

                rows.Add(row);
            }

            return rows;
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

        // Write all individual scores to CSV
        public void ExportFullScoresToCsv(string filePath)
        {
            var exportData = GetFullComfortDataForExport();
            using (var writer = new StreamWriter(filePath))
            {
                foreach (var row in exportData)
                {
                    writer.WriteLine(string.Join(",", row));
                }
            }
        }
    }
}