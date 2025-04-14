using System.Collections;
using UnityEngine;
using System;
using BCIEssentials.Controllers;
using BCIEssentials.StimulusEffects;
using UnityEngine.UI;
using System.Collections.Generic;
using System.IO;

namespace BCIEssentials.ControllerBehaviors
{
    public class ComfortControllerBehavior2 : BCIControllerBehavior
    {
        public override BCIBehaviorType BehaviorType => BCIBehaviorType.TVEP;

        [SerializeField] private float setFreqFlash;
        [SerializeField] private float realFreqFlash;

        private int[] frames_on = new int[99];
        private int[] frame_count = new int[99];
        private float period;
        private int[] frame_off_count = new int[99];
        private int[] frame_on_count = new int[99];

        public Camera mainCam;
        public Text _displayMarker1;
        public Text _displayMarker2;
        public Text _displayText;
        private string stimulusString = "";
        private string markerString = "";
        private Dictionary<int, string> orderDict = new Dictionary<int, string>();


        // Variables for presenting 2 at a time
        private DoubleEliminationBracket bracket;
        private SpriteRenderer stim1;
        private SpriteRenderer stim2;
        private GameObject stim1Object;
        private GameObject stim2Object;
        private bool? preference;
        private int pairNum;

        public class StimulusData
        {
            // Dictionary to hold stimulus indices and their corresponding names
            public Dictionary<int, string> StimulusIndex { get; private set; }

            // List of pairs for comparisons, with a pair number as a string
            public List<(string pairNumber, string item1, string item2)> ComparisonPairs { get; private set; }

            // List to hold winners
            public List<int> Winners { get; private set; }

            // Constructor to initialize the data structure
            public StimulusData()
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

        public StimulusData bracketInfo = new StimulusData();

        protected override void Start()
        {
            base.Start();
            
            mainCam = Camera.main;
            mainCam.enabled = true;
        
            _displayMarker1 = GameObject.Find("Marker1").GetComponent<Text>();
            _displayMarker2 = GameObject.Find("Marker2").GetComponent<Text>();
            _displayText = GameObject.Find("Text").GetComponent<Text>();

            // randomize order of stimulus presentation 
            Randomize();
            // populate bracket with randomized stimuli
            bracket = new DoubleEliminationBracket(orderDict);  

            //set first frequency
            setFreqFlash = 10;

            PopulateObjectList();
            RunStimulus();
        }

        private void GetSPOs()
        {
            stim1 = _selectableSPOs[0].GetComponent<SpriteRenderer>();
            stim1Object = GameObject.Find("Object 1");

            stim2 = _selectableSPOs[1].GetComponent<SpriteRenderer>();
            stim2Object = GameObject.Find("Object 2");
        }

        public override void PopulateObjectList(SpoPopulationMethod populationMethod = SpoPopulationMethod.Tag)
        {
            base.PopulateObjectList(populationMethod);
            for (int i = 0; i < _selectableSPOs.Count; i++)
            {
                frames_on[i] = 0;
                frame_count[i] = 0;
                period = targetFrameRate / setFreqFlash;
                frame_off_count[i] = (int)Math.Ceiling(period / 2);
                frame_on_count[i] = (int)Math.Floor(period / 2);
                realFreqFlash = targetFrameRate / (float)(frame_off_count[i] + frame_on_count[i]);
            }
        }

        protected override IEnumerator OnStimulusRunBehavior()
        {
            for (int i = 0; i < _selectableSPOs.Count; i++)
            {
                frame_count[i]++;
                if (frames_on[i] == 1)
                {
                    if (frame_count[i] >= frame_on_count[i])
                    {
                        _selectableSPOs[i].StopStimulus();
                        frames_on[i] = 0;
                        frame_count[i] = 0;
                    }
                }
                else
                {
                    if (frame_count[i] >= frame_off_count[i])
                    {
                        _selectableSPOs[i].StartStimulus();
                        frames_on[i] = 1;
                        frame_count[i] = 0;
                    }
                }
            } 
            yield return null;
        }

        protected override IEnumerator OnStimulusRunComplete()
        {
            foreach (var spo in _selectableSPOs)
            {
                if (spo != null)
                {
                    spo.StopStimulus();
                }
            }
            yield return null;
        }
            
        protected override IEnumerator RunStimulus()
        {
            // Assign SPO sprite renderers and gameobjects to variables so we can turn them on/off and move them
            GetSPOs();

            // Camera setup for transitions
            var rotateAway = Vector3.zero;
            rotateAway.y = 90f;

            var rotateBack = Vector3.zero;
            rotateBack.y = -90f;

            pairNum = 1;

            StartCoroutine(DisplayTextOnScreen("RS Open"));
            marker.Write("Resting State, Eyes Open");
            yield return new WaitForSecondsRealtime(60.0f);


            StartCoroutine(DisplayTextOnScreen("RS Closed"));
            marker.Write("Resting State, Eyes Closed");
            yield return new WaitForSecondsRealtime(60.0f);

            // Loop through the double elimination bracket
            while (!bracket.IsComplete())
            {
                //marker.Write("Pair number" + pairNum);
                // get the next pair in the bracket
                var currentPair = bracket.GetCurrentMatch(); 
                if (currentPair == null) break;

                // Extract stimuli indices
                int stim1Index = currentPair.Stimulus1;
                int stim2Index = currentPair.Stimulus2 ?? -1;

                // Get names from orderDict
                string stim1Name = orderDict[stim1Index];
                string stim2Name = orderDict[stim2Index];

                bracketInfo.AddComparisonPair(pairNum, stim1Name, stim2Name);

                // Set the materials for each stimulus
                SetMaterialStim1(stim1Index);
                SetMaterialStim2(stim2Index);

                // Turn off stimulus 2 for now & move stim 1 to center of screen
                stim2.enabled = false;
                stim1Object.transform.position = new Vector3(1, 0, 0);
        
                // Reset + marker positions
                RectTransform marker1Rect = _displayMarker1.GetComponent<RectTransform>();
                RectTransform marker2Rect = _displayMarker2.GetComponent<RectTransform>();
            
                // Present Stimulus 1
                StartCoroutine(DisplayTextOnScreen("5")); // 5-second countdown
                yield return new WaitForSecondsRealtime(5f);

                stimulusString = ", "  + stim1Name;
                markerString = "ssvep," + _selectableSPOs.Count.ToString() + "," + windowLength.ToString() + "," + realFreqFlash.ToString() + stimulusString;
                marker.Write(markerString);


                for(var flash = 0; flash <100*10; flash++) //(StimulusRunning)
                //the number that flash is less than is the amount of seconds to flash for 
                //100 = 1 second (frame rate is 100 Hz) so 10 seconds = flash < 100*10s
                {
                    ScalePlusSignToStimulus(stim1Object, false);
                    StartCoroutine(DisplayTextOnScreen("+1"));
                    yield return OnStimulusRunBehavior();
                }

                marker.Write("stimulus ended");

                // Turn off stimulus 1 and turn on stimulus 2 and move stim2 to center of screen
                stim1.enabled = false;
                stim2.enabled = true;
                stim2Object.transform.position = new Vector3(1, 0, 0);


                mainCam.transform.Rotate(rotateAway);
                StartCoroutine(DisplayTextOnScreen("Comfort"));
                yield return new WaitForSecondsRealtime(7f);
                StartCoroutine(DisplayTextOnScreen("3"));
                yield return new WaitForSecondsRealtime(3f); 

                marker.Write("baseline ended");

                // Present Stimulus 2
                mainCam.transform.Rotate(rotateBack);
                stimulusString = ", "  + stim2Name;
                markerString = "ssvep," + _selectableSPOs.Count.ToString() + "," + windowLength.ToString() + "," + realFreqFlash.ToString() + stimulusString;
                marker.Write(markerString);

                for(var flash = 0; flash <100*10; flash++) //(StimulusRunning)
                //the number that flash is less than is the amount of seconds to flash for 
                //100 = 1 second (frame rate is 100 Hz) so 10 seconds = flash < 100*10s
                {
                    ScalePlusSignToStimulus(stim2Object, false);
                    StartCoroutine(DisplayTextOnScreen("+2"));
                    yield return OnStimulusRunBehavior();
                }

                marker.Write("stimulus ended");

                mainCam.transform.Rotate(rotateAway);
                StartCoroutine(DisplayTextOnScreen("Comfort"));
                yield return new WaitForSecondsRealtime(7f);
                StartCoroutine(DisplayTextOnScreen("3"));
                yield return new WaitForSecondsRealtime(3f); 

                marker.Write("baseline ended");

                //Turn both stimuli on and move stimuli to either side of the screen
                stim1.enabled = true;
                stim1Object.transform.position = new Vector3(0, 0, 0);
                stim2Object.transform.position = new Vector3(2, 0, 0);

                mainCam.transform.Rotate(rotateBack);
                stimulusString = ", Stim1: " + stim1Name + ", Stim2: " + stim2Name;
                markerString = "ssvep," + _selectableSPOs.Count.ToString() + "," + windowLength.ToString() + "," + realFreqFlash.ToString() + stimulusString;
                marker.Write(markerString);

                for(var flash = 0; flash <100*10; flash++) //(StimulusRunning)
                //the number that flash is less than is the amount of seconds to flash for 
                //100 = 1 second (frame rate is 100 Hz) so 10 seconds = flash < 100*10s
                {
                    ScalePlusSignToStimulus(stim1Object, true);
                    ScalePlusSignToStimulus(stim2Object, true);
                    StartCoroutine(DisplayTextOnScreen("+1"));
                    StartCoroutine(DisplayTextOnScreen("+2"));
                    yield return OnStimulusRunBehavior();
                }
                
                marker.Write("off");

                StartCoroutine(GetUserPreferenceCoroutine());
                yield return new WaitUntil(() => preference != null);

                // Record the winner in the bracket: stim1 = 'true' recorded, Stim2 = 'false' recorded
                if (preference.HasValue)
                {
                    int winnerIndex = preference.Value ? stim1Index : stim2Index;
                    bracketInfo.AddWinner(winnerIndex);
                    bracket.RecordMatchResult(preference.Value ? stim1Index : stim2Index);
                    StartCoroutine(DisplayTextOnScreen("Break"));
                    yield return new WaitForSecondsRealtime(10.0f); //will be combined with a 5 second countdown for a total 15 second break
                }
                else
                {
                    Debug.Log("No preference chosen");
                }

                // Reset preference to null so the value doesn't carry over to the next pair
                preference = null;
                pairNum = pairNum + 1;
            }
                
            Debug.Log($"Winner overall: {bracket.GetWinner()}");

            // Finalize
            mainCam.transform.Rotate(rotateAway);
            StartCoroutine(DisplayTextOnScreen("EndOfSession"));
            StopCoroutineReference(ref _runStimulus);
            StopCoroutineReference(ref _sendMarkers);

            string filepath = "D:\\Users\\BCI-Morpheus\\Documents\\ES-Masters\\Data\\Bracket\\P001-S001.csv";
            bracketInfo.ExportToCsv(filepath);
        }

        private IEnumerator GetUserPreferenceCoroutine()
        {
            bool preferenceCaptured = false;
            //marker write "start pref collection
            // Wait for user input
            while (!preferenceCaptured)
            {
                StartCoroutine(DisplayTextOnScreen("Choose"));
                if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1))
                {
                    preference = true; // Stimulus 1 selected
                    preferenceCaptured = true;
                    //marker write chose 1
                    //Debug.Log("Stimulus 1 selected successfully in the controller");
                }
                else if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2))
                {
                    preference = false; // Stimulus 2 selected
                    preferenceCaptured = true;
                    //marker write chose 2
                    //Debug.Log("Stimulus 2 selected successfully in the controller");
                }

                // Yield until the next frame to prevent freezing
                yield return null;
            }
        }

        //Helper Methods
        private void ScalePlusSignToStimulus(GameObject stimulus, bool bothDisplayed)
        {
            if (stimulus != null && _displayMarker1 != null && _displayMarker2 != null)
            {
                var currentPair = bracket.GetCurrentMatch(); 

                // Extract stimuli indices
                int stim1Index = currentPair.Stimulus1;
                int stim2Index = currentPair.Stimulus2 ?? -1;

                // Get names from orderDict
                string stim1Name = orderDict[stim1Index];
                string stim2Name = orderDict[stim2Index];

                RectTransform marker1Rect = _displayMarker1.GetComponent<RectTransform>();
                RectTransform marker2Rect = _displayMarker2.GetComponent<RectTransform>();

                if(bothDisplayed)
                {                    
                    marker1Rect.localPosition = new Vector3(-340f,20f,0f);
                    marker2Rect.localPosition = new Vector3(340f,20f,0f);
                }
                else
                {
                    if (stim1Name.Contains("Size1"))
                    {
                        _displayMarker1.fontSize = 50;
                        marker1Rect.localPosition = new Vector3(0f,20f,0f);
                    }
                    else if (stim1Name.Contains("Size2"))
                    {
                        _displayMarker1.fontSize = 50;
                        marker1Rect.localPosition = new Vector3(0f,20f,0f);
                    }
                    else
                    {
                        _displayMarker1.fontSize = 50;
                        marker1Rect.localPosition = new Vector3(0f,20f,0f);
                    }

                    if (stim2Name.Contains("Size1"))
                    {
                        _displayMarker2.fontSize = 50;
                        marker2Rect.localPosition = new Vector3(0f,20f,0f);
                    }
                    else if (stim2Name.Contains("Size2"))
                    {
                        _displayMarker2.fontSize = 50;
                        marker2Rect.localPosition = new Vector3(0f,20f,0f);
                    }
                    else
                    {
                        _displayMarker2.fontSize = 50;
                        marker2Rect.localPosition = new Vector3(0f,20f,0f);
                    } 
                }
            }
        }

        public IEnumerator DisplayTextOnScreen(string textOption)
        {
            if(textOption == "3")
            {
                _displayText.text = "3";
                yield return new WaitForSecondsRealtime(1.0f);
                _displayText.text = "2";
                yield return new WaitForSecondsRealtime(1.0f);
                _displayText.text = "1";
                yield return new WaitForSecondsRealtime(1.0f);
                _displayText.text = "";
            }
            else if(textOption == "5")
            {
                _displayText.text = "Starting in...";
                yield return new WaitForSecondsRealtime(2.0f);
                _displayText.text = "3 seconds";
                yield return new WaitForSecondsRealtime(1.0f);
                _displayText.text = "2 seconds";
                yield return new WaitForSecondsRealtime(1.0f);
                _displayText.text = "1 second";
                yield return new WaitForSecondsRealtime(1.0f);
               _displayText.text = "";
            }
            else if(textOption == "EndOfSession")
            {
                _displayText.text = "End";
                yield return new WaitForSecondsRealtime(2.0f);
            }
            else if(textOption == "+1")
            {
                _displayMarker1.text = "+";
                yield return new WaitForSecondsRealtime(1.0f);
                _displayMarker1.text = "";
            }
            else if(textOption == "+2")
            {
                _displayMarker2.text = "+";
                yield return new WaitForSecondsRealtime(1.0f);
                _displayMarker2.text = "";
            }
            else if (textOption == "Choose")
            {
                _displayText.text = "Press 1 or 2";
                yield return new WaitForSecondsRealtime(1.0f);
            }
            else if (textOption == "Comfort")
            {
                _displayText.text = "Rate Comfort 1 - 7";
                yield return new WaitForSecondsRealtime(5.0f);
                _displayText.text = "";
            }
            else if (textOption == "Break")
            {
                _displayText.text = "Break";
                yield return new WaitForSecondsRealtime(10.0f);
                _displayText.text = "";
            }
            else if (textOption == "RS Open")
            {
                _displayText.text = "+";
                yield return new WaitForSecondsRealtime(60.0f);
                _displayText.text = "";
            }
            else if (textOption == "RS Closed")
            {
                _displayText.text = "Eyes Closed";
                yield return new WaitForSecondsRealtime(60.0f);
                _displayText.text = "";
            }
        } 

        private void SetMaterialStim1(int key)
        {
            ColorFlashEffect3 spoEffect = _selectableSPOs[0].GetComponent<ColorFlashEffect3>();
            if (orderDict.TryGetValue(key, out string material))
            {       
                if (material == "Contrast1Size1")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast1);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size1);
                    stimulusString = ", Contrast1 Size1";
                }
                else if (material == "Contrast1Size2")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast1);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size2);
                    stimulusString = ", Contrast1 Size2";
                }
                else if (material == "Contrast1Size3")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast1);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                    stimulusString = ", Contrast1 Size3";
                }
                else if (material == "Contrast2Size1")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast2);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size1);
                    stimulusString = ", Contrast2 Size1";
                }
                else if (material == "Contrast2Size2")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast2);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size2);
                    stimulusString = ", Contrast2 Size2";
                }
                else if (material == "Contrast2Size3")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast2);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                    stimulusString = ", Contrast2 Size3";
                }
                else if (material == "Contrast3Size1")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast3);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size1);
                    stimulusString = ", Contrast3 Size1";
                }
                else if (material == "Contrast3Size2")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast3);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size2);
                    stimulusString = ", Contrast3 Size2";
                }
                else if (material == "Contrast3Size3")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast3);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                    stimulusString = ", Contrast3 Size3";
                }
                else if (material == "Contrast4Size1")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast4);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size1);
                    stimulusString = ", Contrast4 Size1";
                }
                else if (material == "Contrast4Size2")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast4);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size2);
                    stimulusString = ", Contrast4 Size2";
                }
                else if (material == "Contrast4Size3")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast4);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                    stimulusString = ", Contrast4 Size3";
                }
            }
        }

        private void SetMaterialStim2(int key)
        {
            ColorFlashEffect3 spoEffect = _selectableSPOs[1].GetComponent<ColorFlashEffect3>();
            if (orderDict.TryGetValue(key, out string material))
            {       
                if (material == "Contrast1Size1")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast1);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size1);
                    stimulusString = ", Contrast1 Size1";
                }
                else if (material == "Contrast1Size2")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast1);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size2);
                    stimulusString = ", Contrast1 Size2";
                }
                else if (material == "Contrast1Size3")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast1);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                    stimulusString = ", Contrast1 Size3";
                }
                else if (material == "Contrast2Size1")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast2);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size1);
                    stimulusString = ", Contrast2 Size1";
                }
                else if (material == "Contrast2Size2")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast2);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size2);
                    stimulusString = ", Contrast2 Size2";
                }
                else if (material == "Contrast2Size3")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast2);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                    stimulusString = ", Contrast2 Size3";
                }
                else if (material == "Contrast3Size1")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast3);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size1);
                    stimulusString = ", Contrast3 Size1";
                }
                else if (material == "Contrast3Size2")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast3);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size2);
                    stimulusString = ", Contrast3 Size2";
                }
                else if (material == "Contrast3Size3")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast3);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                    stimulusString = ", Contrast3 Size3";
                }
                else if (material == "Contrast4Size1")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast4);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size1);
                    stimulusString = ", Contrast4 Size1";
                }
                else if (material == "Contrast4Size2")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast4);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size2);
                    stimulusString = ", Contrast4 Size2";
                }
                else if (material == "Contrast4Size3")
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast4);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                    stimulusString = ", Contrast4 Size3";
                }
            }
        }

        private void Randomize()
        {
            orderDict.Add(0, "Contrast1Size1");
            orderDict.Add(1, "Contrast1Size2");
            orderDict.Add(2, "Contrast1Size3");
            orderDict.Add(3, "Contrast2Size1");
            orderDict.Add(4, "Contrast2Size2");
            orderDict.Add(5, "Contrast2Size3");
            orderDict.Add(6, "Contrast3Size1");
            orderDict.Add(7, "Contrast3Size2");
            orderDict.Add(8, "Contrast3Size3");
            orderDict.Add(9, "Contrast4Size1");
            orderDict.Add(10, "Contrast4Size2");
            orderDict.Add(11, "Contrast4Size3");  

            System.Random random = new System.Random();
            List<int> keys = new List<int>(orderDict.Keys);
            int num = keys.Count;

            while (num > 1)
            {
                num--;
                int k = random.Next(num + 1);
                int temp = keys[k];
                keys[k] = keys[num];
                keys[num] = temp;
            }

            List<string> values = new List<string>(orderDict.Values);        
            int n = values.Count;
         
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);
                string temp = values[k];
                values[k] = values[n];
                values[n] = temp;
            }

            Dictionary<int, string> intDict = new Dictionary<int, string>();
                    
            for (int i = 0; i < keys.Count; i++)
            {
                intDict.Add(keys[i], values[i]);
            }

            List<KeyValuePair<int, string>> keyValuePairs = new List<KeyValuePair<int, string>>(intDict);
            int c = keyValuePairs.Count;
                
            while (c > 1)
            {
                c--;
                int k = random.Next(c + 1);
                KeyValuePair<int, string> temp = keyValuePairs[k];
                keyValuePairs[k] = keyValuePairs[c];
                keyValuePairs[c] = temp;
            }

            Dictionary<int, string> randomDict = new Dictionary<int, string>();

            foreach (var k in keyValuePairs)
                randomDict.Add(k.Key, k.Value);

            orderDict = new Dictionary<int, string>(randomDict);    
            bracketInfo.SetStimulusIndex(orderDict);
        }    
    }
}


    

