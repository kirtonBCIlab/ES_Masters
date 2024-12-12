using System.Collections;
using UnityEngine;
using System;
using BCIEssentials.Controllers;
using BCIEssentials.StimulusEffects;
using UnityEngine.UI;
using System.Collections.Generic;

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
        public Text _displayText;
        private string stimulusString = "";
        private string markerString = "";
        private Dictionary<int, string> orderDict = new Dictionary<int, string>();

        private DoubleEliminationBracket bracket;

        private SpriteRenderer stim1;
        private SpriteRenderer stim2;

        private GameObject stim1Object;
        private GameObject stim2Object;

        protected override void Start()
        {
            base.Start();
            
            mainCam = Camera.main;
            mainCam.enabled = true;
        
            _displayText = GameObject.Find("TextToDisplay").GetComponent<Text>();

            //randomize order of stimulus presentation 
            Randomize();
            bracket = new DoubleEliminationBracket(orderDict);  

            //set first frequency
            setFreqFlash = 10;
            PopulateObjectList();
            RunStimulus();
        }

        private void GetSPOs()
        {
            marker.Write("inside getSPOs");
            stim1 = _selectableSPOs[0].GetComponent<SpriteRenderer>();
            stim1Object = GameObject.Find("Object 1");

            if (stim1Object != null)
            {
                marker.Write("object 1 found");
            }

            stim2 = _selectableSPOs[1].GetComponent<SpriteRenderer>();
            stim2Object = GameObject.Find("Object 2");
            if (stim2Object != null)
            {
                 marker.Write("object 2 found");
            }
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
            //Assign SPO sprite renderers to variables so we can turn them on and off
            GetSPOs();

            // Camera setup for transitions
            var rotateAway = Vector3.zero;
            rotateAway.y = 90f;

            var rotateBack = Vector3.zero;
            rotateBack.y = -90f;

            marker.Write("RunStimulus started");
            mainCam.transform.Rotate(rotateBack);
            StartCoroutine(DisplayTextOnScreen("5"));
            yield return new WaitForSecondsRealtime(5f);
            mainCam.transform.Rotate(rotateAway);

            // Loop through the double elimination bracket
            while (!bracket.IsComplete())
            {
                var currentPair = bracket.GetCurrentMatch(); // Get the next stimulus pair
                if (currentPair == null) break;

                // Extract stimuli indices
                int stim1Index = currentPair.Stimulus1;
                int stim2Index = currentPair.Stimulus2 ?? -1;


                // Get names from orderDict
                string stim1Name = orderDict[stim1Index];
                string stim2Name = orderDict[stim2Index];

                // Set the materials for each stimulus
                SetMaterialStim1(stim1Index);
                SetMaterialStim2(stim2Index);

                //Turn off stimulus 2 for now
                stim2.enabled = false;

                // Move stim 1 to center of screen
                stim1Object.transform.position = new Vector3(1, 0, 0);
        
                // Present Stimulus 1
                StartCoroutine(DisplayTextOnScreen("5")); // 5-second countdown

                stimulusString = ", "  + stim1Name;
                markerString = "ssvep," + _selectableSPOs.Count.ToString() + "," + windowLength.ToString() + "," + realFreqFlash.ToString() + stimulusString;
                marker.Write(markerString);


                for(var flash = 0; flash <100*10; flash++) //(StimulusRunning)
                //the number that flash is less than is the amount of seconds to flash for 
                //100 = 1 second (frame rate is 100 Hz) so 10 seconds = flash < 100*10s
                {
                    StartCoroutine(DisplayTextOnScreen("+"));
                    yield return OnStimulusRunBehavior();
                }

                marker.Write("off");

                // Turn off stimulus 1 and turn on stimulus 2
                stim1.enabled = false;
                stim2.enabled = true;

                mainCam.transform.Rotate(rotateAway);
                yield return new WaitForSecondsRealtime(2f);
                StartCoroutine(DisplayTextOnScreen("3"));
                yield return new WaitForSecondsRealtime(3f); 

                //Move stim2 to middle of screen
                stim2Object.transform.position = new Vector3(1, 0, 0);


                // Present Stimulus 2
                mainCam.transform.Rotate(rotateBack);
                stimulusString = ", "  + stim2Name;
                markerString = "ssvep," + _selectableSPOs.Count.ToString() + "," + windowLength.ToString() + "," + realFreqFlash.ToString() + stimulusString;
                marker.Write(markerString);

                for(var flash = 0; flash <100*10; flash++) //(StimulusRunning)
                //the number that flash is less than is the amount of seconds to flash for 
                //100 = 1 second (frame rate is 100 Hz) so 10 seconds = flash < 100*10s
                {
                    StartCoroutine(DisplayTextOnScreen("+"));
                    yield return OnStimulusRunBehavior();
                }

                marker.Write("off");

                mainCam.transform.Rotate(rotateAway);
                yield return new WaitForSecondsRealtime(2f);
                StartCoroutine(DisplayTextOnScreen("3"));
                yield return new WaitForSecondsRealtime(3f); 

                //Turn both stimuli on
                stim1.enabled = true;

                //Move stimuli to either side of the screen
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
                    StartCoroutine(DisplayTextOnScreen("+"));
                    yield return OnStimulusRunBehavior();
                }
                
                marker.Write("off");

                marker.Write("Pair 1 done, collect input");

                // Capture preference with keypress input
                StartCoroutine(DisplayTextOnScreen("Choose"));

                StartCoroutine(GetUserPreferenceCoroutine());
                yield return new WaitUntil(() => preference != null);

                // Record the winner in the bracket
                // Stim1 = 'true' recorded, Stim2 = 'false' recorded
                bracket.RecordMatchResult(preference ? stim1Index : stim2Index);


                // Pause before next match
                yield return new WaitForSecondsRealtime(15f);
            }

                // Finalize
                mainCam.transform.Rotate(rotateAway);
                StartCoroutine(DisplayTextOnScreen("EndOfSession"));
                StopCoroutineReference(ref _runStimulus);
                StopCoroutineReference(ref _sendMarkers);
        }
    
        private bool preference = false; // Store the user's preference

        private IEnumerator GetUserPreferenceCoroutine()
        {
            bool preferenceCaptured = false;

            // Wait for user input
            while (!preferenceCaptured)
            {
                if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1))
                {
                    preference = true; // Stimulus 1 selected
                    preferenceCaptured = true;
                    Debug.Log("Stimulus 1 selected successfully in the controller");
                }
                else if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2))
                {
                    preference = false; // Stimulus 2 selected
                    preferenceCaptured = true;
                    Debug.Log("Stimulus 2 selected successfully in the controller");
                }

                // Yield until the next frame to prevent freezing
                yield return null;
            }
        }



        //Helper Methods
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
            else if(textOption == "+")
            {
                _displayText.text = "+";
                yield return new WaitForSecondsRealtime(1.0f);
                _displayText.text = "";
            }
            else if (textOption == "Choose")
            {
                _displayText.text = "Press 1 or 2";
                yield return new WaitForSecondsRealtime(5.0f);
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
            }    

          

 }
}


    

