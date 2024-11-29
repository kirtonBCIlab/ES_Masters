using System.Collections;
using UnityEngine;
using System;
using BCIEssentials.Controllers;
using BCIEssentials.StimulusEffects;
using UnityEngine.UI;
using System.Collections.Generic;

namespace BCIEssentials.ControllerBehaviors
{
    public class ComfortControllerBehavior : BCIControllerBehavior
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
        private bool _offMessages;
        private bool _restingState;
        private bool _open;
        private bool _closed;
        private string stimulusString = "";
        private Dictionary<int, string> orderDict = new Dictionary<int, string>();


        protected override void Start()
        {
            base.Start();
            
            mainCam = Camera.main;
            mainCam.enabled = true;
        
            _displayText = GameObject.Find("TextToDisplay").GetComponent<Text>();

            //randomize order of stimulus presentation 
            Randomize();

            //set first frequency
            setFreqFlash = 10;
            PopulateObjectList();
            RunStimulus();
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

        protected override IEnumerator SendMarkers(int trainingIndex = 99)
        {
            while (StimulusRunning)
            {
                string freqString = "";
                string markerString=  "";
                string trainingString;
                
                if(!_offMessages)
                {
                    freqString = freqString + "," + realFreqFlash.ToString();
                    trainingString = (trainingIndex <= _selectableSPOs.Count) ? trainingIndex.ToString() : "-1";
                    
                    markerString = "tvep," + _selectableSPOs.Count.ToString() + "," + trainingString + "," +
                                        windowLength.ToString() + freqString + stimulusString;
                }

                if(_offMessages)
                {
                    markerString = "Stimulus Off";
                }

                if(_restingState && _open)
                {
                    markerString = "Resting state, eyes open";
                }
                if(_restingState && _closed)
                {
                    markerString = "Resting state, eyes closed";
                }

                marker.Write(markerString);

                yield return new WaitForSecondsRealtime(windowLength + interWindowInterval);     
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
            //setup variables for camera rotation 
            var _rotateAway = Vector3.zero;
            _rotateAway.y = 90f;

            var _rotateBack = Vector3.zero;
            _rotateBack.y = -90f;
            
            //5 seconds count down before starting
            _offMessages = true;                    
            mainCam.transform.Rotate(_rotateBack);
            StartCoroutine(DisplayTextOnScreen("5"));
            yield return new WaitForSecondsRealtime(5f);
            mainCam.transform.Rotate(_rotateAway);
            _offMessages = false;

            for (var timesShown = 0; timesShown < 2; timesShown++)
            {                
                //set initial color and contrast
                ColorFlashEffect3 spoEffect = _selectableSPOs[0].GetComponent<ColorFlashEffect3>();
                SetMaterial(0);
                stimulusString = ", "  + orderDict[0];

                for(var l = 0 ; l < 12; l++)
                {
                    for(var i = 0; i <100*10; i++) //(StimulusRunning)
                    //the number that i is less than is the amount of seconds to flash for 
                    //100 = 1 second (frame rate is 144 Hz) so 12 seconds = i < 144*12
                    {
                        yield return OnStimulusRunBehavior();
                    }

                    //rotate the camera away from the stimuli objects when they are off
                    mainCam.transform.Rotate(_rotateAway);
                    _offMessages = true;

                    if(!(l == 11 && timesShown == 1))
                    {
                        yield return new WaitForSecondsRealtime(2f);
                        StartCoroutine(DisplayTextOnScreen("3"));
                        yield return new WaitForSecondsRealtime(3f); 
                    }

                    SetMaterial(l+1);
                    
                    //rotate the camera back to facing the stimulus objects 
                    mainCam.transform.Rotate(_rotateBack);
                    _offMessages = false;
                }
                //cLear out the dict and re-randomize stmulus order to show them again
                orderDict.Clear();
                Randomize();
            }

            mainCam.transform.Rotate(_rotateAway);
            StartCoroutine(DisplayTextOnScreen("EndOfSession"));
            StopCoroutineReference(ref _runStimulus);
            
            StopCoroutineReference(ref _runStimulus);
            StopCoroutineReference(ref _sendMarkers);
        }



//////Helper Methods
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
            else if(textOption == "End")
            {
                _displayText. text = "Look at the plus sign";
                yield return new WaitForSecondsRealtime(2.0f);
                _displayText.text = "";
            }
            else if(textOption == "EndOfSession")
            {
                _displayText. text = "End";
                yield return new WaitForSecondsRealtime(2.0f);
            }
            else if(textOption == "Survey")
            {
                _displayText.text = "Survey";
                yield return new WaitForSecondsRealtime(5.0f);
                _displayText.text = "";
            }
            else if(textOption == "+")
            {
                _displayText.text = "+";
                yield return new WaitForSecondsRealtime(60.0f);
                _displayText.text = "";
            }
        } 

        private void SetMaterial(int key)
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

    

