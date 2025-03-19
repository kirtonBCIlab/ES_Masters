using System.Collections;
using UnityEngine;
using System;
using BCIEssentials.Controllers;
using BCIEssentials.StimulusEffects;
using UnityEngine.UI;
using System.Collections.Generic;

namespace BCIEssentials.ControllerBehaviors
{
    public class SSVEPControllerBehavior : BCIControllerBehavior_variant
    {
        public override BCIBehaviorType BehaviorType => BCIBehaviorType.SSVEP;

        [SerializeField] private float[] setFreqFlash;
        [SerializeField] private float[] realFreqFlash;

        private int[] frames_on = new int[99];
        private int[] frame_count = new int[99];
        private float period;
        private int[] frame_off_count = new int[99];
        private int[] frame_on_count = new int[99];

        public enum StimulusType
        {
            BW,
            Custom
        }
        public enum ContrastLevel
        {
            Contrast1,
            Contrast2,
            Contrast3,
            Contrast4,            
        }
        public enum Size
        {
            Size1,
            Size2,
            Size3,
        }

        [Header("Stimulus Parameters")]
        [SerializeField] public StimulusType _stimulusType;
        public ContrastLevel _contrastLevel;
        public Size _size;

        [Header("Text Object")]
        [SerializeField] public Text _displayText;

        // Start is called before the first frame update
        protected override void Start()
        {
            base.Start();
            
            PopulateObjectList();
            RunStimulus();
        }

       public override void PopulateObjectList(SpoPopulationMethod populationMethod = SpoPopulationMethod.Tag)
        {
            base.PopulateObjectList(populationMethod);

            realFreqFlash = new float[_selectableSPOs.Count];

            for (int i = 0; i < _selectableSPOs.Count; i++)
            {
                frames_on[i] = 0;
                frame_count[i] = 0;
                period = targetFrameRate / setFreqFlash[i];
                frame_off_count[i] = (int)Math.Ceiling(period / 2);
                frame_on_count[i] = (int)Math.Floor(period / 2);
                realFreqFlash[i] = (targetFrameRate / (float)(frame_off_count[i] + frame_on_count[i]));

                Debug.Log($"frequency {i + 1} : {realFreqFlash[i]}");
            }
        }

        protected override IEnumerator SendMarkers(int trainingIndex = 99)
        {
            // Make the marker string, this will change based on the paradigm
            while (StimulusRunning)
            {
                // Desired format is: ["ssvep", number of options, training target (-1 if n/a), window length, frequencies]
                string freqString = "";
                for (int i = 0; i < realFreqFlash.Length; i++)
                {
                    freqString = freqString + "," + realFreqFlash[i].ToString();
                }

                string trainingString;
                if (trainingIndex <= _selectableSPOs.Count)
                {
                    trainingString = trainingIndex.ToString();
                }
                else
                {
                    trainingString = "-1";
                }

                string markerString = "ssvep," + _selectableSPOs.Count.ToString() + "," + trainingString + "," +
                                      windowLength.ToString() + freqString;

                // Send the marker
                marker.Write(markerString);

                // Wait the window length + the inter-window interval
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

        protected IEnumerator CueStimulus()
        {
            if (_selectableSPOs.Count > 0)
            {
                int randomIndex = UnityEngine.Random.Range(0, _selectableSPOs.Count);
                _selectableSPOs[randomIndex].Cue();
                yield return new WaitForSecondsRealtime(1f);
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
            //Set StimulusRunning to false to prevent markers from being sent before the stimulus starts
            StimulusRunning = false;

            //flash the stimulus to look at to cue the user
            yield return CueStimulus();

            // Set the stimulus type from the option chosen in the inspector
            SetStimType();

            //Set StimulusRunning to true and call the coroutine to send markers
            StimulusRunning = true;
            StartCoroutine(SendMarkers());

            //this currently displays the 2 stimuli for 10 seconds
            //want it to display until a prediction is made and sent back by python
            for(var flash = 0; flash <100*10; flash++) 
                {
                    yield return OnStimulusRunBehavior();
                }

            //Since not all stimuli flash an even number of times in 10 seconds, some end up with the 'flashOnColor" showing at the end of the 10 seconds
            TurnStimuliBlack();
                
            //Set StimulusRunning to false and stop the coroutine to send markers
            StimulusRunning = false;
            StopCoroutine(SendMarkers());

            _displayText.text = "Stimulus Complete";
            yield return new WaitForSecondsRealtime(2f);
            _displayText.text = "Next Stim";
            yield return new WaitForSecondsRealtime(2f);
            _displayText.text = " ";
            
            //flash the stimulus to look at to cue the user
            yield return CueStimulus();

            // Set the stimulus type from the option chosen in the inspector
            SetStimType();

            //Set StimulusRunning to true and call the coroutine to send markers
            StimulusRunning = true;
            StartCoroutine(SendMarkers());

            //this currently displays the 2 stimuli for 10 seconds
            //want it to display until a prediction is made and sent back by python
            for(var flash = 0; flash <100*10; flash++) 
                {
                    yield return OnStimulusRunBehavior();
                }

            //Since not all stimuli flash an even number of times in 10 seconds, some end up with the 'flashOnColor" showing at the end of the 10 seconds
            TurnStimuliBlack();
                
            //Set StimulusRunning to false and stop the coroutine to send markers
            StimulusRunning = false;
            StopCoroutine(SendMarkers());

            _displayText.text = "Stimulus Complete";
            yield return new WaitForSecondsRealtime(2f);
            _displayText.text = "Done";

           
            //display next Q 5 seconds
            //display T/F
            //Flash and wait for prediction
            //Repeat for 10 questions
        }

        private void SetStimType()
        {
            ColorFlashEffect3 spoEffect;

            if (_stimulusType == StimulusType.BW)
            {
                foreach (var spo in _selectableSPOs)
                {
                    spoEffect = spo.GetComponent<ColorFlashEffect3>();
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.White);
                }
            }
            else
            {
                foreach (var spo in _selectableSPOs)
                {
                    spoEffect = spo.GetComponent<ColorFlashEffect3>();
                    if (_contrastLevel == ContrastLevel.Contrast1)
                    {
                        spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast1);
                    }
                    else if (_contrastLevel == ContrastLevel.Contrast2)
                    {
                        spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast2);
                    }
                    else if (_contrastLevel == ContrastLevel.Contrast3)
                    {
                        spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast3);
                    }
                    else if (_contrastLevel == ContrastLevel.Contrast4)
                    {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast4);
                    }
                
                    if (_size == Size.Size1)
                    {
                        spoEffect.SetSize(ColorFlashEffect3.Size.Size1);
                    }
                    else if (_size == Size.Size2)
                    {
                        spoEffect.SetSize(ColorFlashEffect3.Size.Size2);
                    }
                    else if (_size == Size.Size3)
                    {
                        spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                    }

                    Debug.Log("Contrast Level: " + _contrastLevel);
                    Debug.Log("Size: " + _size);
                }
            }
        }

        private void TurnStimuliBlack()
        {
            ColorFlashEffect3 spoEffect;

            foreach (var spo in _selectableSPOs)
            {
                spoEffect = spo.GetComponent<ColorFlashEffect3>();
                spoEffect.SetBlack(); 
            }
        }
    }
}
