using System.Collections;
using UnityEngine;
using System;
using BCIEssentials.Controllers;
using BCIEssentials.StimulusEffects;
using UnityEngine.UI;
using System.Collections.Generic;
using BCIEssentials.LSLFramework;

namespace BCIEssentials.ControllerBehaviors
{
    public class SSVEPControllerBehavior : BCIControllerBehavior
    {
        public override BCIBehaviorType BehaviorType => BCIBehaviorType.SSVEP;
        
        [FoldoutGroup("Stimulus Frequencies")]
        [SerializeField]
        [Tooltip("User-defined set of target stimulus frequencies [Hz]")]
        private float[] requestedFlashingFrequencies;
        [SerializeField, EndFoldoutGroup, InspectorReadOnly]
        [Tooltip("Calculated best-match achievable frequencies based on the application framerate [Hz]")]
        private float[] realFlashingFrequencies;

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

            realFlashingFrequencies = new float[_selectableSPOs.Count];

            for (int i = 0; i < _selectableSPOs.Count; i++)
            {
                frames_on[i] = 0;
                frame_count[i] = 0;
                period = targetFrameRate / requestedFlashingFrequencies[i];
                // could add duty cycle selection here, but for now we will just get a duty cycle as close to 0.5 as possible
                frame_off_count[i] = (int)Math.Ceiling(period / 2);
                frame_on_count[i] = (int)Math.Floor(period / 2);
                realFlashingFrequencies[i] = (targetFrameRate / (float)(frame_off_count[i] + frame_on_count[i]));
            }
        }

        protected override IEnumerator SendMarkers(int trainingIndex = 99)
        {
            int cue = GetCueStimulus();
            // Make the marker string, this will change based on the paradigm
            while (StimulusRunning)
            {
                
                // Send the marker
                OutStream.PushSSVEPMarker(_selectableSPOs.Count, windowLength, realFlashingFrequencies, cue);

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

        protected int GetCueStimulus()
        {
            int cuedIndex = -1000;
            if (_selectableSPOs.Count > 0)
            {
                cuedIndex = UnityEngine.Random.Range(0, _selectableSPOs.Count);
                Debug.Log($"Cued index: {cuedIndex}");
            }
            return cuedIndex;
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
            //Arbitrarily do 5 runs
            for (int i = 0; i < 5; i++)
            {
                //Set StimulusRunning to false to prevent markers from being sent before the stimulus starts
                StimulusRunning = false;

                //Flash the stimulus to look at to cue the user
                //yield return CueStimulus();

                //Set the stimulus type from the option chosen in the inspector
                SetStimType();

                //Added this so the marker sends every time
                OutStream.PushTrialStartedMarker();

                //Set StimulusRunning to true and call the coroutine to send markers
                StimulusRunning = true;
                StartCoroutine(SendMarkers());

                //This currently displays the 2 stimuli for 10 seconds
                //Want it to display until a prediction is made and sent back by python
                for(var flash = 0; flash <100*5; flash++) 
                {
                    yield return OnStimulusRunBehavior();
                }

                //Set StimulusRunning to false and stop the coroutine to send markers
                StimulusRunning = false;
                StopCoroutine(SendMarkers());
                StopStimulusRun();
                yield return OnStimulusRunComplete();

                //Display text for the user after every run except the last one
                if (i < 4)
                {
                    yield return new WaitForSecondsRealtime(2f); //this is enough to to see feedback
                    _displayText.text = "Stimulus Complete";
                    yield return new WaitForSecondsRealtime(2f);
                    _displayText.text = "Next Stim";
                    yield return new WaitForSecondsRealtime(2f);
                    _displayText.text = " ";
                }
            
                LastSelectedSPO = null;
            }       
            _displayText.text = "Done";
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
                }
            }
        }
    }
}
