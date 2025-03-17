using System.Collections;
using UnityEngine;
using System;
using BCIEssentials.Controllers;
using BCIEssentials.StimulusEffects;
using UnityEngine.UI;
using System.Collections.Generic;

namespace BCIEssentials.ControllerBehaviors
{
    public class SSVEPControllerBehavior : BCIControllerBehavior
    {
        public override BCIBehaviorType BehaviorType => BCIBehaviorType.SSVEP;

        [SerializeField] private float[] setFreqFlash;
        [SerializeField] private float[] realFreqFlash;

        [SerializeField] private enum StimulusType
        {
            BW,
            Custom
        };

        [SerializeField] private StimulusType stimulusType;

        private int[] frames_on = new int[99];
        private int[] frame_count = new int[99];
        private float period;
        private int[] frame_off_count = new int[99];
        private int[] frame_on_count = new int[99];

        public Text _displayText;

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
                // could add duty cycle selection here, but for now we will just get a duty cycle as close to 0.5 as possible
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
            // Set the stimulus type from the option chosen in the inspector
            SetStimType();
            
            _displayText.text = "Running Stimulus";


            //this currently displays the 2 stimuli for 10 seconds
            //want it to display until a prediction is made and sent back by python
            for(var flash = 0; flash <100*10; flash++) 
                {
                    yield return OnStimulusRunBehavior();
                }
                
            _displayText.text = "Stimulus Complete";
        }

        private void SetStimType()
        {
            ColorFlashEffect3 spoEffect;

            if (stimulusType == StimulusType.BW)
            {
                foreach (var spo in _selectableSPOs)
                {
                    spoEffect = spo.GetComponent<ColorFlashEffect3>();
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.White);
                }

            }
        }

    }
}
