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

        public enum StimulusType { BW, Custom }

        public enum ContrastLevel { Contrast1, Contrast2, Contrast3, Contrast4 }

        public enum Size { Size1, Size2, Size3 }

        [Header("Stimulus Parameters")]
        [SerializeField] public StimulusType _stimulusType;
        public ContrastLevel _contrastLevel;
        public Size _size;

        [Header("Text Object")]
        [SerializeField] public Text _displayText;

        public int cuedIndex = -1000;

        // New: Balanced cue tracking
        private List<int> randomizedCueOrder;
        private int cueIndexCounter = 0;

        protected override void Start()
        {
            base.Start();
            PopulateObjectList();
            GenerateBalancedRandomCueOrder();
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
                frame_off_count[i] = (int)Math.Ceiling(period / 2);
                frame_on_count[i] = (int)Math.Floor(period / 2);
                realFlashingFrequencies[i] = (targetFrameRate / (float)(frame_off_count[i] + frame_on_count[i]));
            }
        }

        private void GenerateBalancedRandomCueOrder()
        {
            randomizedCueOrder = new List<int>();

            // Add each of the 4 stimuli 8 times (32 trials)
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    randomizedCueOrder.Add(i);
                }
            }

            // Shuffle the list
            System.Random rng = new System.Random();
            int n = randomizedCueOrder.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                int temp = randomizedCueOrder[k];
                randomizedCueOrder[k] = randomizedCueOrder[n];
                randomizedCueOrder[n] = temp;
            }
        }

        protected int GetCueStimulus()
        {
            if (cueIndexCounter < randomizedCueOrder.Count)
            {
                cuedIndex = randomizedCueOrder[cueIndexCounter];
                cueIndexCounter++;
                OutStream.PushString($"Cued index: {cuedIndex}");
            }
            else
            {
                Debug.LogWarning("Cue index out of bounds.");
                cuedIndex = -1;
            }
            return cuedIndex;
        }

        protected override IEnumerator SendMarkers(int trainingIndex = 99)
        {
            while (StimulusRunning)
            {
                OutStream.PushSSVEPMarker(_selectableSPOs.Count, windowLength, realFlashingFrequencies, -1);
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
            for (int i = 0; i < 32; i++)  // 32 total trials
            {
                StopCoroutineReference(ref _sendMarkers);
                StimulusRunning = false;

                GetCueStimulus();
                SendCue(cuedIndex);
                yield return new WaitForSecondsRealtime(1f);

                SetStimType();
                OutStream.PushTrialStartedMarker();

                StimulusRunning = true;
                Coroutine markerSendingCoroutine = StartCoroutine(SendMarkers());

                for (int flash = 0; flash < 100 * 5; flash++)
                {
                    yield return OnStimulusRunBehavior();
                }

                StimulusRunning = false;
                StopCoroutine(markerSendingCoroutine);
                StopStimulusRun();
                yield return OnStimulusRunComplete();

                if (i < 31)
                {
                    yield return new WaitForSecondsRealtime(4f);
                    _displayText.text = "Next Trial";
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

            foreach (var spo in _selectableSPOs)
            {
                spoEffect = spo.GetComponent<ColorFlashEffect3>();
                if (_stimulusType == StimulusType.BW)
                {
                    spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.White);
                    spoEffect.SetSize(ColorFlashEffect3.Size.Size3);
                }
                else
                {
                    switch (_contrastLevel)
                    {
                        case ContrastLevel.Contrast1: spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast1); break;
                        case ContrastLevel.Contrast2: spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast2); break;
                        case ContrastLevel.Contrast3: spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast3); break;
                        case ContrastLevel.Contrast4: spoEffect.SetContrast(ColorFlashEffect3.ContrastLevel.Contrast4); break;
                    }

                    switch (_size)
                    {
                        case Size.Size1: spoEffect.SetSize(ColorFlashEffect3.Size.Size1); break;
                        case Size.Size2: spoEffect.SetSize(ColorFlashEffect3.Size.Size2); break;
                        case Size.Size3: spoEffect.SetSize(ColorFlashEffect3.Size.Size3); break;
                    }
                }
            }
        }

        private void SendCue(int index)
        {
            if (index >= 0 && index < _selectableSPOs.Count)
            {
                var spoEffect = _selectableSPOs[index].GetComponent<ColorFlashEffect3>();
                spoEffect.CueColorChange();
            }
            else
            {
                Debug.LogWarning("Invalid index for cue stimulus.");
            }
        }
    }
}
