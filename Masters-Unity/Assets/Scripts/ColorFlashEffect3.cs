using System.Collections;
using UnityEngine;
using BCIEssentials.Utilities;
using System.Collections.Generic;

namespace BCIEssentials.StimulusEffects
{
    public class ColorFlashEffect3 : StimulusEffect
    {
        [SerializeField]
        [Tooltip("The renderer to assign the material color to")]
        public Renderer _renderer;

        [SerializeField]
        public Material materialList;

        [Header("Flash Settings")]
        private Color _flashOnColor = Color.red;
        private Color _flashOffColor = Color.black;
        
        private float _flashDurationSeconds = 0.2f;

        private int _flashAmount = 3;

        public bool IsPlaying => _effectRoutine != null;

        private Coroutine _effectRoutine;

        public enum ContrastLevel
        {
            Contrast1,
            Contrast2,
            Contrast3,
            Contrast4
        }

        public enum Size
        {
            Size1,
            Size2,
            Size3,
        }

        public ContrastLevel _contrastLevel;
        public Size _size;
    
        private void Start()
        {
            if (_renderer == null && !gameObject.TryGetComponent(out _renderer))
            {
                Debug.LogWarning($"No Renderer component found for {gameObject.name}");
                return;
            }

            if (_renderer.material == null)
            {
                Debug.LogWarning($"No material assigned to renderer component on {gameObject.name}.");
            }

            _renderer.material = materialList;
            AssignMaterialColor(_flashOffColor);
        }

        public override void SetOn()
        {
            if (_renderer == null || _renderer.material == null)
                return;

            AssignMaterialColor(_flashOnColor);
            IsOn = true;
        }

        public override void SetOff()
        {
            if (_renderer == null || _renderer.material == null)
                return;
            
            AssignMaterialColor(_flashOffColor);
            IsOn = false;
        }

        public void Play()
        {
            Stop();
            _effectRoutine = StartCoroutine(RunEffect());
        }

        public void Stop()
        {
            if (!IsPlaying)
                return;

            SetOff();
            StopCoroutine(_effectRoutine);
            _effectRoutine = null;
        }

        private IEnumerator RunEffect()
        {
            if (_renderer != null && _renderer.material != null)
            {
                IsOn = true;
                
                for (var i = 0; i < _flashAmount; i++)
                {
                    AssignMaterialColor(_flashOnColor);
                    yield return new WaitForSecondsRealtime(_flashDurationSeconds);

                    AssignMaterialColor(_flashOffColor);
                    yield return new WaitForSecondsRealtime(_flashDurationSeconds);
                }
            }

            SetOff();
            _effectRoutine = null;
        }

/// <summary>
/// //////////Helper methods
/// </summary>
        private void ContrastController()
        {
            ColorContrast colorContrast = GetComponent<ColorContrast>();
            int contrastIntValue = ConvertContrastLevel(_contrastLevel);
            colorContrast.SetContrast(contrastIntValue);
            _flashOnColor = colorContrast.Grey();
        }

        private void SizeController()
        {
            Vector3 newSize = Vector3.one; // Default size is (1, 1, 1)

            switch (_size)
            {
                case Size.Size1:
                    newSize = new Vector3(157.01093974f / 100f, 157.01093974f / 100f, 157.01093974f / 100f); // Scaled by 100 to match "Reference Pixels per Unit" setting
                    // Debugging the calculated physical size in cm
                    float pixels = 157.01093974f;
                    float sizeInInches = pixels / 108.7855392633234f ; // DPI is 108.7855392633234  (calculated for 27" 2560 x 1440 resolution monitor)
                    float sizeInCm = sizeInInches * 2.54f;
                    Debug.Log($"Size in physical units: {sizeInCm} cm"); // Should print approximately 3.666 cm
                    break;

                case Size.Size2:
                    newSize = new Vector3(261.7919719485197f / 100f, 261.7919719485197f / 100f, 261.7919719485197f / 100f); // Scaled by 100 to match "Reference Pixels per Unit" setting
                    // Debugging the calculated physical size in cm
                    float pixels2 = 261.7919719485197f;
                    float sizeInInches2 = pixels2 / 108.7855392633234f ; // DPI is 108.7855392633234  (calculated for 27" 2560 x 1440 resolution monitor)
                    float sizeInCm2 = sizeInInches2 * 2.54f;
                    Debug.Log($"Size in physical units: {sizeInCm2} cm"); // Should print approximately 6.1125 cm
                    break;

                case Size.Size3:
                    newSize = new Vector3(366.7357541789091f / 100f, 366.7357541789091f / 100f, 366.7357541789091f / 100f); // Scaled by 100 to match "Reference Pixels per Unit" setting
                    // Debugging the calculated physical size in cm
                    float pixels3 = 366.7357541789091f;
                    float sizeInInches3 = pixels3 / 108.7855392633234f ; // DPI is 108.7855392633234  (calculated for 27" 2560 x 1440 resolution monitor)
                    float sizeInCm3 = sizeInInches3 * 2.54f;
                    Debug.Log($"Size in physical units: {sizeInCm3} cm"); // Should print approximately 8.5628 cm
                    break;
            }

            // Apply the size change
            transform.localScale = newSize;
        }


        public int ConvertContrastLevel(ContrastLevel _contrastLevel)
        {
            if(_contrastLevel == ContrastLevel.Contrast1)
                return 100;
            else if (_contrastLevel == ContrastLevel.Contrast2)
                return 50;
            else if (_contrastLevel == ContrastLevel.Contrast3)
                return 25;
            else if (_contrastLevel == ContrastLevel.Contrast4)
                return 10;
            else return 0;
        }

        public void AssignMaterialColor(Color color)
        {
            _renderer.material.SetColor("_color", color);
        }

        public void SetContrast(ContrastLevel x)
        {
            _contrastLevel = x;
            ContrastController();
        }

        public void SetSize(Size x)
        {
            _size = x;
            SizeController();
        }
    }
 }