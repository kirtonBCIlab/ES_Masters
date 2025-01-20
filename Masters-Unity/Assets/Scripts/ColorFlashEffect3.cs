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
                    newSize = new Vector3(150 / 325f, 150f / 325f, 150f / 325f); // this scaling gives stim of 3.6cm on the screen
                    break;

                case Size.Size2:
                    newSize = new Vector3(255f / 325f, 255f / 325f, 255f / 325f); // this scaling gives stim of 6.1cm on the screen
                    break;

                case Size.Size3:
                    newSize = new Vector3(355f / 325f, 355f / 325f, 355f / 325f); // this scaling gives stim of 8.5cm on the screen
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