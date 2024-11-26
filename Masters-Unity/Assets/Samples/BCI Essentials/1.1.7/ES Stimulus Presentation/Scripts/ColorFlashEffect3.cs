using System.Collections;
using UnityEngine;
using BCIEssentials.Utilities;
using System.Collections.Generic;

namespace BCIEssentials.StimulusEffects
{
      /// <summary>
    /// Assign or Flash a renderers material color.
    /// </summary>
    public class ColorFlashEffect3 : StimulusEffect
    {
        [SerializeField]
        [Tooltip("The renderer to assign the material color to")]
        public Renderer _renderer;

        [SerializeField]
        public Material[] materialList;

        [Header("Flash Settings")]
        private Color _flashOnColor = Color.red;
        private Color _flashOffColor = Color.black;
        
        private float _flashDurationSeconds = 0.2f;

        private int _flashAmount = 3;

        public bool IsPlaying => _effectRoutine != null;

        private Coroutine _effectRoutine;

        public enum ContrastLevel
        {
            Max,
            OneStepDown,
            TwoStepsDown,
            Min
        }
        public ContrastLevel _contrastLevel;
        public Material[] setMaterials;

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

            _renderer.material = materialList[6];
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
            //_flashOffColor = Color.black;
            ColorContrast colorContrast = GetComponent<ColorContrast>();
            int contrastIntValue = ConvertContrastLevel(_contrastLevel);
            colorContrast.SetContrast(contrastIntValue);
            _flashOnColor = colorContrast.Grey();
        }

        public int ConvertContrastLevel(ContrastLevel _contrastLevel)
        {
            if(_contrastLevel == ContrastLevel.Max)
                return 100;
            else if (_contrastLevel == ContrastLevel.OneStepDown)
                return 50;
            else if (_contrastLevel == ContrastLevel.TwoStepsDown)
                return 25;
            else if (_contrastLevel == ContrastLevel.Min)
                return 10;
            else return 0;
        }


        private void AssignMaterialColor(Color color)
        {
            _renderer.material.color = color;
        }

        public void SetContrast(ContrastLevel x)
        {
            _contrastLevel = x;
            ContrastController();
            if(setMaterials[0] != null)
            {
                setMaterials[0] = materialList[6]; 
                setMaterials[1] = null;
                _renderer.materials = setMaterials;
                _flashOffColor = Color.black;
            }
        }
    }
 }