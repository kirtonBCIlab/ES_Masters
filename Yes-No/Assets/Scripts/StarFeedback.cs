using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace BCIEssentials.Utilities
{
    public class StarFeedback : MonoBehaviour 
    {
        Color starOnColor = new Color(255f / 255f, 205f / 255f, 40f / 255f, 255f / 255f);
        Color starOffColor = new Color(255, 205, 40, 0);
        public Renderer _renderer;
        void Start()
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
            
            _renderer.material.SetColor("_Color", starOffColor);
        }

        public void ShowFeedback()
        {
            _renderer.material.SetColor("_Color", starOnColor);
            Invoke("TurnOffFeedback", 0.5f);
        }

        void TurnOffFeedback()
        {
            _renderer.material.SetColor("_Color", starOffColor);
        }
    }

}